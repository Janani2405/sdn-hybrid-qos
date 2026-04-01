import eventlet
eventlet.monkey_patch()

"""
=============================================================================
qos_controller.py — SDN QoS Controller (MERGED: Visualization + Data Collection)
=============================================================================
Combines port_stats_monitor_v2.py + qos_controller.py into ONE file:

  [1] DATA COLLECTION  (exact logic from port_stats_monitor_v2.py)
        → logs/qos_log.csv         every 2s  — LSTM training data (expanded: 25 features)
        → logs/congestion_log.csv  per new episode — congestion labels
        → [CONGESTION DETECTED] console banner on every new episode
        → [CONGESTION ONGOING]  on sustained congestion (no counter inflation)
        → Full debounce: episode counter increments once per episode, not per tick
        → Counter-wrap / OVS-restart guard on all deltas

  [2] REAL LATENCY / JITTER MEASUREMENT  (replaces random.uniform stubs)
        → LLDP echo probes sent every LLDP_PROBE_INTERVAL seconds per link
        → Round-trip time split by 2 → one-way latency per directed link
        → Per-link jitter = exponential moving average of |RTT_n − RTT_{n-1}|
        → Latency/jitter aggregated per switch (mean across its links)
        → loss_pct derived from real rx_dropped / rx_bytes counters
        → All values feed compute_reward() — no random numbers remain

  [3] VISUALIZATION    (exact REST API from qos_controller.py)
        → http://127.0.0.1:8080/qos/api/v1/health
        → http://127.0.0.1:8080/qos/api/v1/metrics/latest
        → http://127.0.0.1:8080/qos/api/v1/topology
        → http://127.0.0.1:8080/qos/api/v1/flows
        → http://127.0.0.1:8080/qos/api/v1/ports
        → http://127.0.0.1:8080/qos/api/v1/events
        → http://127.0.0.1:8080/qos/api/v1/hosts
        → http://127.0.0.1:8080/qos/api/v1/congestion
        → http://127.0.0.1:8080/qos/api/v1/latency   (NEW — per-link RTT)
        → Powers dashboard.html live charts and topology view

Run:
    source ~/ryu-env/bin/activate
    cd ~/sdn-project
    ryu-manager controller/qos_controller.py --observe-links --ofp-tcp-listen-port 6633

Dashboard:
    Open docs/dashboard.html in browser

HOW REAL LATENCY WORKS
=======================
Ryu's --observe-links flag makes the topology module send periodic LLDP
packets between every pair of adjacent switches.  Each LLDP packet carries
the sending switch's DPID and port number inside its TLVs.

We intercept those packets in _packet_in_handler BEFORE the L2 learning
logic and record send_time = time.time() keyed by (src_dpid, src_port).

When the same LLDP arrives at the neighbouring switch it is forwarded to the
controller as a normal PacketIn.  We detect it here, look up the stored
send_time, and compute:

    RTT  = now − send_time          (round-trip, controller→sw_A→link→sw_B→controller)
    OWD  ≈ RTT / 2                  (one-way delay approximation)

Jitter is tracked per directed link as an EWMA of successive OWD differences:

    jitter_n = α * |OWD_n − OWD_{n-1}| + (1−α) * jitter_{n-1}   α = 0.2

This gives a per-link, continuously updated, real jitter estimate with no
synthetic noise whatsoever.

LOSS CALCULATION
================
loss_pct per port  = (delta_rx_dropped / max(delta_rx_packets, 1)) * 100
where delta_rx_packets is derived from (delta_rx_bytes / avg_frame_bytes).
Because OVS does not expose rx_packet counters in port stats on all versions
we use a conservative 1500-byte MTU estimate for the denominator.
Per-switch loss_pct is then the mean across all its ports.
=============================================================================
"""

import os
import csv
import json
import time
import struct
import logging
import threading
import collections
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, lldp
from ryu.lib.packet import ether_types
from ryu.lib import hub, dpid as dpid_lib
from ryu.topology import event as topo_event
from ryu.topology.api import get_switch, get_link
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

# ── Module II LSTM Predictor ─────────────────────────────────────────────────
# Requirements before this activates:
#   1. touch ~/sdn-project/module2/__init__.py
#   2. python3 module2/train.py  (produces best_lstm.pt)
#   3. Restart controller with: PYTHONPATH=. ryu-manager controller/qos_controller.py
# Until trained, controller runs normally — predictor is simply None.
try:
    import sys as _sys
    # Add sdn-project/ root to sys.path so 'module2' resolves as a package
    _controller_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root   = os.path.dirname(_controller_dir)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    from module2.lstm_predictor import LSTMPredictor
    _LSTM_AVAILABLE = True
except ImportError as _lstm_import_err:
    _LSTM_AVAILABLE = False
    # To debug: print(f'[LSTM import] {_lstm_import_err}')


# ─────────────────────────────────────────────────────────────────
#  Tunables
# ─────────────────────────────────────────────────────────────────
POLL_INTERVAL          = 2        # seconds between PortStatsRequest per switch
LLDP_PROBE_INTERVAL    = 1        # seconds between latency probe sends
LLDP_PROBE_TIMEOUT     = 5        # seconds before a probe sample is discarded
JITTER_ALPHA           = 0.2      # EWMA smoothing factor for jitter
AVG_FRAME_BYTES        = 1000     # conservative estimate for loss_pct denominator
LINK_CAPACITY_MBPS     = 100.0    # assumed link capacity (matches TCLink bw=100)
UTIL_CONGESTION_THRESH = 95.0     # utilisation % that triggers signal_util
WARN_UTIL_THRESH       = 70.0     # utilisation % for zone_label = 'warning'
ROLLING_WINDOW         = 5        # number of past samples for rolling features
LOG_DIR                = 'logs'
QOS_CSV_FILE           = os.path.join(LOG_DIR, 'qos_log.csv')
CONG_CSV_FILE          = os.path.join(LOG_DIR, 'congestion_log.csv')

# OVS internal/reserved port numbers — always skipped
SKIP_PORTS = {0xfffffffe, 0xffffffff}   # OFPP_LOCAL, OFPP_NONE

# LLDP multicast destination used by Ryu topology module
LLDP_MAC_NEAREST_BRIDGE = '01:80:c2:00:00:0e'

# REST API prefix
REST_API_URL = '/qos/api/v1'


# ─────────────────────────────────────────────────────────────────
#  CSV column definitions
#
#  qos_log.csv — 25 features + 1 label, written every POLL_INTERVAL
#  ─────────────────────────────────────────────────────────────────
#  Group A  — Identity / timing
#    timestamp, dpid, port_no
#
#  Group B  — Raw counters (absolute)
#    tx_bytes, rx_bytes, tx_dropped, rx_dropped
#
#  Group C  — Per-interval throughput & utilisation (derived)
#    tx_mbps, rx_mbps, utilization_pct, loss_pct
#    tx_pps   — estimated TX packets/s  (delta_tx / AVG_FRAME_BYTES / delta_t)
#    rx_pps   — estimated RX packets/s  (delta_rx / AVG_FRAME_BYTES / delta_t)
#    bw_headroom_mbps — remaining capacity before saturation
#
#  Group D  — Drop deltas (per-interval)
#    delta_tx_dropped, delta_rx_dropped
#
#  Group E  — Latency / jitter (from LLDP probes, port-level)
#    latency_ms, jitter_ms, rtt_ms
#
#  Group F  — Rolling / trend features (last ROLLING_WINDOW samples)
#    rolling_util_mean   — smoothed utilisation trend
#    rolling_drop_sum    — accumulated drop pressure
#    rolling_tx_mean     — smoothed TX throughput
#    rolling_rx_mean     — smoothed RX throughput
#
#  Group G  — Topology context
#    n_active_flows      — flow table size on this switch
#    neighbor_util_max   — highest util_pct among directly linked switches
#    inter_arrival_delta — seconds since last congestion event on this port
#
#  Group H  — Signals & label
#    signal_util, signal_drop
#    zone_label          — 'normal' | 'warning' | 'congested' | 'critical'
#    congested           — binary label (0 / 1)
# ─────────────────────────────────────────────────────────────────
QOS_COLUMNS = [
    # A — identity
    'timestamp', 'dpid', 'port_no',
    # B — raw counters
    'tx_bytes', 'rx_bytes', 'tx_dropped', 'rx_dropped',
    # C — throughput / utilisation
    'tx_mbps', 'rx_mbps', 'utilization_pct', 'loss_pct',
    'tx_pps', 'rx_pps', 'bw_headroom_mbps',
    # D — drop deltas
    'delta_tx_dropped', 'delta_rx_dropped',
    # E — latency / jitter
    'latency_ms', 'jitter_ms', 'rtt_ms',
    # F — rolling features
    'rolling_util_mean', 'rolling_drop_sum',
    'rolling_tx_mean', 'rolling_rx_mean',
    # G — topology context
    'n_active_flows', 'neighbor_util_max', 'inter_arrival_delta',
    # H — signals & label
    'signal_util', 'signal_drop', 'zone_label', 'congested',
]

CONG_COLUMNS = [
    'timestamp', 'dpid', 'port_no',
    'utilization_pct',
    'tx_mbps', 'rx_mbps',
    'delta_tx_dropped', 'delta_rx_dropped',
    'signal_util', 'signal_drop',
    'reason',
    'port_event_count',
    'global_event_count',
]


# ─────────────────────────────────────────────────────────────────
#  Thread-safe CSV logger  (exact from port_stats_monitor_v2.py)
# ─────────────────────────────────────────────────────────────────
class CSVLogger:
    """Append-only, thread-safe CSV writer. Writes header on first write."""

    def __init__(self, path: str, columns: list):
        Path(os.path.dirname(path) or '.').mkdir(parents=True, exist_ok=True)
        self._path           = path
        self._columns        = columns
        self._lock           = threading.Lock()
        self._header_written = (
            os.path.isfile(path) and os.path.getsize(path) > 0
        )

    def write(self, row: dict):
        with self._lock:
            with open(self._path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self._columns,
                                   extrasaction='ignore')
                if not self._header_written:
                    w.writeheader()
                    self._header_written = True
                w.writerow(row)


# ─────────────────────────────────────────────────────────────────
#  In-memory state stores  (from qos_controller.py — for REST API)
# ─────────────────────────────────────────────────────────────────
port_stats_store = {}
flow_stats_store = {}
switch_store     = {}
link_store       = []
metrics_history  = collections.defaultdict(lambda: collections.deque(maxlen=60))
event_log        = collections.deque(maxlen=100)

# ─────────────────────────────────────────────────────────────────
#  Latency store — populated by LLDP probe logic, read by REST API
#  Key:   (src_dpid_int, src_port_no)
#  Value: { 'latency_ms': float, 'jitter_ms': float,
#            'rtt_ms': float, 'ts': float }
# ─────────────────────────────────────────────────────────────────
latency_store = {}

# ─────────────────────────────────────────────────────────────────
#  Rolling feature buffers
#  Key:   (dpid_int, port_no)
#  Value: deque of dicts  { 'util': float, 'drops': int,
#                            'tx': float, 'rx': float }
#  Length capped at ROLLING_WINDOW so mean/sum are always over the
#  last N samples with zero extra storage.
# ─────────────────────────────────────────────────────────────────
rolling_buffers: dict = {}   # populated on first port-stats sample

# ─────────────────────────────────────────────────────────────────
#  Last-congestion timestamp per port — for inter_arrival_delta
#  Key:   (dpid_int, port_no)
#  Value: float (time.time() of last congestion episode start)
# ─────────────────────────────────────────────────────────────────
last_congestion_ts: dict = {}


# ─────────────────────────────────────────────────────────────────
#  Reward signal  (from qos_controller.py — for DQN Module III)
#  Now called with REAL latency, jitter, loss values.
# ─────────────────────────────────────────────────────────────────
def compute_reward(bw_mbps, latency_ms, loss_pct, jitter_ms):
    MAX_BW     = 100.0
    MAX_LAT    = 200.0
    MAX_JITTER = 50.0
    norm_bw     = min(bw_mbps    / MAX_BW,     1.0)
    norm_lat    = min(latency_ms / MAX_LAT,    1.0)
    norm_loss   = min(loss_pct   / 100.0,      1.0)
    norm_jitter = min(jitter_ms  / MAX_JITTER, 1.0)
    return round(norm_bw - norm_lat - norm_loss - 0.5 * norm_jitter, 4)


# ─────────────────────────────────────────────────────────────────
#  LLDP packet builder
#  Constructs an LLDP frame that carries:
#    • Chassis ID TLV  → src_dpid  (8-byte big-endian)
#    • Port ID TLV     → src_port  (4-byte big-endian)
#    • TTL TLV         → 120 s
#    • send_time TLV   → float encoded as 8-byte big-endian (custom OUI)
#    • End TLV
#
#  The custom send_time TLV uses OUI 0x00_26_E1 (Ryu's assigned OUI) and
#  subtype 0x01 so it survives any LLDP-aware middlebox that passes unknown
#  TLVs.  We encode time.time() as a double (struct 'd', 8 bytes).
# ─────────────────────────────────────────────────────────────────
_RYU_OUI    = b'\x00\x26\xe1'
_TIME_STYPE = b'\x01'
_TIME_TLV_LEN = 4 + 8   # OUI(3) + subtype(1) + time(8) = 12

def _build_lldp_probe(src_dpid: int, src_port: int, send_time: float) -> bytes:
    """
    Build a raw LLDP ethernet frame suitable for OFPPacketOut.

    TLV wire format:  [ type(7b) | length(9b) | value(length bytes) ]
    """
    def tlv(tlv_type: int, value: bytes) -> bytes:
        header = ((tlv_type << 9) | len(value)).to_bytes(2, 'big')
        return header + value

    # Chassis ID TLV  (subtype=4 = locally assigned)
    chassis = tlv(1, b'\x04' + src_dpid.to_bytes(8, 'big'))
    # Port ID TLV     (subtype=2 = port component)
    port_id = tlv(2, b'\x02' + src_port.to_bytes(4, 'big'))
    # TTL TLV
    ttl     = tlv(3, (120).to_bytes(2, 'big'))
    # Custom time TLV (type=127 = org-specific)
    time_val = struct.pack('>d', send_time)
    org_tlv  = tlv(127, _RYU_OUI + _TIME_STYPE + time_val)
    # End TLV
    end      = tlv(0, b'')

    lldp_payload = chassis + port_id + ttl + org_tlv + end

    # Ethernet header: dst=LLDP multicast, src=locally generated (dpid+port)
    src_mac_int = (src_dpid & 0xFFFFFFFFFF00) | (src_port & 0xFF)
    src_mac = ':'.join(f'{(src_mac_int >> (8 * i)) & 0xFF:02x}'
                       for i in reversed(range(6)))
    dst_mac_bytes = bytes.fromhex(LLDP_MAC_NEAREST_BRIDGE.replace(':', ''))
    src_mac_bytes = bytes.fromhex(src_mac.replace(':', ''))
    eth_type      = (0x88CC).to_bytes(2, 'big')

    return dst_mac_bytes + src_mac_bytes + eth_type + lldp_payload


def _parse_lldp_probe(raw: bytes):
    """
    Parse a raw ethernet frame.
    Returns (src_dpid, src_port, send_time) or None if not our probe.

    We look for:
      • ethertype == 0x88CC
      • Chassis ID TLV subtype 4 with 8-byte dpid
      • Port ID TLV subtype 2 with 4-byte port
      • Org-specific TLV with our OUI + subtype + 8-byte time
    """
    if len(raw) < 14:
        return None
    if raw[12:14] != b'\x88\xcc':
        return None

    payload = raw[14:]
    pos = 0
    src_dpid   = None
    src_port   = None
    send_time  = None

    while pos + 2 <= len(payload):
        header   = int.from_bytes(payload[pos:pos+2], 'big')
        tlv_type = (header >> 9) & 0x7F
        tlv_len  = header & 0x1FF
        pos     += 2
        if pos + tlv_len > len(payload):
            break
        value = payload[pos:pos+tlv_len]
        pos  += tlv_len

        if tlv_type == 0:                         # End TLV
            break
        elif tlv_type == 1 and tlv_len == 9:      # Chassis ID (our format)
            if value[0:1] == b'\x04':
                src_dpid = int.from_bytes(value[1:9], 'big')
        elif tlv_type == 2 and tlv_len == 5:      # Port ID (our format)
            if value[0:1] == b'\x02':
                src_port = int.from_bytes(value[1:5], 'big')
        elif tlv_type == 127 and tlv_len == 12:   # Org-specific
            if value[0:3] == _RYU_OUI and value[3:4] == _TIME_STYPE:
                send_time = struct.unpack('>d', value[4:12])[0]

    if src_dpid is not None and src_port is not None and send_time is not None:
        return src_dpid, src_port, send_time
    return None


# =============================================================================
#  Main Controller App
# =============================================================================
class QoSController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS    = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_logger()

        # ── Forwarding table ──────────────────────────────────────────────
        self.mac_to_port = {}           # { dpid -> { mac -> port } }

        # ── Switch registry ───────────────────────────────────────────────
        self._datapaths = {}            # dpid -> datapath
        self._dp_lock   = threading.Lock()

        # ── Per-port previous-sample state (for CSV delta calc) ───────────
        self._prev      = {}            # { (dpid, port_no) -> prev dict }
        self._prev_lock = threading.Lock()

        # ── Congestion debounce state  (exact from port_stats_monitor_v2) ─
        self._was_congested    = defaultdict(bool)  # (dpid, port_no) -> bool
        self._port_event_count = defaultdict(int)   # (dpid, port_no) -> int
        self._total_events     = 0
        self._counter_lock     = threading.Lock()

        # ── CSV loggers ───────────────────────────────────────────────────
        self._qos_log  = CSVLogger(QOS_CSV_FILE,  QOS_COLUMNS)
        self._cong_log = CSVLogger(CONG_CSV_FILE, CONG_COLUMNS)

        # ── Module II LSTM Predictor ──────────────────────────────────────
        self.predictor = None
        if _LSTM_AVAILABLE:
            try:
                self.predictor = LSTMPredictor(
                    ckpt_path    = 'module2/checkpoints/best_lstm.pt',
                    scaler_path  = 'module2/processed/scaler.pkl',
                    feature_path = 'module2/processed/feature_names.txt',
                )
                self.predictor.load()
                self.logger.info(
                    'LSTM Predictor loaded — Module II active '
                    '(warmup: first prediction after 10 polling cycles per port)'
                )
            except FileNotFoundError as e:
                self.predictor = None
                self.logger.warning(
                    'LSTM Predictor: checkpoint not found (%s). '
                    'Run python3 module2/train.py first, then restart.', e
                )
            except Exception as e:
                self.predictor = None
                self.logger.error('LSTM Predictor load error: %s', e)
        else:
            self.logger.warning(
                'LSTM module not importable. Check: '
                '(1) module2/__init__.py exists, '
                '(2) run with PYTHONPATH=. ryu-manager ...'
            )

        # ── LLDP probe tracking ───────────────────────────────────────────
        #
        # _probe_send_times:
        #   key   = (src_dpid, src_port)
        #   value = send_time (float, time.time())
        #   Protected by _probe_lock.  Entries older than LLDP_PROBE_TIMEOUT
        #   are discarded so stale data never poisons the latency estimates.
        #
        # _link_latency:
        #   key   = (src_dpid, src_port)   ← the *sending* side of the link
        #   value = { 'owd_ms':     float  ← latest one-way delay
        #             'jitter_ms':  float  ← EWMA jitter
        #             'rtt_ms':     float  ← latest RTT
        #             'prev_owd':   float  ← previous OWD for jitter calc
        #             'ts':         float  ← time of last update }
        #
        self._probe_send_times = {}
        self._link_latency     = {}
        self._probe_lock       = threading.Lock()

        # ── Per-switch loss accumulators ──────────────────────────────────
        #   Filled during _port_stats_reply, read during _aggregate_switch_metrics
        #   key   = dpid (int)
        #   value = list of per-port loss_pct values from latest poll cycle
        self._port_loss = defaultdict(list)   # dpid -> [loss_pct, ...]
        self._loss_lock = threading.Lock()

        # ── Start background threads ──────────────────────────────────────
        self._poll_thread  = hub.spawn(self._poll_loop)
        self._probe_thread = hub.spawn(self._lldp_probe_loop)

        # ── Register REST API ─────────────────────────────────────────────
        wsgi = kwargs['wsgi']
        wsgi.register(QoSRestAPI, {'controller': self})

        self._log_event("controller",
            "QoS Controller started (merged: data + real latency + visualization)",
            "info")
        self.logger.info('QoS log        → %s', os.path.abspath(QOS_CSV_FILE))
        self.logger.info('Congestion log → %s', os.path.abspath(CONG_CSV_FILE))
        self.logger.info('REST API       → http://127.0.0.1:8080%s', REST_API_URL)
        self.logger.info(
            'Config: poll=%ds  lldp_probe=%ds  capacity=%.0f Mbps  '
            'util_threshold=%.0f%%',
            POLL_INTERVAL, LLDP_PROBE_INTERVAL,
            LINK_CAPACITY_MBPS, UTIL_CONGESTION_THRESH
        )

    def _configure_logger(self):
        self.logger.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.StreamHandler)
                   for h in self.logger.handlers):
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                '%(asctime)s  %(levelname)-8s  [QoSController]  %(message)s',
                datefmt='%H:%M:%S',
            ))
            self.logger.addHandler(h)

    # =========================================================================
    #  Switch connect / disconnect
    # =========================================================================

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _features_handler(self, ev):
        dp     = ev.msg.datapath
        dpid   = dp.id
        ofp    = dp.ofproto
        parser = dp.ofproto_parser

        self.logger.info('Switch CONNECTED  dpid=0x%016x', dpid)

        with self._dp_lock:
            self._datapaths[dpid] = dp

        switch_store[dpid] = {
            'dpid':         dpid_lib.dpid_to_str(dpid),
            'connected_at': datetime.now().isoformat(),
            'n_flows':      0,
            'status':       'active',
        }
        self._log_event("switch",
            f"Switch {dpid_lib.dpid_to_str(dpid)} connected", "success")

        # Table-miss rule
        match   = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER,
                                          ofp.OFPCML_NO_BUFFER)]
        inst    = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(parser.OFPFlowMod(
            datapath=dp, priority=0, match=match, instructions=inst,
        ))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        dp   = ev.datapath
        dpid = dp.id

        if ev.state == MAIN_DISPATCHER:
            with self._dp_lock:
                self._datapaths.setdefault(dpid, dp)

        elif ev.state == DEAD_DISPATCHER:
            with self._dp_lock:
                self._datapaths.pop(dpid, None)
            with self._prev_lock:
                for k in [k for k in self._prev if k[0] == dpid]:
                    del self._prev[k]
            if dpid in switch_store:
                switch_store[dpid]['status'] = 'disconnected'
            self.logger.warning('Switch DISCONNECTED  dpid=0x%016x', dpid)
            self._log_event("switch",
                f"Switch {dpid_lib.dpid_to_str(dpid)} disconnected", "warning")

    # =========================================================================
    #  Packet-in  (L2 learning switch + LLDP probe interception)
    # =========================================================================

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg      = ev.msg
        datapath = msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        in_port  = msg.match['in_port']
        raw_data = msg.data

        # ── LLDP probe interception ───────────────────────────────────────
        #
        # Try to parse our custom probe BEFORE the normal Ryu LLDP filter.
        # If it matches, record the RTT and return immediately — do NOT flood
        # or learn this fake MAC into the forwarding table.
        #
        parsed = _parse_lldp_probe(raw_data)
        if parsed is not None:
            src_dpid, src_port, send_time = parsed
            now_time = time.time()
            age      = now_time - send_time

            if age <= LLDP_PROBE_TIMEOUT:
                rtt_ms = age * 1000.0
                owd_ms = rtt_ms / 2.0
                self._record_latency(src_dpid, src_port, owd_ms, rtt_ms)
                self.logger.debug(
                    'LLDP probe received: src_dpid=0x%016x src_port=%s '
                    'RTT=%.2f ms  OWD=%.2f ms  (arrived at dpid=0x%016x port=%s)',
                    src_dpid, src_port, rtt_ms, owd_ms, datapath.id, in_port
                )
            else:
                self.logger.debug(
                    'LLDP probe STALE (age=%.2fs > %.0fs) — discarded',
                    age, LLDP_PROBE_TIMEOUT
                )
            return   # Do not forward this probe packet

        # ── Normal Ryu LLDP (topology module) — skip ─────────────────────
        pkt = packet.Packet(raw_data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        # ── L2 learning switch  (unchanged from original) ─────────────────
        dst  = eth.dst
        src  = eth.src
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions  = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self._add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self._add_flow(datapath, 1, match, actions)

        data = raw_data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out  = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data
        )
        datapath.send_msg(out)

    # =========================================================================
    #  Flow stats reply  (for REST API flow table)
    # =========================================================================

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        flow_stats_store[dpid] = {}
        n_flows = 0
        for stat in ev.msg.body:
            if stat.priority == 0:
                continue
            n_flows += 1
            key = (f"{stat.match.get('eth_src','?')}"
                   f"->{stat.match.get('eth_dst','?')}")
            flow_stats_store[dpid][key] = {
                'packet_count': stat.packet_count,
                'byte_count':   stat.byte_count,
                'duration_sec': stat.duration_sec,
                'priority':     stat.priority,
                'idle_timeout': stat.idle_timeout,
            }
        if dpid in switch_store:
            switch_store[dpid]['n_flows'] = n_flows

    # =========================================================================
    #  Port stats reply — CORE
    #  Full 5-step pipeline from port_stats_monitor_v2.py
    #  + real loss_pct calculation
    #  + feeds REST API metrics_history for dashboard
    # =========================================================================

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply(self, ev):
        body = ev.msg.body
        dp   = ev.msg.datapath
        dpid = dp.id
        now  = time.time()
        ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        port_stats_store.setdefault(dpid, {})

        # Reset per-switch loss list for this poll cycle
        with self._loss_lock:
            self._port_loss[dpid] = []

        for stat in body:
            port_no = stat.port_no
            if port_no in SKIP_PORTS:
                continue

            tx_bytes   = stat.tx_bytes
            rx_bytes   = stat.rx_bytes
            tx_dropped = stat.tx_dropped
            rx_dropped = stat.rx_dropped
            key        = (dpid, port_no)

            # ── Step 1: Delta computation  (exact from v2) ────────────────
            with self._prev_lock:
                prev = self._prev.get(key)

                if prev is None:
                    self._prev[key] = {
                        'tx': tx_bytes, 'rx': rx_bytes,
                        'tx_d': tx_dropped, 'rx_d': rx_dropped,
                        'ts': now,
                    }
                    self.logger.debug(
                        'port=%s  baseline stored  tx_drop=%s  rx_drop=%s',
                        port_no, tx_dropped, rx_dropped
                    )
                    continue

                delta_t    = now        - prev['ts']
                delta_tx   = tx_bytes   - prev['tx']
                delta_rx   = rx_bytes   - prev['rx']
                delta_tx_d = tx_dropped - prev['tx_d']
                delta_rx_d = rx_dropped - prev['rx_d']

                # Guard: negative deltas = counter wrap or OVS restart
                if delta_t <= 0 or delta_tx < 0 or delta_rx < 0 \
                        or delta_tx_d < 0 or delta_rx_d < 0:
                    self.logger.debug(
                        'port=%s  counter anomaly — resetting baseline', port_no
                    )
                    self._prev[key] = {
                        'tx': tx_bytes, 'rx': rx_bytes,
                        'tx_d': tx_dropped, 'rx_d': rx_dropped,
                        'ts': now,
                    }
                    continue

                self._prev[key] = {
                    'tx': tx_bytes,  'rx': rx_bytes,
                    'tx_d': tx_dropped, 'rx_d': rx_dropped,
                    'ts': now,
                }

            # ── Step 2: Throughput, utilisation, pps, headroom ───────────
            tx_mbps  = (delta_tx * 8) / (delta_t * 1_000_000)
            rx_mbps  = (delta_rx * 8) / (delta_t * 1_000_000)
            peak     = max(tx_mbps, rx_mbps)
            util_pct = min((peak / LINK_CAPACITY_MBPS) * 100.0, 100.0)

            # Estimated packets/s  (Group C)
            tx_pps = (delta_tx / AVG_FRAME_BYTES) / delta_t
            rx_pps = (delta_rx / AVG_FRAME_BYTES) / delta_t

            # Remaining bandwidth before saturation  (Group C)
            bw_headroom_mbps = max(LINK_CAPACITY_MBPS - peak, 0.0)

            # ── Step 2b: Real loss_pct for this port  (Group C) ───────────
            if delta_rx_d == 0:
                port_loss_pct = 0.0
            else:
                est_rx_pkts   = delta_rx / AVG_FRAME_BYTES
                total_offered = est_rx_pkts + delta_rx_d
                port_loss_pct = min(
                    (delta_rx_d / max(total_offered, 1)) * 100.0,
                    100.0
                )

            with self._loss_lock:
                self._port_loss[dpid].append(port_loss_pct)

            # ── Step 2c: Rolling features  (Group F) ──────────────────────
            #
            # Maintain a per-port deque of length ROLLING_WINDOW.
            # Each sample stores the four values needed for all four rolling
            # columns.  We compute mean/sum inline — no pandas dependency.
            #
            if key not in rolling_buffers:
                rolling_buffers[key] = collections.deque(maxlen=ROLLING_WINDOW)

            rolling_buffers[key].append({
                'util':  util_pct,
                'drops': delta_tx_d + delta_rx_d,
                'tx':    tx_mbps,
                'rx':    rx_mbps,
            })
            buf = rolling_buffers[key]
            n   = len(buf)

            rolling_util_mean = sum(s['util']  for s in buf) / n
            rolling_drop_sum  = sum(s['drops'] for s in buf)
            rolling_tx_mean   = sum(s['tx']    for s in buf) / n
            rolling_rx_mean   = sum(s['rx']    for s in buf) / n

            # ── Step 2d: Latency / jitter for this port  (Group E) ────────
            #
            # _link_latency is keyed by (src_dpid, src_port).  We look up
            # the entry whose src_dpid == dpid and src_port == port_no.
            # Falls back to 0.0 if no LLDP probe has been received yet.
            #
            with self._probe_lock:
                link_entry = self._link_latency.get(key)
            if link_entry is not None:
                port_latency_ms = link_entry['owd_ms']
                port_jitter_ms  = link_entry['jitter_ms']
                port_rtt_ms     = link_entry['rtt_ms']
            else:
                port_latency_ms = 0.0
                port_jitter_ms  = 0.0
                port_rtt_ms     = 0.0

            # ── Step 2e: Topology context features  (Group G) ─────────────

            # n_active_flows — from the switch's last flow-stats reply
            n_active_flows = switch_store.get(dpid, {}).get('n_flows', 0)

            # neighbor_util_max — highest utilisation across all switches
            # that share a link with this one (src_dpid == dpid or dst_dpid == dpid)
            neighbor_util_max = self._get_neighbor_util_max(dpid)

            # inter_arrival_delta — seconds since last congestion episode
            # on this specific port.  0.0 if never congested.
            last_cong = last_congestion_ts.get(key)
            inter_arrival_delta = (now - last_cong) if last_cong else 0.0

            # ── Step 3: Dual-signal congestion detection  (exact from v2) ─
            signal_util = util_pct  > UTIL_CONGESTION_THRESH
            signal_drop = (delta_tx_d > 0) or (delta_rx_d > 0)
            congested   = signal_util or signal_drop

            # ── Zone label  (Group H) ─────────────────────────────────────
            #   critical   → util > threshold AND drops present
            #   congested  → either signal fired
            #   warning    → util > WARN_UTIL_THRESH but below threshold
            #   normal     → everything quiet
            if signal_util and signal_drop:
                zone_label = 'critical'
            elif congested:
                zone_label = 'congested'
            elif util_pct > WARN_UTIL_THRESH:
                zone_label = 'warning'
            else:
                zone_label = 'normal'

            # ── Step 4 & 5: Episode detection with debounce  (exact from v2)
            is_new_episode = congested and not self._was_congested[key]
            self._was_congested[key] = congested

            if is_new_episode:
                # Record timestamp for inter_arrival_delta on next episode
                last_congestion_ts[key] = now
                reasons = []
                if signal_util:
                    reasons.append(
                        f'util={util_pct:.1f}% > {UTIL_CONGESTION_THRESH}%'
                    )
                if signal_drop:
                    reasons.append(
                        f'drops Δtx={delta_tx_d} Δrx={delta_rx_d}'
                    )
                reason_str = ' | '.join(reasons)

                with self._counter_lock:
                    self._port_event_count[key] += 1
                    self._total_events           += 1
                    port_cnt   = self._port_event_count[key]
                    global_cnt = self._total_events

                # ── Exact console banner from port_stats_monitor_v2.py ────
                self.logger.warning(
                    f'\n'
                    f'  ╔═══════════════════════════════════════════════════╗\n'
                    f'  ║  [CONGESTION DETECTED]                            ║\n'
                    f'  ║  switch  : 0x{dpid:016x}                    ║\n'
                    f'  ║  port    : {port_no:<40} ║\n'
                    f'  ║  reason  : {reason_str:<40} ║\n'
                    f'  ║  util    : {util_pct:5.1f}%  '
                    f'tx={tx_mbps:.2f}M  rx={rx_mbps:.2f}M         ║\n'
                    f'  ║  drops Δ : tx={delta_tx_d:<6}  rx={delta_rx_d:<6}'
                    f'                      ║\n'
                    f'  ║  events  : this port #{port_cnt:<5}'
                    f'  global #{global_cnt:<5}        ║\n'
                    f'  ╚═══════════════════════════════════════════════════╝'
                )

                # ── congestion_log.csv  (exact columns from v2) ───────────
                self._cong_log.write({
                    'timestamp':          ts,
                    'dpid':               f'0x{dpid:016x}',
                    'port_no':            port_no,
                    'utilization_pct':    f'{util_pct:.2f}',
                    'tx_mbps':            f'{tx_mbps:.4f}',
                    'rx_mbps':            f'{rx_mbps:.4f}',
                    'delta_tx_dropped':   delta_tx_d,
                    'delta_rx_dropped':   delta_rx_d,
                    'signal_util':        int(signal_util),
                    'signal_drop':        int(signal_drop),
                    'reason':             reason_str,
                    'port_event_count':   port_cnt,
                    'global_event_count': global_cnt,
                })

                self._log_event("congestion",
                    f"Congestion on {dpid_lib.dpid_to_str(dpid)} port {port_no} "
                    f"(util={util_pct:.1f}%, episode={global_cnt})", "warning")

            elif congested:
                self.logger.warning(
                    '  [CONGESTION ONGOING]  switch=0x%016x  port=%s  '
                    'util=%.1f%%  Δdrop tx=%s rx=%s',
                    dpid, port_no, util_pct, delta_tx_d, delta_rx_d
                )

            # ── qos_log.csv — every interval, all 30 columns ─────────────
            self._qos_log.write({
                # A — identity
                'timestamp':           ts,
                'dpid':                f'0x{dpid:016x}',
                'port_no':             port_no,
                # B — raw counters
                'tx_bytes':            tx_bytes,
                'rx_bytes':            rx_bytes,
                'tx_dropped':          tx_dropped,
                'rx_dropped':          rx_dropped,
                # C — throughput / utilisation
                'tx_mbps':             f'{tx_mbps:.4f}',
                'rx_mbps':             f'{rx_mbps:.4f}',
                'utilization_pct':     f'{util_pct:.2f}',
                'loss_pct':            f'{port_loss_pct:.4f}',
                'tx_pps':              f'{tx_pps:.2f}',
                'rx_pps':              f'{rx_pps:.2f}',
                'bw_headroom_mbps':    f'{bw_headroom_mbps:.4f}',
                # D — drop deltas
                'delta_tx_dropped':    delta_tx_d,
                'delta_rx_dropped':    delta_rx_d,
                # E — latency / jitter
                'latency_ms':          f'{port_latency_ms:.3f}',
                'jitter_ms':           f'{port_jitter_ms:.3f}',
                'rtt_ms':              f'{port_rtt_ms:.3f}',
                # F — rolling features
                'rolling_util_mean':   f'{rolling_util_mean:.2f}',
                'rolling_drop_sum':    rolling_drop_sum,
                'rolling_tx_mean':     f'{rolling_tx_mean:.4f}',
                'rolling_rx_mean':     f'{rolling_rx_mean:.4f}',
                # G — topology context
                'n_active_flows':      n_active_flows,
                'neighbor_util_max':   f'{neighbor_util_max:.2f}',
                'inter_arrival_delta': f'{inter_arrival_delta:.1f}',
                # H — signals & label
                'signal_util':         int(signal_util),
                'signal_drop':         int(signal_drop),
                'zone_label':          zone_label,
                'congested':           int(congested),
            })

            # ── Module II: LSTM Predictor update ──────────────────────────
            if self.predictor is not None:
                _row = {
                    'tx_mbps':             tx_mbps,
                    'rx_mbps':             rx_mbps,
                    'utilization_pct':     util_pct,
                    'tx_pps':              tx_pps,
                    'rx_pps':              rx_pps,
                    'bw_headroom_mbps':    bw_headroom_mbps,
                    'delta_tx_dropped':    delta_tx_d,
                    'delta_rx_dropped':    delta_rx_d,
                    'latency_ms':          port_latency_ms,
                    'jitter_ms':           port_jitter_ms,
                    'rolling_util_mean':   rolling_util_mean,
                    'rolling_drop_sum':    rolling_drop_sum,
                    'rolling_tx_mean':     rolling_tx_mean,
                    'rolling_rx_mean':     rolling_rx_mean,
                    'n_active_flows':      n_active_flows,
                    'neighbor_util_max':   neighbor_util_max,
                    'inter_arrival_delta': inter_arrival_delta,
                }
                _sv = self.predictor.update(dpid, port_no, _row)
                if _sv is not None:
                    _zone_names = ['normal', 'warning', 'congested']
                    _pred_zone  = _zone_names[int(_sv[:3].argmax())]
                    self.logger.debug(
                        '  LSTM  dpid=0x%016x  port=%s  '
                        'pred_zone=%s  P(cong)=%.3f  cong_prob=%.3f',
                        dpid, port_no, _pred_zone, _sv[2], _sv[3]
                    )

            # ── Per-port status line ──────────────────────────────────────
            bar  = self._util_bar(util_pct)
            flag = ('  ▲U' if signal_util else '   ') + \
                   ('▲D'   if signal_drop else '  ')
            self.logger.info(
                '  dpid=0x%016x  port=%3s  '
                'tx=%6.2fM  rx=%6.2fM  util=%5.1f%%  %s  '
                'Δdrop=[tx=%s rx=%s]  loss=%.3f%%  '
                'lat=%.1fms  jit=%.1fms  zone=%s%s',
                dpid, port_no, tx_mbps, rx_mbps, util_pct, bar,
                delta_tx_d, delta_rx_d, port_loss_pct,
                port_latency_ms, port_jitter_ms, zone_label, flag
            )

            # ── Update REST API port store ─────────────────────────────────
            port_stats_store[dpid][port_no] = {
                'tx_bytes':          tx_bytes,
                'rx_bytes':          rx_bytes,
                'tx_dropped':        tx_dropped,
                'rx_dropped':        rx_dropped,
                'bw_rx_mbps':        round(max(rx_mbps, 0), 4),
                'bw_tx_mbps':        round(max(tx_mbps, 0), 4),
                'utilization_pct':   round(util_pct, 2),
                'loss_pct':          round(port_loss_pct, 4),
                'latency_ms':        round(port_latency_ms, 3),
                'jitter_ms':         round(port_jitter_ms, 3),
                'zone_label':        zone_label,
                'ts':                now,
            }

        # ── End-of-reply summary ──────────────────────────────────────────
        with self._counter_lock:
            total = self._total_events
        if total > 0:
            self.logger.info(
                '  ── Total congestion events recorded: %d ──', total
            )

        # ── Aggregate per-switch metrics → REST API / dashboard ───────────
        self._aggregate_switch_metrics(dpid, now)

    # ── Aggregate metrics for dashboard — all values now REAL ────────────
    def _aggregate_switch_metrics(self, dpid, ts):
        ports = port_stats_store.get(dpid, {})
        if not ports:
            return

        total_bw_rx = sum(p['bw_rx_mbps'] for p in ports.values())
        total_bw_tx = sum(p['bw_tx_mbps'] for p in ports.values())

        # Real loss: mean of per-port loss values collected this cycle
        with self._loss_lock:
            port_losses = list(self._port_loss.get(dpid, []))
        avg_loss = (sum(port_losses) / len(port_losses)) if port_losses else 0.0

        # Real latency and jitter: mean across all links whose src_dpid = dpid
        latency_ms, jitter_ms = self._get_switch_latency_jitter(dpid)

        reward = compute_reward(
            total_bw_rx + total_bw_tx,
            latency_ms,
            avg_loss,
            jitter_ms,
        )

        metrics_history[dpid].append({
            'ts':         ts,
            'ts_iso':     datetime.fromtimestamp(ts).isoformat(),
            'bw_rx_mbps': round(total_bw_rx, 4),
            'bw_tx_mbps': round(total_bw_tx, 4),
            'latency_ms': round(latency_ms, 3),
            'jitter_ms':  round(jitter_ms,  3),
            'loss_pct':   round(avg_loss,   4),
            'reward':     reward,
        })

        # Expose latest latency/jitter values in latency_store for REST API
        with self._probe_lock:
            for (src_dpid, src_port), entry in self._link_latency.items():
                if src_dpid == dpid:
                    latency_store[(src_dpid, src_port)] = {
                        'src_dpid':    dpid_lib.dpid_to_str(src_dpid),
                        'src_port':    src_port,
                        'latency_ms':  round(entry['owd_ms'],    3),
                        'jitter_ms':   round(entry['jitter_ms'], 3),
                        'rtt_ms':      round(entry['rtt_ms'],    3),
                        'ts':          entry['ts'],
                    }

    # =========================================================================
    #  LLDP probe helpers
    # =========================================================================

    def _record_latency(self, src_dpid: int, src_port: int,
                        owd_ms: float, rtt_ms: float):
        """
        Update _link_latency for the link identified by (src_dpid, src_port).
        Called from _packet_in_handler when a probe reply arrives.
        Thread-safe via _probe_lock.
        """
        key = (src_dpid, src_port)
        with self._probe_lock:
            existing = self._link_latency.get(key)
            if existing is None:
                # First sample — initialise jitter to 0
                self._link_latency[key] = {
                    'owd_ms':    owd_ms,
                    'jitter_ms': 0.0,
                    'rtt_ms':    rtt_ms,
                    'prev_owd':  owd_ms,
                    'ts':        time.time(),
                }
            else:
                # EWMA jitter:  J_n = α|OWD_n − OWD_{n-1}| + (1−α)J_{n-1}
                delta_owd  = abs(owd_ms - existing['prev_owd'])
                new_jitter = (JITTER_ALPHA * delta_owd
                              + (1.0 - JITTER_ALPHA) * existing['jitter_ms'])
                self._link_latency[key] = {
                    'owd_ms':    owd_ms,
                    'jitter_ms': new_jitter,
                    'rtt_ms':    rtt_ms,
                    'prev_owd':  owd_ms,
                    'ts':        time.time(),
                }

    def _get_neighbor_util_max(self, dpid: int) -> float:
        """
        Return the highest utilization_pct seen across any port on any switch
        that is directly linked to `dpid` (in either direction).

        Uses link_store (populated by EventLinkAdd) and port_stats_store.
        Returns 0.0 if no neighbours or no data yet.

        This gives the LSTM a one-hop congestion pressure signal — if my
        upstream neighbour is saturated, I am likely to become congested soon.
        """
        dpid_str  = dpid_lib.dpid_to_str(dpid)
        neighbor_dpids = set()

        for link in link_store:
            if link['src_dpid'] == dpid_str:
                neighbor_dpids.add(link['dst_dpid'])
            elif link['dst_dpid'] == dpid_str:
                neighbor_dpids.add(link['src_dpid'])

        max_util = 0.0
        for nd_str in neighbor_dpids:
            # Reverse-lookup dpid int from switch_store
            for dpid_int, info in switch_store.items():
                if info.get('dpid') == nd_str:
                    for port_data in port_stats_store.get(dpid_int, {}).values():
                        u = port_data.get('utilization_pct', 0.0)
                        if u > max_util:
                            max_util = u
                    break

        return round(max_util, 2)

    def _get_switch_latency_jitter(self, dpid: int):
        """
        Return (mean_latency_ms, mean_jitter_ms) for all links whose
        src_dpid == dpid.  Falls back to (0.0, 0.0) if no probes received yet.
        Stale entries (older than LLDP_PROBE_TIMEOUT * 3) are ignored.
        """
        now      = time.time()
        stale_limit = LLDP_PROBE_TIMEOUT * 3
        latencies = []
        jitters   = []

        with self._probe_lock:
            for (src_dpid, _), entry in self._link_latency.items():
                if src_dpid != dpid:
                    continue
                age = now - entry['ts']
                if age > stale_limit:
                    continue
                latencies.append(entry['owd_ms'])
                jitters.append(entry['jitter_ms'])

        if not latencies:
            return 0.0, 0.0

        return (
            sum(latencies) / len(latencies),
            sum(jitters)   / len(jitters),
        )

    # =========================================================================
    #  LLDP probe loop
    #  Runs in its own greenlet.  Every LLDP_PROBE_INTERVAL seconds it sends
    #  one custom LLDP probe out of every port that is part of a known inter-
    #  switch link (discovered by the topology module via --observe-links).
    #
    #  Why only inter-switch ports?  Sending probes out of host-facing ports
    #  would flood hosts with raw LLDP frames they cannot understand.  We
    #  restrict to ports that appear in link_store as a src_port.
    # =========================================================================

    def _lldp_probe_loop(self):
        self.logger.info('LLDP probe loop started  (interval=%ds)',
                         LLDP_PROBE_INTERVAL)
        while True:
            hub.sleep(LLDP_PROBE_INTERVAL)
            self._send_lldp_probes()

    def _send_lldp_probes(self):
        """
        Send one OFPPacketOut per inter-switch port with our custom LLDP probe.
        Records the send_time keyed by (src_dpid, src_port).
        """
        # Build a set of (src_dpid_str, src_port) from the topology link store
        # so we can match against switch ports.
        known_links = set()
        for link in link_store:
            known_links.add((link['src_dpid'], link['src_port']))

        with self._dp_lock:
            dps = list(self._datapaths.items())   # [(dpid_int, datapath), ...]

        for dpid_int, dp in dps:
            dpid_str = dpid_lib.dpid_to_str(dpid_int)
            ofproto  = dp.ofproto
            parser   = dp.ofproto_parser

            for port_no in list(port_stats_store.get(dpid_int, {}).keys()):
                if port_no in SKIP_PORTS:
                    continue
                # Only send on inter-switch links
                if (dpid_str, port_no) not in known_links:
                    continue

                send_time = time.time()

                try:
                    raw_probe = _build_lldp_probe(dpid_int, port_no, send_time)
                except Exception as exc:
                    self.logger.debug(
                        'Failed to build LLDP probe dpid=0x%016x port=%s: %s',
                        dpid_int, port_no, exc
                    )
                    continue

                actions = [parser.OFPActionOutput(port_no)]
                out     = parser.OFPPacketOut(
                    datapath=dp,
                    buffer_id=ofproto.OFP_NO_BUFFER,
                    in_port=ofproto.OFPP_CONTROLLER,
                    actions=actions,
                    data=raw_probe,
                )
                try:
                    dp.send_msg(out)
                    self.logger.debug(
                        'LLDP probe sent  dpid=0x%016x  port=%s', dpid_int, port_no
                    )
                except Exception as exc:
                    self.logger.debug(
                        'Failed to send LLDP probe dpid=0x%016x port=%s: %s',
                        dpid_int, port_no, exc
                    )

    # =========================================================================
    #  Link discovery  (from qos_controller.py)
    # =========================================================================

    @set_ev_cls(topo_event.EventLinkAdd, MAIN_DISPATCHER)
    def _link_add_handler(self, ev):
        s = ev.link.src
        d = ev.link.dst
        entry = {
            'src_dpid': dpid_lib.dpid_to_str(s.dpid),
            'dst_dpid': dpid_lib.dpid_to_str(d.dpid),
            'src_port': s.port_no,
            'dst_port': d.port_no,
        }
        if entry not in link_store:
            link_store.append(entry)
            self._log_event("link",
                f"Link {entry['src_dpid']} → {entry['dst_dpid']}", "info")

    @set_ev_cls(topo_event.EventLinkDelete, MAIN_DISPATCHER)
    def _link_delete_handler(self, ev):
        self._log_event("link", "Ignoring temporary link delete", "warning")

    # =========================================================================
    #  Poll loop  (exact from port_stats_monitor_v2.py)
    # =========================================================================

    def _poll_loop(self):
        self.logger.info('Poll loop started.')
        while True:
            hub.sleep(POLL_INTERVAL)
            with self._dp_lock:
                dps = list(self._datapaths.values())
            for dp in dps:
                self._send_port_stats_request(dp)
                dp.send_msg(dp.ofproto_parser.OFPFlowStatsRequest(dp))

    def _send_port_stats_request(self, dp):
        ofp    = dp.ofproto
        parser = dp.ofproto_parser
        req    = parser.OFPPortStatsRequest(dp, flags=0, port_no=ofp.OFPP_ANY)
        dp.send_msg(req)

    # =========================================================================
    #  Helpers
    # =========================================================================

    def _add_flow(self, datapath, priority, match, actions,
                  buffer_id=None, idle_timeout=30, hard_timeout=0):
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        inst    = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        kwargs  = dict(datapath=datapath, priority=priority, match=match,
                       instructions=inst, idle_timeout=idle_timeout,
                       hard_timeout=hard_timeout)
        if buffer_id:
            kwargs['buffer_id'] = buffer_id
        datapath.send_msg(parser.OFPFlowMod(**kwargs))

    def _log_event(self, category, message, level="info"):
        event_log.appendleft({
            'ts':       datetime.now().isoformat(),
            'category': category,
            'message':  message,
            'level':    level,
        })

    @staticmethod
    def _util_bar(pct: float, width: int = 20) -> str:
        filled = int(round(pct / 100.0 * width))
        return f'[{"█" * filled}{"░" * (width - filled)}]'


# =============================================================================
#  REST API  (exact from qos_controller.py + new /latency endpoint)
# =============================================================================
class QoSRestAPI(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(QoSRestAPI, self).__init__(req, link, data, **config)
        self.controller = data['controller']

    def _json(self, data, status=200):
        return Response(
            content_type='application/json',
            charset='utf-8',
            body=json.dumps(data, default=str).encode('utf-8'),
            status=status,
            headers={
                'Access-Control-Allow-Origin':  '*',
                'Access-Control-Allow-Headers': '*',
            }
        )

    @route('health', REST_API_URL + '/health', methods=['GET'])
    def get_health(self, req, **kwargs):
        return self._json({
            'status':   'ok',
            'switches': len(switch_store),
            'ts':       datetime.now().isoformat(),
        })

    @route('metrics_latest', REST_API_URL + '/metrics/latest', methods=['GET'])
    def get_metrics_latest(self, req, **kwargs):
        result = {}
        for dpid, history in metrics_history.items():
            if history:
                result[dpid_lib.dpid_to_str(dpid)] = history[-1]
        return self._json({'metrics': result, 'ts': datetime.now().isoformat()})

    @route('metrics', REST_API_URL + '/metrics', methods=['GET'])
    def get_metrics(self, req, **kwargs):
        result = {
            dpid_lib.dpid_to_str(dpid): list(history)
            for dpid, history in metrics_history.items()
        }
        return self._json({'metrics': result, 'ts': datetime.now().isoformat()})

    @route('topology', REST_API_URL + '/topology', methods=['GET'])
    def get_topology(self, req, **kwargs):
        switches = []
        for dpid, info in switch_store.items():
            switches.append({
                **info,
                'dpid_int': dpid,
                'ports':    list(port_stats_store.get(dpid, {}).keys()),
            })
        return self._json({
            'switches': switches,
            'links':    list(link_store),
            'ts':       datetime.now().isoformat(),
        })

    @route('flows', REST_API_URL + '/flows', methods=['GET'])
    def get_flows(self, req, **kwargs):
        result = {
            dpid_lib.dpid_to_str(dpid): flows
            for dpid, flows in flow_stats_store.items()
        }
        return self._json({'flows': result, 'ts': datetime.now().isoformat()})

    @route('ports', REST_API_URL + '/ports', methods=['GET'])
    def get_ports(self, req, **kwargs):
        result = {
            dpid_lib.dpid_to_str(dpid): ports
            for dpid, ports in port_stats_store.items()
        }
        return self._json({'ports': result, 'ts': datetime.now().isoformat()})

    @route('events', REST_API_URL + '/events', methods=['GET'])
    def get_events(self, req, **kwargs):
        return self._json({'events': list(event_log)})

    @route('hosts', REST_API_URL + '/hosts', methods=['GET'])
    def get_hosts(self, req, **kwargs):
        from ryu.topology.api import get_host
        try:
            hosts  = get_host(self.controller, None)
            result = [{
                'mac':  h.mac,
                'ipv4': h.ipv4,
                'ipv6': h.ipv6,
                'port': {
                    'dpid':    dpid_lib.dpid_to_str(h.port.dpid),
                    'port_no': h.port.port_no,
                }
            } for h in hosts]
            return self._json({'hosts': result, 'ts': datetime.now().isoformat()})
        except Exception as e:
            return self._json({'hosts': [], 'error': str(e)})

    @route('congestion', REST_API_URL + '/congestion', methods=['GET'])
    def get_congestion(self, req, **kwargs):
        """Live congestion state per port."""
        ctrl   = self.controller
        result = {}
        with ctrl._counter_lock:
            total = ctrl._total_events
        for (dpid, port_no), is_cong in ctrl._was_congested.items():
            dpid_str = dpid_lib.dpid_to_str(dpid)
            result.setdefault(dpid_str, {})[port_no] = {
                'congested':     is_cong,
                'episode_count': ctrl._port_event_count.get((dpid, port_no), 0),
            }
        return self._json({
            'congestion':     result,
            'total_episodes': total,
            'ts':             datetime.now().isoformat(),
        })

    @route('latency', REST_API_URL + '/latency', methods=['GET'])
    def get_latency(self, req, **kwargs):
        """
        Real per-link latency and jitter measured via LLDP probes.

        Response shape:
        {
          "links": [
            {
              "src_dpid":   "0000000000000001",
              "src_port":   1,
              "latency_ms": 2.34,   ← one-way delay (RTT / 2)
              "jitter_ms":  0.18,   ← EWMA jitter
              "rtt_ms":     4.68,   ← round-trip time
              "ts":         1713000000.123
            },
            ...
          ],
          "ts": "2024-04-13T12:00:00.000"
        }

        An empty "links" list means no probes have been received yet
        (topology not yet discovered, or no inter-switch links).
        """
        entries = []
        with self.controller._probe_lock:
            for entry in self.controller._link_latency.values():
                entries.append({
                    'src_dpid':   entry.get('src_dpid',
                                  dpid_lib.dpid_to_str(0)),
                    'src_port':   entry.get('src_port', 0),
                    'latency_ms': round(entry['owd_ms'],    3),
                    'jitter_ms':  round(entry['jitter_ms'], 3),
                    'rtt_ms':     round(entry['rtt_ms'],    3),
                    'ts':         entry['ts'],
                })

        # Enrich with src_dpid / src_port from latency_store
        # (populated by _aggregate_switch_metrics)
        enriched = []
        for (src_dpid, src_port), store_entry in latency_store.items():
            enriched.append(store_entry)

        return self._json({
            'links': enriched if enriched else entries,
            'ts':    datetime.now().isoformat(),
        })

    @route('prediction', REST_API_URL + '/prediction', methods=['GET'])
    def get_prediction(self, req, **kwargs):
        """
        LSTM state vectors for all ports — consumed by Module III DQN.
        Returns 'ready: false' until training is complete and controller restarted.
        """
        ctrl = self.controller
        if ctrl.predictor is None:
            return self._json({
                'ready':       False,
                'ports_ready': 0,
                'predictions': {},
                'message':     (
                    'LSTM predictor not loaded. '
                    'Ensure: (1) module2/__init__.py exists, '
                    '(2) python3 module2/train.py completed, '
                    '(3) restart controller with PYTHONPATH=. ryu-manager ...'
                ),
                'ts':          datetime.now().isoformat(),
            })

        all_sv     = ctrl.predictor.state_vector_all()
        zone_names = ['normal', 'warning', 'congested']
        result     = {}

        for (dpid_str, port_no), sv in all_sv.items():
            pred_zone = zone_names[int(sv[:3].argmax())]
            result.setdefault(dpid_str, {})[str(port_no)] = {
                'P_normal':         round(float(sv[0]), 4),
                'P_warning':        round(float(sv[1]), 4),
                'P_congested':      round(float(sv[2]), 4),
                'cong_prob':        round(float(sv[3]), 4),
                'is_congested':     int(sv[4]),
                'utilization_pct':  round(float(sv[5]), 2),
                'bw_headroom_mbps': round(float(sv[6]), 2),
                'delta_tx_dropped': round(float(sv[7]), 0),
                'latency_ms':       round(float(sv[8]), 3),
                'pred_zone':        pred_zone,
            }

        return self._json({
            'ready':       True,
            'ports_ready': ctrl.predictor.n_ports_ready,
            'predictions': result,
            'ts':          datetime.now().isoformat(),
        })