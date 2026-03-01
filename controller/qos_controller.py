import eventlet
eventlet.monkey_patch()

"""
=============================================================================
qos_controller.py — SDN QoS Controller (MERGED: Visualization + Data Collection)
=============================================================================
Combines port_stats_monitor_v2.py + qos_controller.py into ONE file:

  [1] DATA COLLECTION  (exact logic from port_stats_monitor_v2.py)
        → logs/qos_log.csv         every 2s  — LSTM training data
        → logs/congestion_log.csv  per new episode — congestion labels
        → [CONGESTION DETECTED] console banner on every new episode
        → [CONGESTION ONGOING]  on sustained congestion (no counter inflation)
        → Full debounce: episode counter increments once per episode, not per tick
        → Counter-wrap / OVS-restart guard on all deltas

  [2] VISUALIZATION    (exact REST API from qos_controller.py)
        → http://127.0.0.1:8080/qos/api/v1/health
        → http://127.0.0.1:8080/qos/api/v1/metrics/latest
        → http://127.0.0.1:8080/qos/api/v1/topology
        → http://127.0.0.1:8080/qos/api/v1/flows
        → http://127.0.0.1:8080/qos/api/v1/ports
        → http://127.0.0.1:8080/qos/api/v1/events
        → http://127.0.0.1:8080/qos/api/v1/hosts
        → http://127.0.0.1:8080/qos/api/v1/congestion  (NEW — live state)
        → Powers dashboard.html live charts and topology view

Run:
    source ~/ryu-env/bin/activate
    cd ~/sdn-project
    ryu-manager controller/qos_controller.py --observe-links --ofp-tcp-listen-port 6633

Dashboard:
    Open docs/dashboard.html in browser
=============================================================================
"""

import os
import csv
import json
import time
import logging
import threading
import collections
import random
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


# ─────────────────────────────────────────────────────────────────
#  Tunables  (from port_stats_monitor_v2.py — unchanged)
# ─────────────────────────────────────────────────────────────────
POLL_INTERVAL          = 2        # seconds between PortStatsRequest per switch
LINK_CAPACITY_MBPS     = 100.0    # assumed link capacity (matches TCLink bw=100)
UTIL_CONGESTION_THRESH = 95.0     # utilisation % that triggers signal_util
LOG_DIR                = 'logs'
QOS_CSV_FILE           = os.path.join(LOG_DIR, 'qos_log.csv')
CONG_CSV_FILE          = os.path.join(LOG_DIR, 'congestion_log.csv')

# OVS internal/reserved port numbers — always skipped
SKIP_PORTS = {0xfffffffe, 0xffffffff}   # OFPP_LOCAL, OFPP_NONE

# REST API prefix
REST_API_URL = '/qos/api/v1'


# ─────────────────────────────────────────────────────────────────
#  CSV column definitions  (exact from port_stats_monitor_v2.py)
# ─────────────────────────────────────────────────────────────────
QOS_COLUMNS = [
    'timestamp', 'dpid', 'port_no',
    'tx_bytes', 'rx_bytes',
    'tx_mbps', 'rx_mbps',
    'tx_dropped', 'rx_dropped',
    'delta_tx_dropped', 'delta_rx_dropped',
    'utilization_pct',
    'signal_util', 'signal_drop', 'congested',
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
#  Reward signal  (from qos_controller.py — for DQN Module III)
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

        # ── Start background poll loop ────────────────────────────────────
        self._poll_thread = hub.spawn(self._poll_loop)

        # ── Register REST API ─────────────────────────────────────────────
        wsgi = kwargs['wsgi']
        wsgi.register(QoSRestAPI, {'controller': self})

        self._log_event("controller",
            "QoS Controller started (merged: data + visualization)", "info")
        self.logger.info('QoS log        → %s', os.path.abspath(QOS_CSV_FILE))
        self.logger.info('Congestion log → %s', os.path.abspath(CONG_CSV_FILE))
        self.logger.info('REST API       → http://127.0.0.1:8080%s', REST_API_URL)
        self.logger.info(
            'Config: poll=%ds  capacity=%.0f Mbps  util_threshold=%.0f%%',
            POLL_INTERVAL, LINK_CAPACITY_MBPS, UTIL_CONGESTION_THRESH
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
    #  Packet-in  (L2 learning switch)
    # =========================================================================

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg      = ev.msg
        datapath = msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        in_port  = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

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

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
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

            # ── Step 2: Throughput and utilisation  (exact from v2) ───────
            tx_mbps  = (delta_tx * 8) / (delta_t * 1_000_000)
            rx_mbps  = (delta_rx * 8) / (delta_t * 1_000_000)
            peak     = max(tx_mbps, rx_mbps)
            util_pct = min((peak / LINK_CAPACITY_MBPS) * 100.0, 100.0)

            # ── Step 3: Dual-signal congestion detection  (exact from v2) ─
            signal_util = util_pct  > UTIL_CONGESTION_THRESH
            signal_drop = (delta_tx_d > 0) or (delta_rx_d > 0)
            congested   = signal_util or signal_drop

            # ── Step 4 & 5: Episode detection with debounce  (exact from v2)
            is_new_episode = congested and not self._was_congested[key]
            self._was_congested[key] = congested

            if is_new_episode:
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

            # ── qos_log.csv — every interval unconditionally  (exact from v2)
            self._qos_log.write({
                'timestamp':        ts,
                'dpid':             f'0x{dpid:016x}',
                'port_no':          port_no,
                'tx_bytes':         tx_bytes,
                'rx_bytes':         rx_bytes,
                'tx_mbps':          f'{tx_mbps:.4f}',
                'rx_mbps':          f'{rx_mbps:.4f}',
                'tx_dropped':       tx_dropped,
                'rx_dropped':       rx_dropped,
                'delta_tx_dropped': delta_tx_d,
                'delta_rx_dropped': delta_rx_d,
                'utilization_pct':  f'{util_pct:.2f}',
                'signal_util':      int(signal_util),
                'signal_drop':      int(signal_drop),
                'congested':        int(congested),
            })

            # ── Per-port status line  (exact from v2) ─────────────────────
            bar  = self._util_bar(util_pct)
            flag = ('  ▲U' if signal_util else '   ') + \
                   ('▲D'   if signal_drop else '  ')
            self.logger.info(
                '  dpid=0x%016x  port=%3s  '
                'tx=%6.2fM  rx=%6.2fM  util=%5.1f%%  %s  '
                'Δdrop=[tx=%s rx=%s]%s',
                dpid, port_no, tx_mbps, rx_mbps, util_pct, bar,
                delta_tx_d, delta_rx_d, flag
            )

            # ── Update REST API port store ─────────────────────────────────
            port_stats_store[dpid][port_no] = {
                'tx_bytes':   tx_bytes,
                'rx_bytes':   rx_bytes,
                'tx_dropped': tx_dropped,
                'rx_dropped': rx_dropped,
                'bw_rx_mbps': round(max(rx_mbps, 0), 4),
                'bw_tx_mbps': round(max(tx_mbps, 0), 4),
                'loss_pct':   0.0,
                'ts':         now,
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

    # ── Aggregate metrics for dashboard  (from qos_controller.py) ────────
    def _aggregate_switch_metrics(self, dpid, ts):
        ports = port_stats_store.get(dpid, {})
        if not ports:
            return
        total_bw_rx = sum(p['bw_rx_mbps'] for p in ports.values())
        total_bw_tx = sum(p['bw_tx_mbps'] for p in ports.values())
        avg_loss    = sum(p['loss_pct']   for p in ports.values()) / max(len(ports), 1)

        latency = random.uniform(2, 15)
        jitter  = random.uniform(0.5, 5)
        reward  = compute_reward(total_bw_rx + total_bw_tx, latency, avg_loss, jitter)

        metrics_history[dpid].append({
            'ts':         ts,
            'ts_iso':     datetime.fromtimestamp(ts).isoformat(),
            'bw_rx_mbps': round(total_bw_rx, 4),
            'bw_tx_mbps': round(total_bw_tx, 4),
            'latency_ms': round(latency, 2),
            'jitter_ms':  round(jitter,  2),
            'loss_pct':   round(avg_loss, 4),
            'reward':     reward,
        })

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
#  REST API  (exact from qos_controller.py)
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
        """Live congestion state per port — new endpoint."""
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