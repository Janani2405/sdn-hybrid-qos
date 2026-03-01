"""
port_stats_monitor_v2.py — Ryu OpenFlow 1.3 Port Statistics Monitor (Extended)
================================================================================
Extends the base monitor with a two-signal congestion detection engine:

  Signal 1 — High utilisation  : util > UTIL_CONGESTION_THRESH (default 95%)
  Signal 2 — Packet drop growth: tx_dropped or rx_dropped increased since
                                  the previous poll interval

Either signal alone is enough to declare a port congested.

Extra outputs vs the base version:
  • [CONGESTION DETECTED] switch X port Y  printed to console on every event
  • Running total congestion event counter  (per-port and global)
  • logs/congestion_log.csv  — one row per congestion episode start

Compatible with:
  • Ryu 4.x / 4.34+
  • Open vSwitch 2.x+
  • Mininet (any version)
  • OpenFlow 1.3

File location:
  sdn-project/
  └── controller/
      └── port_stats_monitor_v2.py   ← this file

Run with L2 forwarding (recommended for Mininet):
  ryu-manager ryu.app.simple_switch_13 controller/port_stats_monitor_v2.py

Run standalone (monitoring only):
  ryu-manager controller/port_stats_monitor_v2.py

──────────────────────────────────────────────────────────────────────────────
HOW THE CONGESTION DETECTION LOGIC WORKS
──────────────────────────────────────────────────────────────────────────────

Every POLL_INTERVAL seconds the app receives an OFPPortStatsReply containing
cumulative counters for every port on every switch:

    tx_bytes    rx_bytes    tx_dropped    rx_dropped    (all monotonically ↑)

Step 1 — Delta computation
    The app stores the counters from the *previous* reply in self._prev.
    On each new reply it computes:

        Δtx_bytes   = tx_bytes_now   − tx_bytes_prev
        Δrx_bytes   = rx_bytes_now   − rx_bytes_prev
        Δtx_dropped = tx_dropped_now − tx_dropped_prev
        Δrx_dropped = rx_dropped_now − rx_dropped_prev
        Δt          = timestamp_now  − timestamp_prev   (seconds)

Step 2 — Throughput & utilisation
        tx_Mbps  = (Δtx_bytes × 8) / (Δt × 1_000_000)
        rx_Mbps  = (Δrx_bytes × 8) / (Δt × 1_000_000)
        util_%   = max(tx_Mbps, rx_Mbps) / LINK_CAPACITY_MBPS × 100

Step 3 — Congestion signals (OR logic)
    ┌──────────────────────────────────────────────────────────┐
    │  signal_util  = util_% > UTIL_CONGESTION_THRESH (95%)   │
    │  signal_drop  = Δtx_dropped > 0  OR  Δrx_dropped > 0   │
    │                                                          │
    │  congested = signal_util  OR  signal_drop                │
    └──────────────────────────────────────────────────────────┘

    Why OR and not AND?
    • High utilisation without drops means the link is saturated but
      the queue is absorbing bursts — still a congestion precursor.
    • Drops without high utilisation can happen on misconfigured queues
      or when a burst exceeds the queue depth on an otherwise lightly
      loaded link — still a real problem that needs attention.
    Using OR catches both scenarios independently.

Step 4 — Event recording
    When congested == True:
    • Print  [CONGESTION DETECTED] switch <dpid> port <N>  with reason
    • Increment per-port counter  self._port_event_count[(dpid, port_no)]
    • Increment global counter    self._total_events
    • Append one row to congestion_log.csv

Step 5 — Cooldown / debounce
    A port is only counted as a *new* event if it was NOT congested in the
    previous poll cycle (self._was_congested).  This prevents a single
    sustained congestion episode from inflating the event counter on every
    2-second tick.  The counter increments once per congestion *episode*,
    not once per polling interval while congested.
    CSV logging still records every interval for full time-series visibility.
"""

import os
import csv
import time
import logging
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ryu.base         import app_manager
from ryu.controller   import ofp_event
from ryu.controller.handler import (
    MAIN_DISPATCHER,
    DEAD_DISPATCHER,
    CONFIG_DISPATCHER,
    set_ev_cls,
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib     import hub


# ─────────────────────────────────────────────────────────────────────────────
#  Tunables
# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL          = 2       # seconds between PortStatsRequest per switch
LINK_CAPACITY_MBPS     = 100.0   # assumed link capacity (matches TCLink bw=100)
UTIL_CONGESTION_THRESH = 95.0    # utilisation % that triggers signal_util
LOG_DIR                = 'logs'
QOS_CSV_FILE           = os.path.join(LOG_DIR, 'qos_log.csv')
CONG_CSV_FILE          = os.path.join(LOG_DIR, 'congestion_log.csv')

# OVS internal/reserved port numbers — always skipped
SKIP_PORTS = {0xfffffffe, 0xffffffff}   # OFPP_LOCAL, OFPP_NONE


# ─────────────────────────────────────────────────────────────────────────────
#  Thread-safe CSV logger (reusable for both output files)
# ─────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    """
    Append-only, thread-safe CSV writer.
    Creates the file and writes the header row on the very first write.
    """

    def __init__(self, path: str, columns: list):
        Path(os.path.dirname(path) or '.').mkdir(parents=True, exist_ok=True)
        self._path    = path
        self._columns = columns
        self._lock    = threading.Lock()
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


# ─────────────────────────────────────────────────────────────────────────────
#  Column definitions
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  Ryu Application
# ─────────────────────────────────────────────────────────────────────────────

class PortStatsMonitor(app_manager.RyuApp):
    """
    OpenFlow 1.3 port-statistics monitor with dual-signal congestion detection.
    See module docstring for full logic description.
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_logger()

        # ── Switch registry ───────────────────────────────────────────────
        self._datapaths: dict = {}          # dpid → datapath
        self._dp_lock         = threading.Lock()

        # ── Per-port previous-sample state ────────────────────────────────
        # Key: (dpid, port_no)
        # Value: {tx, rx, tx_dropped, rx_dropped, ts}
        self._prev: dict  = {}
        self._prev_lock   = threading.Lock()

        # ── Congestion debounce state ─────────────────────────────────────
        # Key: (dpid, port_no)  →  bool (was congested last cycle?)
        self._was_congested: dict = defaultdict(bool)

        # ── Event counters ────────────────────────────────────────────────
        # Per-port: (dpid, port_no) → int
        self._port_event_count: dict = defaultdict(int)
        # Global: total congestion episodes across all ports/switches
        self._total_events: int      = 0
        self._counter_lock           = threading.Lock()

        # ── CSV loggers ───────────────────────────────────────────────────
        self._qos_log  = CSVLogger(QOS_CSV_FILE,  QOS_COLUMNS)
        self._cong_log = CSVLogger(CONG_CSV_FILE, CONG_COLUMNS)

        self.logger.info(f'QoS log        → {os.path.abspath(QOS_CSV_FILE)}')
        self.logger.info(f'Congestion log → {os.path.abspath(CONG_CSV_FILE)}')
        self.logger.info(
            f'Config: poll={POLL_INTERVAL}s  '
            f'capacity={LINK_CAPACITY_MBPS} Mbps  '
            f'util_threshold={UTIL_CONGESTION_THRESH}%'
        )

        # ── Background polling green-thread ──────────────────────────────
        self._poll_thread = hub.spawn(self._poll_loop)

    def _configure_logger(self):
        self.logger.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.StreamHandler)
                   for h in self.logger.handlers):
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                '%(asctime)s  %(levelname)-8s  [PSMonitor]  %(message)s',
                datefmt='%H:%M:%S',
            ))
            self.logger.addHandler(h)

    # ── Switch connect / disconnect ───────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _features_handler(self, ev):
        dp     = ev.msg.datapath
        dpid   = dp.id
        ofp    = dp.ofproto
        parser = dp.ofproto_parser

        self.logger.info(f'Switch CONNECTED  dpid=0x{dpid:016x}')
        with self._dp_lock:
            self._datapaths[dpid] = dp

        # Table-miss rule: forward unknown packets to controller
        match   = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER,
                                          ofp.OFPCML_NO_BUFFER)]
        inst    = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                actions)]
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
            self.logger.warning(f'Switch DISCONNECTED  dpid=0x{dpid:016x}')

    # ── Polling loop ──────────────────────────────────────────────────────────

    def _poll_loop(self):
        """
        Ryu green-thread: wakes every POLL_INTERVAL seconds and fires
        one OFPPortStatsRequest per connected switch.
        """
        self.logger.info('Poll loop started.')
        while True:
            hub.sleep(POLL_INTERVAL)
            with self._dp_lock:
                dps = list(self._datapaths.values())
            for dp in dps:
                self._send_port_stats_request(dp)

    def _send_port_stats_request(self, dp):
        ofp    = dp.ofproto
        parser = dp.ofproto_parser
        req    = parser.OFPPortStatsRequest(dp, flags=0, port_no=ofp.OFPP_ANY)
        dp.send_msg(req)

    # ── Stats reply handler ───────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply(self, ev):
        """
        Core processing: called by Ryu on every OFPPortStatsReply.
        Implements the 5-step detection pipeline from the module docstring.
        """
        body = ev.msg.body
        dp   = ev.msg.datapath
        dpid = dp.id
        now  = time.time()
        ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        for stat in body:
            port_no = stat.port_no
            if port_no in SKIP_PORTS:
                continue

            # ── Raw cumulative counters from OFP reply ────────────────────
            tx_bytes   = stat.tx_bytes
            rx_bytes   = stat.rx_bytes
            tx_dropped = stat.tx_dropped    # cumulative hardware TX drops
            rx_dropped = stat.rx_dropped    # cumulative hardware RX drops
            key        = (dpid, port_no)

            # ────────────────────────────────────────────────────────────
            # Step 1 — Delta computation
            # ────────────────────────────────────────────────────────────
            with self._prev_lock:
                prev = self._prev.get(key)

                if prev is None:
                    # Very first sample for this port — store baseline only.
                    # We cannot compute a rate without two data points.
                    self._prev[key] = {
                        'tx':  tx_bytes,   'rx':  rx_bytes,
                        'tx_d': tx_dropped, 'rx_d': rx_dropped,
                        'ts':  now,
                    }
                    self.logger.debug(
                        f'port={port_no}  baseline stored  '
                        f'tx_drop={tx_dropped}  rx_drop={rx_dropped}'
                    )
                    continue

                delta_t    = now        - prev['ts']
                delta_tx   = tx_bytes   - prev['tx']
                delta_rx   = rx_bytes   - prev['rx']
                delta_tx_d = tx_dropped - prev['tx_d']  # new TX drops this interval
                delta_rx_d = rx_dropped - prev['rx_d']  # new RX drops this interval

                # Guard: negative deltas mean counter wrap or OVS restart
                if delta_t <= 0 or delta_tx < 0 or delta_rx < 0 \
                        or delta_tx_d < 0 or delta_rx_d < 0:
                    self.logger.debug(
                        f'port={port_no}  counter anomaly '
                        f'(Δt={delta_t:.3f} Δtx={delta_tx} '
                        f'Δrx={delta_rx} Δtx_d={delta_tx_d} '
                        f'Δrx_d={delta_rx_d}) — resetting baseline'
                    )
                    self._prev[key] = {
                        'tx': tx_bytes, 'rx': rx_bytes,
                        'tx_d': tx_dropped, 'rx_d': rx_dropped,
                        'ts': now,
                    }
                    continue

                # Commit updated baseline for next cycle
                self._prev[key] = {
                    'tx': tx_bytes,  'rx': rx_bytes,
                    'tx_d': tx_dropped, 'rx_d': rx_dropped,
                    'ts': now,
                }

            # ────────────────────────────────────────────────────────────
            # Step 2 — Throughput (Mbps) and utilisation %
            #
            #   throughput = Δbytes × 8 bits/byte
            #                ──────────────────────
            #                Δt(s)  × 1,000,000 b/Mb
            # ────────────────────────────────────────────────────────────
            tx_mbps  = (delta_tx * 8) / (delta_t * 1_000_000)
            rx_mbps  = (delta_rx * 8) / (delta_t * 1_000_000)
            peak     = max(tx_mbps, rx_mbps)
            util_pct = min((peak / LINK_CAPACITY_MBPS) * 100.0, 100.0)

            # ────────────────────────────────────────────────────────────
            # Step 3 — Dual-signal congestion detection (OR logic)
            #
            #   signal_util  — bandwidth saturation signal
            #   signal_drop  — queue-overflow / hardware-drop signal
            #   congested    — either or both signals fired
            # ────────────────────────────────────────────────────────────
            signal_util = util_pct  > UTIL_CONGESTION_THRESH
            signal_drop = (delta_tx_d > 0) or (delta_rx_d > 0)
            congested   = signal_util or signal_drop

            # ────────────────────────────────────────────────────────────
            # Step 4 & 5 — Episode detection with debounce
            #
            # is_new_episode is True only on the FIRST poll cycle where
            # congested transitions from False → True.  Subsequent cycles
            # where the port remains congested are labelled "ONGOING" but
            # do NOT increment the event counter.
            # ────────────────────────────────────────────────────────────
            is_new_episode = congested and not self._was_congested[key]
            self._was_congested[key] = congested   # update state for next poll

            if is_new_episode:
                # Compose reason string from whichever signals fired
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

                # Thread-safe counter update
                with self._counter_lock:
                    self._port_event_count[key] += 1
                    self._total_events           += 1
                    port_cnt   = self._port_event_count[key]
                    global_cnt = self._total_events

                # ── Required console output ───────────────────────────────
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

                # ── congestion_log.csv ────────────────────────────────────
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

            elif congested:
                # Ongoing episode — brief log line, no counter increment
                self.logger.warning(
                    f'  [CONGESTION ONGOING]  '
                    f'switch=0x{dpid:016x}  port={port_no}  '
                    f'util={util_pct:.1f}%  '
                    f'Δdrop tx={delta_tx_d} rx={delta_rx_d}'
                )

            # ── qos_log.csv — written every interval unconditionally ──────
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

            # ── Per-port status line ──────────────────────────────────────
            bar  = self._util_bar(util_pct)
            flag = ('  ▲U' if signal_util else '   ') + \
                   ('▲D'   if signal_drop else '  ')
            self.logger.info(
                f'  dpid=0x{dpid:016x}  port={port_no:>3}  '
                f'tx={tx_mbps:6.2f}M  rx={rx_mbps:6.2f}M  '
                f'util={util_pct:5.1f}%  {bar}  '
                f'Δdrop=[tx={delta_tx_d} rx={delta_rx_d}]{flag}'
            )

        # ── End-of-reply summary ──────────────────────────────────────────
        with self._counter_lock:
            total = self._total_events
        if total > 0:
            self.logger.info(
                f'  ── Total congestion events recorded: {total} ──'
            )

    # ── Packet-in no-op ───────────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        pass   # handled by simple_switch_13 when used alongside it

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def _util_bar(pct: float, width: int = 20) -> str:
        filled = int(round(pct / 100.0 * width))
        return f'[{"█" * filled}{"░" * (width - filled)}]'