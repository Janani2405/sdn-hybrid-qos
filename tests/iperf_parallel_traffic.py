#!/usr/bin/env python
"""
iperf_parallel_traffic.py — Parallel iperf Flow Engine for Mininet
===================================================================
Launches multiple overlapping UDP/TCP iperf flows to create realistic
congestion on 100 Mbps links. Safe for Mininet — uses popen (non-blocking),
never calls net.stop(), and cleans up all child processes on exit.

Placement:  sdn-project/tests/iperf_parallel_traffic.py

── How to run ────────────────────────────────────────────────────────────────

  Option A  – from the Mininet CLI (recommended):
      mininet> py exec(open('tests/iperf_parallel_traffic.py').read())

  Option B  – as a standalone script (outside Mininet, h2 must be reachable):
      sudo python tests/iperf_parallel_traffic.py --server-ip 10.0.0.2

  Option C  – called programmatically from topology.py after CLI(net):
      from tests.iperf_parallel_traffic import run_parallel_traffic
      run_parallel_traffic(net)

── Congestion model ──────────────────────────────────────────────────────────

  Flows are grouped into "waves".  Within each wave every flow starts at a
  slightly staggered time (STAGGER_S apart) so that 3–5 flows are alive and
  competing on the 100 Mbps link simultaneously.

  Wave 1  (low contention)   : 2 flows × 30 Mbps  →  60 Mbps aggregate
  Wave 2  (moderate)         : 3 flows × 35 Mbps  → 105 Mbps  ← congestion
  Wave 3  (high contention)  : 4 flows × 28 Mbps  → 112 Mbps  ← heavy loss
  Wave 4  (burst spike)      : 2 flows × 50 Mbps  → 100 Mbps
  Wave 5  (mixed protocol)   : 2 UDP + 1 TCP, varying BW

  You can replace FLOW_PLAN with rows from your CSV — see load_csv_plan().
"""

import os
import sys
import csv
import time
import signal
import logging
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  Tunables  — change these without touching the rest of the script
# ─────────────────────────────────────────────────────────────────────────────

SERVER_HOST   = 'h2'
CLIENT_HOST   = 'h1'
SERVER_IP     = '10.0.0.2'       # IP of h2 in your topology
IPERF_PORT    = 5201              # base port; each flow gets PORT + flow_id
IPERF_BIN     = 'iperf3'         # or 'iperf'
LOG_DIR       = 'logs'

STAGGER_S     = 0.4              # seconds between flow starts within a wave
WAVE_GAP_S    = 2.0              # cooldown between waves
FLOW_TIMEOUT  = 120              # hard kill after N seconds per flow (safety)
MAX_PARALLEL  = 5                # absolute max simultaneous flows

# ── Built-in flow plan (wave, duration_s, bw_mbps, protocol) ─────────────────
FLOW_PLAN = [
    # wave  dur   bw    proto
    (1,     10,   30,  'udp'),   # Wave 1 — low contention
    (1,     12,   30,  'udp'),

    (2,     15,   35,  'udp'),   # Wave 2 — moderate congestion (105 Mbps total)
    (2,     15,   35,  'udp'),
    (2,     15,   35,  'udp'),

    (3,     10,   28,  'udp'),   # Wave 3 — heavy congestion (112 Mbps total)
    (3,     12,   28,  'udp'),
    (3,     10,   28,  'udp'),
    (3,     14,   28,  'udp'),

    (4,     8,    50,  'udp'),   # Wave 4 — burst spike (100 Mbps total)
    (4,     8,    50,  'udp'),

    (5,     12,   40,  'udp'),   # Wave 5 — mixed protocol
    (5,     12,   40,  'udp'),
    (5,     12,   20,  'tcp'),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    log = os.path.join(LOG_DIR, f'parallel_traffic_{ts}.log')

    fmt = logging.Formatter(
        '%(asctime)s.%(msecs)03d  %(levelname)-8s  [%(threadName)s]  %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger('ParallelTraffic')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'Log → {log}')
    return logger


# ─────────────────────────────────────────────────────────────────────────────
#  Thread-safe statistics collector
# ─────────────────────────────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self._lock    = threading.Lock()
        self.started  = 0
        self.finished = 0
        self.failed   = 0
        self._active  = 0           # currently running flows
        self._peak    = 0           # peak simultaneous flows seen

    def on_start(self):
        with self._lock:
            self.started += 1
            self._active += 1
            if self._active > self._peak:
                self._peak = self._active

    def on_end(self, success: bool):
        with self._lock:
            self._active -= 1
            self.finished += 1
            if not success:
                self.failed += 1

    @property
    def active(self):
        with self._lock:
            return self._active

    @property
    def peak(self):
        with self._lock:
            return self._peak


# ─────────────────────────────────────────────────────────────────────────────
#  Flow worker — runs in its own thread
# ─────────────────────────────────────────────────────────────────────────────

# Registry so the shutdown hook can kill stragglers
_procs_lock = threading.Lock()
_procs: dict = {}           # flow_id → subprocess.Popen

def _register(flow_id: int, proc):
    with _procs_lock:
        _procs[flow_id] = proc

def _deregister(flow_id: int):
    with _procs_lock:
        _procs.pop(flow_id, None)

def kill_all_flows(logger: logging.Logger):
    """Emergency cleanup — called by signal handler and finally block."""
    with _procs_lock:
        alive = list(_procs.items())
    for fid, proc in alive:
        try:
            if proc.poll() is None:
                proc.kill()
                logger.debug(f'Killed stray flow {fid}')
        except Exception:
            pass


def _build_cmd(flow_id: int, duration: int, bw_mbps: float,
               protocol: str, server_ip: str) -> list:
    """Build iperf3 command for one flow."""
    port = IPERF_PORT + flow_id          # unique port per flow avoids collisions
    cmd = [
        IPERF_BIN,
        '-c', server_ip,
        '-p', str(port),
        '-t', str(int(duration)),
        '-b', f'{bw_mbps}M',
        '-i', '0',                       # suppress per-interval output (cleaner logs)
        '--json',                         # structured output for parsing
    ]
    if protocol == 'udp':
        cmd.append('-u')
    return cmd


def flow_worker(flow_id: int, wave: int, duration: float, bw_mbps: float,
                protocol: str, net_or_none, stats: Stats,
                semaphore: threading.Semaphore, logger: logging.Logger):
    """
    Runs one iperf flow.
    - Acquires semaphore slot (respects MAX_PARALLEL)
    - Starts iperf via Mininet popen or subprocess
    - Logs precise start/end wall-clock times
    - Always releases semaphore and records stats (never crashes caller)
    """
    tag = f'F{flow_id:03d}|W{wave}|{protocol.upper()}|{bw_mbps}M/{duration}s'

    with semaphore:
        stats.on_start()
        t_start = time.time()
        wall_start = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        logger.info(f'▶ START  {tag}  wall={wall_start}  active={stats.active}')

        cmd = _build_cmd(flow_id, duration, bw_mbps, protocol,
                         SERVER_IP if net_or_none is None else SERVER_IP)
        logger.debug(f'  CMD: {" ".join(cmd)}')

        proc   = None
        rc     = -99
        stdout = ''
        stderr = ''

        try:
            if net_or_none is not None:
                # ── Mininet path ───────────────────────────────────────────
                h1   = net_or_none.get(CLIENT_HOST)
                proc = h1.popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
            else:
                # ── Standalone path ────────────────────────────────────────
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)

            _register(flow_id, proc)

            # Wait with hard timeout to prevent zombie threads
            stdout_b, stderr_b = proc.communicate(
                timeout=float(duration) + FLOW_TIMEOUT
            )
            rc     = proc.returncode
            stdout = stdout_b.decode('utf-8', errors='replace')
            stderr = stderr_b.decode('utf-8', errors='replace')

        except subprocess.TimeoutExpired:
            logger.error(f'✖ TIMEOUT {tag} — killing')
            if proc:
                proc.kill()
                proc.communicate()
            rc     = -1
            stderr = 'TIMEOUT'

        except Exception as exc:
            logger.error(f'✖ EXCEPTION {tag}: {exc}')
            rc     = -2
            stderr = str(exc)
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass

        finally:
            _deregister(flow_id)

        # ── Timing ────────────────────────────────────────────────────────
        elapsed    = time.time() - t_start
        wall_end   = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        success    = (rc == 0)
        status_sym = '✔' if success else '✖'

        logger.info(
            f'{status_sym} END    {tag}  '
            f'wall_start={wall_start}  wall_end={wall_end}  '
            f'elapsed={elapsed:.2f}s  rc={rc}  active_after={stats.active - 1}'
        )
        if not success and stderr:
            logger.warning(f'  STDERR: {stderr[:300].strip()}')

        # ── Per-flow log file ─────────────────────────────────────────────
        flow_log = os.path.join(
            LOG_DIR,
            f'flow_{flow_id:03d}_w{wave}_{protocol}_{int(bw_mbps)}M.log'
        )
        try:
            with open(flow_log, 'w') as f:
                f.write(f'FLOW     : {tag}\n')
                f.write(f'START    : {wall_start}\n')
                f.write(f'END      : {wall_end}\n')
                f.write(f'ELAPSED  : {elapsed:.3f}s\n')
                f.write(f'RC       : {rc}\n')
                f.write(f'CMD      : {" ".join(cmd)}\n\n')
                f.write('=== STDOUT ===\n')
                f.write(stdout or '(empty)\n')
                f.write('\n=== STDERR ===\n')
                f.write(stderr or '(empty)\n')
        except Exception as e:
            logger.debug(f'  Could not write flow log: {e}')

        stats.on_end(success)


# ─────────────────────────────────────────────────────────────────────────────
#  iperf3 server management (one server per unique port)
# ─────────────────────────────────────────────────────────────────────────────

_server_procs: list = []

def start_servers(flow_plan: list, net_or_none, logger: logging.Logger):
    """
    Start one iperf3 server per unique port used across all flows.
    Servers run in daemon mode so they accept multiple sequential connections.
    """
    ports = sorted({IPERF_PORT + fid for fid, (_, _, _, _) in enumerate(flow_plan)})
    logger.info(f'Starting {len(ports)} iperf server(s) on {SERVER_HOST}: ports {ports[0]}–{ports[-1]}')

    for port in ports:
        cmd = [IPERF_BIN, '-s', '-p', str(port), '-D', '--one-off']

        try:
            if net_or_none is not None:
                h2 = net_or_none.get(SERVER_HOST)
                # Use popen for non-blocking start; save PID via cmd wrapper
                h2.cmd(f'{IPERF_BIN} -s -p {port} -D --one-off 2>/dev/null')
            else:
                proc = subprocess.Popen(
                    [IPERF_BIN, '-s', '-p', str(port), '--one-off'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                _server_procs.append(proc)
        except Exception as e:
            logger.warning(f'  Server on port {port} failed to start: {e}')

    time.sleep(1.5)   # give servers time to bind before first client
    logger.info('All iperf servers ready')


def stop_servers(net_or_none, logger: logging.Logger):
    """Kill all iperf server processes."""
    logger.info('Stopping iperf servers')
    for proc in _server_procs:
        try:
            proc.terminate()
        except Exception:
            pass
    if net_or_none is not None:
        h2 = net_or_none.get(SERVER_HOST)
        h2.cmd(f'pkill -f {IPERF_BIN} 2>/dev/null; true')
    else:
        subprocess.call(['pkill', '-f', IPERF_BIN],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────────────────────────────────────────────────────
#  Optional CSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_plan(csv_path: str, logger: logging.Logger) -> list:
    """
    Load flow plan from simulation_traffic_profile.csv.
    Returns list of (wave, duration, bw_mbps, protocol) tuples,
    assigning wave numbers by load_zone ordering.
    """
    if not os.path.isfile(csv_path):
        logger.warning(f'CSV not found: {csv_path} — using built-in FLOW_PLAN')
        return FLOW_PLAN

    rows = []
    zone_to_wave = {}
    wave_counter = [1]

    def get_wave(zone):
        if zone not in zone_to_wave:
            zone_to_wave[zone] = wave_counter[0]
            wave_counter[0] += 1
        return zone_to_wave[zone]

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]

        for row in reader:
            row = {k.strip().lower(): v.strip() for k, v in row.items()}
            try:
                rows.append((
                    get_wave(row.get('load_zone', 'default')),
                    float(row['duration']),
                    float(row['bandwidth']),
                    row.get('protocol_type', 'udp').lower(),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f'Skipping bad CSV row {row}: {e}')

    logger.info(f'Loaded {len(rows)} flows from {csv_path}  zones→waves: {zone_to_wave}')
    return rows if rows else FLOW_PLAN


# ─────────────────────────────────────────────────────────────────────────────
#  Core entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_parallel_traffic(net_or_none=None, csv_path: str = None):
    logger = setup_logging()
    stats  = Stats()
    sem    = threading.Semaphore(MAX_PARALLEL)

    # ── Load plan ─────────────────────────────────────────────────────────
    if csv_path and os.path.isfile(csv_path):
        plan = load_csv_plan(csv_path, logger)
    else:
        plan = FLOW_PLAN
        logger.info('Using built-in FLOW_PLAN')

    # Group by wave
    waves: dict = {}
    for item in plan:
        wave, dur, bw, proto = item
        waves.setdefault(wave, []).append((dur, bw, proto))

    # ── Graceful shutdown hook ─────────────────────────────────────────────
    def _shutdown(signum, frame):
        logger.warning('Signal received — killing all flows and exiting cleanly')
        kill_all_flows(logger)
        stop_servers(net_or_none, logger)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Header ────────────────────────────────────────────────────────────
    total_flows = len(plan)
    logger.info('═' * 65)
    logger.info('  Parallel iperf Traffic Engine')
    logger.info(f'  Flows       : {total_flows}  across {len(waves)} wave(s)')
    logger.info(f'  Max parallel: {MAX_PARALLEL}')
    logger.info(f'  Server      : {SERVER_IP}:{IPERF_PORT}+')
    logger.info(f'  Stagger     : {STAGGER_S}s  |  Wave gap: {WAVE_GAP_S}s')
    logger.info('═' * 65)

    # ── Start servers ─────────────────────────────────────────────────────
    start_servers(plan, net_or_none, logger)

    sim_t0     = time.time()
    flow_id    = 0
    all_threads: list = []

    try:
        for wave_num in sorted(waves.keys()):
            wave_flows = waves[wave_num]
            flow_count = len(wave_flows)
            logger.info(
                f'\n── Wave {wave_num} ── ({flow_count} flow(s), '
                f'aggregate ≈ {sum(b for _,b,_ in wave_flows):.0f} Mbps)'
            )

            wave_threads = []

            for dur, bw, proto in wave_flows:
                flow_id += 1
                t = threading.Thread(
                    target=flow_worker,
                    args=(flow_id, wave_num, dur, bw, proto,
                          net_or_none, stats, sem, logger),
                    name=f'F{flow_id:03d}',
                    daemon=True,          # won't block process exit
                )
                wave_threads.append(t)
                all_threads.append(t)
                t.start()
                logger.debug(f'  Thread {t.name} started, sleeping {STAGGER_S}s')
                time.sleep(STAGGER_S)    # stagger within wave → overlapping flows

            # Optional: wait for current wave before starting next
            # Comment out these two lines to let ALL waves overlap freely
            logger.info(f'  Wave {wave_num}: all {flow_count} flow(s) launched — '
                        f'waiting for wave to finish...')
            for t in wave_threads:
                t.join()

            logger.info(f'  Wave {wave_num} done.  Sleeping {WAVE_GAP_S}s before next wave.')
            time.sleep(WAVE_GAP_S)

        # ── Wait for any remaining stragglers ─────────────────────────────
        logger.info('\nAll waves dispatched — waiting for stragglers...')
        for t in all_threads:
            t.join(timeout=FLOW_TIMEOUT + 10)
            if t.is_alive():
                logger.warning(f'Thread {t.name} still alive after timeout — may be stuck')

    except Exception as exc:
        logger.error(f'Fatal error in main loop: {exc}', exc_info=True)
        kill_all_flows(logger)

    finally:
        stop_servers(net_or_none, logger)

    # ── Final summary ─────────────────────────────────────────────────────
    sim_elapsed = time.time() - sim_t0
    logger.info('')
    logger.info('═' * 65)
    logger.info('  SIMULATION COMPLETE')
    logger.info(f'  Total flows    : {stats.started}')
    logger.info(f'  Succeeded      : {stats.finished - stats.failed}')
    logger.info(f'  Failed         : {stats.failed}')
    logger.info(f'  Peak parallel  : {stats.peak}   (target ≥ 3)')
    logger.info(f'  Wall time      : {sim_elapsed:.1f}s')
    logger.info(f'  Flow logs      : {LOG_DIR}/')
    logger.info('═' * 65)

    if stats.peak < 3:
        logger.warning(
            'Peak parallel flows was < 3. Reduce STAGGER_S or '
            'increase flow durations to guarantee overlap.'
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Entry points
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Standalone / sudo python ───────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Parallel iperf traffic generator for Mininet'
    )
    parser.add_argument('--server-ip', default=SERVER_IP)
    parser.add_argument('--csv',       default='simulation_traffic_profile.csv')
    parser.add_argument('--max-par',   type=int, default=MAX_PARALLEL)
    args = parser.parse_args()

    SERVER_IP    = args.server_ip
    MAX_PARALLEL = args.max_par

    net = None
    try:
        import __main__
        net = getattr(__main__, 'net', None)
    except Exception:
        pass

    run_parallel_traffic(net_or_none=net, csv_path=args.csv)

elif 'net' in dir():   # noqa: F821  — injected by Mininet CLI
    # ── Mininet CLI: py exec(open('tests/iperf_parallel_traffic.py').read()) ─
    run_parallel_traffic(net_or_none=net)   # noqa: F821