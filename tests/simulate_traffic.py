#!/usr/bin/env python3
"""
simulate_traffic.py — Mininet Traffic Simulation from CSV Profile
=================================================================
Reads a traffic profile CSV and replays each row as an iperf3 flow
from h1 → h2 inside a running Mininet network.

CSV columns expected:
    duration        (int/float)  – iperf -t value in seconds
    bandwidth       (int/float)  – target bandwidth in Mbps
    protocol_type   (str)        – 'udp' or 'tcp'
    load_zone       (str)        – label used for logging/grouping

KEY FIX: Uses a port_pool (Queue) of 4 iperf3 servers on ports 5201–5204.
Each concurrent flow gets an exclusive port — eliminates "server busy" rc=1.

Run from Mininet CLI via the launcher:
    mininet> py exec(open('tests/run_simulation.py').read())
"""

import os
import sys
import csv
import time
import queue
import random
import logging
import argparse
import threading
import subprocess
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────

DEFAULT_CSV     = 'tests/simulation_traffic_profile.csv'
LOG_DIR         = 'logs'
DEFAULT_PORTS   = [5201, 5202, 5203, 5204]   # port pool — one per concurrent slot
IPERF_CMD       = 'iperf3'
SERVER_HOST     = 'h2'
CLIENT_HOST     = 'h1'
SERVER_IP       = '10.0.0.2'

MIN_INTER_DELAY = 0.3    # seconds between flow launches (min)
MAX_INTER_DELAY = 1.0    # seconds between flow launches (max)
MAX_CONCURRENT  = 4      # must equal len(DEFAULT_PORTS)
BW_HARD_CAP     = 100    # Mbps hard cap (0 = no cap)

# Minimum duration — iperf3 needs at least 1 second to be reliable
MIN_DURATION_S  = 1

# Timeout for a single flow (seconds). Flows that hang longer are killed.
FLOW_TIMEOUT_S  = 60


# ─────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'traffic_sim_{ts}.log')

    fmt = logging.Formatter(
        '%(asctime)s  %(levelname)-8s  %(threadName)-18s  %(message)s',
        datefmt='%H:%M:%S'
    )

    # Use a unique logger name to avoid duplicate handlers across re-runs
    logger = logging.getLogger(f'TrafficSim_{ts}')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f'Log file → {log_file}')
    return logger


# ─────────────────────────────────────────────────────────────────
#  CSV Loader & Validator
# ─────────────────────────────────────────────────────────────────

REQUIRED_COLS = {'duration', 'bandwidth', 'protocol_type', 'load_zone'}


def load_profile(csv_path: str, logger: logging.Logger) -> list:
    """Load and validate the traffic profile CSV. Returns list of dicts."""
    if not os.path.isfile(csv_path):
        logger.error(f'CSV not found: {csv_path}')
        sys.exit(1)

    rows   = []
    errors = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]

        missing = REQUIRED_COLS - set(reader.fieldnames)
        if missing:
            logger.error(f'CSV missing required columns: {missing}')
            sys.exit(1)

        for lineno, row in enumerate(reader, start=2):
            row = {k.strip().lower(): v.strip() for k, v in row.items()}

            # ── duration ─────────────────────────────────────────
            try:
                duration = float(row['duration'])
                if duration <= 0:
                    raise ValueError('must be > 0')
                if duration < MIN_DURATION_S:
                    duration = MIN_DURATION_S
            except ValueError as e:
                errors.append(f'Row {lineno}: bad duration "{row["duration"]}" — {e}')
                continue

            # ── bandwidth ─────────────────────────────────────────
            try:
                bandwidth = float(row['bandwidth'])
                if bandwidth <= 0:
                    raise ValueError('must be > 0')
                if BW_HARD_CAP and bandwidth > BW_HARD_CAP:
                    logger.warning(
                        f'Row {lineno}: bandwidth {bandwidth:.2f}M exceeds '
                        f'hard cap {BW_HARD_CAP}M — clamping.'
                    )
                    bandwidth = float(BW_HARD_CAP)
            except ValueError as e:
                errors.append(f'Row {lineno}: bad bandwidth "{row["bandwidth"]}" — {e}')
                continue

            # ── protocol ─────────────────────────────────────────
            protocol = row.get('protocol_type', 'udp').lower().strip()
            if protocol not in ('udp', 'tcp'):
                logger.warning(f'Row {lineno}: unknown protocol "{protocol}", defaulting to udp')
                protocol = 'udp'

            rows.append({
                'lineno':    lineno,
                'duration':  duration,
                'bandwidth': bandwidth,
                'protocol':  protocol,
                'load_zone': row['load_zone'],
            })

    if errors:
        logger.error(f'{len(errors)} validation error(s):')
        for e in errors:
            logger.error(f'  {e}')

    logger.info(f'Loaded {len(rows)} valid flow(s) from {csv_path}')
    return rows


# ─────────────────────────────────────────────────────────────────
#  Thread-safe stats collector
# ─────────────────────────────────────────────────────────────────

class FlowStats:
    def __init__(self):
        self._lock   = threading.Lock()
        self.success = 0
        self.failed  = 0
        self.results = []

    def record(self, row: dict, rc: int, stdout: str, stderr: str):
        with self._lock:
            ok = (rc == 0)
            if ok:
                self.success += 1
            else:
                self.failed += 1
            self.results.append({**row, 'ok': ok, 'rc': rc,
                                  'stdout': stdout, 'stderr': stderr})


# ─────────────────────────────────────────────────────────────────
#  iperf3 command builder
# ─────────────────────────────────────────────────────────────────

def build_iperf_cmd(row: dict, server_ip: str, port: int) -> list:
    """Build iperf3 command list for one flow."""
    cmd = [
        IPERF_CMD,
        '-c', server_ip,
        '-p', str(port),
        '-t', str(int(max(row['duration'], MIN_DURATION_S))),
        '-b', f"{row['bandwidth']}M",
        '-i', '0',       # suppress per-interval output
    ]
    if row['protocol'] == 'udp':
        cmd.append('-u')
    return cmd


# ─────────────────────────────────────────────────────────────────
#  Flow worker — uses port_pool to get exclusive server port
# ─────────────────────────────────────────────────────────────────

def run_flow(row: dict, net_or_none, port_pool: queue.Queue,
             stats: FlowStats, logger: logging.Logger,
             log_dir: str, server_ip: str):
    """
    1. Blocks on port_pool.get() to acquire an exclusive iperf3 server port.
    2. Runs iperf3 inside Mininet h1 namespace using h1.popen().
    3. Returns port to pool when done.

    This ensures max MAX_CONCURRENT flows and each gets its own server port
    — eliminates rc=1 'server busy' failures completely.
    """
    tag = (f"[row={row['lineno']} zone={row['load_zone']} "
           f"bw={row['bandwidth']}M dur={row['duration']}s "
           f"proto={row['protocol']}]")

    # Block here until a port slot is free
    port = port_pool.get()

    try:
        logger.info(f'START {tag} port={port}')
        cmd     = build_iperf_cmd(row, server_ip, port)
        cmd_str = ' '.join(cmd)
        logger.debug(f'CMD: {cmd_str}')

        t0     = time.time()
        rc     = -99
        stdout = ''
        stderr = ''

        try:
            timeout = max(row['duration'] + 15, FLOW_TIMEOUT_S)

            if net_or_none is not None:
                h1   = net_or_none.get(CLIENT_HOST)
                proc = h1.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            try:
                out, err = proc.communicate(timeout=timeout)
                stdout   = out.decode('utf-8', errors='replace')
                stderr   = err.decode('utf-8', errors='replace')
                rc       = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                stdout   = out.decode('utf-8', errors='replace')
                stderr   = 'TIMEOUT'
                rc       = -1
                logger.error(f'TIMEOUT {tag}')

        except Exception as exc:
            rc     = -2
            stderr = str(exc)
            logger.error(f'EXCEPTION {tag}: {exc}')

        elapsed = time.time() - t0

        # ── Per-flow log ──────────────────────────────────────────
        ts_str   = datetime.now().strftime('%H%M%S_%f')
        flow_log = os.path.join(
            log_dir,
            f"flow_r{row['lineno']}_{row['load_zone']}_{ts_str}.log"
        )
        try:
            with open(flow_log, 'w') as f:
                f.write(f'CMD    : {cmd_str}\n')
                f.write(f'PORT   : {port}\n')
                f.write(f'RC     : {rc}\n')
                f.write(f'ELAPSED: {elapsed:.3f}s\n\n')
                f.write('=== STDOUT ===\n')
                f.write(stdout or '(empty)\n')
                f.write('\n=== STDERR ===\n')
                f.write(stderr or '(empty)\n')
        except Exception:
            pass

        stats.record(row, rc, stdout, stderr)

        if rc == 0:
            logger.info(f'DONE  {tag}  elapsed={elapsed:.2f}s')
        else:
            logger.warning(f'FAIL  {tag}  rc={rc}  elapsed={elapsed:.2f}s')
            if stderr and stderr not in ('TIMEOUT',):
                logger.debug(f'  STDERR: {stderr[:200].strip()}')

    finally:
        # ALWAYS return port to pool, even if flow failed
        port_pool.put(port)


# ─────────────────────────────────────────────────────────────────
#  Server management
# ─────────────────────────────────────────────────────────────────

def start_iperf_servers(net_or_none, ports: list, logger: logging.Logger):
    """
    Start one persistent iperf3 server per port on h2.
    Called only when skip_server_start=False (standalone mode).
    """
    logger.info(f'Starting {len(ports)} iperf3 servers on {SERVER_HOST}: ports {ports}')

    if net_or_none is not None:
        h2 = net_or_none.get(SERVER_HOST)
        h2.cmd('pkill -f iperf3 2>/dev/null; true')
        time.sleep(0.5)
        for port in ports:
            h2.cmd(f'iperf3 -s -p {port} -D --logfile /tmp/iperf3_server_{port}.log 2>/dev/null; true')
            time.sleep(0.3)
            logger.info(f'  Server started on port {port}')
    else:
        subprocess.call(['pkill', '-f', IPERF_CMD],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        for port in ports:
            subprocess.Popen(
                [IPERF_CMD, '-s', '-p', str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(0.3)
            logger.info(f'  Server started on port {port}')

    time.sleep(1.5)
    logger.info('All iperf3 servers ready')


def verify_servers(net_or_none, server_ip: str, ports: list, logger: logging.Logger) -> bool:
    """
    Quick 1-second test on each port to confirm all servers are reachable.
    Returns True only if ALL ports respond.
    """
    logger.info(f'Verifying iperf3 servers on ports {ports}...')
    all_ok = True

    for port in ports:
        try:
            cmd = [IPERF_CMD, '-c', server_ip, '-p', str(port), '-t', '1', '-i', '0']
            if net_or_none is not None:
                h1   = net_or_none.get(CLIENT_HOST)
                proc = h1.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            out, err = proc.communicate(timeout=10)
            out_str  = out.decode('utf-8', errors='replace')
            rc       = proc.returncode

            if rc == 0:
                logger.info(f'  Port {port} — OK')
            else:
                logger.error(f'  Port {port} — NOT reachable (rc={rc})')
                logger.error(f'  stderr: {err.decode("utf-8", errors="replace")[:200]}')
                all_ok = False

        except Exception as e:
            logger.error(f'  Port {port} — check failed: {e}')
            all_ok = False

    return all_ok


def stop_iperf_servers(net_or_none, logger: logging.Logger):
    """Kill all iperf3 processes on h2."""
    logger.info('Stopping iperf3 servers')
    if net_or_none is not None:
        h2 = net_or_none.get(SERVER_HOST)
        h2.cmd('pkill -f iperf3 2>/dev/null; true')
    else:
        subprocess.call(['pkill', '-f', IPERF_CMD],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────────────────────────────────────────
#  Main simulation loop
# ─────────────────────────────────────────────────────────────────

def run_simulation(
    csv_path: str,
    net_or_none=None,
    server_ip: str = SERVER_IP,
    server_ports: list = None,
    max_concurrent: int = MAX_CONCURRENT,
    log_dir: str = LOG_DIR,
    skip_server_start: bool = False,
):
    """
    Main entry point.

    server_ports: list of iperf3 server ports (default: [5201,5202,5203,5204])
                  Must have >= max_concurrent entries.
    skip_server_start: set True if run_simulation.py already started the servers.
    """
    if server_ports is None:
        server_ports = DEFAULT_PORTS

    # Safety: port pool must be >= concurrency
    if len(server_ports) < max_concurrent:
        max_concurrent = len(server_ports)

    logger = setup_logging(log_dir)
    logger.info('═' * 62)
    logger.info('  Mininet Traffic Simulator — starting')
    logger.info(f'  CSV        : {csv_path}')
    logger.info(f'  Server     : {server_ip}')
    logger.info(f'  Ports      : {server_ports}')
    logger.info(f'  MaxConc    : {max_concurrent}')
    logger.info(f'  Min dur    : {MIN_DURATION_S}s  (enforced)')
    logger.info('═' * 62)

    # Load and validate CSV
    profile = load_profile(csv_path, logger)
    if not profile:
        logger.error('No valid flows to run. Exiting.')
        return

    # Start servers (only if not already started by launcher)
    if not skip_server_start:
        start_iperf_servers(net_or_none, server_ports, logger)

    # Verify all ports before starting 300 flows
    if not verify_servers(net_or_none, server_ip, server_ports, logger):
        logger.error('One or more iperf3 servers are not reachable.')
        logger.error('Run scripts/cleanup.sh and restart controller + topology.')
        stop_iperf_servers(net_or_none, logger)
        return

    # ── Port pool: queue with one slot per server port ────────────
    # Each flow thread blocks on get() until a port is free,
    # then returns the port via put() when done.
    # This completely prevents "server busy" (rc=1) errors.
    port_pool = queue.Queue()
    for p in server_ports[:max_concurrent]:
        port_pool.put(p)

    stats     = FlowStats()
    threads   = []
    sim_start = time.time()

    for idx, row in enumerate(profile):
        zone = row['load_zone']
        logger.info(
            f'Scheduling flow {idx+1}/{len(profile)}  '
            f'zone={zone}  bw={row["bandwidth"]}M  '
            f'dur={row["duration"]}s  proto={row["protocol"]}'
        )

        t = threading.Thread(
            target=run_flow,
            args=(row, net_or_none, port_pool, stats, logger, log_dir, server_ip),
            name=f'Flow-{idx+1:03d}-{zone}',
            daemon=True,
        )
        threads.append(t)
        t.start()

        # Staggered delay between launches to create realistic overlapping flows
        delay = random.uniform(MIN_INTER_DELAY, MAX_INTER_DELAY)
        time.sleep(delay)

    logger.info('All flows scheduled — waiting for completion...')
    for t in threads:
        t.join()

    sim_elapsed = time.time() - sim_start
    stop_iperf_servers(net_or_none, logger)

    # ── Summary ───────────────────────────────────────────────────
    logger.info('═' * 62)
    logger.info('  SIMULATION COMPLETE')
    logger.info(f'  Total flows : {len(profile)}')
    logger.info(f'  Succeeded   : {stats.success}')
    logger.info(f'  Failed      : {stats.failed}')
    logger.info(f'  Wall time   : {sim_elapsed:.1f}s')
    logger.info('═' * 62)

    if stats.failed:
        logger.warning(f'{stats.failed} flow(s) failed — check logs in {log_dir}/')

    # Per-zone summary
    zones = {}
    for r in stats.results:
        z = r['load_zone']
        zones.setdefault(z, {'ok': 0, 'fail': 0})
        if r['ok']:
            zones[z]['ok'] += 1
        else:
            zones[z]['fail'] += 1

    logger.info('  Per-zone results:')
    for z, counts in sorted(zones.items()):
        logger.info(f'    {z:<20}  ok={counts["ok"]}  fail={counts["fail"]}')
    logger.info('═' * 62)


# ─────────────────────────────────────────────────────────────────
#  Entry point (standalone / testing without Mininet)
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Replay a traffic profile CSV as iperf3 flows in Mininet'
    )
    parser.add_argument('--csv',       default=DEFAULT_CSV)
    parser.add_argument('--server-ip', default=SERVER_IP)
    parser.add_argument('--ports',     default='5201,5202,5203,5204',
                        help='Comma-separated list of iperf3 server ports')
    parser.add_argument('--max-conc',  default=MAX_CONCURRENT, type=int)
    parser.add_argument('--log-dir',   default=LOG_DIR)
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(',')]

    run_simulation(
        csv_path=args.csv,
        net_or_none=None,
        server_ip=args.server_ip,
        server_ports=ports,
        max_concurrent=args.max_conc,
        log_dir=args.log_dir,
    )


if __name__ == '__main__':
    main()