#!/usr/bin/env python3
"""
simulate_traffic_v2.py — Mininet Traffic Simulation (32-host topology)
=======================================================================
Compatible with: tree,fanout=2,depth=5 → 32 hosts (h1–h32)
CSV columns (10):
  duration, bandwidth, protocol_type, load_zone,
  src_host, dst_host, flow_count, inter_arrival,
  traffic_type, packet_size_avg

STABILITY FIXES (v2.1)
=======================
  FIX 1 — Persistent iperf3 servers (no --one-off):
        Servers stay alive between clients. Eliminates the 50-200ms
        restart gap that caused immediate 'Connection refused' FAILs
        when a second client arrived while the server was relaunching.

  FIX 2 — All 16 ports available per dst host:
        Port pool now puts all server_ports into each host's queue
        instead of slots_per_host=1 (which left ports 5202-5216 unused
        and serialized all flows to the same dst behind a single slot).

  FIX 3 — Bandwidth hard-capped at 98 Mbps:
        Prevents any single flow from exceeding the 100 Mbps link
        capacity, which caused OVS to drop every packet above the cap
        → massive tx_dropped spikes → PacketIn flood → controller overload.

Run from Mininet CLI:
    mininet> py exec(open('tests/run_simulation.py').read())
"""

import os
import sys
import csv
import time
import queue
import logging
import threading
import subprocess
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────

DEFAULT_CSV    = 'tests/simulation_traffic_profile.csv'
LOG_DIR        = 'logs'
IPERF_CMD      = 'iperf3'
DEFAULT_PORTS  = list(range(5201, 5217))   # 16 ports
MAX_CONCURRENT = 16                        # scale up for 32 hosts
BW_HARD_CAP    = 98     # hard cap at 98 Mbps — keeps flows below 100 Mbps link capacity
MIN_DURATION_S = 1
MAX_DURATION_S = 30    # cap so ports recycle quickly
FLOW_TIMEOUT_S = 60    # duration + 30s margin

# Auto-generate HOST_IPS for h1–h32 (Mininet tree assigns in order)
HOST_IPS = {f'h{i}': f'10.0.0.{i}' for i in range(1, 33)}

REQUIRED_COLS = {
    'duration', 'bandwidth', 'protocol_type', 'load_zone',
    'src_host', 'dst_host', 'flow_count', 'inter_arrival',
    'traffic_type', 'packet_size_avg'
}

# ─────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────

def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'traffic_sim_v2_{ts}.log')
    fmt = logging.Formatter(
        '%(asctime)s  %(levelname)-8s  %(threadName)-22s  %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(f'TrafficSimV2_{ts}')
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
    logger.info(f'Log → {log_file}')
    return logger

# ─────────────────────────────────────────────────────────────────
#  CSV Loader
# ─────────────────────────────────────────────────────────────────

def load_profile(csv_path, logger):
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
            logger.error(f'CSV is missing required columns: {missing}')
            logger.error('Please regenerate simulation_traffic_profile.csv for the 32-host topology.')
            sys.exit(1)

        for lineno, row in enumerate(reader, start=2):
            row = {k.strip().lower(): v.strip() for k, v in row.items()}
            try:
                parsed = _parse_row(row, lineno)
                rows.append(parsed)
            except ValueError as e:
                errors.append(f'Row {lineno}: {e}')

    if errors:
        for e in errors[:10]:
            logger.warning(e)
        if len(errors) > 10:
            logger.warning(f'... and {len(errors)-10} more errors')

    logger.info(f'Loaded {len(rows)} valid flow(s) from {csv_path}')
    return rows

def _parse_row(row, lineno):
    duration = max(float(row['duration']), MIN_DURATION_S)
    duration = min(duration, MAX_DURATION_S)
    bandwidth = min(float(row['bandwidth']), BW_HARD_CAP)
    if bandwidth <= 0:
        raise ValueError(f'bandwidth must be > 0, got {bandwidth}')

    src_host = row.get('src_host', 'h1').strip()
    dst_host = row.get('dst_host', 'h2').strip()

    # Validate against full 32-host map
    if src_host not in HOST_IPS:
        raise ValueError(f'Unknown src_host "{src_host}" — must be h1–h32')
    if dst_host not in HOST_IPS:
        raise ValueError(f'Unknown dst_host "{dst_host}" — must be h1–h32')
    if src_host == dst_host:
        raise ValueError(f'src_host and dst_host cannot be the same ({src_host})')

    flow_count    = max(1, min(int(float(row.get('flow_count', 1))), 4))
    inter_arrival = max(0.0, float(row.get('inter_arrival', 0.5)))
    traffic_type  = row.get('traffic_type', 'web').strip()
    pkt_size      = int(max(64, min(float(row.get('packet_size_avg', 1400)), 1450)))
    protocol      = row.get('protocol_type', 'udp').lower()
    if protocol not in ('tcp', 'udp'):
        protocol = 'udp'

    return {
        'lineno':        lineno,
        'duration':      duration,
        'bandwidth':     bandwidth,
        'protocol':      protocol,
        'load_zone':     row.get('load_zone', 'LOW'),
        'src_host':      src_host,
        'dst_host':      dst_host,
        'server_ip':     HOST_IPS[dst_host],
        'flow_count':    flow_count,
        'inter_arrival': inter_arrival,
        'traffic_type':  traffic_type,
        'pkt_size':      pkt_size,
    }

# ─────────────────────────────────────────────────────────────────
#  iperf3 Command Builder
# ─────────────────────────────────────────────────────────────────

def build_iperf_cmd(row, port):
    cmd = [
        IPERF_CMD,
        '-c', row['server_ip'],
        '-p', str(port),
        '-t', str(int(max(row['duration'], MIN_DURATION_S))),
        '-b', f"{row['bandwidth']}M",
        '-P', str(row['flow_count']),
        '-l', str(row['pkt_size']),
        '-i', '0',
    ]
    if row['protocol'] == 'udp':
        cmd.append('-u')
    return cmd

# ─────────────────────────────────────────────────────────────────
#  Thread-safe stats
# ─────────────────────────────────────────────────────────────────

class FlowStats:
    def __init__(self):
        self._lock   = threading.Lock()
        self.success = 0
        self.failed  = 0
        self.results = []

    def record(self, row, rc, elapsed, stderr=''):
        with self._lock:
            ok = (rc == 0)
            if ok: self.success += 1
            else:  self.failed  += 1
            self.results.append({
                **row, 'ok': ok, 'rc': rc,
                'elapsed': elapsed, 'stderr': stderr[:200]
            })

# ─────────────────────────────────────────────────────────────────
#  Flow worker
# ─────────────────────────────────────────────────────────────────

def run_flow(row, net_or_none, port_pool, stats, logger, log_dir, global_sem):
    tag = (f"[#{row['lineno']} {row['load_zone']:<10} "
           f"{row['src_host']}→{row['dst_host']} "
           f"{row['bandwidth']:.1f}M {row['duration']}s "
           f"P={row['flow_count']} {row['traffic_type']}]")

    global_sem.acquire()
    port = port_pool.get()   # blocks until a port slot is free on this dst host
    try:
        cmd     = build_iperf_cmd(row, port)
        cmd_str = ' '.join(cmd)
        logger.info(f'START {tag} port={port}')
        logger.debug(f'CMD: {cmd_str}')

        t0      = time.time()
        rc      = -99
        err     = ''
        timeout = max(row['duration'] + 20, FLOW_TIMEOUT_S)

        try:
            if net_or_none is not None:
                src  = net_or_none.get(row['src_host'])
                proc = src.popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            else:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)

            try:
                out, err_b = proc.communicate(timeout=timeout)
                err = err_b.decode('utf-8', errors='replace')
                rc  = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                err = 'TIMEOUT'
                rc  = -1
                logger.error(f'TIMEOUT {tag}')

        except Exception as exc:
            rc  = -2
            err = str(exc)
            logger.error(f'EXCEPTION {tag}: {exc}')

        elapsed = round(time.time() - t0, 2)
        stats.record(row, rc, elapsed, err)

        if rc == 0:
            logger.info(f'DONE  {tag}  elapsed={elapsed}s')
        else:
            logger.warning(f'FAIL  {tag}  rc={rc}  elapsed={elapsed}s')
            if err and err != 'TIMEOUT':
                logger.debug(f'  STDERR: {err[:200].strip()}')

    finally:
        port_pool.put(port)
        global_sem.release()

# ─────────────────────────────────────────────────────────────────
#  Server management — starts servers ONLY on dst hosts in CSV
# ─────────────────────────────────────────────────────────────────

def start_servers(net_or_none, dst_hosts, ports, logger):
    """
    Start persistent iperf3 servers on every dst host that appears in the CSV.

    FIX: Uses plain 'iperf3 -s' WITHOUT --one-off.
    A persistent server accepts multiple sequential clients without restarting,
    eliminating the 50-200ms restart gap that caused 'Connection refused' FAILs
    when two clients arrived during --one-off's relaunch window.

    Each port gets one server process. The global semaphore in run_simulation()
    ensures at most max_concurrent clients run simultaneously across all hosts.
    """
    logger.info(f'Starting persistent iperf3 servers on {sorted(dst_hosts)} ports {ports}')

    for hname in sorted(dst_hosts):
        if net_or_none is not None:
            h = net_or_none.get(hname)
            # NOTE: Do NOT pkill here. Mininet hosts share a PID namespace,
            # so pkill -f iperf3 on hN kills ALL previous hosts' servers too.
            # The caller is responsible for killing leftover iperf3 before
            # calling start_servers() (run_simulation.py Step 1 handles this).
            for port in ports:
                # Use h.popen() NOT h.cmd('... &') — h.cmd with & is unreliable
                # in Mininet: subsequent h.cmd() calls can kill the backgrounded
                # process or block waiting for its output. h.popen() creates a
                # real persistent subprocess in the host's network namespace.
                logfile = f'/tmp/iperf3_{hname}_{port}.log'
                with open(logfile, 'a') as lf:
                    h.popen(['iperf3', '-s', '-p', str(port)],
                            stdout=lf, stderr=lf)
            logger.info(f'  {hname}: {len(ports)} persistent servers started')
        else:
            for port in ports:
                subprocess.Popen(
                    [IPERF_CMD, '-s', '-p', str(port)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

    time.sleep(2.0)   # allow servers to bind their ports
    logger.info('All servers ready')


def verify_servers(net_or_none, src_dst_pairs, ports, logger):
    """
    Quick 1-second test flow for each unique (src, dst) pair found in the CSV.
    Verifies actual end-to-end connectivity through the Ryu controller.
    """
    logger.info(f'Verifying {len(src_dst_pairs)} src→dst pair(s)...')
    all_ok  = True
    port    = ports[0]
    checked = set()

    for src, dst in src_dst_pairs:
        if (src, dst) in checked:
            continue
        checked.add((src, dst))

        server_ip = HOST_IPS[dst]
        cmd = [IPERF_CMD, '-c', server_ip, '-p', str(port), '-t', '1', '-i', '0']
        try:
            if net_or_none is not None:
                h    = net_or_none.get(src)
                proc = h.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)

            out, err = proc.communicate(timeout=12)
            # Mininet popen may return str instead of bytes — handle both
            out_str  = out.decode('utf-8', errors='replace') if isinstance(out, bytes) else (out or '')
            err_str  = err.decode('utf-8', errors='replace') if isinstance(err, bytes) else (err or '')

            if proc.returncode == 0 or any(k in out_str for k in ('Mbits','sender','receiver')):
                logger.info(f'  {src}→{dst} ({server_ip}:{port}) — OK')
            else:
                logger.error(f'  {src}→{dst} — FAILED')
                if err_str.strip():
                    logger.error(f'    stderr: {err_str[:200].strip()}')
                if out_str.strip():
                    logger.error(f'    stdout: {out_str[:200].strip()}')
                all_ok = False

        except subprocess.TimeoutExpired:
            logger.error(f'  {src}→{dst} — TIMEOUT')
            all_ok = False
        except Exception as e:
            logger.error(f'  {src}→{dst} — ERROR: {e}')
            all_ok = False

    return all_ok


def stop_servers(net_or_none, dst_hosts, logger):
    logger.info('Stopping iperf3 servers')
    for hname in dst_hosts:
        if net_or_none is not None:
            net_or_none.get(hname).cmd('pkill -f iperf3 2>/dev/null; true')
        else:
            subprocess.call(['pkill', '-f', IPERF_CMD],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ─────────────────────────────────────────────────────────────────
#  Main simulation loop
# ─────────────────────────────────────────────────────────────────

def run_simulation(
    csv_path          = DEFAULT_CSV,
    net_or_none       = None,
    server_ports      = None,
    max_concurrent    = MAX_CONCURRENT,
    log_dir           = LOG_DIR,
    skip_server_start = False,
):
    if server_ports is None:
        server_ports = DEFAULT_PORTS

    logger = setup_logging(log_dir)
    logger.info('═' * 65)
    logger.info('  SDN Traffic Simulator v2 — 32-host topology')
    logger.info(f'  CSV         : {csv_path}')
    logger.info(f'  Ports       : {server_ports}')
    logger.info(f'  MaxConc     : {max_concurrent}')
    logger.info('═' * 65)

    profile = load_profile(csv_path, logger)
    if not profile:
        logger.error('No valid flows. Exiting.')
        return

    # Derive the actual dst and src hosts present in the CSV
    dst_hosts = sorted(set(r['dst_host'] for r in profile),
                       key=lambda h: int(h[1:]))
    src_dst_pairs = list({(r['src_host'], r['dst_host']) for r in profile})

    logger.info(f'  Dst hosts   : {dst_hosts}')
    logger.info(f'  Unique pairs: {len(src_dst_pairs)}')

    if len(server_ports) < max_concurrent:
        max_concurrent = len(server_ports)

    if not skip_server_start:
        start_servers(net_or_none, dst_hosts, server_ports, logger)

    if not verify_servers(net_or_none, src_dst_pairs, server_ports, logger):
        logger.warning('Server verification had failures — continuing anyway.')
        logger.warning('Some flows may fail. Check ryu-manager is running if needed.')

    # Per-dst-host port pools — give each dst host multiple port slots so
    # flows don't all queue behind port 5201.
    # With 25 dst hosts and 16 ports, we assign ALL ports to each host's pool
    # (each port can only serve 1 client at a time via the persistent server).
    # The global semaphore caps total truly-running threads to max_concurrent.
    host_port_pools = {}
    for dst in dst_hosts:
        q = queue.Queue()
        for p in server_ports:          # all 16 ports available per host
            q.put(p)
        host_port_pools[dst] = q

    # Global semaphore caps total truly-running threads
    global_sem = threading.Semaphore(max_concurrent)

    stats     = FlowStats()
    threads   = []
    sim_start = time.time()

    for idx, row in enumerate(profile):
        logger.info(
            f'Scheduling {idx+1}/{len(profile)}  '
            f'{row["load_zone"]:<10} {row["src_host"]}→{row["dst_host"]}  '
            f'{row["bandwidth"]:.1f}M  {row["duration"]}s  '
            f'P={row["flow_count"]}  {row["traffic_type"]}'
        )

        host_pool = host_port_pools[row['dst_host']]
        t = threading.Thread(
            target=run_flow,
            args=(row, net_or_none, host_pool, stats, logger, log_dir, global_sem),
            name=f'Flow-{idx+1:03d}-{row["load_zone"]}',
            daemon=True,
        )
        threads.append(t)
        t.start()

        delay = row['inter_arrival']
        if delay > 0:
            time.sleep(delay)

    logger.info(f'All {len(profile)} flows scheduled — waiting for completion...')
    for t in threads:
        t.join()

    elapsed = time.time() - sim_start
    stop_servers(net_or_none, dst_hosts, logger)

    # ── Summary ────────────────────────────────────────────────────
    logger.info('═' * 65)
    logger.info('  SIMULATION COMPLETE')
    logger.info(f'  Total : {len(profile)}  ✓ {stats.success}  ✗ {stats.failed}')
    logger.info(f'  Time  : {elapsed:.1f}s')

    zones = {}
    for r in stats.results:
        z = r['load_zone']
        zones.setdefault(z, {'ok': 0, 'fail': 0})
        zones[z]['ok' if r['ok'] else 'fail'] += 1

    logger.info('  Per-zone:')
    for z, c in sorted(zones.items()):
        logger.info(f'    {z:<12}  ok={c["ok"]}  fail={c["fail"]}')

    ttypes = {}
    for r in stats.results:
        tt = r['traffic_type']
        ttypes.setdefault(tt, {'ok': 0, 'fail': 0})
        ttypes[tt]['ok' if r['ok'] else 'fail'] += 1

    logger.info('  Per-traffic-type:')
    for tt, c in sorted(ttypes.items()):
        logger.info(f'    {tt:<10}  ok={c["ok"]}  fail={c["fail"]}')
    logger.info('═' * 65)


# ─────────────────────────────────────────────────────────────────
#  Standalone entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',      default=DEFAULT_CSV)
    parser.add_argument('--ports',    default=','.join(str(p) for p in DEFAULT_PORTS))
    parser.add_argument('--max-conc', default=MAX_CONCURRENT, type=int)
    parser.add_argument('--log-dir',  default=LOG_DIR)
    args  = parser.parse_args()
    ports = [int(p.strip()) for p in args.ports.split(',')]
    run_simulation(csv_path=args.csv, server_ports=ports,
                   max_concurrent=args.max_conc, log_dir=args.log_dir)