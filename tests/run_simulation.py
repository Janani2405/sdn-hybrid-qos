"""
run_simulation.py — Simulation Launcher for SDN QoS Project (32-host topology)
===============================================================================
Run from inside the Mininet CLI ONLY:

    mininet> py exec(open('tests/run_simulation.py').read())

Topology: sudo mn --controller remote --topo tree,fanout=2,depth=5
          → 31 switches, 32 hosts (h1–h32), IPs 10.0.0.1–10.0.0.32

What this script does:
  Step 1 — Kill any leftover iperf3 processes on ALL 32 hosts
  Step 2 — Read the CSV to find which hosts are actually used as dst
  Step 3 — Start iperf3 servers ONLY on those dst hosts
  Step 4 — Verify a sample of src→dst pairs (quick 1s test flow)
  Step 5 — Load simulate_traffic_v2.py and call run_simulation()
"""

import os
import sys
import time
import types
import subprocess

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
CSV_PATH       = 'tests/simulation_traffic_profile.csv'
#CSV_PATH = 'tests/simulation_traffic_profile_test.csv'
SIM_SCRIPT     = 'tests/simulate_traffic_v2.py'
LOG_DIR        = 'logs'
SERVER_PORTS   = list(range(5201, 5217))   # 16 ports
MAX_CONCURRENT = 16

# Full 32-host IP map — Mininet tree assigns IPs in host-number order
HOST_IPS = {f'h{i}': f'10.0.0.{i}' for i in range(1, 33)}
ALL_HOSTS = [f'h{i}' for i in range(1, 33)]

print('\n' + '='*65)
print('  [run_simulation] SDN Traffic Simulation Launcher (32-host)')
print('='*65)

# ── Detect Mininet net object ────────────────────────────────────
try:
    net_obj = net   # injected by Mininet CLI's py exec
    print(f'  Mininet net object found — running in real network mode')
except NameError:
    net_obj = None
    print('  WARNING: no Mininet net object — standalone mode (testing only)')

# ─────────────────────────────────────────────────────────────────
#  STEP 1 — Kill leftover iperf3 on ALL 32 hosts
# ─────────────────────────────────────────────────────────────────
print('\n[Step 1] Killing leftover iperf3 on all 32 hosts...')
for hname in ALL_HOSTS:
    if net_obj is not None:
        h = net_obj.get(hname)
        h.cmd('pkill -f iperf3 2>/dev/null; true')
    else:
        subprocess.call(['pkill', '-f', 'iperf3'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        break   # in standalone mode, one pkill call is enough
time.sleep(0.5)
print('  Done.')

# ─────────────────────────────────────────────────────────────────
#  STEP 2 — Read CSV to find dst hosts and src→dst pairs
# ─────────────────────────────────────────────────────────────────
print(f'\n[Step 2] Reading CSV: {CSV_PATH}')
if not os.path.isfile(CSV_PATH):
    print(f'  ERROR: {CSV_PATH} not found. Aborting.')
    raise SystemExit(1)

import csv as _csv
dst_hosts    = set()
src_dst_pairs = []
with open(CSV_PATH, newline='') as f:
    reader = _csv.DictReader(f)
    reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
    for row in reader:
        src = row.get('src_host','').strip()
        dst = row.get('dst_host','').strip()
        if src and dst and src != dst:
            dst_hosts.add(dst)
            src_dst_pairs.append((src, dst))

dst_hosts     = sorted(dst_hosts, key=lambda h: int(h[1:]))
unique_pairs  = list({(s,d) for s,d in src_dst_pairs})
print(f'  Dst hosts ({len(dst_hosts)}): {dst_hosts}')
print(f'  Unique src→dst pairs: {len(unique_pairs)}')

# ─────────────────────────────────────────────────────────────────
#  STEP 3 — Start iperf3 servers ONLY on dst hosts from the CSV
# ─────────────────────────────────────────────────────────────────
print(f'\n[Step 3] Starting iperf3 servers on {len(dst_hosts)} dst hosts, '
      f'{len(SERVER_PORTS)} ports each...')
os.makedirs(LOG_DIR, exist_ok=True)

for hname in dst_hosts:
    for port in SERVER_PORTS:
        logfile = f'/tmp/iperf3_{hname}_{port}.log'
        cmd = (f'while true; do iperf3 -s -p {port} --one-off '
               f'>> {logfile} 2>&1; done')
        if net_obj is not None:
            h = net_obj.get(hname)
            h.cmd(cmd + ' &')
        else:
            subprocess.Popen(
                ['bash', '-c', f'while true; do iperf3 -s -p {port} --one-off; done'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        time.sleep(0.03)
    print(f'  {hname}: servers started on {len(SERVER_PORTS)} ports')

print('  Waiting 2.5s for servers to settle...')
time.sleep(2.5)

# ─────────────────────────────────────────────────────────────────
#  STEP 4 — Verify a sample of src→dst pairs
# ─────────────────────────────────────────────────────────────────
print(f'\n[Step 4] Verifying connectivity (sample of src→dst pairs)...')

# Sample up to 8 unique pairs for the check — covers fixed + long-distance
import random as _random
_random.seed(0)
sample_pairs = unique_pairs if len(unique_pairs) <= 8 else _random.sample(unique_pairs, 8)

all_ok = True
for src_name, dst_name in sample_pairs:
    if dst_name not in HOST_IPS or src_name not in HOST_IPS:
        print(f'  SKIP {src_name}→{dst_name} (not in HOST_IPS)')
        continue
    server_ip = HOST_IPS[dst_name]
    port      = SERVER_PORTS[0]
    cmd       = ['iperf3', '-c', server_ip, '-p', str(port), '-t', '1', '-i', '0']

    try:
        if net_obj is not None:
            src_host = net_obj.get(src_name)
            proc = src_host.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out, err = proc.communicate(timeout=12)
        out_str  = (out or b'').decode('utf-8', errors='replace')
        err_str  = (err or b'').decode('utf-8', errors='replace')

        if proc.returncode == 0 or any(k in out_str for k in ('Mbits','sender','receiver')):
            print(f'  {src_name}→{dst_name} ({server_ip}:{port}) — OK')
        else:
            print(f'  {src_name}→{dst_name} — FAILED')
            print(f'    stdout: {out_str[:120].strip()}')
            print(f'    stderr: {err_str[:120].strip()}')
            all_ok = False

    except subprocess.TimeoutExpired:
        print(f'  {src_name}→{dst_name} — TIMEOUT')
        all_ok = False
    except Exception as exc:
        print(f'  {src_name}→{dst_name} — ERROR: {exc}')
        all_ok = False

if not all_ok:
    print('\n  WARNING: Some pairs failed verification.')
    print('  Check: ryu-manager is running, pingall shows 0% drop, then retry.')
else:
    print('  All sampled pairs reachable — OK')

# ─────────────────────────────────────────────────────────────────
#  STEP 5 — Load simulate_traffic_v2.py and run
# ─────────────────────────────────────────────────────────────────
print(f'\n[Step 5] Loading {SIM_SCRIPT}...')

if not os.path.isfile(SIM_SCRIPT):
    print(f'  ERROR: {SIM_SCRIPT} not found.')
    raise SystemExit(1)

mod = types.ModuleType('simulate_traffic_v2')
mod.__file__ = SIM_SCRIPT
with open(SIM_SCRIPT, 'r') as f:
    source = f.read()

try:
    exec(compile(source, SIM_SCRIPT, 'exec'), mod.__dict__)
except Exception as e:
    print(f'  ERROR compiling {SIM_SCRIPT}: {e}')
    raise

print(f'  Module loaded — calling run_simulation()...\n')

mod.run_simulation(
    csv_path          = CSV_PATH,
    net_or_none       = net_obj,
    server_ports      = SERVER_PORTS,
    max_concurrent    = MAX_CONCURRENT,
    log_dir           = LOG_DIR,
    skip_server_start = True,   # servers already started in Step 3
)

print('\n[run_simulation] Done. Check logs/ for qos_log.csv and congestion_log.csv.')
print('='*65 + '\n')