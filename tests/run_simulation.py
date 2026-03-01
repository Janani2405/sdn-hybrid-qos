#!/usr/bin/env python3
"""
run_simulation.py — Mininet CLI launcher for simulate_traffic.py
================================================================
Handles the scope problem of Mininet's py command by loading
simulate_traffic as a module and passing the live net object.

Starts 4 persistent iperf3 servers on h2 (ports 5201–5204) so that
up to 4 concurrent flows never hit a "server busy" conflict.

Run from Mininet CLI:
    mininet> py exec(open('tests/run_simulation.py').read())
"""

import os
import sys
import time
import types

# ── Step 1: Kill any leftover iperf3 from previous runs ──────────
print('[run_simulation] Cleaning up any leftover iperf3 processes...')
h2 = net.get('h2')
h1 = net.get('h1')
h2.cmd('pkill -f iperf3 2>/dev/null; true')
h1.cmd('pkill -f iperf3 2>/dev/null; true')
time.sleep(0.8)

# ── Step 2: Start 4 persistent iperf3 servers on h2 ─────────────
print('[run_simulation] Step 2: Starting 4 iperf3 servers on h2...')
PORTS = [5201, 5202, 5203, 5204]
for port in PORTS:
    h2.cmd(f'iperf3 -s -p {port} -D --logfile /tmp/iperf3_server_{port}.log 2>/dev/null; true')
    time.sleep(0.4)
    print(f'  Server started on port {port}')

time.sleep(1.5)   # allow all servers to bind

# ── Step 3: Verify all 4 ports are reachable from h1 ─────────────
print('[run_simulation] Step 3: Verifying all 4 ports from h1...')
all_ok = True
for port in PORTS:
    result = h1.cmd(f'iperf3 -c 10.0.0.2 -p {port} -t 1 -i 0 2>&1')
    if 'receiver' in result or 'sender' in result or 'Mbits' in result:
        print(f'  Port {port} — OK')
    else:
        print(f'  Port {port} — FAILED')
        print(f'    Output: {result[:200].strip()}')
        all_ok = False

if not all_ok:
    print('[run_simulation] WARNING: One or more ports not reachable.')
    print('[run_simulation] Tip: Run bash scripts/cleanup.sh and restart.')
    # Do not abort — partial success is better than nothing
else:
    print('[run_simulation] All 4 servers reachable — OK')

# ── Step 4: Load simulate_traffic as a fresh module ──────────────
print('[run_simulation] Step 4: Loading simulate_traffic.py...')
mod          = types.ModuleType('simulate_traffic')
mod.__file__ = 'tests/simulate_traffic.py'

with open('tests/simulate_traffic.py') as _f:
    exec(compile(_f.read(), 'tests/simulate_traffic.py', 'exec'), mod.__dict__)

# ── Step 5: Run the simulation ────────────────────────────────────
# skip_server_start=True because we already started all 4 servers above
mod.run_simulation(
    csv_path='tests/simulation_traffic_profile.csv',
    net_or_none=net,
    server_ip='10.0.0.2',
    server_ports=PORTS,          # pass the 4-port pool
    max_concurrent=4,
    log_dir='logs',
    skip_server_start=True,
)