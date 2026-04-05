"""
=============================================================================
traffic_gen.py  —  Realistic traffic generator for Mininet training
=============================================================================
Drives iperf3 flows between random host pairs in the tree topology to create
varied traffic patterns — burst, sustained, background — so the DQN sees
diverse states during training.

Run INSIDE the Mininet CLI or as a separate terminal:
    python3 traffic_gen.py

Requirements: iperf3 installed on all Mininet hosts (default in Ubuntu).

Tree topology  (depth=5, fanout=2):
  Hosts: h1..h32
  Each host assigned IP 10.0.0.1 through 10.0.0.32
=============================================================================
"""

import subprocess
import random
import time
import threading
import logging
import sys

log = logging.getLogger('traffic')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  [TRAFFIC]  %(message)s',
                    datefmt='%H:%M:%S',
                    stream=sys.stdout)

# All 32 hosts in the topology
HOSTS = [f'10.0.0.{i}' for i in range(1, 33)]

# Traffic patterns
PATTERNS = {
    'background': {'bw_mbps': 1,   'duration': 30},
    'sustained':  {'bw_mbps': 20,  'duration': 20},
    'burst':      {'bw_mbps': 90,  'duration': 5},
    'flood':      {'bw_mbps': 100, 'duration': 3},
}


def run_iperf(src_ip: str, dst_ip: str, bw_mbps: int, duration: int, port: int = 5201):
    """Run one iperf3 client flow from src to dst."""
    cmd = [
        'iperf3', '-c', dst_ip,
        '-b', f'{bw_mbps}M',
        '-t', str(duration),
        '-p', str(port),
        '-u',          # UDP — more predictable for SDN testing
        '--connect-timeout', '1000',
        '-q',          # quiet
    ]
    try:
        subprocess.run(cmd, timeout=duration + 5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def traffic_wave(n_flows: int, pattern_name: str):
    """Launch n_flows simultaneous iperf flows with the given pattern."""
    p = PATTERNS[pattern_name]
    threads = []
    used_ports = set()
    log.info('Starting %d %s flows  (bw=%dMbps  dur=%ds)',
             n_flows, pattern_name, p['bw_mbps'], p['duration'])

    for _ in range(n_flows):
        src, dst = random.sample(HOSTS, 2)
        port     = random.randint(5201, 5300)
        while port in used_ports:
            port = random.randint(5201, 5300)
        used_ports.add(port)

        t = threading.Thread(
            target=run_iperf,
            args=(src, dst, p['bw_mbps'], p['duration'], port),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(0.1)   # stagger starts slightly

    # Wait for all flows to finish
    for t in threads:
        t.join(timeout=p['duration'] + 10)


def main():
    log.info('Traffic generator started  (32 hosts, tree depth=5 fanout=2)')
    log.info('Ctrl+C to stop')

    cycle = 0
    while True:
        cycle += 1
        log.info('─── Cycle %d ───', cycle)

        # Background traffic — always present
        t_bg = threading.Thread(
            target=traffic_wave, args=(4, 'background'), daemon=True)
        t_bg.start()

        # Vary intensity each cycle
        phase = cycle % 6
        if phase == 0:
            # Burst — triggers congestion
            traffic_wave(8, 'burst')
        elif phase == 1:
            # Sustained medium load
            traffic_wave(6, 'sustained')
        elif phase == 2:
            # Flood — heavy congestion
            traffic_wave(4, 'flood')
        elif phase == 3:
            # Light load — calm period
            traffic_wave(2, 'background')
        elif phase == 4:
            # Mixed burst + sustained
            threading.Thread(target=traffic_wave, args=(3, 'burst'),     daemon=True).start()
            threading.Thread(target=traffic_wave, args=(3, 'sustained'), daemon=True).start()
            time.sleep(8)
        else:
            # Very light — test do-nothing action
            traffic_wave(1, 'background')

        time.sleep(5)   # gap between cycles


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.info('Traffic generator stopped.')
