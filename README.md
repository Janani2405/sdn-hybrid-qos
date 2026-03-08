# A Hybrid Learning Approach for Dynamic Bandwidth Allocation and Load Balancing in SDN Based Cloud Networks

**Puducherry Technological University — Department of Information Technology**  
**VIII Semester — IT233 Project Work — 2025–26**

**Team:**
- Janani A (Reg. No: 2201112023)
- Raja Hariharan K (Reg. No: 2201112033)
- Selvaganapathi S (Reg. No: 2201112039)

**Guide:** Dr. Santhi G (Professor)

---

## Project Overview

This project implements a hybrid SDN-based intelligent traffic management framework that combines:
- **LSTM-based traffic prediction** for proactive congestion detection (Module II)
- **DQN-based intelligent load balancing** for dynamic bandwidth allocation (Module III)
- **OpenFlow 1.3 QoS enforcement** via a Ryu SDN controller (Module IV)
- **Real-time visualization** dashboard with live network monitoring (Module I)

The system runs on a Mininet-emulated tree topology and collects rich network metrics — including real measured latency and jitter via LLDP probes — to train and evaluate the ML models.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                             │
│   Web / VoIP / Gaming / IoT / Cloud API Traffic (TCP/UDP)   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  MODULE I — Network Traffic Monitoring & Data Collection  ✅ │
│  • Ryu OpenFlow 1.3 controller (qos_controller.py)          │
│  • Per-port stats every 2s → qos_log.csv (30 columns)       │
│  • Real latency/jitter via LLDP probes (no random stubs)    │
│  • Dual-signal congestion detection → congestion_log.csv    │
│  • 4-zone labels: normal / warning / congested / critical   │
│  • REST API (9 endpoints) + live dashboard                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  MODULE II — Traffic Analysis & Congestion Prediction       │
│  • LSTM trained on qos_log.csv (30 features)   [PENDING]    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  MODULE III — Intelligent Load Balancing & BW Optimization  │
│  • DQN agent (state/action/reward already defined) [PENDING]│
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  MODULE IV — SDN Control & QoS Enforcement                  │
│  • OpenFlow rule installation & HTB queue enforcement       │
│                                                  [PENDING]  │
└─────────────────────────────────────────────────────────────┘
```

---

## Network Topology

```
Tree topology — fanout=2, depth=3 → 7 switches, 8 hosts

                    s1  (root)
              ┌──────┴──────┐
             s2             s3
          ┌───┴───┐       ┌───┴───┐
         s4      s5      s6      s7
       ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐
      h1  h2 h3  h4 h5  h6 h7  h8
   .0.1 .0.2 .0.3 .0.4 .0.5 .0.6 .0.7 .0.8

Link parameters:
  Bandwidth   : 100 Mbps (TCLink / HTB)
  Delay       : 2ms
  Queue depth : 1000 packets
  Queues      : 0 = BestEffort (100 Mbps)
                1 = Priority   (50–100 Mbps guaranteed)
                2 = Bulk       (max 20 Mbps)
  Protocol    : OpenFlow 1.3
```

---

## Project Structure

```
sdn-project/
│
├── controller/
│   ├── qos_controller.py          # Active: merged controller (data + latency + REST API)
│   ├── port_stats_monitor.py      # Original v1 stats monitor (reference)
│   └── port_stats_monitor_v2.py   # Original v2 with dual-signal detection (reference)
│
├── topology/
│   ├── topology.py                # Active: tree topology (7 switches, 8 hosts)
│   └── topology_linear.py         # Original: linear topology (2 switches, 4 hosts)
│
├── tests/
│   ├── simulate_traffic.py        # Traffic generator with 4-port pool fix
│   ├── run_simulation.py          # Simulation launcher (4 iperf3 servers on h2)
│   ├── iperf_parallel_traffic.py  # Alternative wave-based traffic generator
│   └── simulation_traffic_profile.csv  # 300 flows (LOW/MEDIUM/HIGH/CONGESTION)
│
├── logs/                          # Runtime output — not committed to git
│   ├── qos_log.csv                # 30-column per-port metrics (every 2s)
│   └── congestion_log.csv         # Per-episode congestion records
│
├── docs/
│   ├── dashboard.html             # Live visualization dashboard (open in browser)
│   └── topology-diagram.png       # Network topology diagram
│
├── scripts/
│   ├── start_qos_controller.sh    # Start merged Ryu controller (use this)
│   ├── start_controller.sh        # Legacy script (replaced by above)
│   ├── start_topology.sh          # Start tree topology
│   └── cleanup.sh                 # Kill Mininet + clean OVS state
│
├── config/                        # Reserved for future config files
├── .gitignore
└── README.md
```

---

## Module Status

| Module | Description | Status |
|--------|-------------|--------|
| Module I | Network Traffic Monitoring & Data Collection | ✅ Complete |
| Module II | Traffic Analysis & Congestion Prediction (LSTM) | 🔄 Pending |
| Module III | Intelligent Load Balancing & Bandwidth Optimization (DQN) | 🔄 Pending |
| Module IV | SDN Control & QoS Enforcement (OpenFlow) | 🔄 Pending |

---

## Environment

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 LTS |
| Python | 3.12.3 |
| Mininet | 2.3.1b4 |
| Open vSwitch | 3.3.4 |
| Ryu Controller | 4.34 |
| OpenFlow | 1.3 |
| iperf3 | 3.16 |

---

## Prerequisites

```bash
# Install Mininet
sudo apt install mininet

# Install Ryu in a virtual environment
python3 -m venv ~/ryu-env
source ~/ryu-env/bin/activate
pip install ryu eventlet webob

# Install iperf3
sudo apt install iperf3
```

---

## How to Run

### Step 1 — Start the Controller (Terminal 1)
```bash
source ~/ryu-env/bin/activate
cd ~/sdn-project
bash scripts/start_qos_controller.sh
```
Wait for:
```
Poll loop started.
LLDP probe loop started  (interval=1s)
Switch CONNECTED  dpid=0x...
```

### Step 2 — Start the Topology (Terminal 2)
```bash
cd ~/sdn-project
bash scripts/cleanup.sh
sudo python3 topology/topology.py
```
Verify connectivity:
```
mininet> pingall
```
Expected: `0% dropped (56/56 received)`

### Step 3 — Run Traffic Simulation (Terminal 2)
```
mininet> py exec(open('tests/run_simulation.py').read())
```

### Step 4 — Open Dashboard (Browser)
```bash
xdg-open ~/sdn-project/docs/dashboard.html
```
Polls `http://127.0.0.1:8080/qos/api/v1/` every 2 seconds.

### Step 5 — Cleanup When Done
```bash
bash scripts/cleanup.sh
```

---

## REST API Endpoints

All served at `http://127.0.0.1:8080/qos/api/v1/`

| Endpoint | Description |
|----------|-------------|
| `/health` | Controller status and connected switch count |
| `/metrics/latest` | Per-switch BW, latency, jitter, loss, reward (all real values) |
| `/metrics` | Full 60-sample rolling history per switch |
| `/topology` | All switches and inter-switch links |
| `/flows` | Active OpenFlow flow table entries per switch |
| `/ports` | Per-port byte/drop/utilisation/latency counters |
| `/events` | Controller event log (last 100 events) |
| `/hosts` | Connected hosts with MAC and IP |
| `/congestion` | Live congestion state + episode count per port |
| `/latency` | Real per-link RTT, OWD, jitter from LLDP probes |

---

## Data Collection — Module I Output

### `logs/qos_log.csv` — 30 columns, written every 2 seconds per port

| Group | Columns | Description |
|-------|---------|-------------|
| A — Identity | `timestamp`, `dpid`, `port_no` | When and where |
| B — Raw counters | `tx_bytes`, `rx_bytes`, `tx_dropped`, `rx_dropped` | Cumulative OVS counters |
| C — Throughput | `tx_mbps`, `rx_mbps`, `utilization_pct`, `loss_pct`, `tx_pps`, `rx_pps`, `bw_headroom_mbps` | Per-interval derived rates |
| D — Drop deltas | `delta_tx_dropped`, `delta_rx_dropped` | Drops this interval only |
| E — Latency | `latency_ms`, `jitter_ms`, `rtt_ms` | **Real** values from LLDP probes |
| F — Rolling | `rolling_util_mean`, `rolling_drop_sum`, `rolling_tx_mean`, `rolling_rx_mean` | Last 5-sample trends |
| G — Topology | `n_active_flows`, `neighbor_util_max`, `inter_arrival_delta` | Network context features |
| H — Labels | `signal_util`, `signal_drop`, `zone_label`, `congested` | Congestion signals and label |

### `logs/congestion_log.csv` — written once per new congestion episode

| Column | Description |
|--------|-------------|
| `timestamp`, `dpid`, `port_no` | Episode location and time |
| `utilization_pct`, `tx_mbps`, `rx_mbps` | Traffic state at episode start |
| `delta_tx_dropped`, `delta_rx_dropped` | Drops that triggered the event |
| `signal_util`, `signal_drop` | Which signals fired |
| `reason` | Human-readable reason string |
| `port_event_count` | Episode count on this specific port |
| `global_event_count` | Total episodes across all ports |

---

## Real Latency Measurement (LLDP Probe System)

The controller measures actual link latency — no synthetic random values.

```
Every LLDP_PROBE_INTERVAL (1s) per inter-switch link:

  1. Controller builds custom LLDP frame containing:
       - src_dpid  (Chassis ID TLV)
       - src_port  (Port ID TLV)
       - send_time (Org-specific TLV, OUI=0x0026E1, encoded as float64)

  2. Frame is injected via OFPPacketOut on the source switch port

  3. Probe travels: Controller → Switch_A → [link] → Switch_B → Controller

  4. On arrival, _packet_in_handler intercepts it, extracts send_time:
       RTT    = time.now() − send_time   (round-trip)
       OWD    = RTT / 2                  (one-way delay approximation)
       Jitter = α × |OWD_n − OWD_{n−1}| + (1−α) × Jitter_{n−1}
                where α = 0.2  (EWMA smoothing)

  5. Values stored in _link_latency and written to qos_log.csv columns
     latency_ms, jitter_ms, rtt_ms
```

---

## Congestion Detection Logic

```
Every 2 seconds per port:

  Δtx_bytes = tx_bytes_now  − tx_bytes_prev
  Δrx_bytes = rx_bytes_now  − rx_bytes_prev
  tx_Mbps   = (Δtx_bytes × 8) / (Δt × 1,000,000)
  rx_Mbps   = (Δrx_bytes × 8) / (Δt × 1,000,000)
  util_%    = max(tx_Mbps, rx_Mbps) / 100 × 100
  loss_%    = (Δrx_dropped / (Δrx_bytes / 1000)) × 100

  signal_util = util_% > 95%
  signal_drop = Δtx_dropped > 0  OR  Δrx_dropped > 0
  congested   = signal_util  OR  signal_drop

  Zone labels:
    critical  → signal_util AND signal_drop both fired
    congested → either signal fired
    warning   → util > 70% but below 95%
    normal    → everything quiet

  Episode debounce:
    → Counter increments only on False→True transition
    → Ongoing congestion logged but never double-counted
    → Counter-wrap and OVS restart guard on all deltas
```

---

## Reward Signal (for Module III DQN)

Computed from **real** measured values (no random stubs):

```
reward = norm_bw − norm_latency − norm_loss − 0.5 × norm_jitter

Where:
  norm_bw      = min(bw_mbps   / 100,  1.0)
  norm_latency = min(latency   / 200,  1.0)   ← real LLDP measurement
  norm_loss    = min(loss_pct  / 100,  1.0)   ← real drop-based calculation
  norm_jitter  = min(jitter_ms / 50,   1.0)   ← real EWMA from LLDP probes

Range: −2.5 (worst) to +1.0 (best)
Exposed via /metrics/latest REST endpoint for each switch
```

---

## Traffic Simulation Profile

300 flows across 4 congestion zones (`simulation_traffic_profile.csv`):

| Zone | Target BW | Purpose |
|------|-----------|---------|
| LOW | 1–5 Mbps | Normal background traffic |
| MEDIUM | 10–40 Mbps | Moderate load |
| HIGH | 50–80 Mbps | Near-saturation |
| CONGESTION | 90–100+ Mbps | Triggers congestion labels for training |

Simulation uses a 4-port iperf3 server pool (ports 5201–5204) on h2 to support up to 4 concurrent flows without "server busy" errors.

---

## Baseline Metrics (Module I — No AI)

Measured from initial simulation run:

| Metric | Value |
|--------|-------|
| Total data rows collected | 10,536 |
| Congestion labeled rows | 48 (0.46%) |
| Max utilization recorded | 100% |
| Avg utilization | 1.42% |
| Congestion detection method | Reactive (at-the-moment threshold) |
| Response to congestion | None (observation only) |
| Latency measurement | Real (LLDP probe, ~2–15ms in Mininet) |
| Jitter measurement | Real (EWMA from LLDP probes) |

These are the **baseline values** that Modules II–IV will improve upon.

---

## References

1. M. Khalid et al., "New learning approach for high-load traffic optimization SDN," J. Intell. Syst. Internet Things, vol. 17, no. 1, pp. 255–270, 2025.
2. K. Somsuk et al., "Dynamic predictive feedback mechanism for intelligent bandwidth control in future SDN networks," Network (MDPI), vol. 5, 2025.
3. M. Kirti et al., "Fault-tolerance approaches for distributed and cloud computing environments," Concurrency Comput. Pract. Exper., vol. 36, no. 13, 2024.
4. P. Tamilarasu et al., "QoS transformation in the cloud: Advancing service quality through innovative resource scheduling," IET Commun., vol. 19, 2025.
5. P. Agrawal et al., "AI-powered predictive models for network fault detection and proactive QoS management," IJSCI, vol. 2, no. 10, 2025.
6. N. McKeown et al., "OpenFlow: Enabling innovation in campus networks," ACM SIGCOMM, 2008.
7. V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, 2015.
8. S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Comput., vol. 9, no. 8, 1997.