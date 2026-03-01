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
- **LSTM-based traffic prediction** for proactive congestion detection
- **DQN-based intelligent load balancing** for dynamic bandwidth allocation
- **OpenFlow 1.3 QoS enforcement** via a Ryu SDN controller
- **Real-time visualization** dashboard for live network monitoring

The system runs on a Mininet-emulated tree topology and collects real network metrics to train and evaluate the ML models.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                             │
│   Web / VoIP / Gaming / IoT / Cloud API Traffic (TCP/UDP)   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  MODULE I — Network Traffic Monitoring & Data Collection    │
│  • Ryu OpenFlow 1.3 controller                              │
│  • Per-port stats every 2s → qos_log.csv                    │
│  • Dual-signal congestion detection → congestion_log.csv    │
│  • REST API + live dashboard                                │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  MODULE II — Traffic Analysis & Congestion Prediction       │
│  • LSTM model trained on qos_log.csv                        │
│  • Sliding window sequence prediction                       │
│  • Congestion probability forecasting          [PENDING]    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  MODULE III — Intelligent Load Balancing & BW Optimization  │
│  • DQN agent (state: network metrics, action: flow rules)   │
│  • Dynamic bandwidth allocation                [PENDING]    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  MODULE IV — SDN Control & QoS Enforcement                  │
│  • OpenFlow rule installation                               │
│  • HTB queue-based bandwidth enforcement       [PENDING]    │
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

Link parameters:
  Bandwidth  : 100 Mbps (TCLink / HTB)
  Delay      : 2ms
  Queue depth: 1000 packets
  Queues     : 0=BestEffort(100M), 1=Priority(50-100M), 2=Bulk(20M)
  Protocol   : OpenFlow 1.3
```

---

## Project Structure

```
sdn-project/
│
├── controller/
│   ├── qos_controller.py          # Merged controller — data collection + REST API
│   └── port_stats_monitor_v2.py   # Original stats monitor (reference only)
│
├── topology/
│   ├── topology.py                # Active: tree topology (7 switches, 8 hosts)
│   └── topology_linear.py         # Original: linear topology (2 switches, 4 hosts)
│
├── tests/
│   ├── simulate_traffic.py        # Traffic generator with port-pool fix
│   ├── run_simulation.py          # Simulation launcher (4-server iperf3 pool)
│   └── simulation_traffic_profile.csv  # 300 flows (LOW/MEDIUM/HIGH/CONGESTION)
│
├── logs/
│   └── .gitkeep                   # qos_log.csv and congestion_log.csv generated at runtime
│
├── docs/
│   └── dashboard.html             # Live visualization dashboard (open in browser)
│
├── scripts/
│   ├── start_qos_controller.sh    # Start merged Ryu controller
│   ├── start_controller.sh        # Legacy (replaced by above)
│   └── cleanup.sh                 # Kill Mininet + clean OVS
│
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
pip install ryu eventlet

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
The dashboard polls `http://127.0.0.1:8080/qos/api/v1/` every 2 seconds.

### Step 5 — Cleanup
```bash
bash scripts/cleanup.sh
```

---

## REST API Endpoints

All endpoints served at `http://127.0.0.1:8080/qos/api/v1/`

| Endpoint | Description | Poll Rate |
|----------|-------------|-----------|
| `/health` | Controller status + switch count | Every 2s |
| `/metrics/latest` | Per-switch BW, latency, loss, jitter, reward | Every 2s |
| `/metrics` | Full 60-sample history per switch | On demand |
| `/topology` | Switches and links | Every 2s |
| `/flows` | Active flow table entries per switch | Every 2s |
| `/ports` | Per-port byte/drop counters | Every 2s |
| `/events` | Controller event log (last 100) | Every 2s |
| `/hosts` | Connected hosts with MAC/IP | Every 2s |
| `/congestion` | Live congestion state per port + episode count | Every 2s |

---

## Data Collection (Module I Output)

### `logs/qos_log.csv`
Written every 2 seconds per port. Used as training data for Module II LSTM.

| Column | Description |
|--------|-------------|
| timestamp | Poll timestamp |
| dpid | Switch datapath ID |
| port_no | Port number |
| tx_bytes / rx_bytes | Cumulative byte counters |
| tx_mbps / rx_mbps | Throughput (computed delta) |
| tx_dropped / rx_dropped | Cumulative drop counters |
| delta_tx_dropped / delta_rx_dropped | Drops this interval |
| utilization_pct | max(tx,rx) / 100 Mbps × 100 |
| signal_util | 1 if utilization > 95% |
| signal_drop | 1 if any drops this interval |
| congested | 1 if signal_util OR signal_drop |

### `logs/congestion_log.csv`
Written once per new congestion episode (debounced — not every tick).

| Column | Description |
|--------|-------------|
| timestamp | Episode start time |
| dpid / port_no | Where congestion occurred |
| utilization_pct | Utilisation at episode start |
| tx_mbps / rx_mbps | Throughput at episode start |
| delta_tx_dropped / delta_rx_dropped | Drops that triggered the event |
| signal_util / signal_drop | Which signals fired |
| reason | Human-readable reason string |
| port_event_count | Episodes on this port so far |
| global_event_count | Total episodes across all ports |

### Collected Dataset (Module I Run)
- **Total rows:** 10,536
- **Congestion labeled rows:** 48
- **Time range:** ~13 hours of simulation data
- **Max utilization recorded:** 100%

---

## Congestion Detection Logic

```
Every 2 seconds per port:

  Δtx_bytes = tx_bytes_now − tx_bytes_prev
  Δrx_bytes = rx_bytes_now − rx_bytes_prev
  tx_Mbps   = (Δtx_bytes × 8) / (Δt × 1,000,000)
  rx_Mbps   = (Δrx_bytes × 8) / (Δt × 1,000,000)
  util_%    = max(tx_Mbps, rx_Mbps) / 100 × 100

  signal_util = util_% > 95%
  signal_drop = Δtx_dropped > 0 OR Δrx_dropped > 0
  congested   = signal_util OR signal_drop

  Episode debounce:
    → New episode counted only on False→True transition
    → Ongoing congestion logged but counter not incremented
```

---

## Reward Signal (Preview for Module III DQN)

```
reward = norm_bw − norm_latency − norm_loss − 0.5 × norm_jitter

Where:
  norm_bw      = min(bw_mbps / 100,  1.0)
  norm_latency = min(latency / 200,  1.0)
  norm_loss    = min(loss%  / 100,   1.0)
  norm_jitter  = min(jitter / 50,    1.0)

Range: −2.5 (worst) to +1.0 (best)
```

---

## Traffic Simulation Profile

300 flows across 4 congestion zones:

| Zone | Target BW | Purpose |
|------|-----------|---------|
| LOW | 1–5 Mbps | Normal background traffic |
| MEDIUM | 10–40 Mbps | Moderate load |
| HIGH | 50–80 Mbps | Near-saturation |
| CONGESTION | 90–100+ Mbps | Triggers congestion labels |

---

## References

1. M. Khalid et al., "New learning approach for high-load traffic optimization SDN," J. Intell. Syst. Internet Things, vol. 17, no. 1, pp. 255–270, 2025.
2. K. Somsuk et al., "Dynamic predictive feedback mechanism for intelligent bandwidth control in future SDN networks," Network (MDPI), vol. 5, 2025.
3. M. Kirti et al., "Fault-tolerance approaches for distributed and cloud computing environments," Concurrency Comput. Pract. Exper., vol. 36, no. 13, 2024.
4. P. Tamilarasu et al., "QoS transformation in the cloud," IET Commun., vol. 19, 2025.
5. P. Agrawal et al., "AI-powered predictive models for network fault detection," IJSCI, vol. 2, no. 10, 2025.
6. N. McKeown et al., "OpenFlow: Enabling innovation in campus networks," ACM SIGCOMM, 2008.
7. V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, 2015.
8. S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Comput., vol. 9, no. 8, 1997.
