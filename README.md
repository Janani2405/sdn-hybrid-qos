# A Hybrid Learning Approach for Dynamic Bandwidth Allocation and Load Balancing in SDN Based Cloud Networks

**Puducherry Technological University — Department of Information Technology**
**VIII Semester — IT233 Project Work — 2025–26**

**Team:** Janani A (2201112023) · Raja Hariharan K (2201112033) · Selvaganapathi S (2201112039)
**Guide:** Dr. Santhi G (Professor)

---

## System Architecture

```
INPUT — Web / VoIP / Gaming / IoT / Cloud API Traffic (TCP/UDP)
    │
    ▼
MODULE I  — Network Traffic Monitoring & Data Collection       ✅ Complete
    Ryu OpenFlow 1.3 · LLDP probes · 30-col CSV · REST API · Dashboard
    │
    ▼
MODULE II — Traffic Analysis & Congestion Prediction           ✅ Complete
    LSTM 220K params · 2 heads · 95.5% zone accuracy · state_vector()
    │
    ▼
MODULE III — Intelligent Load Balancing & BW Optimization      🔲 Pending
    DQN agent · consumes state_vector from LSTM
    │
    ▼
MODULE IV  — SDN Control & QoS Enforcement                     🔲 Pending
    OpenFlow rule install · HTB queues · feedback loop
```

---

## Network Topology

```
sudo mn --controller remote --topo tree,fanout=2,depth=5
→ 31 switches · 32 hosts (h1–h32) · IPs 10.0.0.1–10.0.0.32
All links: 100 Mbps · 2ms delay · OpenFlow 1.3
```

---

## Project Structure

```
sdn-project/
├── controller/
│   └── qos_controller.py          ← Ryu controller + LSTM hook + /prediction API
│
├── module2/
│   ├── preprocess.py              ← Task 1: clean, scale, encode
│   ├── window.py                  ← Task 2: sliding windows (seq_len=10)
│   ├── split.py                   ← Task 3: port-level stratified split
│   ├── model.py                   ← Task 4: SDNTrafficLSTM (220K params)
│   ├── train.py                   ← Task 5: training loop
│   ├── evaluate.py                ← Task 6: metrics + plots
│   ├── lstm_predictor.py          ← Task 7: live inference + state_vector()
│   └── processed/                 ← auto-created
│       ├── X_windows.npy          (73508 × 10 × 17)
│       ├── scaler.pkl · label_encoder.pkl · feature_names.txt
│       ├── train_idx / val_idx / test_idx .npy
│       └── checkpoints/
│           ├── best_lstm.pt
│           ├── confusion_matrix.png · training_curves.png
│           └── evaluation_report.txt
│
├── tests/
│   ├── simulate_traffic_v2.py
│   ├── run_simulation.py
│   ├── simulation_traffic_profile.csv      ← 1000 flows, h1–h32
│   └── simulation_traffic_profile_test.csv ← 36 flows (connectivity check)
│
├── logs/
│   ├── qos_log.csv                ← 74,336 rows, 30 cols (LSTM training data)
│   └── congestion_log.csv         ← 3,248 episodes (DQN reward reference)
│
└── docs/
    └── dashboard.html
```

---

## Module Status

| Module | Status | Key Output |
|--------|--------|------------|
| Module I — Monitoring | ✅ Complete | `qos_log.csv` 74,336 rows |
| CICIDS Preprocessing | ✅ Complete | `simulation_traffic_profile.csv` |
| Traffic Simulation | ✅ Complete | 910/1000 flows · 32 hosts |
| Module II — LSTM | ✅ Complete | `best_lstm.pt` · 95.5% accuracy |
| Module III — DQN | 🔲 Pending | — |
| Module IV — Enforcement | 🔲 Pending | — |

---

## How to Run

```bash
# 1 — Controller
source ~/ryu-env/bin/activate && cd ~/sdn-project
ryu-manager controller/qos_controller.py --observe-links --ofp-tcp-listen-port 6633

# 2 — Topology
sudo mn --controller remote --topo tree,fanout=2,depth=5
mininet> pingall   # must show 0% drop

# 3 — Traffic simulation
mininet> py exec(open('tests/run_simulation.py').read())

# 4 — Train Module II (run once after collecting logs)
python3 module2/preprocess.py
python3 module2/window.py
python3 module2/split.py
python3 module2/train.py      # ~19 min on CPU
python3 module2/evaluate.py

# 5 — Restart controller to activate LSTM predictions
ryu-manager controller/qos_controller.py --observe-links --ofp-tcp-listen-port 6633
# After 20s: curl http://127.0.0.1:8080/qos/api/v1/prediction
```

---

## REST API

`http://127.0.0.1:8080/qos/api/v1/`

| Endpoint | Description |
|----------|-------------|
| `/health` | Controller status |
| `/metrics/latest` | Per-switch BW, latency, jitter, reward |
| `/metrics` | 60-sample history |
| `/topology` | Switches and links |
| `/flows` | Flow table entries |
| `/ports` | Per-port counters |
| `/events` | Event log |
| `/hosts` | Host MAC/IP |
| `/congestion` | Live congestion state |
| `/latency` | LLDP-measured RTT/OWD/jitter |
| `/prediction` | **NEW** LSTM state vectors → Module III DQN |

---

## Module II — LSTM Architecture

```
Input (batch, 10, 17)
→ LSTM layer 1 (17→128, dropout=0.3)
→ LSTM layer 2 (128→128)
→ h_n[-1]  →  BatchNorm1d(128)
→ Head A: Linear(128→64)→ReLU→Dropout→Linear(64→3)   [zone: normal/warning/congested]
→ Head B: Linear(128→32)→ReLU→Dropout→Linear(32→1)   [cong_prob ∈ [0,1]]
Total: 220,228 parameters
```

## Module II — State Vector for DQN

```
state_vector() → float32 (9,) per port per 2s

[0] P(normal)         LSTM zone softmax
[1] P(warning)        LSTM zone softmax
[2] P(congested)      LSTM zone softmax
[3] cong_prob         LSTM Head B sigmoid
[4] is_congested      hard 0/1 from zone pred
[5] utilization_pct   raw controller value
[6] bw_headroom_mbps  raw controller value
[7] delta_tx_dropped  raw controller value
[8] latency_ms        LLDP measurement
```

## Module II — Results

| Metric | Value |
|--------|-------|
| Zone accuracy | 95.51% |
| Binary F1 | 95.27% |
| ROC-AUC | 99.97% |
| Brier score | 0.0052 |
| Training data | 73,508 windows |
| Test set | 11,186 windows |

---

## References

1. M. Khalid et al., J. Intell. Syst. IoT, vol. 17, 2025.
2. K. Somsuk et al., Network (MDPI), vol. 5, 2025.
3. M. Kirti et al., Concurrency Comput., vol. 36, 2024.
4. P. Tamilarasu et al., IET Commun., vol. 19, 2025.
5. P. Agrawal et al., IJSCI, vol. 2, 2025.
6. N. McKeown et al., ACM SIGCOMM, 2008.
7. V. Mnih et al., Nature, vol. 518, 2015.
8. S. Hochreiter & J. Schmidhuber, Neural Comput., vol. 9, 1997.