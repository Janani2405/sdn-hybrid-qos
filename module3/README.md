# DQN Agent — SDN QoS Optimization (Module III)

## File structure

```
sdn-project/
├── controller/
│   └── qos_controller.py     ← your existing controller (unchanged)
├── module2/
│   └── lstm_predictor.py     ← your existing LSTM module (unchanged)
├── module3/                  ← THIS module
│   ├── dqn_agent.py          ← main DQN agent
│   ├── traffic_gen.py        ← training traffic generator
│   ├── eval_dqn.py           ← evaluation script
│   └── README.md
└── logs/
    ├── qos_log.csv           ← controller writes this
    ├── congestion_log.csv    ← controller writes this
    └── dqn_agent.log         ← DQN agent writes this
```

## Install

```bash
pip install torch requests numpy
```

## How to run (3 terminals)

### Terminal 1 — Ryu controller (already running)
```bash
source ~/ryu-env/bin/activate
cd ~/sdn-project
ryu-manager controller/qos_controller.py \
    --observe-links \
    --ofp-tcp-listen-port 6633
```

### Terminal 2 — Mininet topology (already running)
```bash
sudo mn --topo tree,depth=5,fanout=2 \
    --controller remote,ip=127.0.0.1,port=6633 \
    --link tc,bw=100,delay=2ms,loss=0 \
    --switch ovsk,protocols=OpenFlow13
```

### Terminal 3 — DQN Agent
```bash
cd ~/sdn-project
python3 module3/dqn_agent.py
```

### Terminal 4 — Traffic generator (for training diversity)
```bash
cd ~/sdn-project
python3 module3/traffic_gen.py
```

## What the DQN agent does

Every 2 seconds (matching the controller's poll cycle):

1. **Reads state** from `/qos/api/v1/prediction` — LSTM congestion predictions per port
2. **Aggregates** per-port data into one 12-feature state vector per switch
3. **Selects action** (epsilon-greedy) per switch:
   - `0` — do nothing
   - `1` — reroute: redirect traffic from congested port to alternate port
   - `2` — throttle: install OpenFlow meter limiting best-effort traffic to 5 Mbps
   - `3` — prioritise: mark all flows with DSCP EF (Expedited Forwarding)
   - `4` — reset: remove all DQN-installed rules, restore normal routing
4. **Enforces action** via Ryu REST API (`/stats/flowentry/add`)
5. **Reads reward** from `/qos/api/v1/metrics/latest` (real throughput/latency/loss)
6. **Trains** on a random batch of 64 past experiences (replay memory)
7. **Saves checkpoint** every 100 steps to `saved_dqn/dqn_ckpt.pt`

## State vector (12 features per switch)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | P_congested_max | Highest LSTM congestion probability across ports |
| 1 | cong_prob_max | Highest raw congestion probability |
| 2 | util_pct_max | Highest link utilization (0–1) |
| 3 | util_pct_mean | Mean link utilization (0–1) |
| 4 | bw_headroom_min | Tightest remaining bandwidth (0–1) |
| 5 | latency_ms_mean | Mean one-way latency (normalised /200ms) |
| 6 | loss_pct_mean | Mean packet loss (0–1) |
| 7 | jitter_ms_mean | Mean jitter (normalised /50ms) |
| 8 | delta_drops_sum | Total drops this step (normalised /1000) |
| 9 | rolling_util_mean | 5-step smoothed utilisation |
| 10 | neighbor_util_max | Upstream neighbour congestion pressure |
| 11 | n_ports_congested_ratio | Fraction of ports predicted congested |

## Reward function

```
R = 1.5 × throughput_norm
  - 1.0 × latency_norm
  - 2.0 × loss_norm         (loss weighted highest — worst QoS metric)
  - 0.5 × jitter_norm
  + 1.0 (bonus if congestion ratio decreased from last step)
  - 0.3 (penalty if non-zero action taken on calm switch)
```

## Evaluation (after training)

```bash
# Pure exploitation — no random actions
python3 module3/eval_dqn.py --steps 100

# Observe-only — no flow rules installed
python3 module3/eval_dqn.py --steps 100 --no-enforce
```

## Architecture details

- **Dueling DQN**: separates value V(s) and advantage A(s,a) streams
  → better action selection when many actions have similar Q-values
- **Double DQN**: action chosen by main net, evaluated by target net
  → reduces Q-value overestimation
- **Huber loss**: less sensitive to reward outliers than MSE
- **Gradient clipping**: max_norm=10 prevents exploding gradients
- **All DQN rules tagged with cookie=0xDEADBEEF** → easy cleanup via action=reset

## Expected training progression

| Steps | Epsilon | Expected behaviour |
|-------|---------|-------------------|
| 0–50 | 1.0–0.86 | Random exploration, building memory |
| 50–200 | 0.86–0.50 | Starts preferring reroute during bursts |
| 200–500 | 0.50–0.22 | Learns calm → do_nothing pattern |
| 500+ | 0.22–0.05 | Settled policy, fine-tuning |

## Requirements file

```
torch>=2.0
requests>=2.28
numpy>=1.24
```
