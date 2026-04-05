"""
=============================================================================
dqn_agent.py  —  DQN Agent for SDN QoS Optimization  (Module III)
=============================================================================
Connects to your running Ryu controller (qos_controller.py) via REST API.
Reads LSTM predictions from /qos/api/v1/prediction every 2s, decides
per-switch actions, enforces them via OpenFlow, trains continuously.

Stack: Pure PyTorch — matches your LSTM training stack exactly.

IMPORTANT — before running:
  Share module2/lstm_predictor.py so state_vector_all() format is confirmed.
  The fetch_state() function below handles both the 9-value format and the
  raw port-metrics fallback automatically.

Run:
    cd ~/sdn-project
    python3 module3/dqn_agent.py

Architecture: Dueling Double DQN
  State  : 12 features per switch (aggregated from all ports)
  Actions: 5  (do_nothing / reroute / throttle / prioritise / reset)
  Reward : real throughput − latency − loss − jitter  (from controller)
=============================================================================
"""

import os, sys, time, json, random, logging, argparse, collections
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────
#  Logging — both stdout and file
# ─────────────────────────────────────────────────────────────────
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  [DQN]  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/dqn_agent.log', mode='a'),
    ]
)
log = logging.getLogger('dqn')

# ─────────────────────────────────────────────────────────────────
#  Configuration — change these to match your deployment
# ─────────────────────────────────────────────────────────────────
RYU_BASE          = 'http://127.0.0.1:8080'
RYU_QOS_BASE      = f'{RYU_BASE}/qos/api/v1'
RYU_STATS_BASE    = f'{RYU_BASE}/stats'

STATE_DIM         = 12      # features per switch (see STATE VECTOR section)
ACTION_DIM        = 5       # actions (see ACTIONS section)

POLL_INTERVAL     = 2.0     # must match controller POLL_INTERVAL = 2
TARGET_SYNC_EVERY = 20      # steps between target net weight copy
CHECKPOINT_EVERY  = 100     # steps between saving to disk
MEMORY_SIZE       = 10_000
BATCH_SIZE        = 64
GAMMA             = 0.97    # high — network QoS effects persist across steps
LR                = 0.0005
EPSILON_START     = 1.0
EPSILON_MIN       = 0.05
EPSILON_DECAY     = 0.997   # reaches 0.05 after ~1000 steps

SAVE_DIR          = Path('saved_dqn')
CKPT_PATH         = SAVE_DIR / 'dqn_ckpt.pt'

# Dashboard REST API — exposes DQN state to dashboard.html on port 8081
DQN_API_PORT      = 8081
MAX_DECISION_LOG  = 200   # keep last N decisions in memory

# Reward weights — must match your project's priority
ALPHA_BW    = 1.5   # reward high throughput
BETA_LAT    = 1.0   # penalise latency
GAMMA_LOSS  = 2.0   # heavily penalise packet loss
DELTA_JIT   = 0.5   # penalise jitter
CONG_BONUS  = 1.0   # extra reward when congestion ratio drops
IDLE_PENALTY = 0.3  # penalise unnecessary action on calm switch

# Enforcement config
REROUTE_PRIORITY   = 200    # higher than normal L2 flows (priority=1)
THROTTLE_KBPS      = 5_000  # 5 Mbps cap for best-effort during throttle
DSCP_EF            = 46     # Expedited Forwarding DSCP value
DQN_COOKIE         = 0xDEADBEEF   # tag all DQN-installed rules for clean removal


# ─────────────────────────────────────────────────────────────────
#  STATE VECTOR  (12 features per switch, all normalised 0–1)
#  ─────────────────────────────────────────────────────────────────
#  [0]  P_congested_max        worst port's LSTM P(congested)
#  [1]  cong_prob_max          worst port's raw congestion probability
#  [2]  util_pct_max           highest link utilization        /100
#  [3]  util_pct_mean          mean link utilization           /100
#  [4]  bw_headroom_min        tightest remaining BW           /100
#  [5]  latency_ms_mean        mean one-way latency            /200
#  [6]  loss_pct_mean          mean packet loss                /100
#  [7]  jitter_ms_mean         mean jitter                     /50
#  [8]  delta_drops_sum        total drops this step           /1000
#  [9]  rolling_util_mean      5-step smoothed utilization     /100
#  [10] neighbor_util_max      upstream congestion pressure    /100
#  [11] n_ports_congested_ratio fraction of ports congested
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
#  ACTIONS
#  0 — do_nothing   no OpenFlow change
#  1 — reroute      redirect from congested port to alternate port
#  2 — throttle     install OpenFlow meter: cap BE traffic to 5 Mbps
#  3 — prioritise   mark all flows DSCP EF (expedited forwarding)
#  4 — reset        delete all DQN-installed rules (cookie=0xDEADBEEF)
# ─────────────────────────────────────────────────────────────────
ACTION_NAMES = ['do_nothing', 'reroute', 'throttle', 'prioritise', 'reset']


# ─────────────────────────────────────────────────────────────────
#  Dueling Double DQN Network
#  Shared trunk → Value stream V(s) + Advantage stream A(s,a)
#  Q(s,a) = V(s) + A(s,a) − mean(A(s,:))
#
#  Why Dueling?  In SDN, many switches are calm most of the time.
#  Separating V(s) from A(s,a) means the network learns "calm state
#  is good" independently from "which action is best" — this
#  converges faster when most actions have similar Q-values.
# ─────────────────────────────────────────────────────────────────
class DuelingQNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f   = self.trunk(x)
        v   = self.value_head(f)                          # (B, 1)
        a   = self.adv_head(f)                            # (B, A)
        return v + a - a.mean(dim=1, keepdim=True)        # (B, A)


# ─────────────────────────────────────────────────────────────────
#  Replay Memory
# ─────────────────────────────────────────────────────────────────
Transition = collections.namedtuple(
    'T', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity: int):
        self.buf = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, n: int) -> List[Transition]:
        return random.sample(self.buf, n)
    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────
#  REST helpers
# ─────────────────────────────────────────────────────────────────
_sess = requests.Session()
_sess.headers['Content-Type'] = 'application/json'

def _get(url: str, timeout=3.0) -> Optional[dict]:
    try:
        r = _sess.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug('GET %s → %s', url, e)
        return None

def _post(url: str, data: dict, timeout=3.0) -> bool:
    try:
        r = _sess.post(url, data=json.dumps(data), timeout=timeout)
        r.raise_for_status()
        return True
    except Exception as e:
        log.debug('POST %s → %s', url, e)
        return False

def _delete(url: str, data: dict, timeout=3.0) -> bool:
    try:
        r = _sess.delete(url, data=json.dumps(data), timeout=timeout)
        r.raise_for_status()
        return True
    except Exception as e:
        log.debug('DELETE %s → %s', url, e)
        return False


# ─────────────────────────────────────────────────────────────────
#  State extraction
#  Handles two cases:
#    Case A — LSTM ready  → reads /prediction  (P_congested etc.)
#    Case B — LSTM not ready → reads /ports only (raw utilization)
#  Both produce the same 12-dim state vector.
# ─────────────────────────────────────────────────────────────────
def fetch_state() -> Dict[str, np.ndarray]:
    """
    Returns { dpid_str -> np.ndarray shape (STATE_DIM,) }
    Empty dict means controller not reachable yet.
    """
    pred_resp = _get(f'{RYU_QOS_BASE}/prediction')
    port_resp = _get(f'{RYU_QOS_BASE}/ports')

    ports_raw = port_resp.get('ports', {}) if port_resp else {}

    # ── Case A: LSTM predictions available ───────────────────────
    if pred_resp and pred_resp.get('ready', False):
        return _state_from_predictions(pred_resp['predictions'], ports_raw)

    # ── Case B: fallback — raw port metrics only ──────────────────
    log.debug('LSTM not ready — using raw port metrics for state')
    return _state_from_ports(ports_raw)


def _state_from_predictions(predictions: dict, ports: dict) -> Dict[str, np.ndarray]:
    """Build state vectors from LSTM /prediction endpoint."""
    states = {}
    for dpid_str, port_preds in predictions.items():
        if not port_preds:
            continue

        # Each port_preds value is a dict like:
        # { P_normal, P_warning, P_congested, cong_prob,
        #   is_congested, utilization_pct, bw_headroom_mbps,
        #   delta_tx_dropped, latency_ms, pred_zone }
        # (from controller's /prediction endpoint)

        p_cong, cong_prob   = [], []
        util_all, head_all  = [], []
        lat_all, loss_all   = [], []
        jit_all, drop_all   = [], []
        roll_all, neigh_all = [], []
        n_cong = 0

        for port_str, p in port_preds.items():
            p_cong.append(  float(p.get('P_congested',      0.0)))
            cong_prob.append(float(p.get('cong_prob',        0.0)))
            util = float(p.get('utilization_pct',  0.0)) / 100.0
            util_all.append(util)
            head_all.append(float(p.get('bw_headroom_mbps', 100.0)) / 100.0)
            lat_all.append( min(float(p.get('latency_ms',   0.0)) / 200.0, 1.0))
            drop_all.append(min(float(p.get('delta_tx_dropped', 0.0)) / 1000.0, 1.0))
            if p.get('pred_zone', 'normal') in ('congested', 'critical'):
                n_cong += 1

            # Augment with real port data
            pd = ports.get(dpid_str, {}).get(port_str, {})
            loss_all.append(min(float(pd.get('loss_pct',   0.0)) / 100.0, 1.0))
            jit_all.append( min(float(pd.get('jitter_ms',  0.0)) / 50.0,  1.0))
            roll_all.append(float(pd.get('utilization_pct', util * 100.0)) / 100.0)
            neigh_all.append(float(pd.get('neighbor_util_max', 0.0)) / 100.0)

        n = max(len(p_cong), 1)
        sv = np.clip(np.array([
            max(p_cong),              # [0]
            max(cong_prob),           # [1]
            max(util_all),            # [2]
            sum(util_all) / n,        # [3]
            min(head_all),            # [4]
            sum(lat_all)  / n,        # [5]
            sum(loss_all) / n,        # [6]
            sum(jit_all)  / n,        # [7]
            min(sum(drop_all), 1.0),  # [8]
            sum(roll_all) / n,        # [9]
            max(neigh_all),           # [10]
            n_cong / n,               # [11]
        ], dtype=np.float32), 0.0, 1.0)

        states[dpid_str] = sv
    return states


def _state_from_ports(ports: dict) -> Dict[str, np.ndarray]:
    """Fallback: build state from raw /ports endpoint when LSTM not ready."""
    states = {}
    for dpid_str, port_dict in ports.items():
        if not port_dict:
            continue
        vals = list(port_dict.values())
        utils = [float(p.get('utilization_pct', 0.0)) / 100.0 for p in vals]
        heads = [float(p.get('bw_headroom_mbps', 100.0)) / 100.0 for p in vals]
        lats  = [min(float(p.get('latency_ms', 0.0)) / 200.0, 1.0) for p in vals]
        loss  = [min(float(p.get('loss_pct',   0.0)) / 100.0, 1.0) for p in vals]
        jits  = [min(float(p.get('jitter_ms',  0.0)) / 50.0,  1.0) for p in vals]
        n     = max(len(utils), 1)
        sv = np.clip(np.array([
            max(utils), max(utils),
            max(utils), sum(utils) / n,
            min(heads),
            sum(lats)  / n,
            sum(loss)  / n,
            sum(jits)  / n,
            0.0, sum(utils) / n, 0.0,
            sum(1 for u in utils if u > 0.95) / n,
        ], dtype=np.float32), 0.0, 1.0)
        states[dpid_str] = sv
    return states


# ─────────────────────────────────────────────────────────────────
#  Reward computation from real controller metrics
# ─────────────────────────────────────────────────────────────────
def fetch_rewards() -> Dict[str, float]:
    """Read /metrics/latest — all real values (LLDP latency, real counters)."""
    resp = _get(f'{RYU_QOS_BASE}/metrics/latest')
    if not resp:
        return {}
    rewards = {}
    for dpid_str, m in resp.get('metrics', {}).items():
        bw  = float(m.get('bw_rx_mbps', 0)) + float(m.get('bw_tx_mbps', 0))
        lat = float(m.get('latency_ms', 0))
        lss = float(m.get('loss_pct',   0))
        jit = float(m.get('jitter_ms',  0))
        r = (ALPHA_BW   * min(bw  / 200.0, 1.0)
           - BETA_LAT   * min(lat / 200.0, 1.0)
           - GAMMA_LOSS * min(lss / 100.0, 1.0)
           - DELTA_JIT  * min(jit / 50.0,  1.0))
        rewards[dpid_str] = round(r, 4)
    return rewards


# ─────────────────────────────────────────────────────────────────
#  Action enforcement via Ryu REST API
# ─────────────────────────────────────────────────────────────────
def dpid_to_int(dpid_str: str) -> int:
    """'0x0000000000000001' or '0000000000000001' → int."""
    return int(dpid_str, 16)

def get_congested_ports(dpid_str: str) -> List[int]:
    resp = _get(f'{RYU_QOS_BASE}/congestion')
    if not resp:
        return []
    switch = resp.get('congestion', {}).get(dpid_str, {})
    return [int(p) for p, info in switch.items() if info.get('congested')]

def get_alternate_port(dpid_int: int, congested_port: int) -> Optional[int]:
    """Find any inter-switch port that is NOT the congested port."""
    resp = _get(f'{RYU_QOS_BASE}/topology')
    if not resp:
        return None
    dpid_hex = f'{dpid_int:016x}'   # matches dpid_to_str() format in controller
    alts = [lk['src_port'] for lk in resp.get('links', [])
            if lk.get('src_dpid') == dpid_hex
            and lk.get('src_port') != congested_port]
    return alts[0] if alts else None

def enforce_action(dpid_str: str, action: int) -> bool:
    """
    Translate DQN action → OpenFlow rules via Ryu REST API.

    All rules use cookie=DQN_COOKIE so action=4 (reset) removes them cleanly.
    Returns True if enforcement was successful or unnecessary.
    """
    if action == 0:
        return True   # do nothing

    dpid_int = dpid_to_int(dpid_str)

    # ── Action 1: Reroute ─────────────────────────────────────────
    if action == 1:
        ok = True
        for c_port in get_congested_ports(dpid_str):
            alt = get_alternate_port(dpid_int, c_port)
            if alt is None:
                log.warning('[reroute] no alternate port on %s for port %s',
                            dpid_str, c_port)
                continue
            rule = {
                'dpid':     dpid_int,
                'cookie':   DQN_COOKIE,
                'priority': REROUTE_PRIORITY,
                'match':    {'in_port': c_port},
                'actions':  [{'type': 'OUTPUT', 'port': alt}],
            }
            if _post(f'{RYU_STATS_BASE}/flowentry/add', rule):
                log.info('[reroute] %s  port %s → %s', dpid_str, c_port, alt)
            else:
                ok = False
        return ok

    # ── Action 2: Throttle best-effort traffic ────────────────────
    elif action == 2:
        # Delete stale meter first — Ryu rejects duplicate meter_id silently,
        # leaving the flow rule pointing at a dead meter (drops all matched pkts).
        _delete(f'{RYU_STATS_BASE}/meterentry/delete',
                {'dpid': dpid_int, 'meter_id': 1})
        meter = {
            'dpid':     dpid_int,
            'meter_id': 1,
            'flags':    'KBPS',
            'bands':    [{'type': 'DROP', 'rate': THROTTLE_KBPS, 'burst_size': 1000}],
        }
        ok = _post(f'{RYU_STATS_BASE}/meterentry/add', meter)
        if ok:
            flow = {
                'dpid':     dpid_int,
                'cookie':   DQN_COOKIE,
                'priority': REROUTE_PRIORITY,
                'match':    {'eth_type': 0x0800, 'ip_dscp': 0},
                'actions':  [
                    {'type': 'METER', 'meter_id': 1},
                    {'type': 'OUTPUT', 'port': 'NORMAL'},
                ],
            }
            ok = _post(f'{RYU_STATS_BASE}/flowentry/add', flow)
            if ok:
                log.info('[throttle] %s  BE traffic capped at %d kbps',
                         dpid_str, THROTTLE_KBPS)
        return ok

    # ── Action 3: Prioritise — DSCP EF marking ───────────────────
    elif action == 3:
        rule = {
            'dpid':     dpid_int,
            'cookie':   DQN_COOKIE,
            'priority': REROUTE_PRIORITY,
            'match':    {'eth_type': 0x0800},
            'actions':  [
                {'type': 'SET_FIELD', 'field': 'ip_dscp', 'value': DSCP_EF},
                {'type': 'OUTPUT', 'port': 'NORMAL'},
            ],
        }
        ok = _post(f'{RYU_STATS_BASE}/flowentry/add', rule)
        if ok:
            log.info('[prioritise] %s  DSCP EF installed', dpid_str)
        return ok

    # ── Action 4: Reset — remove all DQN rules ───────────────────
    elif action == 4:
        rule = {
            'dpid':        dpid_int,
            'cookie':      DQN_COOKIE,
            'cookie_mask': 0xFFFFFFFFFFFFFFFF,
            'table_id':    0,
            'priority':    0,
            'match':       {},
            'actions':     [],
        }
        ok = _delete(f'{RYU_STATS_BASE}/flowentry/delete', rule)
        if ok:
            log.info('[reset] %s  all DQN rules removed', dpid_str)
        return ok

    return True



# ─────────────────────────────────────────────────────────────────
#  Global decision log — written by DQNAgent, read by REST API
#  Each entry: { ts, step, dpid, action, action_name, q_values,
#                state, reward, is_congested, epsilon }
# ─────────────────────────────────────────────────────────────────
_decision_log:  collections.deque = collections.deque(maxlen=MAX_DECISION_LOG)
_agent_status:  dict = {
    'running': False, 'step': 0, 'epsilon': 1.0,
    'memory': 0, 'avg_reward': 0.0, 'avg_loss': 0.0,
    'action_counts': {n: 0 for n in ACTION_NAMES},
}
_log_lock = threading.Lock()


def _log_decision(step, dpid, action, q_values, state_vec, reward, epsilon):
    """Record one DQN decision for the dashboard."""
    with _log_lock:
        _decision_log.appendleft({
            'ts':          time.strftime('%H:%M:%S'),
            'step':        step,
            'dpid':        dpid,
            'action':      action,
            'action_name': ACTION_NAMES[action],
            'q_values':    {ACTION_NAMES[i]: round(float(v), 4)
                            for i, v in enumerate(q_values)},
            'state': {
                'P_congested':    round(float(state_vec[0]), 4),
                'cong_prob':      round(float(state_vec[1]), 4),
                'util_max_pct':   round(float(state_vec[2]) * 100, 2),
                'util_mean_pct':  round(float(state_vec[3]) * 100, 2),
                'bw_headroom':    round(float(state_vec[4]) * 100, 2),
                'latency_norm':   round(float(state_vec[5]), 4),
                'loss_norm':      round(float(state_vec[6]), 4),
                'cong_ratio':     round(float(state_vec[11]), 4),
            },
            'reward':       round(float(reward), 4),
            'is_congested': bool(state_vec[0] > 0.4),
            'epsilon':      round(epsilon, 4),
        })
        _agent_status['action_counts'][ACTION_NAMES[action]] += 1


# ─────────────────────────────────────────────────────────────────
#  Tiny HTTP server on port 8081 — serves dashboard API requests
#  Endpoints:
#    GET /decisions  → last N decisions (JSON)
#    GET /status     → agent status (JSON)
#    GET /health     → { ok: true }
# ─────────────────────────────────────────────────────────────────
class _DQNApiHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass   # suppress access logs

    def _send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.end_headers()

    def do_GET(self):
        path = self.path.split('?')[0].rstrip('/')
        with _log_lock:
            if path == '/decisions':
                self._send_json({'decisions': list(_decision_log)})
            elif path == '/status':
                self._send_json(_agent_status)
            elif path == '/health':
                self._send_json({'ok': True})
            else:
                self.send_response(404); self.end_headers()


def _start_api_server():
    """Start the DQN REST API server in a daemon thread."""
    try:
        srv = HTTPServer(('0.0.0.0', DQN_API_PORT), _DQNApiHandler)
        t   = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        log.info('DQN API server → http://127.0.0.1:%d', DQN_API_PORT)
    except Exception as e:
        log.warning('DQN API server failed to start: %s', e)


# ─────────────────────────────────────────────────────────────────
#  DQN Agent
# ─────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self):
        self.device = torch.device('cpu')

        self.main_net   = DuelingQNet(STATE_DIM, ACTION_DIM).to(self.device)
        self.target_net = DuelingQNet(STATE_DIM, ACTION_DIM).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.main_net.parameters(), lr=LR)
        self.memory     = ReplayMemory(MEMORY_SIZE)

        self.epsilon    = EPSILON_START
        self.step_count = 0

        # Per-switch rolling history for transitions
        self._prev_states:  Dict[str, np.ndarray] = {}
        self._prev_actions: Dict[str, int]         = {}

        # Metrics tracking
        self.ep_rewards: List[float] = []
        self.losses:     List[float] = []

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_checkpoint()

    # ── Epsilon-greedy action selection ──────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        # When clearly calm, bias toward do-nothing even during exploration
        is_calm = state[0] < 0.10 and state[2] < 0.15 and state[11] < 0.10
        if random.random() < self.epsilon:
            return 0 if is_calm else random.randrange(ACTION_DIM)

        with torch.no_grad():
            q = self.main_net(
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )
        return int(q.argmax(dim=1).item())

    # ── One training step on a random batch ──────────────────────
    def train_step(self) -> Optional[float]:
        if len(self.memory) < BATCH_SIZE:
            return None

        batch       = self.memory.sample(BATCH_SIZE)
        states      = torch.FloatTensor(np.array([t.state      for t in batch])).to(self.device)
        actions     = torch.LongTensor( np.array([t.action     for t in batch])).to(self.device)
        rewards     = torch.FloatTensor(np.array([t.reward     for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones       = torch.FloatTensor(np.array([t.done       for t in batch])).to(self.device)

        # Current Q-values for actions taken
        current_q = self.main_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        #   action = argmax of MAIN net on next state
        #   value  = TARGET net evaluated at that action
        with torch.no_grad():
            next_acts = self.main_net(next_states).argmax(dim=1)
            next_q    = self.target_net(next_states).gather(
                            1, next_acts.unsqueeze(1)).squeeze(1)
            target_q  = rewards + GAMMA * next_q * (1.0 - dones)

        # Huber loss — less sensitive to reward outliers than MSE
        loss = F.huber_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.main_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    # ── One full observe→act→reward→train cycle ──────────────────
    def step(self):
        self.step_count += 1

        # Mark running immediately so dashboard shows ACTIVE on step 1
        with _log_lock:
            _agent_status['running'] = True
            _agent_status['step']    = self.step_count

        # 1. Observe
        states  = fetch_state()
        rewards = fetch_rewards()

        if not states:
            log.warning('No state from controller — waiting...')
            return

        step_reward = 0.0
        for dpid_str, state in states.items():

            # 2. Store transition from previous step
            if dpid_str in self._prev_states:
                reward = rewards.get(dpid_str, 0.0)

                # Congestion-clearing bonus
                prev_cong = self._prev_states[dpid_str][11]
                curr_cong = state[11]
                if curr_cong < prev_cong - 0.10:
                    reward += CONG_BONUS
                    log.info('  [+bonus] %s congestion cleared  Δ=%.2f',
                             dpid_str, prev_cong - curr_cong)

                # Idle penalty for acting on calm switch
                if self._prev_actions.get(dpid_str, 0) != 0 and prev_cong < 0.10:
                    reward -= IDLE_PENALTY

                self.memory.push(
                    self._prev_states[dpid_str],
                    self._prev_actions[dpid_str],
                    reward,
                    state,
                    False,   # continuous — no terminal state
                )
                step_reward += reward

            # 3. Select action
            action = self.select_action(state)

            # Get Q-values for logging (always, not just when exploiting)
            with torch.no_grad():
                q_vals = self.main_net(
                    torch.FloatTensor(state).unsqueeze(0).to(self.device)
                ).numpy()[0]

            # 4. Enforce on network
            enforce_action(dpid_str, action)

            # 4b. Log decision for dashboard
            _log_decision(
                self.step_count, dpid_str, action,
                q_vals, state,
                rewards.get(dpid_str, 0.0),
                self.epsilon
            )

            # 5. Save for next step
            self._prev_states[dpid_str]  = state
            self._prev_actions[dpid_str] = action

        # 6. Train one batch
        loss = self.train_step()
        if loss is not None:
            self.losses.append(loss)

        # 7. Sync target network
        if self.step_count % TARGET_SYNC_EVERY == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
            log.info('[target sync]  step=%d', self.step_count)

        # 8. Decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        # 9. Track and log
        n = max(len(states), 1)
        self.ep_rewards.append(step_reward / n)

        if self.step_count % 10 == 0:
            avg_r = np.mean(self.ep_rewards[-100:]) if self.ep_rewards else 0
            avg_l = np.mean(self.losses[-50:])       if self.losses     else 0
            log.info(
                'step=%4d | switches=%2d | ε=%.3f | '
                'avg_reward(100)=%+.3f | loss=%.4f | mem=%d',
                self.step_count, n, self.epsilon, avg_r, avg_l, len(self.memory)
            )
            # Update status for dashboard REST API
            with _log_lock:
                _agent_status.update({
                    'running':    True,
                    'step':       self.step_count,
                    'epsilon':    round(self.epsilon, 4),
                    'memory':     len(self.memory),
                    'avg_reward': round(float(avg_r), 4),
                    'avg_loss':   round(float(avg_l), 4),
                    'ep_rewards': list(self.ep_rewards[-60:]),
                    'losses':     list(self.losses[-60:]),
                })

        # 10. Checkpoint
        if self.step_count % CHECKPOINT_EVERY == 0:
            self._save()

    # ── Checkpoint ───────────────────────────────────────────────
    def _save(self):
        torch.save({
            'step':        self.step_count,
            'epsilon':     self.epsilon,
            'main':        self.main_net.state_dict(),
            'target':      self.target_net.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'ep_rewards':  self.ep_rewards[-1000:],
            'losses':      self.losses[-1000:],
        }, CKPT_PATH)
        log.info('[checkpoint] saved  step=%d  path=%s', self.step_count, CKPT_PATH)

    def _load_checkpoint(self):
        if not CKPT_PATH.exists():
            log.info('[checkpoint] none found — starting fresh')
            return
        try:
            # weights_only=True — safe loading, avoids pickle warning in PyTorch 2.x
            data = torch.load(CKPT_PATH, map_location=self.device,
                              weights_only=False)
            self.main_net.load_state_dict(data['main'])
            self.target_net.load_state_dict(data['target'])
            self.optimizer.load_state_dict(data['optimizer'])
            self.step_count = data.get('step', 0)
            self.epsilon    = data.get('epsilon', EPSILON_START)
            self.ep_rewards = data.get('ep_rewards', [])
            self.losses     = data.get('losses', [])
            log.info('[checkpoint] loaded  step=%d  ε=%.3f',
                     self.step_count, self.epsilon)
        except Exception as e:
            log.error('[checkpoint] load failed: %s — starting fresh', e)

    # ── Main loop ────────────────────────────────────────────────
    def run(self):
        log.info('=' * 58)
        log.info('DQN Agent  |  state=%d  actions=%d  ε=%.2f→%.2f',
                 STATE_DIM, ACTION_DIM, EPSILON_START, EPSILON_MIN)
        log.info('Controller : %s', RYU_BASE)
        log.info('=' * 58)

        # Start dashboard REST API server
        _start_api_server()

        # Wait for controller
        log.info('Waiting for Ryu controller at %s ...', RYU_BASE)
        while True:
            r = _get(f'{RYU_QOS_BASE}/health')
            if r and r.get('status') == 'ok':
                log.info('Controller ready — %d switch(es) connected',
                         r.get('switches', 0))
                break
            time.sleep(3)

        # Main loop
        while True:
            t0 = time.time()
            try:
                self.step()
            except KeyboardInterrupt:
                log.info('Interrupted — saving...')
                self._save()
                break
            except Exception as e:
                log.error('step error: %s', e, exc_info=True)
            time.sleep(max(0.0, POLL_INTERVAL - (time.time() - t0)))

        log.info('Agent stopped.')


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='SDN DQN Agent')
    p.add_argument('--controller', default='http://127.0.0.1:8080')
    p.add_argument('--epsilon',    type=float, default=None,
                   help='Override epsilon (e.g. 0.0 for eval-only mode)')
    args = p.parse_args()

    RYU_BASE       = args.controller
    RYU_QOS_BASE   = f'{RYU_BASE}/qos/api/v1'
    RYU_STATS_BASE = f'{RYU_BASE}/stats'

    agent = DQNAgent()
    if args.epsilon is not None:
        agent.epsilon = args.epsilon
        log.info('Epsilon overridden to %.3f', agent.epsilon)

    agent.run()