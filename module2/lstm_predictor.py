"""
lstm_predictor.py — Module II Task 7: Inference Module
========================================================
This file is the BRIDGE between Module II (LSTM) and Module III (DQN).

How it fits into the live system:
    qos_controller.py (Ryu)
        │
        │  every 2 seconds, for each (dpid, port_no):
        │  calls predictor.update(dpid, port_no, row_dict)
        ▼
    LSTMPredictor
        │  maintains a rolling window of 10 rows per port
        │  once 10 rows accumulated → runs LSTM forward pass
        ▼
    state_vector(dpid, port_no)
        │  returns 9-element numpy array
        ▼
    Module III DQN agent
        receives state, selects action (reroute / throttle / do-nothing)

State vector structure (9 elements per port):
    [0] P(normal)       — probability this port is in normal zone
    [1] P(warning)      — probability this port is approaching congestion
    [2] P(congested)    — probability this port is congested
    [3] cong_prob       — Head B soft congestion score (0.0–1.0)
    [4] is_congested    — 1.0 if zone_pred==congested, else 0.0
    [5] utilization_pct — current utilization % (raw, not scaled)
    [6] bw_headroom_mbps— remaining bandwidth capacity (raw)
    [7] delta_tx_dropped— new drops this interval (raw)
    [8] latency_ms      — current LLDP-measured latency (raw)

Elements [0–4] come from the LSTM (learned, temporal).
Elements [5–8] come directly from the controller (instantaneous, raw).
Together they give Module III both the trend (LSTM) and the snapshot (raw).

File placement: sdn-project/module2/lstm_predictor.py
Imported by  : sdn-project/controller/qos_controller.py
"""

import os
import sys
import pickle
import logging
import numpy as np
from collections import deque

import torch

# Allow import of model.py from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SDNTrafficLSTM, N_FEATURES, SEQ_LEN, N_ZONES
ZONE_NAMES = ['normal', 'warning', 'congested']

logger = logging.getLogger(__name__)

# ── Default paths (relative to sdn-project/) ─────────────────────
DEFAULT_CKPT    = 'module2/checkpoints/best_lstm.pt'
DEFAULT_SCALER  = 'module2/processed/scaler.pkl'
DEFAULT_FEATURES= 'module2/processed/feature_names.txt'

# These 4 raw features are appended to the LSTM output in state_vector
# They must exist in the row_dict passed to update()
RAW_STATE_COLS = [
    'utilization_pct',    # [5] current load %
    'bw_headroom_mbps',   # [6] remaining capacity
    'delta_tx_dropped',   # [7] packet drops this interval
    'latency_ms',         # [8] LLDP-measured delay
]

# BW clip applied during preprocessing — must match preprocess.py
BW_CLIP = 500.0

# ─────────────────────────────────────────────────────────────────

class LSTMPredictor:
    """
    Stateful inference wrapper around SDNTrafficLSTM.

    Maintains a rolling window of the last SEQ_LEN (10) rows for
    each (dpid, port_no) pair seen by the controller. Once a port
    has accumulated 10 rows, every subsequent call to update()
    produces a fresh prediction.

    Usage (inside qos_controller.py):
        # At controller startup:
        predictor = LSTMPredictor()
        predictor.load()

        # Inside _port_stats_handler, for each port row:
        predictor.update(dpid, port_no, row_dict)

        # To get the state vector for Module III DQN:
        sv = predictor.state_vector(dpid, port_no)
        # sv is a numpy float32 array of shape (9,)
        # Returns None if fewer than SEQ_LEN rows seen yet for this port
    """

    def __init__(
        self,
        ckpt_path    = DEFAULT_CKPT,
        scaler_path  = DEFAULT_SCALER,
        feature_path = DEFAULT_FEATURES,
        device       = 'cpu',
    ):
        self.ckpt_path    = ckpt_path
        self.scaler_path  = scaler_path
        self.feature_path = feature_path
        self.device       = torch.device(device)

        self.model        = None
        self.scaler       = None
        self.feature_cols = None
        self.loaded       = False

        # Per-port rolling window: (dpid, port_no) → deque of raw row dicts
        # deque(maxlen=SEQ_LEN) auto-drops oldest row when full
        self._windows = {}

        # Per-port latest prediction cache
        # (dpid, port_no) → state_vector numpy array (9,)
        self._cache = {}

    # ── Loading ───────────────────────────────────────────────────

    def load(self):
        """
        Load model checkpoint, scaler, and feature names from disk.
        Call this once at controller startup — not on every polling cycle.
        """
        if self.loaded:
            return

        # Load feature names
        if not os.path.isfile(self.feature_path):
            raise FileNotFoundError(
                f'Feature names not found: {self.feature_path}\n'
                f'Run python3 module2/preprocess.py first.'
            )
        with open(self.feature_path) as f:
            self.feature_cols = [line.strip() for line in f if line.strip()]
        assert len(self.feature_cols) == N_FEATURES, \
            f'Expected {N_FEATURES} features, got {len(self.feature_cols)}'

        # Load scaler
        if not os.path.isfile(self.scaler_path):
            raise FileNotFoundError(
                f'Scaler not found: {self.scaler_path}\n'
                f'Run python3 module2/preprocess.py first.'
            )
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load model
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(
                f'Checkpoint not found: {self.ckpt_path}\n'
                f'Run python3 module2/train.py first.'
            )
        ckpt = torch.load(self.ckpt_path, map_location=self.device,
                          weights_only=False)
        self.model = SDNTrafficLSTM().to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        epoch     = ckpt.get('epoch', '?')
        val_loss  = ckpt.get('val_loss', '?')
        logger.info(
            f'LSTMPredictor loaded — epoch={epoch}  val_loss={val_loss:.6f}'
            if isinstance(val_loss, float) else
            f'LSTMPredictor loaded — epoch={epoch}'
        )
        self.loaded = True

    # ── Core update ───────────────────────────────────────────────

    def update(self, dpid, port_no, row_dict):
        """
        Called every 2 seconds by the controller for each port.
        Appends the new row to that port's rolling window.
        If the window is full (10 rows), runs LSTM and updates cache.

        Args:
            dpid     : switch datapath ID (int or hex string)
            port_no  : port number (int)
            row_dict : dict with at minimum all 17 feature columns
                       + the 4 RAW_STATE_COLS
                       (exactly what qos_controller.py writes to CSV)

        Returns:
            state_vector numpy (9,) if window is full, else None
        """
        if not self.loaded:
            logger.warning('LSTMPredictor.update() called before load()')
            return None

        key = (str(dpid), int(port_no))

        # Initialise deque for new port
        if key not in self._windows:
            self._windows[key] = deque(maxlen=SEQ_LEN)

        self._windows[key].append(row_dict)

        # Only predict once we have a full window
        if len(self._windows[key]) < SEQ_LEN:
            return None

        sv = self._predict(key, row_dict)
        self._cache[key] = sv
        return sv

    # ── State vector ─────────────────────────────────────────────

    def state_vector(self, dpid, port_no):
        """
        Return the latest cached state vector for a port.

        Args:
            dpid    : switch datapath ID
            port_no : port number

        Returns:
            numpy float32 array of shape (9,):
                [P(normal), P(warning), P(congested),
                 cong_prob, is_congested,
                 utilization_pct, bw_headroom_mbps,
                 delta_tx_dropped, latency_ms]
            None if fewer than SEQ_LEN rows have been seen yet.
        """
        key = (str(dpid), int(port_no))
        return self._cache.get(key, None)

    def state_vector_all(self):
        """
        Return state vectors for ALL ports that have predictions.
        Useful for Module III when it needs a global network view.

        Returns:
            dict: {(dpid, port_no): numpy array (9,)}
        """
        return dict(self._cache)

    # ── Internal prediction ───────────────────────────────────────

    def _predict(self, key, latest_row):
        """
        Build the feature matrix from the window, scale it,
        run the LSTM forward pass, and assemble the state vector.

        Args:
            key        : (dpid, port_no) tuple
            latest_row : the most recent row dict (for raw state cols)

        Returns:
            numpy float32 array (9,)
        """
        window = self._windows[key]   # deque of SEQ_LEN row dicts

        # ── Build raw feature matrix (SEQ_LEN × N_FEATURES) ──────
        raw = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
        for t, row in enumerate(window):
            for j, col in enumerate(self.feature_cols):
                val = float(row.get(col, 0.0))
                # Apply same clip used in preprocess.py
                if col in ('tx_mbps', 'rx_mbps'):
                    val = min(val, BW_CLIP)
                raw[t, j] = val

        # ── Scale using fitted StandardScaler ─────────────────────
        # scaler expects shape (N, n_features) → reshape, scale, reshape
        scaled = self.scaler.transform(raw)   # (SEQ_LEN, N_FEATURES)

        # ── Run LSTM ──────────────────────────────────────────────
        x = torch.from_numpy(scaled).unsqueeze(0)   # (1, SEQ_LEN, N_FEATURES)
        lstm_sv = self.model.state_vector(x)         # numpy (5,)
        # lstm_sv = [P(normal), P(warning), P(congested), cong_prob, is_congested]

        # ── Append raw snapshot features ──────────────────────────
        raw_snap = np.array([
            float(latest_row.get('utilization_pct',    0.0)),
            float(latest_row.get('bw_headroom_mbps',   0.0)),
            float(latest_row.get('delta_tx_dropped',   0.0)),
            float(latest_row.get('latency_ms',         0.0)),
        ], dtype=np.float32)

        # ── Concatenate into final state vector ───────────────────
        sv = np.concatenate([lstm_sv, raw_snap])   # (9,)
        return sv

    # ── Utilities ─────────────────────────────────────────────────

    def reset_port(self, dpid, port_no):
        """
        Clear the rolling window for a port.
        Call this if a switch reconnects or a port goes down/up.
        """
        key = (str(dpid), int(port_no))
        self._windows.pop(key, None)
        self._cache.pop(key, None)
        logger.info(f'LSTMPredictor: reset port {key}')

    def reset_all(self):
        """Clear all windows and cache (e.g. on controller restart)."""
        self._windows.clear()
        self._cache.clear()
        logger.info('LSTMPredictor: all ports reset')

    @property
    def n_ports_ready(self):
        """Number of ports that have a full window and can predict."""
        return len(self._cache)

    @property
    def n_ports_warming(self):
        """Number of ports seen but not yet at full window length."""
        return sum(
            1 for key, dq in self._windows.items()
            if len(dq) < SEQ_LEN and key not in self._cache
        )

    def window_fill_pct(self, dpid, port_no):
        """
        How full is this port's rolling window? (0–100%)
        Useful for logging during controller warmup.
        """
        key = (str(dpid), int(port_no))
        dq  = self._windows.get(key)
        if dq is None:
            return 0.0
        return 100.0 * len(dq) / SEQ_LEN

    def summary(self):
        """Print a one-line status summary."""
        print(f'LSTMPredictor | loaded={self.loaded} | '
              f'ports_ready={self.n_ports_ready} | '
              f'ports_warming={self.n_ports_warming}')


# ─────────────────────────────────────────────────────────────────
# Integration snippet for qos_controller.py
# ─────────────────────────────────────────────────────────────────
INTEGRATION_GUIDE = """
HOW TO INTEGRATE INTO qos_controller.py
=========================================

1. At the top of qos_controller.py, add:
       from module2.lstm_predictor import LSTMPredictor

2. In QoSController.__init__(), add:
       self.predictor = LSTMPredictor()
       self.predictor.load()

3. At the END of your _port_stats_handler (after writing to CSV),
   add this block for each port row:
       row_dict = {
           'tx_mbps':          tx_mbps,
           'rx_mbps':          rx_mbps,
           'utilization_pct':  util_pct,
           'tx_pps':           tx_pps,
           'rx_pps':           rx_pps,
           'bw_headroom_mbps': bw_headroom,
           'delta_tx_dropped': delta_tx_drop,
           'delta_rx_dropped': delta_rx_drop,
           'latency_ms':       latency_ms,
           'jitter_ms':        jitter_ms,
           'rolling_util_mean':rolling_util_mean,
           'rolling_drop_sum': rolling_drop_sum,
           'rolling_tx_mean':  rolling_tx_mean,
           'rolling_rx_mean':  rolling_rx_mean,
           'n_active_flows':   n_active_flows,
           'neighbor_util_max':neighbor_util_max,
           'inter_arrival_delta': inter_arrival_delta,
       }
       sv = self.predictor.update(dpid, port_no, row_dict)
       if sv is not None:
           # sv is ready for Module III DQN
           # sv[0]=P(normal)  sv[1]=P(warning)  sv[2]=P(congested)
           # sv[3]=cong_prob  sv[4]=is_congested
           # sv[5]=util%      sv[6]=bw_headroom  sv[7]=drops  sv[8]=latency
           pass
"""


# ─────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 65)
    print('  Module II — Task 7: Inference Module Smoke Test')
    print('=' * 65)

    import pandas as pd

    # Load a sample of real rows from qos_log.csv for testing
    log_path = 'logs/qos_log.csv'
    if not os.path.isfile(log_path):
        log_path = '/mnt/user-data/uploads/qos_log.csv'

    print(f'\n[Step 1] Loading sample rows from {log_path} ...')
    df = pd.read_csv(log_path)
    print(f'  Loaded {len(df)} rows')

    # Pick one port to simulate live streaming
    sample_dpid = df['dpid'].iloc[0]
    sample_port = int(df['port_no'].iloc[0])
    port_rows   = df[(df['dpid'] == sample_dpid) &
                     (df['port_no'] == sample_port)].reset_index(drop=True)
    print(f'  Using dpid={sample_dpid}  port={sample_port}  '
          f'({len(port_rows)} rows available)')

    # Instantiate predictor
    print('\n[Step 2] Instantiating LSTMPredictor ...')
    predictor = LSTMPredictor()
    predictor.load()
    predictor.summary()

    # Simulate streaming rows one at a time
    print('\n[Step 3] Simulating live streaming (one row per poll cycle) ...')
    print(f'\n  {"Row":>4}  {"Fill":>6}  {"Ready":>6}  '
          f'{"P(norm)":>8}  {"P(warn)":>8}  {"P(cong)":>8}  '
          f'{"CongProb":>9}  {"IsCong":>7}  {"Util%":>6}')
    print('  ' + '-' * 80)

    sv_final = None
    for i, (_, row) in enumerate(port_rows.iterrows()):
        row_dict = row.to_dict()
        sv = predictor.update(sample_dpid, sample_port, row_dict)

        fill = predictor.window_fill_pct(sample_dpid, sample_port)
        ready = sv is not None

        if i < SEQ_LEN + 3 or i == len(port_rows) - 1:
            if ready:
                print(f'  {i+1:>4}  {fill:>5.0f}%  {"YES":>6}  '
                      f'{sv[0]:>8.4f}  {sv[1]:>8.4f}  {sv[2]:>8.4f}  '
                      f'{sv[3]:>9.4f}  {sv[4]:>7.0f}  {sv[5]:>6.1f}%')
            else:
                print(f'  {i+1:>4}  {fill:>5.0f}%  {"no":>6}  '
                      f'{"(warming up)":>46}')
        if sv is not None:
            sv_final = sv

    print('\n[Step 4] Final state vector shape and values ...')
    print(f'  Shape : {sv_final.shape}  (expected (9,))')
    print(f'  Values:')
    names = ['P(normal)', 'P(warning)', 'P(congested)', 'cong_prob',
             'is_congested', 'utilization_pct', 'bw_headroom_mbps',
             'delta_tx_dropped', 'latency_ms']
    for name, val in zip(names, sv_final):
        print(f'    [{names.index(name)}] {name:<20} = {val:.4f}')

    assert sv_final.shape == (9,), f'Expected (9,), got {sv_final.shape}'
    assert 0.0 <= sv_final[3] <= 1.0, 'cong_prob must be in [0,1]'
    assert sv_final[4] in (0.0, 1.0), 'is_congested must be 0 or 1'
    assert abs(sv_final[:3].sum() - 1.0) < 1e-4, 'Zone probs must sum to 1'
    print('\n  All assertions passed')

    print('\n[Step 5] state_vector_all() ...')
    all_sv = predictor.state_vector_all()
    print(f'  Ports with predictions: {len(all_sv)}')
    predictor.summary()

    print('\n[Step 6] Integration guide for qos_controller.py ...')
    print(INTEGRATION_GUIDE)

    print('=' * 65)
    print('  Task 7 complete. Module II is fully implemented.')
    print('=' * 65)
