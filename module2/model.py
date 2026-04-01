"""
model.py — Module II Task 4: LSTM Model Definition
====================================================
Architecture: 2-layer stacked LSTM → BatchNorm → 2 output heads

  Input  : (batch, seq_len=10, n_features=17)
  Output :
    zone_logits  — (batch, 3)   raw logits for CrossEntropyLoss
    cong_logit   — (batch, 1)   raw logit  for BCEWithLogitsLoss

Head design (why 2 heads, not 4):
  Original plan had 4 heads. Analysis of actual data showed:
    • Head 2 (prob)  — duplicate of Head 1 logit via sigmoid. Dropped.
    • Head 3 (util)  — utilization is already input feature [02].
                       Predicting it back adds no learning signal. Dropped.
    • Head 1 (binary)— zone is a strict superset of binary.
                       binary can be derived from zone: congested=(zone==2).
                       Merged into Head A.
  Result: 2 heads, no redundancy, same predictive power.

  Head A — Zone (3-class):
    Predicts: normal=0 / warning=1 / congested=2
    Gives Module III full network state, not just a binary flag.
    Warning class (util 70–95%) acts as an early congestion signal.

  Head B — Congestion probability (scalar):
    Predicts: P(congested) in [0, 1] via sigmoid
    Gives Module III a continuous confidence score for its state vector.
    Different from the old duplicate Head 2 — this is trained jointly
    with Head A and provides a calibrated soft score.

Also contains:
  SDNDataset     — PyTorch Dataset wrapping the windowed arrays
  get_dataloaders — builds train/val/test DataLoaders
  count_parameters — utility to inspect model size
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Constants (must match preprocess.py and window.py) ───────────
N_FEATURES  = 17
SEQ_LEN     = 10
N_ZONES     = 3       # normal, warning, congested
HIDDEN_SIZE = 128
N_LSTM_LAYERS = 2
DROPOUT     = 0.3

# ─────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────

class SDNDataset(Dataset):
    """
    PyTorch Dataset that wraps the windowed arrays produced by Task 2/3.

    Args:
        X        : float32 array of shape (N, seq_len, n_features)
        y_zone   : int64  array of shape (N,)  — 0/1/2
        y_binary : int64  array of shape (N,)  — 0/1  (for Head B)
        indices  : int64 array of window indices (from split.py)
                   If None, uses all rows.
    """

    def __init__(self, X, y_zone, y_binary, indices=None):
        if indices is not None:
            self.X        = torch.from_numpy(X[indices])
            self.y_zone   = torch.from_numpy(y_zone[indices].astype(np.int64))
            self.y_binary = torch.from_numpy(y_binary[indices].astype(np.float32))
        else:
            self.X        = torch.from_numpy(X)
            self.y_zone   = torch.from_numpy(y_zone.astype(np.int64))
            self.y_binary = torch.from_numpy(y_binary.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_zone[idx], self.y_binary[idx]


# ─────────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────────

def get_dataloaders(data_dir='module2/processed', batch_size=256, num_workers=0):
    """
    Load all arrays, apply split indices, return train/val/test DataLoaders.

    Args:
        data_dir    : path to folder containing Task 1–3 outputs
        batch_size  : mini-batch size (256 recommended for CPU training)
        num_workers : 0 = load in main process (safest on Mininet machine)

    Returns:
        train_loader, val_loader, test_loader, class_weights dict
    """
    # Load arrays
    X          = np.load(f'{data_dir}/X_windows.npy')
    y_zone     = np.load(f'{data_dir}/y_zone_w.npy')
    y_binary   = np.load(f'{data_dir}/y_binary_w.npy')
    train_idx  = np.load(f'{data_dir}/train_idx.npy')
    val_idx    = np.load(f'{data_dir}/val_idx.npy')
    test_idx   = np.load(f'{data_dir}/test_idx.npy')

    # Build datasets
    train_ds = SDNDataset(X, y_zone, y_binary, train_idx)
    val_ds   = SDNDataset(X, y_zone, y_binary, val_idx)
    test_ds  = SDNDataset(X, y_zone, y_binary, test_idx)

    # Build loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False)

    # Compute class weights for CrossEntropyLoss (Head A)
    # Weight = total_samples / (n_classes × count_of_class)
    # This down-weights normal (majority) and up-weights warning (rare).
    y_train_zone   = y_zone[train_idx]
    n_train        = len(y_train_zone)
    zone_counts    = np.bincount(y_train_zone, minlength=N_ZONES).astype(float)
    zone_weights   = n_train / (N_ZONES * zone_counts)
    zone_weights_t = torch.tensor(zone_weights, dtype=torch.float32)

    # pos_weight for BCEWithLogitsLoss (Head B)
    n_neg      = (y_binary[train_idx] == 0).sum()
    n_pos      = (y_binary[train_idx] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    class_weights = {
        'zone_weights': zone_weights_t,   # shape (3,)
        'pos_weight':   pos_weight,        # shape (1,)
    }

    return train_loader, val_loader, test_loader, class_weights


# ─────────────────────────────────────────────────────────────────
#  LSTM Model
# ─────────────────────────────────────────────────────────────────

class SDNTrafficLSTM(nn.Module):
    """
    2-layer stacked LSTM with 2 output heads for SDN congestion prediction.

    Forward pass returns (zone_logits, cong_logit):
      zone_logits : (batch, 3)  — raw scores, pass to CrossEntropyLoss
      cong_logit  : (batch, 1)  — raw score,  pass to BCEWithLogitsLoss
                                  sigmoid(cong_logit) = P(congested)

    Args:
        n_features  : number of input features per timestep (17)
        hidden_size : LSTM hidden dimension (128)
        n_layers    : number of stacked LSTM layers (2)
        n_zones     : number of zone classes (3)
        dropout     : dropout rate between LSTM layers and in heads (0.3)
    """

    def __init__(
        self,
        n_features  = N_FEATURES,
        hidden_size = HIDDEN_SIZE,
        n_layers    = N_LSTM_LAYERS,
        n_zones     = N_ZONES,
        dropout     = DROPOUT,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        # ── LSTM encoder ─────────────────────────────────────────
        # batch_first=True  → input shape is (batch, seq, features)
        #                      (more readable than the default seq-first)
        # dropout applies between stacked layers (not after the last layer)
        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )

        # ── Batch normalisation on the LSTM encoding ─────────────
        # Applied to the (batch, hidden_size) vector from the last timestep.
        # Stabilises training: prevents one direction of the hidden state
        # from dominating both heads during early epochs.
        self.bn = nn.BatchNorm1d(hidden_size)

        # ── Head A — Zone classification ─────────────────────────
        # 128 → 64 → 3
        # Two linear layers with ReLU and dropout give the head capacity
        # to learn a non-linear boundary between normal/warning/congested.
        self.head_zone = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_zones),
        )

        # ── Head B — Congestion probability ──────────────────────
        # 128 → 32 → 1
        # Smaller than Head A because binary classification needs
        # less capacity than 3-class. Sigmoid is NOT applied here —
        # BCEWithLogitsLoss applies it internally for numerical stability.
        self.head_prob = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Initialise weights for stable training
        self._init_weights()

    def _init_weights(self):
        """
        Xavier uniform init for linear layers.
        Orthogonal init for LSTM weights (standard practice for RNNs —
        helps gradients flow through many timesteps without vanishing).
        Biases zeroed except LSTM forget gate biases set to 1.0 —
        this is a well-known trick: starting the forget gate open means
        the LSTM can remember long sequences from the first epoch.
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.zero_()
                # Forget gate bias is in the second quarter of the bias vector
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in [self.head_zone, self.head_prob]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Args:
            x : (batch, seq_len, n_features)  float32

        Returns:
            zone_logits : (batch, 3)   raw logits for CrossEntropyLoss
            cong_logit  : (batch, 1)   raw logit  for BCEWithLogitsLoss
        """
        # LSTM returns output at every timestep + final (h_n, c_n)
        # output shape : (batch, seq_len, hidden_size)
        # h_n shape    : (n_layers, batch, hidden_size)
        lstm_out, (h_n, _) = self.lstm(x)

        # Take the hidden state from the LAST layer, LAST timestep.
        # h_n[-1] = (batch, hidden_size) — the final sequence encoding.
        # Equivalent to lstm_out[:, -1, :] but slightly more efficient.
        encoding = h_n[-1]   # (batch, hidden_size)

        # Batch normalise the encoding
        encoding = self.bn(encoding)   # (batch, hidden_size)

        # Pass through both heads
        zone_logits = self.head_zone(encoding)   # (batch, 3)
        cong_logit  = self.head_prob(encoding)   # (batch, 1)

        return zone_logits, cong_logit

    def predict(self, x):
        """
        Convenience method for inference (no gradients).
        Returns human-readable predictions instead of raw logits.

        Args:
            x : (batch, seq_len, n_features)  float32 tensor

        Returns:
            zone_pred  : (batch,)  int — predicted zone class (0/1/2)
            cong_prob  : (batch,)  float — P(congested) in [0, 1]
            is_congested: (batch,) bool — True if zone_pred == 2
        """
        self.eval()
        with torch.no_grad():
            zone_logits, cong_logit = self.forward(x)
            zone_pred    = zone_logits.argmax(dim=1)           # (batch,)
            cong_prob    = torch.sigmoid(cong_logit).squeeze(1) # (batch,)
            is_congested = (zone_pred == 2)
        return zone_pred, cong_prob, is_congested

    def state_vector(self, x):
        """
        Produces the compact state vector for Module III DQN.
        Called once per controller polling cycle (every 2 seconds).

        Args:
            x : (1, seq_len, n_features)  float32 — one port's window

        Returns:
            numpy array of shape (5,):
              [zone_0_prob, zone_1_prob, zone_2_prob, cong_prob, is_congested]
              zone_0_prob  — P(normal)
              zone_1_prob  — P(warning)
              zone_2_prob  — P(congested)
              cong_prob    — soft congestion confidence from Head B
              is_congested — 1.0 if zone_pred==2 else 0.0
        """
        self.eval()
        with torch.no_grad():
            zone_logits, cong_logit = self.forward(x)
            zone_probs   = torch.softmax(zone_logits, dim=1).squeeze(0)  # (3,)
            cong_prob    = torch.sigmoid(cong_logit).squeeze()            # scalar
            is_congested = float(zone_probs.argmax().item() == 2)

        return np.array([
            zone_probs[0].item(),   # P(normal)
            zone_probs[1].item(),   # P(warning)
            zone_probs[2].item(),   # P(congested)
            cong_prob.item(),       # Head B soft score
            is_congested,           # hard binary from zone head
        ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────

def count_parameters(model):
    """Print a parameter count breakdown by component."""
    total = 0
    print(f'\n{"Component":<30} {"Parameters":>12}')
    print('-' * 44)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        total += params
        print(f'{name:<30} {params:>12,}')
    print('-' * 44)
    print(f'{"TOTAL":<30} {total:>12,}')
    return total


# ─────────────────────────────────────────────────────────────────
#  Smoke test (run this file directly to verify the model)
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 55)
    print('  Module II — Task 4: LSTM Model Definition')
    print('=' * 55)

    # Step 1 — Instantiate model
    print('\n[Step 1] Instantiating SDNTrafficLSTM ...')
    model = SDNTrafficLSTM()
    print(f'  Model created successfully')

    # Step 2 — Parameter count
    print('\n[Step 2] Parameter breakdown:')
    total_params = count_parameters(model)

    # Step 3 — Forward pass with dummy batch
    print('\n[Step 3] Forward pass smoke test ...')
    batch_size = 8
    dummy_x    = torch.randn(batch_size, SEQ_LEN, N_FEATURES)
    zone_logits, cong_logit = model(dummy_x)
    print(f'  Input shape       : {tuple(dummy_x.shape)}')
    print(f'  zone_logits shape : {tuple(zone_logits.shape)}  (expected ({batch_size}, 3))')
    print(f'  cong_logit  shape : {tuple(cong_logit.shape)}   (expected ({batch_size}, 1))')
    assert zone_logits.shape == (batch_size, N_ZONES)
    assert cong_logit.shape  == (batch_size, 1)
    print('  Shape assertions passed')

    # Step 4 — predict() method
    print('\n[Step 4] predict() method test ...')
    zone_pred, cong_prob, is_cong = model.predict(dummy_x)
    print(f'  zone_pred    : {zone_pred.tolist()}')
    print(f'  cong_prob    : {[round(p,3) for p in cong_prob.tolist()]}')
    print(f'  is_congested : {is_cong.tolist()}')
    assert all(p >= 0 and p <= 1 for p in cong_prob.tolist())
    assert all(z in [0, 1, 2] for z in zone_pred.tolist())
    print('  Output range assertions passed')

    # Step 5 — state_vector() method
    print('\n[Step 5] state_vector() method test ...')
    single_x = torch.randn(1, SEQ_LEN, N_FEATURES)
    sv = model.state_vector(single_x)
    print(f'  state_vector shape : {sv.shape}  (expected (5,))')
    print(f'  state_vector values: {np.round(sv, 4)}')
    print(f'  Components:')
    names = ['P(normal)', 'P(warning)', 'P(congested)', 'cong_prob', 'is_congested']
    for name, val in zip(names, sv):
        print(f'    {name:<16} = {val:.4f}')
    assert sv.shape == (5,)
    assert abs(sv[:3].sum() - 1.0) < 1e-5, 'Zone probs must sum to 1'
    print('  state_vector assertions passed')

    # Step 6 — DataLoader test
    print('\n[Step 6] DataLoader test ...')
    try:
        train_loader, val_loader, test_loader, cw = get_dataloaders(
            data_dir='module2/processed', batch_size=256
        )
        xb, yz, yb = next(iter(train_loader))
        print(f'  train_loader batch: X={tuple(xb.shape)}  '
              f'y_zone={tuple(yz.shape)}  y_binary={tuple(yb.shape)}')
        print(f'  zone_weights : {cw["zone_weights"].numpy().round(4)}')
        print(f'  pos_weight   : {cw["pos_weight"].item():.4f}')
        print(f'  Train batches: {len(train_loader)}')
        print(f'  Val   batches: {len(val_loader)}')
        print(f'  Test  batches: {len(test_loader)}')
    except FileNotFoundError:
        print('  (Skipped — processed data not found in current directory)')

    print('\n' + '=' * 55)
    print('  Task 4 complete. Ready for Task 5 (Training Loop).')
    print('=' * 55)
