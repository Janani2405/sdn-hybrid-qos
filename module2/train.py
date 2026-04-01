"""
train.py — Module II Task 5: Training Loop
============================================
Run from sdn-project/ directory:
    python3 module2/train.py

What this script does (in order):
  Step 1 — Load data via get_dataloaders() from model.py
  Step 2 — Instantiate SDNTrafficLSTM and both loss functions
  Step 3 — Set up AdamW optimizer + CosineAnnealingLR scheduler
  Step 4 — Run training loop (train + validate each epoch)
  Step 5 — Early stopping: save best model, stop if no improvement
  Step 6 — Save training history and final report

Loss function (combined):
  total_loss = 0.6 × zone_loss + 0.4 × prob_loss

  zone_loss — CrossEntropyLoss with class weights [0.54, 6.01, 1.03]
    The 6× weight on warning ensures the model learns the early-warning
    signal even though warning is only 5.7% of training data.

  prob_loss — BCEWithLogitsLoss with pos_weight=2.10
    Weights each congested example 2.1× more than normal/warning.

  Why 0.6/0.4 split:
    Zone classification is the primary task (Module III needs zone state).
    Probability is secondary (soft score for DQN confidence).
    0.6/0.4 gives zone the dominant gradient without silencing prob head.

Optimizer: AdamW (Adam + weight decay)
  lr=1e-3, weight_decay=1e-4
  Weight decay prevents overfitting on the 220K parameter model.
  AdamW separates weight decay from gradient update (unlike Adam+L2).

Scheduler: CosineAnnealingLR
  T_max = n_epochs, eta_min = 1e-5
  Learning rate follows a cosine curve from 1e-3 down to 1e-5.
  Prevents the model from getting stuck in sharp minima late in training.

Early stopping:
  patience=10 — if validation loss does not improve for 10 consecutive
  epochs, training stops. The best checkpoint is already saved.
  This prevents overfitting and saves time on CPU.
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn

# Allow imports from sdn-project/module2/ regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import SDNTrafficLSTM, get_dataloaders, count_parameters

# ── Config ───────────────────────────────────────────────────────
DATA_DIR    = 'module2/processed'
CKPT_DIR    = 'module2/checkpoints'
N_EPOCHS    = 50
BATCH_SIZE  = 256
LR          = 1e-3
WEIGHT_DECAY= 1e-4
PATIENCE    = 10          # early stopping patience
ZONE_WEIGHT = 0.6         # weight of zone loss in combined loss
PROB_WEIGHT = 0.4         # weight of prob loss in combined loss
SEED        = 42

# ─────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, optimizer, zone_criterion, prob_criterion, device):
    """
    Run one full pass over the training set.
    Returns dict of average losses and accuracy for this epoch.
    """
    model.train()
    total_loss   = 0.0
    total_zloss  = 0.0
    total_ploss  = 0.0
    correct_zone = 0
    total_samples= 0

    for X_batch, y_zone_batch, y_bin_batch in loader:
        X_batch      = X_batch.to(device)
        y_zone_batch = y_zone_batch.to(device)
        y_bin_batch  = y_bin_batch.to(device)

        optimizer.zero_grad()

        zone_logits, cong_logit = model(X_batch)

        # Head A loss — zone classification
        z_loss = zone_criterion(zone_logits, y_zone_batch)

        # Head B loss — congestion probability
        # y_bin_batch is float32 (0.0 or 1.0); cong_logit is (batch,1)
        p_loss = prob_criterion(cong_logit.squeeze(1), y_bin_batch)

        loss = ZONE_WEIGHT * z_loss + PROB_WEIGHT * p_loss
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        # Clips the global norm of all gradients to max 1.0
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate metrics
        batch_size    = X_batch.size(0)
        total_loss   += loss.item()   * batch_size
        total_zloss  += z_loss.item() * batch_size
        total_ploss  += p_loss.item() * batch_size
        preds         = zone_logits.argmax(dim=1)
        correct_zone += (preds == y_zone_batch).sum().item()
        total_samples+= batch_size

    n = total_samples
    return {
        'loss':      total_loss  / n,
        'zone_loss': total_zloss / n,
        'prob_loss': total_ploss / n,
        'zone_acc':  correct_zone / n,
    }


@torch.no_grad()
def validate(model, loader, zone_criterion, prob_criterion, device):
    """
    Run one full pass over the validation set (no gradients).
    Returns same metric dict as train_one_epoch.
    """
    model.eval()
    total_loss   = 0.0
    total_zloss  = 0.0
    total_ploss  = 0.0
    correct_zone = 0
    total_samples= 0

    for X_batch, y_zone_batch, y_bin_batch in loader:
        X_batch      = X_batch.to(device)
        y_zone_batch = y_zone_batch.to(device)
        y_bin_batch  = y_bin_batch.to(device)

        zone_logits, cong_logit = model(X_batch)

        z_loss = zone_criterion(zone_logits, y_zone_batch)
        p_loss = prob_criterion(cong_logit.squeeze(1), y_bin_batch)
        loss   = ZONE_WEIGHT * z_loss + PROB_WEIGHT * p_loss

        batch_size    = X_batch.size(0)
        total_loss   += loss.item()   * batch_size
        total_zloss  += z_loss.item() * batch_size
        total_ploss  += p_loss.item() * batch_size
        preds         = zone_logits.argmax(dim=1)
        correct_zone += (preds == y_zone_batch).sum().item()
        total_samples+= batch_size

    n = total_samples
    return {
        'loss':      total_loss  / n,
        'zone_loss': total_zloss / n,
        'prob_loss': total_ploss / n,
        'zone_acc':  correct_zone / n,
    }


def main():
    set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device('cpu')   # CPU-only machine

    print('=' * 65)
    print('  Module II — Task 5: Training Loop')
    print(f'  Device     : {device}')
    print(f'  Epochs     : {N_EPOCHS}  (early stopping patience={PATIENCE})')
    print(f'  Batch size : {BATCH_SIZE}')
    print(f'  LR         : {LR}  →  cosine anneal to 1e-5')
    print(f'  Loss       : {ZONE_WEIGHT}×zone  +  {PROB_WEIGHT}×prob')
    print('=' * 65)

    # ── Step 1 — Data ────────────────────────────────────────────
    print('\n[Step 1] Loading data ...')
    train_loader, val_loader, _, class_weights = get_dataloaders(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE
    )
    print(f'  Train batches : {len(train_loader)}  '
          f'({len(train_loader.dataset)} windows)')
    print(f'  Val   batches : {len(val_loader)}  '
          f'({len(val_loader.dataset)} windows)')
    print(f'  zone_weights  : {class_weights["zone_weights"].numpy().round(4)}')
    print(f'  pos_weight    : {class_weights["pos_weight"].item():.4f}')

    # ── Step 2 — Model and loss functions ────────────────────────
    print('\n[Step 2] Building model and loss functions ...')
    model = SDNTrafficLSTM().to(device)
    count_parameters(model)

    zone_criterion = nn.CrossEntropyLoss(
        weight=class_weights['zone_weights'].to(device)
    )
    prob_criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_weights['pos_weight'].to(device)
    )

    # ── Step 3 — Optimizer and scheduler ─────────────────────────
    print('\n[Step 3] Setting up optimizer and scheduler ...')
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5
    )
    print(f'  AdamW  lr={LR}  weight_decay={WEIGHT_DECAY}')
    print(f'  CosineAnnealingLR  T_max={N_EPOCHS}  eta_min=1e-5')

    # ── Step 4+5 — Training loop with early stopping ─────────────
    print('\n[Step 4] Training ...')
    print(f'\n{"Ep":>4}  {"Tr Loss":>9}  {"Tr ZAcc":>9}  '
          f'{"Vl Loss":>9}  {"Vl ZAcc":>9}  {"LR":>9}  {"":>6}')
    print('-' * 70)

    history = {
        'train_loss': [], 'train_zone_acc': [],
        'train_zone_loss': [], 'train_prob_loss': [],
        'val_loss':   [], 'val_zone_acc': [],
        'val_zone_loss':   [], 'val_prob_loss': [],
        'lr': [],
    }

    best_val_loss   = float('inf')
    epochs_no_improve = 0
    best_epoch      = 0
    training_start  = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        ep_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            zone_criterion, prob_criterion, device
        )
        val_metrics = validate(
            model, val_loader,
            zone_criterion, prob_criterion, device
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_zone_acc'].append(train_metrics['zone_acc'])
        history['train_zone_loss'].append(train_metrics['zone_loss'])
        history['train_prob_loss'].append(train_metrics['prob_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_zone_acc'].append(val_metrics['zone_acc'])
        history['val_zone_loss'].append(val_metrics['zone_loss'])
        history['val_prob_loss'].append(val_metrics['prob_loss'])
        history['lr'].append(current_lr)

        ep_time = time.time() - ep_start

        # Check for improvement
        improved = val_metrics['loss'] < best_val_loss
        if improved:
            best_val_loss    = val_metrics['loss']
            best_epoch       = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':   best_val_loss,
                'val_zone_acc': val_metrics['zone_acc'],
                'history':    history,
            }, f'{CKPT_DIR}/best_lstm.pt')
            tag = '  ✓ saved'
        else:
            epochs_no_improve += 1
            tag = f'  ({epochs_no_improve}/{PATIENCE})'

        print(f'{epoch:>4}  '
              f'{train_metrics["loss"]:>9.4f}  '
              f'{train_metrics["zone_acc"]:>9.4f}  '
              f'{val_metrics["loss"]:>9.4f}  '
              f'{val_metrics["zone_acc"]:>9.4f}  '
              f'{current_lr:>9.2e}  '
              f'{tag}')

        # Early stopping check
        if epochs_no_improve >= PATIENCE:
            print(f'\n  Early stopping at epoch {epoch} '
                  f'(no improvement for {PATIENCE} epochs)')
            break

    total_time = time.time() - training_start

    # ── Step 6 — Save history and report ─────────────────────────
    print('\n[Step 5] Saving training history ...')

    np.save(f'{CKPT_DIR}/history.npy', history)

    report = [
        '=' * 65,
        '  Module II — Task 5: Training Report',
        '=' * 65,
        f'  Total epochs run  : {epoch}',
        f'  Best epoch        : {best_epoch}',
        f'  Best val loss     : {best_val_loss:.6f}',
        f'  Best val zone acc : {history["val_zone_acc"][best_epoch-1]:.4f}',
        f'  Total train time  : {total_time:.1f}s  ({total_time/60:.1f} min)',
        f'  Time per epoch    : {total_time/epoch:.1f}s',
        '',
        '  Final epoch metrics:',
        f'    Train loss      : {history["train_loss"][-1]:.6f}',
        f'    Train zone acc  : {history["train_zone_acc"][-1]:.4f}',
        f'    Val   loss      : {history["val_loss"][-1]:.6f}',
        f'    Val   zone acc  : {history["val_zone_acc"][-1]:.4f}',
        '',
        '  Saved files:',
        f'    {CKPT_DIR}/best_lstm.pt',
        f'    {CKPT_DIR}/history.npy',
    ]

    with open(f'{CKPT_DIR}/training_report.txt', 'w') as f:
        f.write('\n'.join(report))

    for line in report:
        print(line)

    print('\n' + '=' * 65)
    print('  Task 5 complete. Ready for Task 6 (Evaluation).')
    print('=' * 65)

    return history


if __name__ == '__main__':
    main()
