"""
evaluate.py — Module II Task 6: Evaluation
============================================
Run from sdn-project/ directory AFTER training is complete:
    python3 module2/evaluate.py

What this script does:
  Step 1 — Load best_lstm.pt and run inference on the test set
  Step 2 — Zone classification metrics (Head A)
  Step 3 — Binary congestion metrics derived from zone (congested = zone==2)
  Step 4 — Congestion probability metrics (Head B)
  Step 5 — Confusion matrix plot
  Step 6 — Training curve plot (loss + accuracy over epochs)
  Step 7 — Save full evaluation report

Outputs (all in module2/checkpoints/):
  evaluation_report.txt   — all numeric metrics
  confusion_matrix.png    — 3×3 confusion matrix heatmap
  training_curves.png     — loss and accuracy curves over epochs

Metrics explained:
  Zone head (3-class):
    Accuracy    — overall % of windows classified correctly
    Per-class F1— harmonic mean of precision and recall per class
                  important because warning class is rare (6.1% of test)
    Macro F1    — average F1 across all 3 classes (treats classes equally)
    Weighted F1 — F1 weighted by class frequency

  Binary (derived from zone==2):
    Precision   — of all predicted congested, how many actually were?
    Recall      — of all actual congested, how many did we catch?
    F1          — harmonic mean of precision and recall
    ROC-AUC     — area under the ROC curve using cong_prob from Head B

  Probability head (Head B):
    Brier score — mean squared error between predicted prob and true binary
                  0.0 = perfect, 0.25 = random, lower is better
"""

import os
import sys
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works without display
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, brier_score_loss, f1_score, accuracy_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import SDNTrafficLSTM, get_dataloaders

# ── Config ───────────────────────────────────────────────────────
DATA_DIR  = 'module2/processed'
CKPT_DIR  = 'module2/checkpoints'
ZONE_NAMES= ['normal', 'warning', 'congested']

# ─────────────────────────────────────────────────────────────────

def run_inference(model, test_loader, device):
    """
    Run the trained model on the entire test set.
    Returns numpy arrays of predictions and ground truth.
    """
    model.eval()
    all_zone_pred = []
    all_zone_true = []
    all_cong_prob = []
    all_bin_true  = []

    with torch.no_grad():
        for X_batch, y_zone_batch, y_bin_batch in test_loader:
            X_batch = X_batch.to(device)
            zone_logits, cong_logit = model(X_batch)

            zone_pred = zone_logits.argmax(dim=1).cpu().numpy()
            cong_prob = torch.sigmoid(cong_logit.squeeze(1)).cpu().numpy()

            all_zone_pred.extend(zone_pred)
            all_zone_true.extend(y_zone_batch.numpy())
            all_cong_prob.extend(cong_prob)
            all_bin_true.extend(y_bin_batch.numpy())

    return (
        np.array(all_zone_pred, dtype=np.int64),
        np.array(all_zone_true, dtype=np.int64),
        np.array(all_cong_prob, dtype=np.float32),
        np.array(all_bin_true,  dtype=np.int64),
    )


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save a labelled confusion matrix heatmap.
    Rows = actual class, Columns = predicted class.
    Each cell shows count and row-normalised percentage.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalise rows to get per-class recall percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_classes = len(class_names)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title('Confusion Matrix — Zone Classification\n(row-normalised)', fontsize=13)

    # Annotate each cell with count and percentage
    thresh = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if cm_norm[i, j] > thresh else 'black'
            ax.text(j, i,
                    f'{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)',
                    ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_training_curves(history, save_path):
    """
    Plot training and validation loss + zone accuracy over epochs.
    Two subplots side by side.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Loss curves ──────────────────────────────────────────────
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=4,
             label='Train loss', linewidth=1.5)
    ax1.plot(epochs, history['val_loss'],   'r-o', markersize=4,
             label='Val loss',   linewidth=1.5)

    # Mark best val epoch
    best_ep = int(np.argmin(history['val_loss'])) + 1
    best_vl = min(history['val_loss'])
    ax1.axvline(x=best_ep, color='green', linestyle='--', linewidth=1.2,
                label=f'Best epoch ({best_ep})')
    ax1.scatter([best_ep], [best_vl], color='green', zorder=5, s=80)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Combined Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ── Zone accuracy curves ─────────────────────────────────────
    ax2.plot(epochs, [v*100 for v in history['train_zone_acc']], 'b-o', markersize=4,
             label='Train zone acc', linewidth=1.5)
    ax2.plot(epochs, [v*100 for v in history['val_zone_acc']],   'r-o', markersize=4,
             label='Val zone acc',   linewidth=1.5)
    ax2.axvline(x=best_ep, color='green', linestyle='--', linewidth=1.2,
                label=f'Best epoch ({best_ep})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Zone Accuracy (%)', fontsize=12)
    ax2.set_title('Zone Classification Accuracy', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.suptitle('SDN Traffic LSTM — Training History', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def main():
    device = torch.device('cpu')

    print('=' * 65)
    print('  Module II — Task 6: Evaluation')
    print('=' * 65)

    # ── Step 1 — Load model and run inference ─────────────────────
    print('\n[Step 1] Loading best checkpoint and running test inference ...')
    ckpt_path = f'{CKPT_DIR}/best_lstm.pt'
    if not os.path.isfile(ckpt_path):
        print(f'  ERROR: {ckpt_path} not found.')
        print('  Run python3 module2/train.py first.')
        return

    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    model   = SDNTrafficLSTM().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    best_epoch    = ckpt.get('epoch', '?')
    best_val_loss = ckpt.get('val_loss', '?')

    print(f'  Checkpoint epoch    : {best_epoch}')
    print(f'  Checkpoint val loss : {best_val_loss:.6f}' if isinstance(best_val_loss, float) else '')

    _, _, test_loader, _ = get_dataloaders(DATA_DIR, batch_size=256)
    print(f'  Test windows        : {len(test_loader.dataset)}')

    zone_pred, zone_true, cong_prob, bin_true = run_inference(
        model, test_loader, device
    )
    bin_pred = (zone_pred == 2).astype(np.int64)

    report_lines = [
        '=' * 65,
        '  Module II — Task 6: Evaluation Report',
        '=' * 65,
        f'  Checkpoint : epoch {best_epoch}',
        f'  Test set   : {len(zone_true)} windows',
        '',
    ]

    def log(msg=''):
        print(msg)
        report_lines.append(msg)

    # ── Step 2 — Zone classification metrics (Head A) ─────────────
    log('\n[Step 2] Zone classification metrics (Head A) ...')

    zone_acc    = accuracy_score(zone_true, zone_pred)
    macro_f1    = f1_score(zone_true, zone_pred, average='macro',    zero_division=0)
    weighted_f1 = f1_score(zone_true, zone_pred, average='weighted', zero_division=0)

    log(f'  Overall accuracy : {zone_acc*100:.2f}%')
    log(f'  Macro F1         : {macro_f1:.4f}')
    log(f'  Weighted F1      : {weighted_f1:.4f}')
    log('')
    log('  Per-class breakdown:')

    cls_report = classification_report(
        zone_true, zone_pred,
        target_names=ZONE_NAMES,
        zero_division=0,
        digits=4
    )
    for line in cls_report.split('\n'):
        log('    ' + line)

    # ── Step 3 — Binary congestion metrics ────────────────────────
    log('\n[Step 3] Binary congestion metrics (zone==2 → congested) ...')

    tp = int(((bin_pred == 1) & (bin_true == 1)).sum())
    fp = int(((bin_pred == 1) & (bin_true == 0)).sum())
    fn = int(((bin_pred == 0) & (bin_true == 1)).sum())
    tn = int(((bin_pred == 0) & (bin_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    bin_acc   = (tp + tn) / len(bin_true)

    log(f'  Confusion (binary):')
    log(f'    TP={tp}  FP={fp}  FN={fn}  TN={tn}')
    log(f'  Accuracy  : {bin_acc*100:.2f}%')
    log(f'  Precision : {precision:.4f}  '
        f'(of predicted congested, {precision*100:.1f}% actually were)')
    log(f'  Recall    : {recall:.4f}  '
        f'(of actual congested, {recall*100:.1f}% were caught)')
    log(f'  F1 score  : {f1:.4f}')

    # ROC-AUC using Head B probability
    try:
        roc_auc = roc_auc_score(bin_true, cong_prob)
        log(f'  ROC-AUC   : {roc_auc:.4f}  (using Head B probability)')
    except Exception as e:
        roc_auc = None
        log(f'  ROC-AUC   : could not compute ({e})')

    # ── Step 4 — Probability metrics (Head B) ─────────────────────
    log('\n[Step 4] Congestion probability metrics (Head B) ...')

    brier = brier_score_loss(bin_true, cong_prob)
    log(f'  Brier score   : {brier:.4f}  '
        f'(0.0=perfect, 0.25=random, lower is better)')
    log(f'  Mean prob when not congested : '
        f'{cong_prob[bin_true==0].mean():.4f}')
    log(f'  Mean prob when congested     : '
        f'{cong_prob[bin_true==1].mean():.4f}')

    # ── Step 5 — Confusion matrix plot ───────────────────────────
    log('\n[Step 5] Plotting confusion matrix ...')
    cm = confusion_matrix(zone_true, zone_pred, labels=[0, 1, 2])
    log(f'  Raw confusion matrix (rows=actual, cols=predicted):')
    log(f'    {"":12}  ' + '  '.join(f'{n:>10}' for n in ZONE_NAMES))
    for i, row in enumerate(cm):
        log(f'    {ZONE_NAMES[i]:<12}  ' +
            '  '.join(f'{v:>10}' for v in row))
    plot_confusion_matrix(cm, ZONE_NAMES,
                          f'{CKPT_DIR}/confusion_matrix.png')

    # ── Step 6 — Training curves ──────────────────────────────────
    log('\n[Step 6] Plotting training curves ...')
    history_path = f'{CKPT_DIR}/history.npy'
    if os.path.isfile(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        plot_training_curves(history, f'{CKPT_DIR}/training_curves.png')
        log(f'  Epochs trained : {len(history["train_loss"])}')
        log(f'  Best val loss  : {min(history["val_loss"]):.6f}  '
            f'at epoch {int(np.argmin(history["val_loss"]))+1}')
        log(f'  Best val ZAcc  : '
            f'{max(history["val_zone_acc"])*100:.2f}%  '
            f'at epoch {int(np.argmax(history["val_zone_acc"]))+1}')
    else:
        log('  history.npy not found — run train.py fully to get curves')

    # ── Step 7 — Save report ──────────────────────────────────────
    log('\n[Step 7] Saving evaluation report ...')
    report_path = f'{CKPT_DIR}/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    log(f'  Saved: {report_path}')

    log('\n' + '=' * 65)
    log('  Task 6 complete. Ready for Task 7 (Inference Module).')
    log('=' * 65)


if __name__ == '__main__':
    main()
