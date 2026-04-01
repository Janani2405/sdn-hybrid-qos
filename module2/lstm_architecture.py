"""
lstm_architecture.py — Final LSTM Architecture Diagram
=======================================================
Run from sdn-project/:
    python3 module2/lstm_architecture.py

Saves: module2/processed/lstm_architecture.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = 'module2/processed/lstm_architecture.png'
os.makedirs('module2/processed', exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────
C_INPUT    = '#2980b9'
C_LSTM1    = '#8e44ad'
C_LSTM2    = '#6c3483'
C_BN       = '#1abc9c'
C_HEAD_A   = '#e74c3c'
C_HEAD_B   = '#e67e22'
C_ARROW    = '#2c3e50'
C_OUT_A    = '#c0392b'
C_OUT_B    = '#d35400'
C_BG       = '#f0f4f8'
C_DARK     = '#1a252f'
C_GREY     = '#7f8c8d'
C_WHITE    = '#ffffff'
C_SHADOW   = '#bdc3c7'

def rounded_box(ax, x, y, w, h, color, text_lines,
                fontsize=9, text_color='white', alpha=1.0,
                bold_first=True):
    """Draw a rounded rectangle with centred multi-line text."""
    shadow = FancyBboxPatch((x+0.01, y-0.012), w, h,
                             boxstyle='round,pad=0.015',
                             facecolor=C_SHADOW, edgecolor='none',
                             alpha=0.35, zorder=1)
    ax.add_patch(shadow)
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.015',
                          facecolor=color, edgecolor=C_WHITE,
                          linewidth=1.8, alpha=alpha, zorder=2)
    ax.add_patch(box)
    total = len(text_lines)
    for i, line in enumerate(text_lines):
        ty = y + h/2 + (total/2 - i - 0.5) * (fontsize * 0.016)
        bold = bold_first and i == 0
        ax.text(x + w/2, ty, line,
                ha='center', va='center',
                fontsize=fontsize,
                fontweight='bold' if bold else 'normal',
                color=text_color, zorder=3)

def arrow(ax, x1, y1, x2, y2, label='', color=C_ARROW, lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=14),
                zorder=4)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.015, my, label,
                ha='left', va='center',
                fontsize=7.5, color=color,
                fontstyle='italic', zorder=5)

fig, ax = plt.subplots(figsize=(14, 17))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# ── Coordinates (x, y = bottom-left of box, w, h) ────────────────
BW   = 0.52   # box width (centred at 0.5)
BX   = (1 - BW) / 2
SH   = 0.058  # standard height
GAP  = 0.028  # gap between boxes
HEAD_W = 0.22
HEAD_GAP = 0.035

# Y positions from bottom to top
y_out_b  = 0.030
y_out_a  = 0.030
y_head_b = 0.115
y_head_a = 0.115
y_bn     = 0.260
y_lstm2  = 0.355
y_dp     = 0.448
y_lstm1  = 0.525
y_input  = 0.640
y_shape  = 0.730

# ── Title ─────────────────────────────────────────────────────────
ax.text(0.5, 0.97,
        'SDNTrafficLSTM — Final Architecture',
        ha='center', va='top',
        fontsize=17, fontweight='bold', color=C_DARK)
ax.text(0.5, 0.945,
        '2-layer Stacked LSTM  ·  2 Output Heads  ·  220,228 Parameters',
        ha='center', va='top',
        fontsize=11, color=C_GREY)

# horizontal separator
ax.plot([0.05, 0.95], [0.93, 0.93], color=C_SHADOW, linewidth=1.2)

# ── INPUT block ───────────────────────────────────────────────────
y_inp_box = 0.82
rounded_box(ax, BX, y_inp_box, BW, SH, C_INPUT,
            ['Input Tensor',
             'Shape: (batch, seq_len=10, features=17)',
             '10 timesteps  ×  17 features  =  170 values'],
            fontsize=9.5)
ax.text(BX + BW + 0.02, y_inp_box + SH/2,
        '20 seconds\nof history',
        ha='left', va='center', fontsize=8,
        color=C_INPUT, fontstyle='italic')

# ── LSTM Layer 1 ──────────────────────────────────────────────────
y_l1 = 0.715
arrow(ax, 0.5, y_inp_box, 0.5, y_l1 + SH + GAP*0.3)
rounded_box(ax, BX, y_l1, BW, SH, C_LSTM1,
            ['LSTM Layer 1',
             'input_size=17  →  hidden_size=128',
             'batch_first=True  ·  Forget gate bias = 1.0'],
            fontsize=9.5)
ax.text(BX + BW + 0.02, y_l1 + SH/2,
        '74,752\nparams',
        ha='left', va='center', fontsize=8, color=C_LSTM1)

# ── Dropout label ─────────────────────────────────────────────────
y_dp_label = (y_l1 + y_l1 - SH*0.15) / 2 - 0.035
ax.annotate('', xy=(0.5, y_l1 - GAP*1.2),
            xytext=(0.5, y_l1),
            arrowprops=dict(arrowstyle='->', color=C_GREY, lw=1.5))
ax.text(0.5 + 0.015, y_l1 - GAP*0.6,
        'Dropout(p=0.3)',
        ha='left', va='center', fontsize=8,
        color=C_GREY, fontstyle='italic')

# ── LSTM Layer 2 ──────────────────────────────────────────────────
y_l2 = 0.610
rounded_box(ax, BX, y_l2, BW, SH, C_LSTM2,
            ['LSTM Layer 2',
             'input_size=128  →  hidden_size=128',
             'Take h_n[-1]  →  (batch, 128)  ← final encoding'],
            fontsize=9.5)
ax.text(BX + BW + 0.02, y_l2 + SH/2,
        '131,584\nparams',
        ha='left', va='center', fontsize=8, color=C_LSTM2)
arrow(ax, 0.5, y_l2, 0.5, y_l2 - GAP*0.9,
      label='(batch, 128)')

# ── BatchNorm ─────────────────────────────────────────────────────
y_bn = 0.505
rounded_box(ax, BX, y_bn, BW, SH*0.85, C_BN,
            ['BatchNorm1d(128)',
             'Stabilises encoding before heads  ·  256 params'],
            fontsize=9.5)
arrow(ax, 0.5, y_bn, 0.5, y_bn - GAP*1.0)

# ── Split arrow to two heads ──────────────────────────────────────
y_split = y_bn - GAP * 1.5
ax.plot([0.5, 0.5], [y_bn, y_split],
        color=C_ARROW, linewidth=1.8, zorder=4)

head_ax = 0.28   # Head A centre x
head_bx = 0.72   # Head B centre x
HW = 0.20        # each head width

# Arrow to Head A
ax.annotate('', xy=(head_ax, y_split - 0.015),
            xytext=(0.5, y_split),
            arrowprops=dict(arrowstyle='->', color=C_HEAD_A, lw=1.8))
# Arrow to Head B
ax.annotate('', xy=(head_bx, y_split - 0.015),
            xytext=(0.5, y_split),
            arrowprops=dict(arrowstyle='->', color=C_HEAD_B, lw=1.8))

# ── Head A — Zone ─────────────────────────────────────────────────
y_ha = y_split - 0.11
rounded_box(ax, head_ax - HW/2, y_ha, HW, 0.09, C_HEAD_A,
            ['Head A — Zone',
             'Linear(128→64)',
             'ReLU + Dropout(0.3)',
             'Linear(64→3)',
             '8,451 params'],
            fontsize=8.5)

# ── Head B — Prob ─────────────────────────────────────────────────
rounded_box(ax, head_bx - HW/2, y_ha, HW, 0.09, C_HEAD_B,
            ['Head B — Prob',
             'Linear(128→32)',
             'ReLU + Dropout(0.3)',
             'Linear(32→1)',
             '4,161 params'],
            fontsize=8.5)

# ── Loss labels ───────────────────────────────────────────────────
y_loss = y_ha - 0.055
ax.text(head_ax, y_loss + 0.01,
        'CrossEntropyLoss\nweights=[0.54, 6.01, 1.03]',
        ha='center', va='top', fontsize=7.8,
        color=C_HEAD_A, fontstyle='italic',
        bbox=dict(facecolor='white', alpha=0.7,
                  edgecolor=C_HEAD_A, boxstyle='round,pad=0.2'))
ax.text(head_bx, y_loss + 0.01,
        'BCEWithLogitsLoss\npos_weight=2.09',
        ha='center', va='top', fontsize=7.8,
        color=C_HEAD_B, fontstyle='italic',
        bbox=dict(facecolor='white', alpha=0.7,
                  edgecolor=C_HEAD_B, boxstyle='round,pad=0.2'))

# ── Output boxes ─────────────────────────────────────────────────
y_out = y_loss - 0.10
OW = 0.22; OH = 0.075

arrow(ax, head_ax, y_ha, head_ax, y_out + OH + 0.005)
arrow(ax, head_bx, y_ha, head_bx, y_out + OH + 0.005)

rounded_box(ax, head_ax - OW/2, y_out, OW, OH, C_OUT_A,
            ['Output A',
             'Shape: (batch, 3)',
             'normal / warning / congested',
             'argmax → zone_pred'],
            fontsize=8, bold_first=True)

rounded_box(ax, head_bx - OW/2, y_out, OW, OH, C_OUT_B,
            ['Output B',
             'Shape: (batch, 1)',
             'sigmoid → P(congested)',
             'range: [0.0, 1.0]'],
            fontsize=8, bold_first=True)

# ── state_vector box at very bottom ──────────────────────────────
y_sv = y_out - 0.10
sv_y_line = y_out
ax.plot([head_ax, head_ax], [y_out, y_sv + 0.05],
        color=C_DARK, linewidth=1.5, linestyle='--')
ax.plot([head_bx, head_bx], [y_out, y_sv + 0.05],
        color=C_DARK, linewidth=1.5, linestyle='--')
ax.plot([head_ax, head_bx], [y_sv + 0.05, y_sv + 0.05],
        color=C_DARK, linewidth=1.5, linestyle='--')
ax.annotate('', xy=(0.5, y_sv + 0.012),
            xytext=(0.5, y_sv + 0.05),
            arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.8))

rounded_box(ax, BX, y_sv - 0.045, BW, 0.07, C_DARK,
            ['state_vector()  →  numpy float32  (9,)',
             '[P(normal), P(warning), P(congested), cong_prob, is_congested,',
             ' utilization_pct, bw_headroom_mbps, delta_drops, latency_ms]'],
            fontsize=8.5, bold_first=True)

ax.text(0.5, y_sv - 0.055,
        '↓  Module III DQN input',
        ha='center', va='top', fontsize=9,
        color=C_DARK, fontweight='bold')

# ── Parameter summary sidebar ─────────────────────────────────────
sx = 0.03
sy = 0.38
ax.text(sx, sy + 0.22, 'Parameters', ha='left', va='top',
        fontsize=9, fontweight='bold', color=C_DARK)
rows = [
    ('LSTM Layer 1', '74,752',  C_LSTM1),
    ('LSTM Layer 2', '131,584', C_LSTM2),
    ('BatchNorm1d',  '256',     C_BN),
    ('Head A',       '8,451',   C_HEAD_A),
    ('Head B',       '4,161',   C_HEAD_B),
    ('─────────', '─────────', C_GREY),
    ('TOTAL',    '220,228',  C_DARK),
]
for i, (name, val, col) in enumerate(rows):
    y_r = sy + 0.17 - i * 0.028
    ax.text(sx, y_r, name, ha='left', va='center',
            fontsize=8, color=col,
            fontweight='bold' if name in ('TOTAL','─────────') else 'normal')
    ax.text(sx + 0.115, y_r, val, ha='right', va='center',
            fontsize=8, color=col,
            fontweight='bold' if name == 'TOTAL' else 'normal')

# ── Training config sidebar ───────────────────────────────────────
tx = 0.03
ty = 0.175
ax.text(tx, ty, 'Training Config', ha='left', va='top',
        fontsize=9, fontweight='bold', color=C_DARK)
cfg = [
    ('Optimizer',   'AdamW  lr=1e-3'),
    ('Scheduler',   'CosineAnnealingLR'),
    ('Epochs',      '50 (early stop p=10)'),
    ('Batch size',  '256'),
    ('Loss ratio',  '0.6×zone + 0.4×prob'),
    ('Grad clip',   '1.0 (global norm)'),
    ('Device',      'CPU'),
]
for i, (k, v) in enumerate(cfg):
    y_c = ty - 0.032 - i * 0.026
    ax.text(tx, y_c, f'{k}:', ha='left', va='center',
            fontsize=7.5, color=C_GREY)
    ax.text(tx + 0.075, y_c, v, ha='left', va='center',
            fontsize=7.5, color=C_DARK)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_PATH, dpi=160, bbox_inches='tight',
            facecolor=C_BG)
plt.close()
print(f'Saved: {OUT_PATH}')

if __name__ == '__main__':
    pass
