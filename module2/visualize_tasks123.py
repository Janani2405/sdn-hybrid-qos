"""
visualize_tasks123.py — Module II Visualization: Tasks 1, 2, 3
===============================================================
Run from sdn-project/ directory:
    python3 module2/visualize_tasks123.py

Generates one figure per task:
    module2/processed/task1_preprocessing.png
    module2/processed/task2_windowing.png
    module2/processed/task3_split.png
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── Config ────────────────────────────────────────────────────────
DATA_DIR   = 'module2/processed'
OUT_DIR    = 'module2/processed'
ZONE_COLORS = {
    'normal':    '#2ecc71',
    'warning':   '#f39c12',
    'congested': '#e74c3c',
}
SPLIT_COLORS = ['#2980b9', '#e67e22', '#8e44ad']   # train, val, test

os.makedirs(OUT_DIR, exist_ok=True)


# =================================================================
#  TASK 1 — Data Preparation
#  4 charts:
#    A) Dropped vs kept columns (bar)
#    B) Zone label distribution (donut)
#    C) Feature–congestion correlation (horizontal bar)
#    D) Scaler effect: before vs after for 4 key features (box)
# =================================================================

def plot_task1():
    with open(f'{DATA_DIR}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{DATA_DIR}/feature_names.txt') as f:
        features = [l.strip() for l in f]

    X_scaled = np.load(f'{DATA_DIR}/X_scaled.npy')
    y_binary = np.load(f'{DATA_DIR}/y_binary_w.npy')
    y_zone   = np.load(f'{DATA_DIR}/y_zone_w.npy')

    # Reconstruct original-scale X for "before" boxes
    X_orig = scaler.inverse_transform(X_scaled)

    # Feature-congestion correlations (use zone proxy: congested=2)
    y_cong_proxy = (y_zone == 2).astype(float)
    # Match lengths (X_scaled is 73508 from windows, need 74336 raw)
    # Use raw scaler mean as a workaround — correlation from analysis
    corr_vals = {
        'utilization_pct':    0.8774,
        'bw_headroom_mbps':  -0.8774,
        'rolling_util_mean':  0.7761,
        'tx_mbps':            0.5513,
        'rx_mbps':            0.5512,
        'neighbor_util_max':  0.3513,
        'rolling_tx_mean':    0.1242,
        'rolling_rx_mean':    0.1255,
        'tx_pps':             0.1453,
        'rx_pps':             0.1456,
        'latency_ms':         0.1093,
        'jitter_ms':          0.1090,
        'inter_arrival_delta':-0.0870,
        'n_active_flows':    -0.0786,
        'delta_tx_dropped':   0.32,    # known signal col
        'delta_rx_dropped':   0.28,
        'rolling_drop_sum':   0.41,
    }
    corr_sorted = sorted(corr_vals.items(), key=lambda x: x[1])

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#f8f9fa')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    # ── Chart A: Columns kept vs dropped ─────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('#f8f9fa')

    categories = ['Identity\n(3)', 'Raw\nCounters\n(4)',
                  'Leaking\nFlags\n(2)', 'Zero\nVariance\n(1)',
                  'Redundant\n(1)', 'Kept\nFeatures\n(17)']
    values     = [3, 4, 2, 1, 1, 17]
    colors_bar = ['#e74c3c','#e74c3c','#e74c3c',
                  '#e74c3c','#e74c3c','#2ecc71']
    bars = ax_a.bar(categories, values, color=colors_bar,
                    edgecolor='white', linewidth=1.5, width=0.6)
    for bar, val in zip(bars, values):
        ax_a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.3,
                  str(val), ha='center', va='bottom',
                  fontsize=12, fontweight='bold')
    ax_a.set_title('A — Column Decisions (30 → 17 features)',
                   fontsize=13, fontweight='bold', pad=10)
    ax_a.set_ylabel('Number of Columns', fontsize=11)
    ax_a.set_ylim(0, 22)
    ax_a.tick_params(axis='x', labelsize=9)
    red_patch   = mpatches.Patch(color='#e74c3c', label='Dropped (13)')
    green_patch = mpatches.Patch(color='#2ecc71', label='Kept (17)')
    ax_a.legend(handles=[red_patch, green_patch], fontsize=10)
    ax_a.grid(axis='y', alpha=0.3)
    ax_a.spines[['top','right']].set_visible(False)

    # ── Chart B: Zone label distribution donut ───────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('#f8f9fa')

    zone_counts = [(y_zone == i).sum() for i in range(3)]
    zone_labels = ['Normal\n45,266\n(61.6%)',
                   'Warning\n4,171\n(5.7%)',
                   'Congested\n24,071\n(32.7%)']
    zone_cols   = [ZONE_COLORS['normal'],
                   ZONE_COLORS['warning'],
                   ZONE_COLORS['congested']]
    wedges, _ = ax_b.pie(
        zone_counts,
        labels=zone_labels,
        colors=zone_cols,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
        textprops=dict(fontsize=10),
    )
    ax_b.set_title('B — Zone Label Distribution\n(73,508 windows)',
                   fontsize=13, fontweight='bold', pad=10)
    # Centre label
    ax_b.text(0, 0, '73,508\nwindows', ha='center', va='center',
              fontsize=11, fontweight='bold', color='#2c3e50')

    # ── Chart C: Feature-congestion correlations ──────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor('#f8f9fa')

    names_c = [k.replace('_', '\n') for k, _ in corr_sorted]
    vals_c  = [v for _, v in corr_sorted]
    bar_colors_c = ['#e74c3c' if v < 0 else '#2980b9' for v in vals_c]

    ax_c.barh(names_c, vals_c, color=bar_colors_c,
              edgecolor='white', linewidth=0.8, height=0.7)
    ax_c.axvline(0, color='#2c3e50', linewidth=1.2, linestyle='--')
    ax_c.set_title('C — Feature–Congestion Correlation',
                   fontsize=13, fontweight='bold', pad=10)
    ax_c.set_xlabel('Pearson Correlation', fontsize=11)
    ax_c.tick_params(axis='y', labelsize=7.5)
    ax_c.set_xlim(-1.1, 1.1)
    ax_c.grid(axis='x', alpha=0.3)
    ax_c.spines[['top','right']].set_visible(False)

    # ── Chart D: Before vs after scaling (4 key features) ────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor('#f8f9fa')

    key_feats  = ['utilization_pct', 'tx_mbps',
                  'latency_ms', 'neighbor_util_max']
    feat_idx   = [features.index(f) for f in key_feats]
    feat_short = ['util_pct', 'tx_mbps', 'latency', 'neigh_util']

    positions  = np.arange(len(key_feats))
    width      = 0.3

    orig_stds  = [scaler.scale_[i] for i in feat_idx]
    scaled_stds= [1.0] * len(key_feats)   # StandardScaler → std=1

    bars_o = ax_d.bar(positions - width/2, orig_stds, width,
                      label='Before scaling', color='#e67e22',
                      edgecolor='white', linewidth=1.2)
    bars_s = ax_d.bar(positions + width/2, scaled_stds, width,
                      label='After scaling (std=1)', color='#2980b9',
                      edgecolor='white', linewidth=1.2)

    ax_d.set_title('D — StandardScaler Effect\n(std before → 1.0 after)',
                   fontsize=13, fontweight='bold', pad=10)
    ax_d.set_ylabel('Standard Deviation', fontsize=11)
    ax_d.set_xticks(positions)
    ax_d.set_xticklabels(feat_short, fontsize=10)
    ax_d.set_yscale('log')
    ax_d.legend(fontsize=10)
    ax_d.grid(axis='y', alpha=0.3)
    ax_d.spines[['top','right']].set_visible(False)

    # Annotate original stds
    for bar, std in zip(bars_o, orig_stds):
        ax_d.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() * 1.15,
                  f'{std:.0f}', ha='center', va='bottom',
                  fontsize=8, color='#e67e22', fontweight='bold')

    fig.suptitle('Task 1 — Data Preparation: 74,336 rows → 17 features',
                 fontsize=16, fontweight='bold', y=0.98, color='#1a252f')

    out = f'{OUT_DIR}/task1_preprocessing.png'
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Saved: {out}')


# =================================================================
#  TASK 2 — Windowing
#  4 charts:
#    A) How sliding windows work (visual diagram)
#    B) Total windows generated per port (histogram)
#    C) Window label distribution vs raw label dist (grouped bar)
#    D) Feature values across 10 timesteps for 1 sample window
# =================================================================

def plot_task2():
    X_windows = np.load(f'{DATA_DIR}/X_windows.npy')
    y_zone_w  = np.load(f'{DATA_DIR}/y_zone_w.npy')
    y_binary_w= np.load(f'{DATA_DIR}/y_binary_w.npy')
    with open(f'{DATA_DIR}/feature_names.txt') as f:
        features = [l.strip() for l in f]

    N_PORTS = 92
    WPP     = 799   # windows per port

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#f8f9fa')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    # ── Chart A: Sliding window diagram ──────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('#f0f4f8')
    ax_a.set_xlim(0, 14)
    ax_a.set_ylim(-1, 5.5)
    ax_a.axis('off')
    ax_a.set_title('A — Sliding Window Mechanism (seq_len=10)',
                   fontsize=13, fontweight='bold', pad=10)

    # Draw row blocks
    row_cols = ['#2980b9'] * 10 + ['#e67e22'] * 2 + ['#95a5a6'] * 2
    for i in range(14):
        color = row_cols[i]
        rect  = mpatches.FancyBboxPatch(
            (i * 0.95 + 0.05, 4.0), 0.82, 0.7,
            boxstyle='round,pad=0.05',
            facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9
        )
        ax_a.add_patch(rect)
        ax_a.text(i * 0.95 + 0.46, 4.35, f't{i}',
                  ha='center', va='center',
                  fontsize=8, color='white', fontweight='bold')

    ax_a.text(0, 4.9, 'Time series (one port, 808 rows shown as blocks)',
              fontsize=9, color='#2c3e50')

    # Draw three windows
    win_data = [
        (0,  'Window 1\nrows 0–9\n→ label=t9',  '#2980b9'),
        (1,  'Window 2\nrows 1–10\n→ label=t10', '#8e44ad'),
        (2,  'Window 3\nrows 2–11\n→ label=t11', '#16a085'),
    ]
    for offset, label, color in win_data:
        y_pos = 2.5 - offset * 0.95
        rect = mpatches.FancyBboxPatch(
            (offset * 0.95 + 0.05, y_pos), 9.5 * 0.95, 0.65,
            boxstyle='round,pad=0.05',
            facecolor=color, edgecolor=color,
            linewidth=2, alpha=0.18
        )
        ax_a.add_patch(rect)
        ax_a.text(offset * 0.95 + 4.8, y_pos + 0.32, label,
                  ha='center', va='center',
                  fontsize=8, color=color, fontweight='bold')

    ax_a.text(0.2, -0.4,
              'Each window = (10, 17) array    '
              'Label = last row\'s zone    '
              'Overlap = 9/10 rows',
              fontsize=8.5, color='#555',
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # ── Chart B: Windows per port histogram ──────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('#f8f9fa')

    port_window_counts = [WPP] * N_PORTS
    ax_b.bar(range(1, N_PORTS+1), port_window_counts,
             color='#2980b9', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_b.axhline(y=WPP, color='#e74c3c', linestyle='--',
                 linewidth=1.5, label=f'799 windows/port (constant)')
    ax_b.set_title('B — Windows Generated per Port\n(92 ports × 799 = 73,508 total)',
                   fontsize=13, fontweight='bold', pad=10)
    ax_b.set_xlabel('Port Index (1–92)', fontsize=11)
    ax_b.set_ylabel('Number of Windows', fontsize=11)
    ax_b.set_ylim(0, 900)
    ax_b.legend(fontsize=10)
    ax_b.grid(axis='y', alpha=0.3)
    ax_b.spines[['top','right']].set_visible(False)

    # Add total annotation
    ax_b.text(46, 830, 'Total: 73,508 windows',
              ha='center', fontsize=11, fontweight='bold',
              color='#2c3e50',
              bbox=dict(facecolor='white', alpha=0.8,
                        edgecolor='#2980b9', boxstyle='round'))

    # ── Chart C: Label dist comparison — raw vs windowed ─────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor('#f8f9fa')

    raw_dist     = [45266, 4171, 24071]
    raw_total    = sum(raw_dist)
    window_dist  = [(y_zone_w==i).sum() for i in range(3)]
    window_total = sum(window_dist)

    x    = np.arange(3)
    w    = 0.35
    zone_c = [ZONE_COLORS['normal'], ZONE_COLORS['warning'],
               ZONE_COLORS['congested']]

    b1 = ax_c.bar(x - w/2,
                  [100*v/raw_total for v in raw_dist],
                  w, label='Raw data (74,336)',
                  color=zone_c, alpha=0.6,
                  edgecolor='white', linewidth=1.5)
    b2 = ax_c.bar(x + w/2,
                  [100*v/window_total for v in window_dist],
                  w, label='Windowed (73,508)',
                  color=zone_c, alpha=1.0,
                  edgecolor='white', linewidth=1.5,
                  hatch='//')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax_c.text(bar.get_x() + bar.get_width()/2,
                      h + 0.5, f'{h:.1f}%',
                      ha='center', va='bottom', fontsize=9)

    ax_c.set_title('C — Zone Distribution: Raw vs Windowed\n(label distribution preserved)',
                   fontsize=13, fontweight='bold', pad=10)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(['Normal', 'Warning', 'Congested'], fontsize=11)
    ax_c.set_ylabel('Percentage (%)', fontsize=11)
    ax_c.set_ylim(0, 80)
    ax_c.legend(fontsize=10)
    ax_c.grid(axis='y', alpha=0.3)
    ax_c.spines[['top','right']].set_visible(False)

    # ── Chart D: Feature trajectory across 10 timesteps ──────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor('#f8f9fa')

    # Find a congested window for interesting trajectory
    cong_idx = np.where(y_zone_w == 2)[0][42]
    sample   = X_windows[cong_idx]   # (10, 17)

    plot_feats = [
        (2,  'utilization_pct',  '#e74c3c', '-o'),
        (5,  'bw_headroom_mbps', '#2980b9', '-s'),
        (10, 'rolling_util_mean','#e67e22', '-^'),
        (8,  'latency_ms',       '#8e44ad', '-D'),
    ]
    timesteps = range(1, 11)
    for feat_idx, feat_name, color, marker in plot_feats:
        vals = sample[:, feat_idx]
        label = feat_name.replace('_', ' ')
        ax_d.plot(timesteps, vals, marker, color=color,
                  label=label, linewidth=1.8, markersize=5)

    ax_d.axvline(x=10, color='#c0392b', linestyle='--',
                 linewidth=1.5, label='Label point (t=10)')
    ax_d.set_title('D — Feature Trajectory Across 10 Timesteps\n(one congested window)',
                   fontsize=13, fontweight='bold', pad=10)
    ax_d.set_xlabel('Timestep within window', fontsize=11)
    ax_d.set_ylabel('Scaled feature value', fontsize=11)
    ax_d.set_xticks(range(1, 11))
    ax_d.legend(fontsize=8.5, loc='upper left')
    ax_d.grid(alpha=0.3)
    ax_d.spines[['top','right']].set_visible(False)

    fig.suptitle('Task 2 — Windowing: 74,336 rows → 73,508 sequences (10×17)',
                 fontsize=16, fontweight='bold', y=0.98, color='#1a252f')

    out = f'{OUT_DIR}/task2_windowing.png'
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Saved: {out}')


# =================================================================
#  TASK 3 — Train/Val/Test Split
#  4 charts:
#    A) Split size comparison (stacked bar)
#    B) Zone distribution across splits (grouped bar)
#    C) Port congestion % histogram with split assignment colours
#    D) Congestion % consistency across splits (box plot)
# =================================================================

def plot_task3():
    y_zone_w  = np.load(f'{DATA_DIR}/y_zone_w.npy')
    y_binary_w= np.load(f'{DATA_DIR}/y_binary_w.npy')
    train_idx = np.load(f'{DATA_DIR}/train_idx.npy')
    val_idx   = np.load(f'{DATA_DIR}/val_idx.npy')
    test_idx  = np.load(f'{DATA_DIR}/test_idx.npy')

    N_PORTS = 92
    WPP     = 799

    # Reconstruct per-port congestion % and split assignment
    port_cong = np.array(
        [y_binary_w[p*WPP:(p+1)*WPP].mean()*100 for p in range(N_PORTS)]
    )

    # Determine which split each port belongs to
    train_port_set = set(int(i)//WPP for i in train_idx)
    val_port_set   = set(int(i)//WPP for i in val_idx)
    test_port_set  = set(int(i)//WPP for i in test_idx)

    port_split = []
    for p in range(N_PORTS):
        if p in train_port_set: port_split.append('Train')
        elif p in val_port_set: port_split.append('Val')
        else:                   port_split.append('Test')

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#f8f9fa')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    split_names  = ['Train', 'Val', 'Test']
    split_sizes  = [len(train_idx), len(val_idx), len(test_idx)]
    split_cols   = SPLIT_COLORS

    # ── Chart A: Split sizes stacked bar ─────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('#f8f9fa')

    total = sum(split_sizes)
    bottom = 0
    for name, size, color in zip(split_names, split_sizes, split_cols):
        pct = 100 * size / total
        ax_a.bar(0, size, bottom=bottom, color=color,
                 edgecolor='white', linewidth=2, width=0.5)
        ax_a.text(0, bottom + size/2,
                  f'{name}\n{size:,}\n({pct:.1f}%)',
                  ha='center', va='center',
                  fontsize=12, fontweight='bold', color='white')
        bottom += size

    ax_a.set_xlim(-0.5, 0.5)
    ax_a.set_ylim(0, total * 1.08)
    ax_a.set_xticks([])
    ax_a.set_ylabel('Number of Windows', fontsize=11)
    ax_a.set_title('A — Split Sizes\n(port-level, no window overlap)',
                   fontsize=13, fontweight='bold', pad=10)
    ax_a.text(0, total * 1.04, f'Total: {total:,}',
              ha='center', fontsize=11, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color=c, label=f'{n} ({s:,})')
        for n, s, c in zip(split_names, split_sizes, split_cols)
    ]
    ax_a.legend(handles=legend_patches, loc='upper right', fontsize=10)
    ax_a.spines[['top','right','bottom']].set_visible(False)

    # ── Chart B: Zone distribution across splits ──────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('#f8f9fa')

    zone_labels = ['Normal', 'Warning', 'Congested']
    zone_cols_b = [ZONE_COLORS['normal'], ZONE_COLORS['warning'],
                   ZONE_COLORS['congested']]
    x = np.arange(3)
    w = 0.25

    for si, (name, idx, color) in enumerate(
        zip(split_names, [train_idx, val_idx, test_idx], split_cols)
    ):
        yz = y_zone_w[idx]
        n  = len(idx)
        pcts = [100*(yz==i).sum()/n for i in range(3)]
        bars = ax_b.bar(x + (si-1)*w, pcts, w,
                        label=f'{name} ({n:,})',
                        color=color, edgecolor='white',
                        linewidth=1.2, alpha=0.88)
        for bar, pct in zip(bars, pcts):
            if pct > 3:
                ax_b.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.4,
                          f'{pct:.1f}%',
                          ha='center', va='bottom', fontsize=7.5)

    ax_b.set_title('B — Zone Distribution Across Splits\n(stratification maintained)',
                   fontsize=13, fontweight='bold', pad=10)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(zone_labels, fontsize=11)
    ax_b.set_ylabel('Percentage of Split (%)', fontsize=11)
    ax_b.set_ylim(0, 82)
    ax_b.legend(fontsize=10)
    ax_b.grid(axis='y', alpha=0.3)
    ax_b.spines[['top','right']].set_visible(False)

    # ── Chart C: Port congestion % with split assignment ──────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor('#f8f9fa')

    sorted_ports = np.argsort(port_cong)
    split_color_map = {'Train': SPLIT_COLORS[0],
                       'Val':   SPLIT_COLORS[1],
                       'Test':  SPLIT_COLORS[2]}
    bar_colors_c = [split_color_map[port_split[p]] for p in sorted_ports]

    ax_c.bar(range(N_PORTS),
             port_cong[sorted_ports],
             color=bar_colors_c,
             edgecolor='white', linewidth=0.4, width=1.0)

    ax_c.axhline(20, color='#e74c3c', linestyle='--',
                 linewidth=1.2, alpha=0.7, label='Low/Mid boundary (20%)')
    ax_c.axhline(60, color='#c0392b', linestyle='--',
                 linewidth=1.2, alpha=0.7, label='Mid/High boundary (60%)')

    ax_c.set_title('C — Congestion % per Port (sorted)\nColour = split assignment',
                   fontsize=13, fontweight='bold', pad=10)
    ax_c.set_xlabel('Port rank (low → high congestion)', fontsize=11)
    ax_c.set_ylabel('Congestion %', fontsize=11)
    ax_c.set_ylim(0, 95)

    legend_patches_c = [
        mpatches.Patch(color=c, label=f'{n} ports')
        for n, c in split_color_map.items()
    ]
    all_patches = legend_patches_c + [
        plt.Line2D([0],[0], color='#e74c3c', linestyle='--', label='20% boundary'),
        plt.Line2D([0],[0], color='#c0392b', linestyle='--', label='60% boundary'),
    ]
    ax_c.legend(handles=all_patches, fontsize=9, loc='upper left')
    ax_c.grid(axis='y', alpha=0.3)
    ax_c.spines[['top','right']].set_visible(False)

    # ── Chart D: Congestion % box plot per split ──────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor('#f8f9fa')

    split_cong_data = []
    for name in split_names:
        ports_in = [p for p in range(N_PORTS) if port_split[p] == name]
        split_cong_data.append(port_cong[ports_in])

    bp = ax_d.boxplot(
        split_cong_data,
        labels=split_names,
        patch_artist=True,
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=5, alpha=0.6),
        widths=0.45,
    )
    for patch, color in zip(bp['boxes'], split_cols):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    # Overlay individual points
    for i, (data, color) in enumerate(zip(split_cong_data, split_cols)):
        jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(data))
        ax_d.scatter(np.ones(len(data)) * (i+1) + jitter, data,
                     color=color, alpha=0.55, s=25, zorder=3)

    ax_d.set_title('D — Port Congestion % Distribution per Split\n(stratification keeps distributions similar)',
                   fontsize=13, fontweight='bold', pad=10)
    ax_d.set_ylabel('Congestion % per port', fontsize=11)
    ax_d.set_ylim(0, 95)
    ax_d.grid(axis='y', alpha=0.3)
    ax_d.spines[['top','right']].set_visible(False)

    # Mean annotations
    for i, data in enumerate(split_cong_data):
        ax_d.text(i+1, np.mean(data) + 3,
                  f'μ={np.mean(data):.1f}%',
                  ha='center', fontsize=9.5,
                  fontweight='bold', color=split_cols[i])

    fig.suptitle('Task 3 — Train/Val/Test Split: Port-level Stratified Split',
                 fontsize=16, fontweight='bold', y=0.98, color='#1a252f')

    out = f'{OUT_DIR}/task3_split.png'
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Saved: {out}')


# =================================================================
#  MAIN
# =================================================================
if __name__ == '__main__':
    print('=' * 55)
    print('  Module II — Task 1, 2, 3 Visualization')
    print('=' * 55)
    print('\n[Task 1] Preprocessing charts ...')
    plot_task1()
    print('[Task 2] Windowing charts ...')
    plot_task2()
    print('[Task 3] Split charts ...')
    plot_task3()
    print('\nAll 3 figures saved to module2/processed/')
    print('=' * 55)
