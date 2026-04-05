"""
=============================================================================
plot_results.py  —  Module III Evaluation: DQN Agent Training Analysis
=============================================================================
Project : A Hybrid Learning Approach for Dynamic Bandwidth Allocation and
          Load Balancing in SDN Based Cloud Networks
Team    : Janani A (2201112023), Raja Hariharan K (2201112033),
          Selvaganapathi S (2201112039)
Guide   : Dr. Santhi G
=============================================================================

Reads  : logs/dqn_agent.log   (produced by module3/dqn_agent.py)
Saves  : logs/plots/           (6 PNG figures, one combined PDF)

Run from ~/sdn-project/:
    python3 module3/plot_results.py

Optional flags:
    --log   PATH   path to log file  (default: logs/dqn_agent.log)
    --out   DIR    output directory  (default: logs/plots)
    --dpi   INT    figure resolution (default: 150)
=============================================================================
"""

import re
import sys
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch

# ── Colour palette (matches project style) ────────────────────────
C_BLUE   = '#185FA5'
C_TEAL   = '#0F6E56'
C_AMBER  = '#BA7517'
C_RED    = '#A32D2D'
C_PURPLE = '#534AB7'
C_GRAY   = '#5F5E5A'
C_LIGHT  = '#F1EFE8'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
    'lines.linewidth':   1.8,
    'figure.dpi':        150,
    'savefig.bbox':      'tight',
    'savefig.dpi':       150,
})


# ═══════════════════════════════════════════════════════════════════
#  1.  LOG PARSER
# ═══════════════════════════════════════════════════════════════════
def parse_log(log_path: Path) -> dict:
    """
    Extract all numeric series from dqn_agent.log.

    Returns a dict with keys:
        steps, epsilon, avg_reward, loss, mem_size,
        bonus_steps, bonus_counts,          # per-step congestion clears
        dropout_steps,                       # steps with controller warning
        reroute_fail_steps                   # steps with reroute warning
    """
    step_pat   = re.compile(
        r'step=\s*(\d+).*?ε=([\d.]+).*?avg_reward\(100\)=([+\-][\d.]+)'
        r'.*?loss=([\d.]+).*?mem=(\d+)')
    bonus_pat  = re.compile(r'\[DQN\]\s+\[.bonus\] .+ congestion cleared\s+Δ=([\d.]+)')
    warn_pat   = re.compile(r'No state from controller')
    reroute_pat= re.compile(r'\[reroute\] no alternate port')
    time_pat   = re.compile(r'^(\d{2}:\d{2}:\d{2})')

    data = defaultdict(list)
    current_step = 0

    with open(log_path) as f:
        for line in f:
            m = step_pat.search(line)
            if m:
                current_step = int(m.group(1))
                data['steps'].append(current_step)
                data['epsilon'].append(float(m.group(2)))
                data['avg_reward'].append(float(m.group(3)))
                data['loss'].append(float(m.group(4)))
                data['mem_size'].append(int(m.group(5)))
                continue

            if bonus_pat.search(line):
                delta = float(bonus_pat.search(line).group(1))
                data['bonus_steps'].append(current_step)
                data['bonus_deltas'].append(delta)
                continue

            if warn_pat.search(line):
                data['dropout_steps'].append(current_step)
                continue

            if reroute_pat.search(line):
                data['reroute_fail_steps'].append(current_step)

    # Convert to numpy
    for k in ('steps','epsilon','avg_reward','loss','mem_size'):
        data[k] = np.array(data[k], dtype=float)

    return dict(data)


def smooth(values, window=5):
    """Simple moving average for trend lines."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='same')


def congestion_events_per_window(bonus_steps, total_steps, window=50):
    """
    Count how many congestion-clearing bonuses occurred in each
    rolling window of `window` steps. Used for before/after comparison.
    """
    if len(bonus_steps) == 0:
        return np.array([]), np.array([])
    bins = np.arange(0, total_steps + window, window)
    counts, edges = np.histogram(bonus_steps, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, counts


# ═══════════════════════════════════════════════════════════════════
#  2.  INDIVIDUAL FIGURES
# ═══════════════════════════════════════════════════════════════════

def fig_reward_curve(data, out_dir, dpi):
    """Figure 1 — Average reward over training steps."""
    steps   = data['steps']
    rewards = data['avg_reward']

    fig, ax = plt.subplots(figsize=(8, 4))

    # Raw values (faint)
    ax.plot(steps, rewards, color=C_BLUE, alpha=0.25, linewidth=1, label='Raw avg reward')

    # Smoothed trend
    if len(rewards) >= 5:
        ax.plot(steps, smooth(rewards, 5), color=C_BLUE, linewidth=2,
                label='Smoothed (window=5)')

    # Zero reference line
    ax.axhline(0, color=C_GRAY, linewidth=0.8, linestyle='--', alpha=0.6)

    # Phase annotations
    ax.axvspan(steps[0], steps[min(4, len(steps)-1)], alpha=0.08,
               color=C_AMBER, label='Exploration (ε>0.85)')
    if len(steps) > 25:
        ax.axvspan(steps[24], steps[-1], alpha=0.06,
                   color=C_TEAL, label='Learning phase')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Avg Reward (last 100 steps)')
    ax.set_title('Figure 1 — DQN Reward Curve (350 Training Steps)')
    ax.legend(fontsize=8, loc='lower right')
    ax.xaxis.set_major_locator(MaxNLocator(10, integer=True))

    # Annotate best and worst
    best_idx = int(np.argmax(rewards))
    worst_idx= int(np.argmin(rewards))
    ax.annotate(f'Peak {rewards[best_idx]:+.3f}',
                xy=(steps[best_idx], rewards[best_idx]),
                xytext=(steps[best_idx]+10, rewards[best_idx]+0.005),
                fontsize=8, color=C_TEAL,
                arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=1))
    ax.annotate(f'Low {rewards[worst_idx]:+.3f}',
                xy=(steps[worst_idx], rewards[worst_idx]),
                xytext=(steps[worst_idx]+10, rewards[worst_idx]-0.005),
                fontsize=8, color=C_RED,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1))

    fig.tight_layout()
    path = out_dir / 'fig1_reward_curve.png'
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


def fig_loss_curve(data, out_dir, dpi):
    """Figure 2 — Training loss (Huber) over steps."""
    steps = data['steps']
    loss  = data['loss']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, loss, color=C_RED, alpha=0.3, linewidth=1)
    if len(loss) >= 5:
        ax.plot(steps, smooth(loss, 5), color=C_RED, linewidth=2, label='Huber loss (smoothed)')
    ax.fill_between(steps, 0, loss, color=C_RED, alpha=0.07)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Huber Loss')
    ax.set_title('Figure 2 — Training Loss (Huber) over Steps')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(10, integer=True))

    # Annotate rising loss phase (expected as memory fills and epsilon decays)
    if len(steps) > 20:
        mid = len(steps) // 2
        ax.annotate('Loss rises as\nmemory diversifies',
                    xy=(steps[mid], loss[mid]),
                    xytext=(steps[mid]-40, loss[mid]+0.006),
                    fontsize=7.5, color=C_GRAY,
                    arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=0.8))

    fig.tight_layout()
    path = out_dir / 'fig2_loss_curve.png'
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


def fig_epsilon_decay(data, out_dir, dpi):
    """Figure 3 — Epsilon decay and memory growth."""
    steps   = data['steps']
    epsilon = data['epsilon']
    mem     = data['mem_size']

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(steps, epsilon, color=C_PURPLE, linewidth=2, label='Epsilon (ε)')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Epsilon (exploration rate)', color=C_PURPLE)
    ax1.tick_params(axis='y', labelcolor=C_PURPLE)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(steps, mem, color=C_TEAL, linewidth=1.5, linestyle='--', label='Memory size')
    ax2.set_ylabel('Replay Memory Size', color=C_TEAL)
    ax2.tick_params(axis='y', labelcolor=C_TEAL)
    ax2.spines['right'].set_visible(True)

    # Phase labels
    for thresh, label, col in [(0.85, 'Random', C_AMBER),
                                (0.5,  'Mixed',  C_BLUE),
                                (0.0,  'Policy', C_TEAL)]:
        idx = np.where(epsilon <= thresh)[0]
        if len(idx):
            ax1.axvline(steps[idx[0]], color=col, linewidth=0.8,
                        linestyle=':', alpha=0.7)
            ax1.text(steps[idx[0]]+2, 0.92, label, fontsize=7.5,
                     color=col, va='top')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper right')
    ax1.set_title('Figure 3 — Epsilon Decay & Replay Memory Growth')
    ax1.xaxis.set_major_locator(MaxNLocator(10, integer=True))

    fig.tight_layout()
    path = out_dir / 'fig3_epsilon_memory.png'
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


def fig_congestion_reduction(data, out_dir, dpi):
    """
    Figure 4 — Before vs After: congestion-clearing events per 50-step window.
    This is the core evaluation metric — shows the DQN's effect over time.
    """
    bonus_steps = np.array(data.get('bonus_steps', []))
    total_steps = int(data['steps'][-1]) if len(data['steps']) else 350

    centres, counts = congestion_events_per_window(bonus_steps, total_steps, window=50)
    if len(centres) == 0:
        print('  [skip] No congestion bonus data found — skipping Fig 4')
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    # Split into "before" (first half) and "after" (second half) phases
    mid   = total_steps // 2
    mask_before = centres <= mid
    mask_after  = centres >  mid

    ax.bar(centres[mask_before], counts[mask_before], width=45,
           color=C_RED,  alpha=0.75, label='Early training (high ε, random)')
    ax.bar(centres[mask_after],  counts[mask_after],  width=45,
           color=C_TEAL, alpha=0.75, label='Late training  (low ε, policy)')

    # Trend line
    if len(counts) >= 4:
        ax.plot(centres, smooth(counts, 3), color=C_GRAY,
                linewidth=1.5, linestyle='--', label='Trend')

    # Averages
    avg_before = counts[mask_before].mean() if mask_before.any() else 0
    avg_after  = counts[mask_after].mean()  if mask_after.any()  else 0
    ax.axhline(avg_before, color=C_RED,  linewidth=1, linestyle=':',
               alpha=0.6, label=f'Avg early = {avg_before:.1f}')
    ax.axhline(avg_after,  color=C_TEAL, linewidth=1, linestyle=':',
               alpha=0.6, label=f'Avg late  = {avg_after:.1f}')

    ax.axvline(mid, color=C_GRAY, linewidth=1, linestyle='--', alpha=0.5)
    ax.text(mid+2, ax.get_ylim()[1]*0.95, 'ε = 0.60\n(policy emerging)',
            fontsize=7.5, color=C_GRAY, va='top')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Congestion-Clearing Events (per 50-step window)')
    ax.set_title('Figure 4 — Congestion Reduction: Early vs Late Training')
    ax.legend(fontsize=8, loc='upper right')
    ax.xaxis.set_major_locator(MaxNLocator(10, integer=True))

    fig.tight_layout()
    path = out_dir / 'fig4_congestion_reduction.png'
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


def fig_training_summary(data, out_dir, dpi):
    """
    Figure 5 — 4-panel summary dashboard (the one to show in viva).
    Reward | Loss | Epsilon | Congestion clears per step window
    """
    steps   = data['steps']
    rewards = data['avg_reward']
    loss    = data['loss']
    epsilon = data['epsilon']
    bonus_steps = np.array(data.get('bonus_steps', []))

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        'Module III — DQN Agent Training Summary\n'
        'SDN-Based Cloud Network Load Balancing  |  350 Steps  |  31 Switches',
        fontsize=13, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel A: Reward ───────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(steps, rewards, color=C_BLUE, alpha=0.2, linewidth=1)
    ax0.plot(steps, smooth(rewards, 5), color=C_BLUE, linewidth=2)
    ax0.axhline(0, color=C_GRAY, linewidth=0.7, linestyle='--', alpha=0.5)
    ax0.fill_between(steps, 0, rewards,
                     where=np.array(rewards) >= 0, color=C_TEAL, alpha=0.15)
    ax0.fill_between(steps, 0, rewards,
                     where=np.array(rewards) < 0,  color=C_RED,  alpha=0.12)
    ax0.set_title('(A) Avg Reward (100-step window)')
    ax0.set_xlabel('Step')
    ax0.set_ylabel('Reward')
    ax0.xaxis.set_major_locator(MaxNLocator(6, integer=True))

    # ── Panel B: Loss ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(steps, loss, color=C_RED, alpha=0.25, linewidth=1)
    ax1.plot(steps, smooth(loss, 5), color=C_RED, linewidth=2)
    ax1.fill_between(steps, 0, loss, color=C_RED, alpha=0.07)
    ax1.set_title('(B) Huber Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.xaxis.set_major_locator(MaxNLocator(6, integer=True))

    # ── Panel C: Epsilon ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(steps, epsilon, color=C_PURPLE, linewidth=2)
    ax2.fill_between(steps, 0, epsilon, color=C_PURPLE, alpha=0.1)
    ax2.axhline(0.5, color=C_GRAY, linewidth=0.7, linestyle=':', alpha=0.6)
    ax2.text(steps[-1]*0.02, 0.52, 'ε=0.5 (mixed)', fontsize=7.5, color=C_GRAY)
    ax2.set_title('(C) Epsilon Decay (1.0 → 0.42)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Epsilon')
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_locator(MaxNLocator(6, integer=True))

    # ── Panel D: Congestion clears per window ────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if len(bonus_steps) > 0:
        total = int(steps[-1])
        centres, counts = congestion_events_per_window(bonus_steps, total, 50)
        mid = total // 2
        ax3.bar(centres[centres <= mid], counts[centres <= mid],
                width=44, color=C_RED, alpha=0.7, label='Early')
        ax3.bar(centres[centres >  mid], counts[centres >  mid],
                width=44, color=C_TEAL, alpha=0.7, label='Late')
        if len(counts) >= 3:
            ax3.plot(centres, smooth(counts, 3), color=C_GRAY,
                     linestyle='--', linewidth=1.2)
        ax3.legend(fontsize=8)
    ax3.set_title('(D) Congestion Clears / 50-step Window')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Count')
    ax3.xaxis.set_major_locator(MaxNLocator(6, integer=True))

    path = out_dir / 'fig5_training_summary.png'
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


def fig_performance_comparison(data, out_dir, dpi):
    """
    Figure 6 — Before/After bar chart: key metrics in early vs late training.
    This is the cleanest figure for a project report or presentation.
    """
    steps   = data['steps']
    rewards = data['avg_reward']
    loss    = data['loss']
    bonus_steps = np.array(data.get('bonus_steps', []))

    n = len(steps)
    half = n // 2

    # Compute metrics for first half vs second half
    r_early = rewards[:half]
    r_late  = rewards[half:]
    l_early = loss[:half]
    l_late  = loss[half:]

    total = int(steps[-1])
    mid   = total // 2
    _, c_early = congestion_events_per_window(
        bonus_steps[bonus_steps <= mid], mid, 50)
    _, c_late  = congestion_events_per_window(
        bonus_steps[bonus_steps >  mid], mid, 50)

    avg_r_early = float(np.mean(r_early))
    avg_r_late  = float(np.mean(r_late))
    avg_l_early = float(np.mean(l_early))
    avg_l_late  = float(np.mean(l_late))
    avg_c_early = float(np.mean(c_early)) if len(c_early) else 0
    avg_c_late  = float(np.mean(c_late))  if len(c_late)  else 0

    metrics = ['Avg Reward', 'Avg Loss (×100)', 'Congestion\nClears/Window']
    early_v = [avg_r_early, avg_l_early * 100, avg_c_early]
    late_v  = [avg_r_late,  avg_l_late  * 100, avg_c_late]

    x    = np.arange(len(metrics))
    w    = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - w/2, early_v, w, label='Early training (steps 10–175)',
                   color=C_RED,  alpha=0.80)
    bars2 = ax.bar(x + w/2, late_v,  w, label='Late training  (steps 180–350)',
                   color=C_TEAL, alpha=0.80)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    # Improvement annotations
    for i, (e, l) in enumerate(zip(early_v, late_v)):
        if abs(e) > 1e-6:
            pct = (l - e) / abs(e) * 100
            sign = '+' if pct > 0 else ''
            col  = C_TEAL if (i == 0 and pct > 0) or (i != 0 and pct < 0) else C_RED
            ax.text(x[i], max(e, l) + 0.015,
                    f'{sign}{pct:.0f}%', ha='center', fontsize=8.5,
                    fontweight='bold', color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title('Figure 6 — Early vs Late Training: Key Metrics Comparison\n'
                 '(Higher reward = better   |   Lower loss = better   |   More clears = better)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.axhline(0, color=C_GRAY, linewidth=0.7, alpha=0.5)

    fig.tight_layout()
    path = out_dir / 'fig6_before_after_comparison.png'
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


# ═══════════════════════════════════════════════════════════════════
#  3.  COMBINED PDF  (all 6 figures in one file)
# ═══════════════════════════════════════════════════════════════════
def save_combined_pdf(figure_paths, out_dir):
    """Save all figures into one multi-page PDF for the project report."""
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = out_dir / 'module3_evaluation_report.pdf'
        with PdfPages(pdf_path) as pdf:
            for img_path in figure_paths:
                if img_path is None:
                    continue
                fig, ax = plt.subplots(figsize=(10, 6))
                img = plt.imread(str(img_path))
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        print(f'  Saved: {pdf_path}')
    except Exception as e:
        print(f'  [PDF skipped] {e}')


# ═══════════════════════════════════════════════════════════════════
#  4.  STATS SUMMARY  (printed to terminal)
# ═══════════════════════════════════════════════════════════════════
def print_summary(data):
    steps   = data['steps']
    rewards = data['avg_reward']
    loss    = data['loss']
    epsilon = data['epsilon']
    bonus   = data.get('bonus_steps', [])
    dropout = data.get('dropout_steps', [])

    n     = len(steps)
    half  = n // 2

    print()
    print('=' * 56)
    print('  MODULE III — DQN TRAINING SUMMARY')
    print('=' * 56)
    print(f'  Total steps logged       : {int(steps[-1])}')
    print(f'  Epsilon range            : {epsilon[0]:.3f} → {epsilon[-1]:.3f}')
    print(f'  Replay memory final size : {int(data["mem_size"][-1]):,}')
    print(f'  Controller dropouts      : {len(dropout)}')
    print()
    print('  Reward (avg of 100-step window):')
    print(f'    Early training  (steps {int(steps[0])}–{int(steps[half])}) : '
          f'{np.mean(rewards[:half]):+.4f}')
    print(f'    Late  training  (steps {int(steps[half])}–{int(steps[-1])}) : '
          f'{np.mean(rewards[half:]):+.4f}')
    print(f'    Best step  : step {int(steps[np.argmax(rewards)])}  '
          f'reward={np.max(rewards):+.4f}')
    print(f'    Worst step : step {int(steps[np.argmin(rewards)])}  '
          f'reward={np.min(rewards):+.4f}')
    print()
    print('  Loss (Huber):')
    print(f'    Early avg : {np.mean(loss[:half]):.5f}')
    print(f'    Late  avg : {np.mean(loss[half:]):.5f}')
    print(f'    Final     : {loss[-1]:.5f}')
    print()
    print(f'  Congestion-clearing bonuses triggered: {len(bonus)}')
    if len(bonus):
        deltas = data.get('bonus_deltas', [])
        if deltas:
            print(f'    Avg Δ per event : {np.mean(deltas):.3f}')
            print(f'    Max Δ per event : {np.max(deltas):.2f}')
    print('=' * 56)
    print()


# ═══════════════════════════════════════════════════════════════════
#  5.  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Module III — DQN Evaluation Plots')
    parser.add_argument('--log', default='logs/dqn_agent.log',
                        help='Path to dqn_agent.log')
    parser.add_argument('--out', default='logs/plots',
                        help='Output directory for PNG/PDF')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Figure DPI (default: 150)')
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir  = Path(args.out)

    if not log_path.exists():
        print(f'ERROR: Log file not found: {log_path}')
        print('Make sure you run from ~/sdn-project/ and the log exists.')
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nParsing log : {log_path}')
    data = parse_log(log_path)

    if len(data.get('steps', [])) == 0:
        print('ERROR: No training step data found in log file.')
        sys.exit(1)

    print(f'  Found {len(data["steps"])} step records, '
          f'{len(data.get("bonus_steps", []))} congestion bonus events.')

    print_summary(data)

    print(f'Generating figures → {out_dir}/')
    paths = [
        fig_reward_curve(data, out_dir, args.dpi),
        fig_loss_curve(data, out_dir, args.dpi),
        fig_epsilon_decay(data, out_dir, args.dpi),
        fig_congestion_reduction(data, out_dir, args.dpi),
        fig_training_summary(data, out_dir, args.dpi),
        fig_performance_comparison(data, out_dir, args.dpi),
    ]

    print('\nSaving combined PDF...')
    save_combined_pdf(paths, out_dir)

    print('\nDone! All outputs in:', out_dir.resolve())
    print()


if __name__ == '__main__':
    main()
