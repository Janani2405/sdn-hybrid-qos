"""
window.py — Module II Task 2: Windowing
=========================================
Input  : module2/processed/
            X_scaled.npy      (74336 × 17)  — normalized features
            y_binary.npy      (74336,)       — binary congestion label
            y_zone.npy        (74336,)       — zone label encoded
            y_util.npy        (74336,)       — utilization regression
            port_index.npy    (74336 × 2)    — (dpid_enc, port_no) per row

Outputs: module2/processed/
            X_windows.npy     (73508 × 10 × 17) — windowed feature sequences
            y_binary_w.npy    (73508,)           — binary label at window end
            y_zone_w.npy      (73508,)           — zone label at window end
            y_util_w.npy      (73508,)           — utilization at window end
            windowing_report.txt

What windowing means and why it is done this way:
-------------------------------------------------
The LSTM learns from SEQUENCES, not individual rows. A single row tells you
the network state at one instant. A sequence of rows tells you how the state
is CHANGING — whether utilization is climbing, whether drops are accumulating,
whether latency is spiking. That trajectory is what predicts future congestion.

Grouping rule (critical):
  Rows from different (dpid, port_no) pairs MUST NEVER be mixed in one window.
  Each switch port is an independent time series. Mixing them would create
  sequences like [port1_row, port2_row, port1_row...] which are meaningless —
  the LSTM would be trying to learn a pattern that does not exist.

Window structure (seq_len = 10 steps = 20 seconds of history):
  For a port with 808 rows, windows are:
    Window 1:  rows  0 –  9  → label = row  9
    Window 2:  rows  1 – 10  → label = row 10
    Window 3:  rows  2 – 11  → label = row 11
    ...
    Window 799: rows 798–807 → label = row 807

  Step = 1 (stride of 1) so consecutive windows overlap by seq_len-1 rows.
  This maximizes training data — 799 windows per port × 92 ports = 73,508 total.

Label assignment:
  The label for each window is taken from the LAST row of that window.
  This means: given the last 10 observations, predict the state RIGHT NOW.
  (Module III will extend this to predict the NEXT state — but that is Task 7.)
"""

import os
import numpy as np

# ── Config ───────────────────────────────────────────────────────
INPUT_DIR  = 'module2/processed'
OUTPUT_DIR = 'module2/processed'
SEQ_LEN    = 10   # 10 steps × 2s polling = 20 seconds of history

# ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = []

    def log(msg=''):
        print(msg)
        report_lines.append(msg)

    log('=' * 65)
    log('  Module II — Task 2: Windowing')
    log(f'  seq_len = {SEQ_LEN} steps ({SEQ_LEN*2}s of history per window)')
    log('=' * 65)

    # ── Load Task 1 outputs ──────────────────────────────────────
    log('\n[Step 1] Loading preprocessed data ...')
    X          = np.load(f'{INPUT_DIR}/X_scaled.npy')
    y_binary   = np.load(f'{INPUT_DIR}/y_binary.npy')
    y_zone     = np.load(f'{INPUT_DIR}/y_zone.npy')
    y_util     = np.load(f'{INPUT_DIR}/y_util.npy')
    port_index = np.load(f'{INPUT_DIR}/port_index.npy')

    log(f'  X          : {X.shape}')
    log(f'  y_binary   : {y_binary.shape}')
    log(f'  y_zone     : {y_zone.shape}')
    log(f'  y_util     : {y_util.shape}')
    log(f'  port_index : {port_index.shape}')

    # ── Group rows by (dpid_enc, port_no) ────────────────────────
    log('\n[Step 2] Grouping rows by (dpid, port_no) ...')

    # Build a dict: key=(dpid_enc, port_no) → sorted list of row indices
    # port_index[:,0] = dpid encoded, port_index[:,1] = port_no
    groups = {}
    for i in range(len(port_index)):
        key = (int(port_index[i, 0]), int(port_index[i, 1]))
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    log(f'  Total unique (dpid, port_no) pairs: {len(groups)}')
    group_sizes = [len(v) for v in groups.values()]
    log(f'  Rows per group — min:{min(group_sizes)}  max:{max(group_sizes)}  '
        f'mean:{sum(group_sizes)/len(group_sizes):.1f}')

    # Verify groups are already in time order
    # (they should be since qos_log.csv is written in polling order)
    all_in_order = True
    for key, indices in groups.items():
        if indices != sorted(indices):
            all_in_order = False
            break
    log(f'  Row indices in time order within all groups: {all_in_order}')

    # ── Slide windows ────────────────────────────────────────────
    log(f'\n[Step 3] Sliding windows (seq_len={SEQ_LEN}, stride=1) ...')

    windows_X        = []
    windows_y_binary = []
    windows_y_zone   = []
    windows_y_util   = []
    windows_per_port = {}

    for key, indices in sorted(groups.items()):
        indices = np.array(indices)   # already in time order
        n       = len(indices)

        if n < SEQ_LEN:
            log(f'  SKIP {key}: only {n} rows, need at least {SEQ_LEN}')
            continue

        n_windows = n - SEQ_LEN + 1
        windows_per_port[key] = n_windows

        for start in range(n_windows):
            window_idx = indices[start : start + SEQ_LEN]

            # Feature sequence: shape (seq_len, n_features)
            windows_X.append(X[window_idx])

            # Labels: taken from the LAST row of the window
            last = window_idx[-1]
            windows_y_binary.append(y_binary[last])
            windows_y_zone.append(y_zone[last])
            windows_y_util.append(y_util[last])

    # ── Stack into arrays ────────────────────────────────────────
    log('\n[Step 4] Stacking into arrays ...')
    X_windows  = np.array(windows_X,        dtype=np.float32)
    y_binary_w = np.array(windows_y_binary,  dtype=np.int64)
    y_zone_w   = np.array(windows_y_zone,    dtype=np.int64)
    y_util_w   = np.array(windows_y_util,    dtype=np.float32)

    log(f'  X_windows  : {X_windows.shape}  '
        f'({X_windows.nbytes/1024/1024:.1f} MB)')
    log(f'  y_binary_w : {y_binary_w.shape}')
    log(f'  y_zone_w   : {y_zone_w.shape}')
    log(f'  y_util_w   : {y_util_w.shape}')

    # ── Verify label distributions ───────────────────────────────
    log('\n[Step 5] Verifying label distributions in windowed dataset ...')

    n_total = len(y_binary_w)
    n_pos   = y_binary_w.sum()
    n_neg   = n_total - n_pos
    log(f'  Binary — total windows : {n_total}')
    log(f'  Binary — not congested : {n_neg} ({100*n_neg/n_total:.1f}%)')
    log(f'  Binary — congested     : {n_pos} ({100*n_pos/n_total:.1f}%)')

    zone_counts = np.bincount(y_zone_w)
    zone_names  = ['normal', 'warning', 'congested']
    log(f'  Zone distribution:')
    for i, count in enumerate(zone_counts):
        name = zone_names[i] if i < len(zone_names) else f'class_{i}'
        log(f'    {i} ({name}): {count} ({100*count/n_total:.1f}%)')

    log(f'  Utilization — mean:{y_util_w.mean():.2f}  '
        f'max:{y_util_w.max():.2f}  min:{y_util_w.min():.2f}')

    # ── Sanity checks ────────────────────────────────────────────
    log('\n[Step 6] Sanity checks ...')

    # Check 1: shape correctness
    assert X_windows.ndim == 3, f"Expected 3D, got {X_windows.ndim}D"
    assert X_windows.shape[1] == SEQ_LEN, "Window length mismatch"
    assert X_windows.shape[2] == X.shape[1], "Feature count mismatch"
    log(f'  Shape check passed: (N={X_windows.shape[0]}, '
        f'seq={X_windows.shape[1]}, features={X_windows.shape[2]})')

    # Check 2: total windows = sum of (group_size - seq_len + 1) per group
    expected = sum(max(0, s - SEQ_LEN + 1) for s in group_sizes)
    assert len(X_windows) == expected, \
        f"Window count mismatch: got {len(X_windows)}, expected {expected}"
    log(f'  Window count check passed: {len(X_windows)} == {expected}')

    # Check 3: no NaN or Inf in feature windows
    has_nan = np.isnan(X_windows).any()
    has_inf = np.isinf(X_windows).any()
    log(f'  NaN in X_windows: {has_nan}')
    log(f'  Inf in X_windows: {has_inf}')
    assert not has_nan and not has_inf, "NaN or Inf found in windowed features"
    log(f'  NaN/Inf check passed')

    # Check 4: verify one window manually
    # Take first port, window 0 — last row label should match y_binary[9]
    first_key   = sorted(groups.keys())[0]
    first_idx   = groups[first_key]
    expected_label = int(y_binary[first_idx[SEQ_LEN - 1]])
    actual_label   = int(y_binary_w[0])
    assert expected_label == actual_label, \
        f"Label mismatch at window 0: expected {expected_label}, got {actual_label}"
    log(f'  Label alignment check passed '
        f'(window[0] last-row label = {actual_label})')

    # ── Save outputs ─────────────────────────────────────────────
    log('\n[Step 7] Saving windowed arrays ...')
    np.save(f'{OUTPUT_DIR}/X_windows.npy',   X_windows)
    np.save(f'{OUTPUT_DIR}/y_binary_w.npy',  y_binary_w)
    np.save(f'{OUTPUT_DIR}/y_zone_w.npy',    y_zone_w)
    np.save(f'{OUTPUT_DIR}/y_util_w.npy',    y_util_w)

    with open(f'{OUTPUT_DIR}/windowing_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    log(f'  X_windows.npy    saved  {X_windows.nbytes/1024/1024:.1f} MB')
    log(f'  y_binary_w.npy   saved')
    log(f'  y_zone_w.npy     saved')
    log(f'  y_util_w.npy     saved')
    log(f'  windowing_report.txt saved')

    log('\n' + '=' * 65)
    log('  Task 2 complete. Ready for Task 3 (Train/Val/Test Split).')
    log('=' * 65)

    return {
        'X_windows':  X_windows,
        'y_binary_w': y_binary_w,
        'y_zone_w':   y_zone_w,
        'y_util_w':   y_util_w,
    }


if __name__ == '__main__':
    main()
