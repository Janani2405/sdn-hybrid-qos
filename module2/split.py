"""
split.py — Module II Task 3: Train / Validation / Test Split
=============================================================
Input  : module2/processed/
            X_windows.npy    (19580 × 10 × 17)
            y_binary_w.npy   (19580,)
            y_zone_w.npy     (19580,)
            y_util_w.npy     (19580,)
            port_index.npy   (20534 × 2)  — used to reconstruct port boundaries

Outputs: module2/processed/
            train_idx.npy    — indices into X_windows for train set
            val_idx.npy      — indices for validation set
            test_idx.npy     — indices for test set
            split_report.txt — full audit trail with distributions

Why we split by PORT, not by window (data leakage explanation):
---------------------------------------------------------------
Consecutive windows from the same port overlap heavily.
  Window[0]  covers rows 0–9
  Window[1]  covers rows 1–10   ← shares 9 of 10 rows with Window[0]
  Window[2]  covers rows 2–11   ← shares 9 of 10 rows with Window[1]

If Window[0] goes to train and Window[1] goes to val, the model has
effectively seen 90% of that val sample during training. The model
would report artificially high validation accuracy — not because it
generalised, but because it memorised overlapping rows.

Correct approach: assign entire ports to splits.
  → All windows of port (dpid=X, port=1) go entirely to TRAIN
  → All windows of port (dpid=Y, port=2) go entirely to VAL
  → No single row ever appears in two different splits

Why we stratify by congestion level:
------------------------------------
Ports have very different congestion profiles. A random port assignment
could cluster all high-congestion ports in train and leave test with
only easy (low-congestion) ports, making test metrics misleadingly
optimistic.

Stratification: sort ports into 3 buckets by congestion %, then
distribute each bucket proportionally across train/val/test.

Split ratios: 70% train / 15% val / 15% test (port-level)

NOTE on non-uniform window counts:
-----------------------------------
The actual dataset has 106 ports where some ports have 184 windows and
others have 185 windows (due to 193 vs 194 raw rows per port). Window
indices are NOT assumed to be uniform — they are derived directly from
the windowed arrays using port_index.npy, ensuring exact alignment.
"""

import os
import numpy as np

# ── Config ───────────────────────────────────────────────────────
INPUT_DIR    = 'module2/processed'
OUTPUT_DIR   = 'module2/processed'
SEQ_LEN      = 10
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO = 0.15
RANDOM_SEED  = 42

# ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = []

    def log(msg=''):
        print(msg)
        report_lines.append(msg)

    log('=' * 65)
    log('  Module II — Task 3: Train / Val / Test Split')
    log(f'  Strategy : port-level split with congestion stratification')
    log(f'  Ratios   : {int(TRAIN_RATIO*100)}% train / '
        f'{int(VAL_RATIO*100)}% val / '
        f'{int((1-TRAIN_RATIO-VAL_RATIO)*100)}% test')
    log(f'  Seed     : {RANDOM_SEED}')
    log('=' * 65)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── Load windowed data ───────────────────────────────────────
    log('\n[Step 1] Loading windowed data ...')
    X_windows  = np.load(f'{INPUT_DIR}/X_windows.npy')
    y_binary_w = np.load(f'{INPUT_DIR}/y_binary_w.npy')
    y_zone_w   = np.load(f'{INPUT_DIR}/y_zone_w.npy')
    y_util_w   = np.load(f'{INPUT_DIR}/y_util_w.npy')

    n_total = len(X_windows)
    log(f'  Total windows : {n_total}')

    # ── Reconstruct port → window index mapping from port_index ──
    # port_index.npy has one row per RAW row (pre-windowing).
    # For each port (dpid_enc, port_no), windows are ordered sequentially
    # in X_windows in the same order window.py processed them (sorted keys).
    # We rebuild the mapping by re-sorting ports and counting their rows,
    # then deriving window counts as (n_rows - SEQ_LEN + 1).
    log('\n[Step 2] Reconstructing port → window index mapping ...')

    port_index = np.load(f'{INPUT_DIR}/port_index.npy')  # (N_raw, 2)
    log(f'  Raw rows loaded from port_index.npy : {len(port_index)}')

    # Count rows per port, preserving sorted order (matches window.py)
    from collections import defaultdict, OrderedDict
    port_row_counts = defaultdict(int)
    for i in range(len(port_index)):
        key = (int(port_index[i, 0]), int(port_index[i, 1]))
        port_row_counts[key] += 1

    # Sort by key to match window.py's sorted(groups.items()) order
    sorted_ports = sorted(port_row_counts.keys())
    N_PORTS = len(sorted_ports)
    log(f'  Unique ports detected : {N_PORTS}')

    # Compute window count per port and starting window index
    port_window_starts = {}   # key → first window index in X_windows
    port_window_counts = {}   # key → number of windows
    cursor = 0
    for key in sorted_ports:
        n_rows = port_row_counts[key]
        n_win  = max(0, n_rows - SEQ_LEN + 1)
        port_window_starts[key] = cursor
        port_window_counts[key] = n_win
        cursor += n_win

    assert cursor == n_total, \
        f'Window count mismatch: reconstructed {cursor}, actual {n_total}'
    log(f'  Window index reconstruction verified: {cursor} == {n_total}')

    win_counts = [port_window_counts[k] for k in sorted_ports]
    log(f'  Windows per port — min:{min(win_counts)}  '
        f'max:{max(win_counts)}  '
        f'mean:{sum(win_counts)/len(win_counts):.1f}')

    # ── Compute per-port congestion % ───────────────────────────
    log('\n[Step 3] Computing congestion % per port for stratification ...')

    port_cong = np.zeros(N_PORTS)
    for pid, key in enumerate(sorted_ports):
        start = port_window_starts[key]
        end   = start + port_window_counts[key]
        if end > start:
            port_cong[pid] = y_binary_w[start:end].mean() * 100

    log(f'  Congestion % across ports:')
    log(f'    min  = {port_cong.min():.1f}%')
    log(f'    max  = {port_cong.max():.1f}%')
    log(f'    mean = {port_cong.mean():.1f}%')

    # ── Stratify ports into 3 buckets ───────────────────────────
    log('\n[Step 4] Stratifying ports by congestion level ...')
    port_ids   = np.arange(N_PORTS)
    low_ports  = port_ids[port_cong < 20]
    mid_ports  = port_ids[(port_cong >= 20) & (port_cong < 60)]
    high_ports = port_ids[port_cong >= 60]

    log(f'  Low  congestion ports (<20%)  : {len(low_ports)}')
    log(f'  Mid  congestion ports (20-60%): {len(mid_ports)}')
    log(f'  High congestion ports (>=60%) : {len(high_ports)}')

    # Shuffle each bucket independently with fixed seed
    rng.shuffle(low_ports)
    rng.shuffle(mid_ports)
    rng.shuffle(high_ports)

    # ── Assign ports to splits ───────────────────────────────────
    log('\n[Step 5] Assigning ports to train / val / test ...')

    def split_bucket(bucket, train_r, val_r):
        n       = len(bucket)
        n_train = max(1, round(n * train_r))
        n_val   = max(1, round(n * val_r))
        n_test  = n - n_train - n_val
        if n_test < 1:          # guarantee at least 1 port in test
            n_val  -= 1
            n_test += 1
        return (bucket[:n_train],
                bucket[n_train : n_train + n_val],
                bucket[n_train + n_val:])

    low_tr,  low_v,  low_te  = split_bucket(low_ports,  TRAIN_RATIO, VAL_RATIO)
    mid_tr,  mid_v,  mid_te  = split_bucket(mid_ports,  TRAIN_RATIO, VAL_RATIO)
    high_tr, high_v, high_te = split_bucket(high_ports, TRAIN_RATIO, VAL_RATIO)

    train_port_ids = np.concatenate([low_tr, mid_tr, high_tr])
    val_port_ids   = np.concatenate([low_v,  mid_v,  high_v])
    test_port_ids  = np.concatenate([low_te, mid_te, high_te])

    log(f'  Train ports : {len(train_port_ids)}  '
        f'(low={len(low_tr)} mid={len(mid_tr)} high={len(high_tr)})')
    log(f'  Val   ports : {len(val_port_ids)}  '
        f'(low={len(low_v)} mid={len(mid_v)} high={len(high_v)})')
    log(f'  Test  ports : {len(test_port_ids)}  '
        f'(low={len(low_te)} mid={len(mid_te)} high={len(high_te)})')

    # ── Convert port assignments → window indices ─────────────────
    log('\n[Step 6] Converting port assignments to window indices ...')

    def ports_to_indices(pid_array):
        idx = []
        for pid in pid_array:
            key   = sorted_ports[pid]
            start = port_window_starts[key]
            count = port_window_counts[key]
            idx.extend(range(start, start + count))
        return np.array(idx, dtype=np.int64)

    train_idx = ports_to_indices(train_port_ids)
    val_idx   = ports_to_indices(val_port_ids)
    test_idx  = ports_to_indices(test_port_ids)

    # Shuffle train indices so batches contain mixed zones / ports
    rng.shuffle(train_idx)

    log(f'  Train windows : {len(train_idx)}')
    log(f'  Val   windows : {len(val_idx)}')
    log(f'  Test  windows : {len(test_idx)}')
    log(f'  Total         : {len(train_idx)+len(val_idx)+len(test_idx)} '
        f'(expected {n_total})')
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_total, \
        f'Window count mismatch: {len(train_idx)+len(val_idx)+len(test_idx)} != {n_total}'

    # ── Verify no overlap between splits ─────────────────────────
    log('\n[Step 7] Verifying zero overlap between splits ...')
    train_set = set(train_idx.tolist())
    val_set   = set(val_idx.tolist())
    test_set  = set(test_idx.tolist())

    tv_overlap = len(train_set & val_set)
    tt_overlap = len(train_set & test_set)
    vt_overlap = len(val_set  & test_set)

    log(f'  Train ∩ Val  overlap : {tv_overlap}')
    log(f'  Train ∩ Test overlap : {tt_overlap}')
    log(f'  Val   ∩ Test overlap : {vt_overlap}')
    assert tv_overlap == 0 and tt_overlap == 0 and vt_overlap == 0, \
        'DATA LEAKAGE: splits share window indices!'
    log('  No overlap confirmed — zero data leakage')

    # ── Verify label distributions ───────────────────────────────
    log('\n[Step 8] Label distributions across splits ...')

    for split_name, idx in [('Train', train_idx),
                             ('Val',   val_idx),
                             ('Test',  test_idx)]:
        yb = y_binary_w[idx]
        yz = y_zone_w[idx]
        n  = len(idx)
        pos = yb.sum()
        log(f'\n  {split_name} ({n} windows):')
        log(f'    Binary — congested    : {pos} ({100*pos/n:.1f}%)')
        log(f'    Binary — not congested: {n-pos} ({100*(n-pos)/n:.1f}%)')
        zone_counts = np.bincount(yz, minlength=3)
        zone_names  = ['normal', 'warning', 'congested']
        for i, cnt in enumerate(zone_counts):
            log(f'    Zone {i} ({zone_names[i]}): {cnt} ({100*cnt/n:.1f}%)')

    # ── Save outputs ─────────────────────────────────────────────
    log('\n[Step 9] Saving split indices ...')
    np.save(f'{OUTPUT_DIR}/train_idx.npy', train_idx)
    np.save(f'{OUTPUT_DIR}/val_idx.npy',   val_idx)
    np.save(f'{OUTPUT_DIR}/test_idx.npy',  test_idx)

    with open(f'{OUTPUT_DIR}/split_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    log(f'  train_idx.npy    saved  ({len(train_idx)} indices)')
    log(f'  val_idx.npy      saved  ({len(val_idx)} indices)')
    log(f'  test_idx.npy     saved  ({len(test_idx)} indices)')
    log(f'  split_report.txt saved')

    log('\n' + '=' * 65)
    log('  Task 3 complete. Ready for Task 4 (LSTM Model Definition).')
    log('=' * 65)

    return {
        'train_idx': train_idx,
        'val_idx':   val_idx,
        'test_idx':  test_idx,
    }


if __name__ == '__main__':
    main()