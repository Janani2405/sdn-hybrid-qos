"""
split.py — Module II Task 3: Train / Validation / Test Split
=============================================================
Input  : module2/processed/
            X_windows.npy    (73508 × 10 × 17)
            y_binary_w.npy   (73508,)
            y_zone_w.npy     (73508,)
            y_util_w.npy     (73508,)

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
  → All 799 windows of port (dpid=X, port=1) go entirely to TRAIN
  → All 799 windows of port (dpid=Y, port=2) go entirely to VAL
  → No single row ever appears in two different splits

Why we stratify by congestion level:
------------------------------------
Ports have very different congestion profiles (10.9%–83.4% congested).
A random port assignment could cluster all high-congestion ports in
train and leave test with only easy (low-congestion) ports, making
test metrics misleadingly optimistic.

Stratification: sort ports into 3 buckets by congestion %, then
distribute each bucket proportionally across train/val/test.

Split ratios: 70% train / 15% val / 15% test (port-level)
  64 train ports → 51,136 windows
  13 val ports   → 10,387 windows
  15 test ports  → 11,985 windows
"""

import os
import numpy as np

# ── Config ───────────────────────────────────────────────────────
INPUT_DIR        = 'module2/processed'
OUTPUT_DIR       = 'module2/processed'
ROWS_PER_PORT    = 808
SEQ_LEN          = 10
WINDOWS_PER_PORT = ROWS_PER_PORT - SEQ_LEN + 1   # 799
N_PORTS          = 92
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO = 0.15
RANDOM_SEED      = 42

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
    log(f'  Ports         : {N_PORTS}  ×  {WINDOWS_PER_PORT} windows each')
    assert n_total == N_PORTS * WINDOWS_PER_PORT, \
        f'Expected {N_PORTS}×{WINDOWS_PER_PORT}={N_PORTS*WINDOWS_PER_PORT}, got {n_total}'

    # ── Compute per-port congestion % ───────────────────────────
    log('\n[Step 2] Computing congestion % per port for stratification ...')
    port_ids   = np.arange(N_PORTS)
    port_cong  = np.zeros(N_PORTS)

    for pid in port_ids:
        start           = pid * WINDOWS_PER_PORT
        end             = start + WINDOWS_PER_PORT
        port_cong[pid]  = y_binary_w[start:end].mean() * 100

    log(f'  Congestion % across ports:')
    log(f'    min  = {port_cong.min():.1f}%')
    log(f'    max  = {port_cong.max():.1f}%')
    log(f'    mean = {port_cong.mean():.1f}%')

    # ── Stratify ports into 3 buckets ───────────────────────────
    log('\n[Step 3] Stratifying ports by congestion level ...')
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
    log('\n[Step 4] Assigning ports to train / val / test ...')

    def split_bucket(bucket, train_r, val_r):
        n      = len(bucket)
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

    train_ports = np.concatenate([low_tr, mid_tr, high_tr])
    val_ports   = np.concatenate([low_v,  mid_v,  high_v])
    test_ports  = np.concatenate([low_te, mid_te, high_te])

    log(f'  Train ports : {len(train_ports)}  '
        f'(low={len(low_tr)} mid={len(mid_tr)} high={len(high_tr)})')
    log(f'  Val   ports : {len(val_ports)}  '
        f'(low={len(low_v)} mid={len(mid_v)} high={len(high_v)})')
    log(f'  Test  ports : {len(test_ports)}  '
        f'(low={len(low_te)} mid={len(mid_te)} high={len(high_te)})')

    # ── Convert port assignments → window indices ─────────────────
    log('\n[Step 5] Converting port assignments to window indices ...')

    def ports_to_indices(ports):
        idx = []
        for pid in ports:
            start = pid * WINDOWS_PER_PORT
            idx.extend(range(start, start + WINDOWS_PER_PORT))
        return np.array(idx, dtype=np.int64)

    train_idx = ports_to_indices(train_ports)
    val_idx   = ports_to_indices(val_ports)
    test_idx  = ports_to_indices(test_ports)

    # Shuffle train indices so batches contain mixed zones / ports
    rng.shuffle(train_idx)

    log(f'  Train windows : {len(train_idx)}')
    log(f'  Val   windows : {len(val_idx)}')
    log(f'  Test  windows : {len(test_idx)}')
    log(f'  Total         : {len(train_idx)+len(val_idx)+len(test_idx)} '
        f'(expected {n_total})')
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_total

    # ── Verify no overlap between splits ─────────────────────────
    log('\n[Step 6] Verifying zero overlap between splits ...')
    train_set = set(train_idx.tolist())
    val_set   = set(val_idx.tolist())
    test_set  = set(test_idx.tolist())

    tv_overlap  = len(train_set & val_set)
    tt_overlap  = len(train_set & test_set)
    vt_overlap  = len(val_set  & test_set)

    log(f'  Train ∩ Val  overlap : {tv_overlap}')
    log(f'  Train ∩ Test overlap : {tt_overlap}')
    log(f'  Val   ∩ Test overlap : {vt_overlap}')
    assert tv_overlap == 0 and tt_overlap == 0 and vt_overlap == 0, \
        'DATA LEAKAGE: splits share window indices!'
    log('  No overlap confirmed — zero data leakage')

    # ── Verify label distributions ───────────────────────────────
    log('\n[Step 7] Label distributions across splits ...')

    for split_name, idx in [('Train', train_idx),
                             ('Val',   val_idx),
                             ('Test',  test_idx)]:
        yb  = y_binary_w[idx]
        yz  = y_zone_w[idx]
        n   = len(idx)
        pos = yb.sum()
        log(f'\n  {split_name} ({n} windows):')
        log(f'    Binary — congested    : {pos} ({100*pos/n:.1f}%)')
        log(f'    Binary — not congested: {n-pos} ({100*(n-pos)/n:.1f}%)')
        zone_counts = np.bincount(yz, minlength=3)
        zone_names  = ['normal', 'warning', 'congested']
        for i, cnt in enumerate(zone_counts):
            log(f'    Zone {i} ({zone_names[i]}): {cnt} ({100*cnt/n:.1f}%)')

    # ── Save outputs ─────────────────────────────────────────────
    log('\n[Step 8] Saving split indices ...')
    np.save(f'{OUTPUT_DIR}/train_idx.npy', train_idx)
    np.save(f'{OUTPUT_DIR}/val_idx.npy',   val_idx)
    np.save(f'{OUTPUT_DIR}/test_idx.npy',  test_idx)

    with open(f'{OUTPUT_DIR}/split_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    log(f'  train_idx.npy   saved  ({len(train_idx)} indices)')
    log(f'  val_idx.npy     saved  ({len(val_idx)} indices)')
    log(f'  test_idx.npy    saved  ({len(test_idx)} indices)')
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
