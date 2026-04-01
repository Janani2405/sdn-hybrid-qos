"""
preprocess.py — Module II Task 1: Data Preparation
====================================================
Input  : logs/qos_log.csv  (74,336 rows, 30 columns)
Outputs: module2/processed/
            X_scaled.npy          — normalized feature matrix  (N × 17)
            y_binary.npy          — binary congestion label    (N,)
            y_zone.npy            — zone label encoded         (N,)
            y_util.npy            — utilization regression     (N,)
            port_index.npy        — (dpid_encoded, port_no) per row (N × 2)
            scaler.pkl            — fitted StandardScaler
            label_encoder.pkl     — fitted LabelEncoder for zone
            feature_names.txt     — ordered list of 17 feature names
            preprocessing_report.txt — full audit trail

What this script does (in order):
  Step 1 — Load raw CSV
  Step 2 — Drop columns that must never enter the model
  Step 3 — Clip tx_mbps / rx_mbps outliers (core switches aggregate traffic)
  Step 4 — Fit and apply StandardScaler on 17 features
  Step 5 — Encode zone_label with LabelEncoder
  Step 6 — Extract targets (binary, zone, utilization)
  Step 7 — Save all outputs + report

Column decisions (full audit):
  DROPPED — identity / index (not features):
    timestamp, dpid, port_no

  DROPPED — raw cumulative counters (delta/rate columns already encode this):
    tx_bytes, rx_bytes, tx_dropped, rx_dropped

  DROPPED — label-leaking signal flags (derived from the same threshold that
             produces `congested` — feeding these in gives the model the answer):
    signal_util, signal_drop

  DROPPED — all-zero column (zero variance, carries no information):
    loss_pct

  DROPPED — exact duplicate (rtt_ms = 2 × latency_ms in every row):
    rtt_ms

  KEPT as TARGET (predict, never input):
    congested, zone_label, utilization_pct (also kept as feature below)

  KEPT as FEATURES (17 total):
    tx_mbps, rx_mbps, utilization_pct, tx_pps, rx_pps, bw_headroom_mbps,
    delta_tx_dropped, delta_rx_dropped, latency_ms, jitter_ms,
    rolling_util_mean, rolling_drop_sum, rolling_tx_mean, rolling_rx_mean,
    n_active_flows, neighbor_util_max, inter_arrival_delta
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Paths ────────────────────────────────────────────────────────
INPUT_CSV  = 'logs/qos_log.csv'
OUTPUT_DIR = 'module2/processed'

# ── Column definitions ───────────────────────────────────────────
DROP_IDENTITY = ['timestamp', 'dpid', 'port_no']

DROP_RAW_COUNTERS = ['tx_bytes', 'rx_bytes', 'tx_dropped', 'rx_dropped']

DROP_LEAKING = ['signal_util', 'signal_drop']

DROP_ZERO_VARIANCE = ['loss_pct']

DROP_REDUNDANT = ['rtt_ms']   # always = 2 × latency_ms

TARGET_COLS = ['congested', 'zone_label', 'utilization_pct']

FEATURE_COLS = [
    'tx_mbps',            # transmission rate — primary load signal
    'rx_mbps',            # receive rate — inbound load
    'utilization_pct',    # max(tx,rx)/link_bw — overall load percentage
    'tx_pps',             # packets per second tx — traffic intensity
    'rx_pps',             # packets per second rx
    'bw_headroom_mbps',   # remaining capacity — how close to saturation
    'delta_tx_dropped',   # new drops this interval (tx) — congestion event signal
    'delta_rx_dropped',   # new drops this interval (rx)
    'latency_ms',         # real LLDP-measured one-way delay
    'jitter_ms',          # EWMA jitter from LLDP probes
    'rolling_util_mean',  # 5-step rolling average of utilization — trend
    'rolling_drop_sum',   # 5-step rolling sum of drops — sustained drop signal
    'rolling_tx_mean',    # 5-step rolling tx mean — traffic trend
    'rolling_rx_mean',    # 5-step rolling rx mean
    'n_active_flows',     # number of active OpenFlow entries — network activity
    'neighbor_util_max',  # max utilization of neighboring ports — spillover risk
    'inter_arrival_delta' # time since last flow arrival — traffic burstiness
]

# Clip threshold for tx_mbps / rx_mbps
# Core switches legitimately carry >100 Mbps aggregated traffic.
# Clipping at 500 Mbps preserves the signal while preventing extreme
# outliers (max observed: 28,926 Mbps) from dominating the scaler.
BW_CLIP = 500.0

ZONE_ORDER = ['normal', 'warning', 'congested', 'critical']

# ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = []

    def log(msg=''):
        print(msg)
        report_lines.append(msg)

    log('=' * 65)
    log('  Module II — Task 1: Data Preparation')
    log('=' * 65)

    # ── Step 1 — Load ────────────────────────────────────────────
    log(f'\n[Step 1] Loading {INPUT_CSV} ...')
    df = pd.read_csv(INPUT_CSV)
    log(f'  Raw shape : {df.shape}  ({df.shape[0]} rows × {df.shape[1]} columns)')
    log(f'  Columns   : {list(df.columns)}')

    # ── Step 2 — Drop columns ────────────────────────────────────
    log('\n[Step 2] Dropping columns ...')

    all_drops = (DROP_IDENTITY + DROP_RAW_COUNTERS +
                 DROP_LEAKING + DROP_ZERO_VARIANCE + DROP_REDUNDANT)

    # verify all drop columns actually exist
    missing_drops = [c for c in all_drops if c not in df.columns]
    if missing_drops:
        log(f'  WARNING: columns not found (already absent): {missing_drops}')
    all_drops = [c for c in all_drops if c in df.columns]

    log(f'  Dropping identity    : {DROP_IDENTITY}')
    log(f'  Dropping raw counters: {DROP_RAW_COUNTERS}')
    log(f'  Dropping leaking     : {DROP_LEAKING}')
    log(f'  Dropping zero-var    : {DROP_ZERO_VARIANCE}')
    log(f'  Dropping redundant   : {DROP_REDUNDANT}')

    # Keep dpid and port_no separately before dropping (needed for windowing in Task 2)
    port_index_df = df[['dpid', 'port_no']].copy()
    dpid_encoder  = LabelEncoder()
    port_index    = np.column_stack([
        dpid_encoder.fit_transform(df['dpid']),
        df['port_no'].astype(int).values
    ])

    df.drop(columns=all_drops, inplace=True)
    log(f'  Shape after drop: {df.shape}')

    # Verify all feature cols are present
    missing_feats = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feats:
        raise ValueError(f'Missing expected feature columns: {missing_feats}')

    # ── Step 3 — Clip bandwidth outliers ─────────────────────────
    log(f'\n[Step 3] Clipping tx_mbps and rx_mbps to {BW_CLIP} Mbps ...')
    before_tx_max = df['tx_mbps'].max()
    before_rx_max = df['rx_mbps'].max()
    tx_clipped    = (df['tx_mbps'] > BW_CLIP).sum()
    rx_clipped    = (df['rx_mbps'] > BW_CLIP).sum()
    df['tx_mbps'] = df['tx_mbps'].clip(upper=BW_CLIP)
    df['rx_mbps'] = df['rx_mbps'].clip(upper=BW_CLIP)
    log(f'  tx_mbps: max before={before_tx_max:.2f}  rows clipped={tx_clipped}')
    log(f'  rx_mbps: max before={before_rx_max:.2f}  rows clipped={rx_clipped}')
    log(f'  bw_headroom_mbps NOT clipped (already controller-bounded to 100)')

    # ── Step 4 — Scale features ───────────────────────────────────
    log(f'\n[Step 4] Fitting StandardScaler on {len(FEATURE_COLS)} features ...')
    X_raw    = df[FEATURE_COLS].values.astype(np.float32)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    log(f'  Feature matrix shape : {X_scaled.shape}')
    log(f'  Features (in order)  :')
    for i, col in enumerate(FEATURE_COLS):
        mean = scaler.mean_[i]
        std  = scaler.scale_[i]
        log(f'    [{i:02d}] {col:<25}  mean={mean:.4f}  std={std:.4f}')

    # ── Step 5 — Encode zone_label ────────────────────────────────
    log('\n[Step 5] Encoding zone_label ...')
    le = LabelEncoder()

    # Fit on the expected fixed order so encoding is always deterministic:
    # normal=0, warning=1, congested=2, critical=3
    # (even if 'critical' doesn't appear in this dataset)
    present_zones = df['zone_label'].unique().tolist()
    fit_zones     = [z for z in ZONE_ORDER if z in present_zones]
    le.classes_ = np.array(fit_zones)  # enforce severity order: normal=0, warning=1, congested=2, critical=3
    y_zone = le.transform(df['zone_label']).astype(np.int64)

    zone_dist = pd.Series(y_zone).value_counts().sort_index()
    log(f'  Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}')
    log(f'  Distribution:')
    for code, count in zone_dist.items():
        name = le.inverse_transform([code])[0]
        log(f'    {code} ({name}): {count} rows ({100*count/len(y_zone):.1f}%)')

    # ── Step 6 — Extract targets ──────────────────────────────────
    log('\n[Step 6] Extracting targets ...')
    y_binary = df['congested'].astype(np.int64).values
    y_util   = df['utilization_pct'].astype(np.float32).values

    log(f'  y_binary  shape={y_binary.shape}  '
        f'positive={y_binary.sum()} ({100*y_binary.mean():.1f}%)')
    log(f'  y_zone    shape={y_zone.shape}')
    log(f'  y_util    shape={y_util.shape}  '
        f'mean={y_util.mean():.2f}  max={y_util.max():.2f}')
    log(f'  port_index shape={port_index.shape}  '
        f'unique ports={len(np.unique(port_index, axis=0))}')

    # ── Step 7 — Class imbalance info (for Task 5 training) ───────
    log('\n[Step 7] Class imbalance analysis ...')
    n_neg     = (y_binary == 0).sum()
    n_pos     = (y_binary == 1).sum()
    pos_weight = n_neg / n_pos
    log(f'  Binary — negative (not congested): {n_neg}')
    log(f'  Binary — positive (congested)    : {n_pos}')
    log(f'  Recommended pos_weight for BCEWithLogitsLoss: {pos_weight:.4f}')
    log(f'  (meaning: weight each positive example {pos_weight:.2f}× more than negative)')

    # ── Step 8 — Save all outputs ────────────────────────────────
    log('\n[Step 8] Saving outputs ...')

    np.save(f'{OUTPUT_DIR}/X_scaled.npy',    X_scaled)
    np.save(f'{OUTPUT_DIR}/y_binary.npy',    y_binary)
    np.save(f'{OUTPUT_DIR}/y_zone.npy',      y_zone)
    np.save(f'{OUTPUT_DIR}/y_util.npy',      y_util)
    np.save(f'{OUTPUT_DIR}/port_index.npy',  port_index)

    with open(f'{OUTPUT_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'{OUTPUT_DIR}/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    with open(f'{OUTPUT_DIR}/feature_names.txt', 'w') as f:
        for name in FEATURE_COLS:
            f.write(name + '\n')

    with open(f'{OUTPUT_DIR}/preprocessing_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    log(f'\n  Saved to {OUTPUT_DIR}/:')
    log(f'    X_scaled.npy        {X_scaled.nbytes/1024:.1f} KB')
    log(f'    y_binary.npy        {y_binary.nbytes/1024:.1f} KB')
    log(f'    y_zone.npy          {y_zone.nbytes/1024:.1f} KB')
    log(f'    y_util.npy          {y_util.nbytes/1024:.1f} KB')
    log(f'    port_index.npy      {port_index.nbytes/1024:.1f} KB')
    log(f'    scaler.pkl')
    log(f'    label_encoder.pkl')
    log(f'    feature_names.txt')
    log(f'    preprocessing_report.txt')

    log('\n' + '=' * 65)
    log('  Task 1 complete. Ready for Task 2 (Windowing).')
    log('=' * 65)

    # Final sanity check
    assert X_scaled.shape == (len(df), len(FEATURE_COLS)), "Shape mismatch"
    assert len(y_binary) == len(y_zone) == len(y_util) == len(X_scaled)
    assert len(port_index) == len(X_scaled)
    log('\n  All assertions passed.')

    return {
        'X_scaled':   X_scaled,
        'y_binary':   y_binary,
        'y_zone':     y_zone,
        'y_util':     y_util,
        'port_index': port_index,
        'scaler':     scaler,
        'le':         le,
    }


if __name__ == '__main__':
    main()
