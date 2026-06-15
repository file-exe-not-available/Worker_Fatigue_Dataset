"""
Shared preprocessing pipeline for all models.

Handles: data loading, cleaning, RPE interpolation, windowing,
feature extraction, and label binning.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

EXCLUDE = {'Sub11', 'Sub12', 'Sub14'}

IMU_COLS_PATTERN = [
    (site, sensor_type, axis)
    for site in ['trunk', 'upper_arm', 'wrist']
    for sensor_type in ['Accelerometer', 'Gyroscope', 'Magnetometer']
    for axis in ['X', 'Y', 'Z']
]

# RPE bins: Low=0 (0-3), Moderate=1 (4-6), High=2 (7-10)
RPE_BIN_EDGES = [-0.01, 3.5, 6.5, 10.01]
RPE_LABELS = ['Low', 'Moderate', 'High']

WINDOW_SIZE = 1000  # 10 s at 100 Hz
STRIDE = 500        # 50% overlap

# MVIC strength tests occur at these minutes from picking start (Vahedi et al. 2023)
MVIC_MINUTES = [9, 18, 27, 36, 45]
MVIC_BUFFER_SECONDS = 60  # exclude 60 s before and after each test


def load_and_clean(data_path):
    print(f"Loading {data_path} ...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    ts_data = {k: v for k, v in data['ts_data'].items() if k[0] not in EXCLUDE}
    anthro = data['anthro_clean']
    anthro = anthro[~anthro['Subject'].isin(EXCLUDE)].reset_index(drop=True)

    # Remove trailing artefact RPE drop in Sub13/Session2
    key = ('Sub13', 'Session2')
    if key in ts_data:
        df = ts_data[key]
        rpe_rows = df[df['RPE_Val'].notna()]
        max_rpe = rpe_rows['RPE_Val'].max()
        last_peak = rpe_rows[rpe_rows['RPE_Val'] == max_rpe].index[-1]
        ts_data[key] = df.loc[:last_peak].copy()

    n_subjects = len(set(k[0] for k in ts_data))
    print(f"  {len(ts_data)} sessions, {n_subjects} subjects after cleaning")
    return ts_data, anthro, data['experiment_settings']


def get_imu_cols(df):
    cols = []
    for site, sensor_type, axis in IMU_COLS_PATTERN:
        col = f'{site}_{sensor_type}.{axis}'
        if col in df.columns:
            cols.append(col)
    return cols


def interpolate_rpe(df):
    """Linearly interpolate sparse RPE readings across all rows of one session."""
    rpe = df['RPE_Val'].copy().astype(float)
    rpe = rpe.interpolate(method='linear')
    rpe = rpe.bfill().ffill()
    return rpe


def bin_rpe(rpe_val):
    """Map a scalar RPE value to class index 0 (Low), 1 (Moderate), or 2 (High)."""
    result = pd.cut([rpe_val], bins=RPE_BIN_EDGES, labels=[0, 1, 2])[0]
    if pd.isna(result):
        if rpe_val <= 3.5:
            return 0
        elif rpe_val <= 6.5:
            return 1
        else:
            return 2
    return int(result)


def get_mvic_exclusion_mask(df, mvic_minutes=MVIC_MINUTES, buffer_seconds=MVIC_BUFFER_SECONDS):
    """
    Return a boolean array (True = exclude) marking rows within buffer_seconds
    of each MVIC strength test. Tests occur at fixed intervals from session start
    (Vahedi et al. 2023: every 9 min, data begins after the baseline MVIC).
    """
    elapsed = df['Timestamp'].values - df['Timestamp'].iloc[0]
    mask = np.zeros(len(df), dtype=bool)
    for t_min in mvic_minutes:
        mask |= np.abs(elapsed - t_min * 60) <= buffer_seconds
    return mask


def extract_features(window, fs=100):
    """
    Extract statistical + frequency-domain + magnitude features from one window.

    Per channel (10): mean, std, min, max, RMS, ZCR,
                      dominant freq, mean freq, spectral entropy, mean abs jerk
    Per (site, sensor_type) magnitude group (6): mean, std, min, max, RMS, ZCR

    window : np.ndarray shape (window_size, n_channels)
    Returns: 1-D float32 vector of length n_channels*10 + (n_channels//3)*6
    """
    n_samples, n_channels = window.shape
    feats = []
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    for i in range(n_channels):
        ch = window[:, i]
        mean = ch.mean()

        feats.append(mean)
        feats.append(ch.std())
        feats.append(ch.min())
        feats.append(ch.max())
        feats.append(np.sqrt(np.mean(ch ** 2)))
        feats.append(np.sum(np.diff(np.sign(ch - mean)) != 0) / n_samples)

        power = np.abs(np.fft.rfft(ch)) ** 2
        total_power = power.sum() + 1e-12
        feats.append(float(freqs[np.argmax(power[1:]) + 1]))
        feats.append(float(np.dot(freqs, power) / total_power))
        psd_norm = power / total_power
        feats.append(float(-np.sum(psd_norm * np.log(psd_norm + 1e-12))))
        feats.append(float(np.abs(np.diff(ch)).mean()))

    for i in range(0, n_channels, 3):
        mag = np.sqrt((window[:, i:i + 3] ** 2).sum(axis=1))
        mag_mean = mag.mean()
        feats.append(mag_mean)
        feats.append(mag.std())
        feats.append(mag.min())
        feats.append(mag.max())
        feats.append(np.sqrt(np.mean(mag ** 2)))
        feats.append(np.sum(np.diff(np.sign(mag - mag_mean)) != 0) / len(mag))

    return np.array(feats, dtype=np.float32)


def get_feature_names(imu_cols):
    per_ch = ['mean', 'std', 'min', 'max', 'rms', 'zcr',
              'dom_freq', 'mean_freq', 'spec_entropy', 'jerk']
    names = [f'{col}_{stat}' for col in imu_cols for stat in per_ch]
    for i in range(0, len(imu_cols), 3):
        group = imu_cols[i].rsplit('.', 1)[0]
        for stat in ['mean', 'std', 'min', 'max', 'rms', 'zcr']:
            names.append(f'{group}_mag_{stat}')
    return names


def build_dataset(ts_data, window_size=WINDOW_SIZE, stride=STRIDE, cache_path=None):
    """
    Slide windows over all sessions and build a labelled feature matrix.

    Returns
    -------
    X          : (n_windows, n_features) float32
    y          : (n_windows,) int32  — 0=Low, 1=Moderate, 2=High
    subjects   : (n_windows,) str
    feat_names : list[str]
    """
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            print(f"Loading cached dataset from {cache} ...")
            d = np.load(cache, allow_pickle=True)
            return d['X'], d['y'], d['subjects'], list(d['feat_names'])
        print(f"Cache not found — building dataset (will save to {cache}) ...")

    X_list, y_list, subj_list = [], [], []

    canonical_cols = None
    for key in sorted(ts_data.keys()):
        cols = get_imu_cols(ts_data[key])
        if cols:
            canonical_cols = cols
            break
    if canonical_cols is None:
        raise ValueError("No IMU columns found in any session")
    feat_names = get_feature_names(canonical_cols)

    for (subj, sess), df in sorted(ts_data.items()):
        missing = [c for c in canonical_cols if c not in df.columns]
        if missing:
            print(f"  WARNING: {subj}/{sess} missing columns {missing}, skipping")
            continue

        rpe = interpolate_rpe(df)
        imu = df[canonical_cols].values.astype(np.float32)
        mvic_mask = get_mvic_exclusion_mask(df)
        n = len(df)

        n_windows = 0
        start = 0
        while start + window_size <= n:
            end = start + window_size

            if mvic_mask[start:end].any():
                start += stride
                continue

            window = imu[start:end]
            if np.isnan(window).any():
                start += stride
                continue

            rpe_slice = rpe.iloc[start:end].dropna()
            if rpe_slice.empty:
                start += stride
                continue

            label = bin_rpe(float(rpe_slice.median()))
            X_list.append(extract_features(window))
            y_list.append(label)
            subj_list.append(subj)
            n_windows += 1
            start += stride

        print(f"  {subj}/{sess}: {n_windows} windows")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subj_list)

    print(f"\nDataset: {len(X):,} windows, {X.shape[1]} features")
    counts = {RPE_LABELS[i]: int((y == i).sum()) for i in range(3)}
    print(f"Class counts: {counts}")

    if cache_path is not None:
        np.savez(cache_path, X=X, y=y, subjects=subjects, feat_names=feat_names)
        print(f"Dataset cached to {cache_path}")

    return X, y, subjects, feat_names
