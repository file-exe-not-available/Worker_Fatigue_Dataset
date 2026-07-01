"""
pipeline.py — shared data loading, cleaning and feature extraction
==================================================================
Imported by  : eda_plots.py, knn_model.py, centralized_rf.py
Exports      : load_and_clean(), build_dataset(), RPE_LABELS
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── label definitions ──────────────────────────────────────────────────────────
RPE_LABELS = ["Low", "Moderate", "High"]   # order matters for confusion matrix

# ── global constants ───────────────────────────────────────────────────────────
EXCLUDE_SUBJECTS = {"Sub11", "Sub12", "Sub14"}
SESSIONS_LIST    = ["Session1", "Session2", "Session3", "Session4"]

BODY_SEGMENTS = ["trunk", "upper_arm", "wrist"]
SENSOR_TYPES  = ["Accelerometer", "Gyroscope"]   # Magnetometer excluded
AXES          = ["X", "Y", "Z"]

SENSOR_COLS = [
    f"{seg}_{st}.{ax}"
    for seg in BODY_SEGMENTS
    for st  in SENSOR_TYPES
    for ax  in AXES
]                                    # 18 channels  (3 segs × 2 sensor types × 3 axes)

STATIC_FEATS = ["Gender_Bin", "BMI", "HWR"]

# Window sizes in seconds
HALF_WINDOW_SEC  = 30   # ±30 s around interior RPE timestamps
EDGE_WINDOW_SEC  = 60   # 60 s one-sided for t=0 and t=45 edges


# ── helpers ────────────────────────────────────────────────────────────────────
def rpe_to_label(v: float) -> str:
    """Borg CR-10 → 3-class string label."""
    if v <= 3:
        return "Low"
    elif v <= 6:
        return "Moderate"
    return "High"


def label_to_int(label: str) -> int:
    return RPE_LABELS.index(label)


def _get_window(df_sess: pd.DataFrame,
                rpe_ts: float,
                elapsed_min: float,
                session_dur: float = 45.0) -> pd.DataFrame:
    """
    Return sensor rows within the appropriate time window around an RPE event.
      • Edge intervals (t ≤ 0 or t ≥ session_dur) → 60-sec one-sided window
      • Interior intervals                         → ±30 sec
    """
    is_edge = (elapsed_min <= 0.0) or (elapsed_min >= session_dur)
    half    = EDGE_WINDOW_SEC if is_edge else HALF_WINDOW_SEC
    return df_sess[
        (df_sess["Timestamp"] >= rpe_ts - half) &
        (df_sess["Timestamp"] <= rpe_ts + half)
    ]


def _extract_features(window: pd.DataFrame,
                      available_cols: list) -> dict:
    """Compute mean, std, RMS for every available sensor column."""
    feats = {}
    for col in available_cols:
        sig = window[col].dropna().values
        if len(sig) == 0:
            feats[f"{col}_Mean"] = np.nan
            feats[f"{col}_Std"]  = np.nan
            feats[f"{col}_RMS"]  = np.nan
        else:
            feats[f"{col}_Mean"] = float(np.mean(sig))
            feats[f"{col}_Std"]  = float(np.std(sig))
            feats[f"{col}_RMS"]  = float(np.sqrt(np.mean(sig ** 2)))
    return feats


# ── public API ─────────────────────────────────────────────────────────────────
def load_and_clean(data_path: str = "NIOSH_Combined_Dataset.pkl"):
    """
    Load the NIOSH pickle and return the three main objects.

    Returns
    -------
    ts_data             : dict  {(subject, session) → DataFrame}
    anthro_clean        : pd.DataFrame
    experiment_settings : dict
    """
    with open(data_path, "rb") as f:
        raw = pickle.load(f)
    return raw["ts_data"], raw["anthro_clean"], raw["experiment_settings"]


def build_dataset(ts_data: dict,
                  anthro_clean: pd.DataFrame = None,
                  cache_path: str = None):
    """
    Extract windowed features for every subject-session pair.

    Parameters
    ----------
    ts_data      : dict {(subject, session) → DataFrame}
    anthro_clean : if provided, appends Gender_Bin, BMI, HWR
    cache_path   : optional path to save/reload the computed arrays

    Returns
    -------
    X          : np.ndarray  (n_samples, n_features)
    y          : np.ndarray  (n_samples,)  int  0=Low 1=Moderate 2=High
    subjects   : np.ndarray  (n_samples,)  str  subject IDs for LOSO grouping
    feat_names : list[str]   feature column names matching X
    """
    # ── try cache ──────────────────────────────────────────────────────────────
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached dataset from {cache_path} …")
        d = np.load(cache_path, allow_pickle=True)
        return (d["X"].astype(float),
                d["y"].astype(int),
                d["subjects"].astype(str),
                list(d["feat_names"]))

    # ── build anthropometric lookup ────────────────────────────────────────────
    anthro_lookup = {}
    if anthro_clean is not None:
        ac = anthro_clean.copy()
        ac["BMI"]        = ac["Weight (kg)"] / ((ac["Height (cm)"] / 100) ** 2)
        ac["HWR"]        = (ac["Waist circumference (cm)"] /
                            ac["Hip circumference (cm)"])
        ac["Gender_Bin"] = ac["Gender"].map({"M": 1, "F": 0})
        anthro_lookup = ac.set_index("Subject")[STATIC_FEATS].to_dict("index")

    # ── extract features ───────────────────────────────────────────────────────
    rows = []
    for (sub, sess), df in ts_data.items():
        if sub in EXCLUDE_SUBJECTS:
            continue

        df_s = df[df["Session"].astype(str).str.strip() == sess].copy()
        if df_s.empty:
            continue

        avail = [c for c in SENSOR_COLS if c in df_s.columns]
        if not avail:
            continue

        t0 = df_s["Timestamp"].min()
        df_s["_elapsed_min"] = (df_s["Timestamp"] - t0) / 60.0

        rpe_rows = df_s.dropna(subset=["RPE_Val"])
        for _, r in rpe_rows.iterrows():
            win = _get_window(df_s, r["Timestamp"], r["_elapsed_min"])
            if len(win) < 10:
                continue

            feat = {"_subject": sub, "_session": sess,
                    "_elapsed_min": r["_elapsed_min"],
                    "RPE_Target": r["RPE_Val"]}
            feat.update(_extract_features(win, avail))

            if sub in anthro_lookup:
                feat.update(anthro_lookup[sub])

            rows.append(feat)

    feat_df = pd.DataFrame(rows)

    # ── assemble X, y, subjects ────────────────────────────────────────────────
    meta_cols   = {"_subject", "_session", "_elapsed_min", "RPE_Target"}
    sensor_feat = [c for c in feat_df.columns if c not in meta_cols]
    feat_df     = feat_df.dropna(subset=sensor_feat + ["RPE_Target"])

    feat_df["_label"] = feat_df["RPE_Target"].apply(rpe_to_label)
    feat_df["_y"]     = feat_df["_label"].apply(label_to_int)

    X        = feat_df[sensor_feat].values.astype(float)
    y        = feat_df["_y"].values.astype(int)
    subjects = feat_df["_subject"].values.astype(str)

    print(f"Dataset: {X.shape[0]} samples | "
          f"{X.shape[1]} features | "
          f"{len(np.unique(subjects))} subjects")
    print("Class distribution:",
          dict(zip(*np.unique(y, return_counts=True))))

    if cache_path:
        np.savez(cache_path, X=X, y=y, subjects=subjects,
                 feat_names=np.array(sensor_feat))
        print(f"Cached → {cache_path}")

    return X, y, subjects, sensor_feat