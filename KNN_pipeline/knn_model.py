"""
knn_model.py — K-Nearest Neighbours fatigue classifier
=======================================================
SELF-CONTAINED: no imports from pipeline.py or data_utils.py

Reads features_extracted.csv produced by eda_plots.py, OR
re-extracts features directly from the pickle if the CSV is missing.

k is tuned via RandomizedSearchCV with GroupKFold (no subject leakage).
Evaluation: Leave-One-Subject-Out (LOSO) cross-validation.

Usage:
    python knn_model.py
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import RandomizedSearchCV, GroupKFold

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH        = "NIOSH_Combined_Dataset.pkl"
FEATURES_CSV     = "features_extracted.csv"     # written by eda_plots.py
FIGURES_DIR      = Path("figures")

EXCLUDE_SUBJECTS = {"Sub11", "Sub12", "Sub14"}
SESSIONS_LIST    = ["Session1", "Session2", "Session3", "Session4"]

BODY_SEGMENTS    = ["trunk", "upper_arm", "wrist"]
SENSOR_TYPES     = ["Accelerometer", "Gyroscope"]   # Magnetometer excluded
AXES             = ["X", "Y", "Z"]

SENSOR_COLS = [
    f"{seg}_{st}.{ax}"
    for seg in BODY_SEGMENTS
    for st  in SENSOR_TYPES
    for ax  in AXES
]

STATIC_FEATS     = ["Gender_Bin", "BMI", "HWR"]
RPE_LABELS       = ["Low", "Moderate", "High"]

HALF_WINDOW_SEC  = 30
EDGE_WINDOW_SEC  = 60


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def rpe_to_label(v: float) -> str:
    """Borg CR-10 → 3-class string label."""
    if v <= 3:
        return "Low"
    elif v <= 6:
        return "Moderate"
    return "High"


def rpe_to_int(v: float) -> int:
    return RPE_LABELS.index(rpe_to_label(v))


def extract_features_from_pickle(data_path: str,
                                  anthro_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: extract windowed features directly from the pickle.
    Called only when features_extracted.csv does not exist.
    """
    print("  Extracting features from pickle (run eda_plots.py first to cache) …")
    with open(data_path, "rb") as f:
        raw = pickle.load(f)
    ts_data = raw["ts_data"]

    # anthropometric lookup
    ac = anthro_clean.copy()
    ac["BMI"]        = ac["Weight (kg)"] / ((ac["Height (cm)"] / 100) ** 2)
    ac["HWR"]        = ac["Waist circumference (cm)"] / ac["Hip circumference (cm)"]
    ac["Gender_Bin"] = ac["Gender"].map({"M": 1, "F": 0})
    anthro_lkp = ac.set_index("Subject")[STATIC_FEATS].to_dict("index")

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
        df_s["_el"] = (df_s["Timestamp"] - t0) / 60.0

        for _, r in df_s.dropna(subset=["RPE_Val"]).iterrows():
            el   = r["_el"]
            ts   = r["Timestamp"]
            rpe  = r["RPE_Val"]
            half = EDGE_WINDOW_SEC if (el <= 0 or el >= 45) else HALF_WINDOW_SEC
            win  = df_s[
                (df_s["Timestamp"] >= ts - half) &
                (df_s["Timestamp"] <= ts + half)
            ]
            if len(win) < 10:
                continue
            row = {"Subject": sub, "Session": sess,
                   "Elapsed_Min": el, "RPE_Target": rpe}
            for col in avail:
                sig = win[col].dropna().values
                if len(sig) == 0:
                    continue
                row[f"{col}_Mean"] = float(np.mean(sig))
                row[f"{col}_Std"]  = float(np.std(sig))
                row[f"{col}_RMS"]  = float(np.sqrt(np.mean(sig ** 2)))
            if sub in anthro_lkp:
                row.update(anthro_lkp[sub])
            rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  LOAD / BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
FIGURES_DIR.mkdir(exist_ok=True)

if Path(FEATURES_CSV).exists():
    print(f"Loading feature matrix from {FEATURES_CSV} …")
    feat_df = pd.read_csv(FEATURES_CSV)
else:
    print(f"{FEATURES_CSV} not found — extracting from pickle …")
    with open(DATA_PATH, "rb") as f:
        raw = pickle.load(f)
    feat_df = extract_features_from_pickle(DATA_PATH, raw["anthro_clean"])

    # build anthropometric columns if missing
    if "BMI" not in feat_df.columns:
        ac = raw["anthro_clean"].copy()
        ac["BMI"]        = ac["Weight (kg)"] / ((ac["Height (cm)"] / 100) ** 2)
        ac["HWR"]        = (ac["Waist circumference (cm)"] /
                            ac["Hip circumference (cm)"])
        ac["Gender_Bin"] = ac["Gender"].map({"M": 1, "F": 0})
        anthro_lkp = ac.set_index("Subject")[STATIC_FEATS].to_dict("index")
        for col in STATIC_FEATS:
            feat_df[col] = feat_df["Subject"].map(
                lambda s: anthro_lkp.get(s, {}).get(col, np.nan)
            )

# ── add anthropometric columns from pickle if still missing ──────────────────
if "BMI" not in feat_df.columns or feat_df["BMI"].isna().all():
    with open(DATA_PATH, "rb") as f:
        raw = pickle.load(f)
    ac = raw["anthro_clean"].copy()
    ac["BMI"]        = ac["Weight (kg)"] / ((ac["Height (cm)"] / 100) ** 2)
    ac["HWR"]        = (ac["Waist circumference (cm)"] /
                        ac["Hip circumference (cm)"])
    ac["Gender_Bin"] = ac["Gender"].map({"M": 1, "F": 0})
    for col in STATIC_FEATS:
        feat_df[col] = feat_df["Subject"].map(
            ac.set_index("Subject")[col]
        )

# ── fatigue class labels ──────────────────────────────────────────────────────
feat_df["RPE_Target"]    = pd.to_numeric(feat_df["RPE_Target"], errors="coerce")
feat_df["Fatigue_Class"] = feat_df["RPE_Target"].apply(rpe_to_label)
feat_df["Fatigue_Int"]   = feat_df["RPE_Target"].apply(rpe_to_int)

print(f"  Rows    : {len(feat_df)}")
print(f"  Subjects: {feat_df['Subject'].nunique()}")
print(f"  Class distribution:\n{feat_df['Fatigue_Class'].value_counts()}\n")

# ── feature column list ───────────────────────────────────────────────────────
sensor_feat_cols = [
    c for c in feat_df.columns
    if any(f"{seg}_{st}" in c
           for seg in BODY_SEGMENTS
           for st  in SENSOR_TYPES)
    and c.endswith(("_Mean", "_Std", "_RMS"))
]
static_available = [c for c in STATIC_FEATS if c in feat_df.columns]
FEATURE_COLS     = sensor_feat_cols + static_available

feat_df_clean = feat_df.dropna(subset=FEATURE_COLS + ["Fatigue_Int"]).copy()
X        = feat_df_clean[FEATURE_COLS].values.astype(float)
y        = feat_df_clean["Fatigue_Int"].values.astype(int)
subjects = feat_df_clean["Subject"].values.astype(str)

print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  K-SENSITIVITY SWEEP  (diagnostic)
# ─────────────────────────────────────────────────────────────────────────────
def k_sensitivity_plot(X, y, subjects, out_dir):
    print("Running k-sensitivity sweep (k = 1 … 21, odd values) …")
    cv       = GroupKFold(n_splits=5)
    k_vals   = list(range(1, 22, 2))
    f1_means = []

    for k in k_vals:
        pipe = Pipeline([
            ("sc",  StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=k, weights="distance", n_jobs=-1
            )),
        ])
        fold_f1s = [
            f1_score(
                y[ti],
                pipe.fit(X[tri], y[tri]).predict(X[ti]),
                average="macro", zero_division=0
            )
            for tri, ti in cv.split(X, y, groups=subjects)
        ]
        f1_means.append(np.mean(fold_f1s))

    best_k = k_vals[int(np.argmax(f1_means))]
    print(f"  Best k in sweep: {best_k}  "
          f"(mean macro-F1 = {max(f1_means):.3f})")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_vals, f1_means, "-o", color="#1D3557",
            linewidth=2, markersize=6)
    ax.axvline(best_k, color="#E63946", linestyle="--",
               label=f"Best k = {best_k}")
    ax.set_xlabel("k (number of neighbours)", fontsize=11)
    ax.set_ylabel("Mean Macro F1 (5-fold GroupKFold)", fontsize=11)
    ax.set_title("KNN Sensitivity to k", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = out_dir / "knn_k_sensitivity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
def tune_hyperparameters(X, y, subjects):
    print("Tuning KNN hyperparameters "
          "(GroupKFold 5-fold, 20 iterations) …")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier(n_jobs=-1)),
    ])
    param_dist = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "knn__weights":     ["uniform", "distance"],
        "knn__metric":      ["euclidean", "manhattan", "minkowski"],
    }
    search = RandomizedSearchCV(
        pipe, param_dist,
        n_iter=20, scoring="f1_macro",
        cv=GroupKFold(n_splits=5),
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X, y, groups=subjects)
    best = {k.replace("knn__", ""): v
            for k, v in search.best_params_.items()
            if k.startswith("knn__")}
    print(f"  Best params : {best}")
    print(f"  Best CV F1  : {search.best_score_:.3f}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LEAVE-ONE-SUBJECT-OUT CV
# ─────────────────────────────────────────────────────────────────────────────
def loso_cv(X, y, subjects, params):
    unique_subjs    = np.unique(subjects)
    all_true, all_pred = [], []

    for test_subj in unique_subjs:
        mask       = subjects == test_subj
        X_tr, y_tr = X[~mask], y[~mask]
        X_te, y_te = X[mask],  y[mask]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        knn = KNeighborsClassifier(n_jobs=-1, **params)
        knn.fit(X_tr, y_tr)
        y_pred = knn.predict(X_te)

        fold_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        print(f"  {test_subj}: macro-F1 = {fold_f1:.3f}  "
              f"(n={mask.sum()})")
        all_true.extend(y_te)
        all_pred.extend(y_pred)

    return np.array(all_true), np.array(all_pred)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SAVE CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, out_dir):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=RPE_LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("KNN — LOSO Confusion Matrix")
    plt.tight_layout()
    path = out_dir / "knn_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE PER-SUBJECT F1 BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def save_per_subject_f1(y_true, y_pred, subjects, params, out_dir):
    unique_subjs = np.unique(subjects)
    f1_scores    = [
        f1_score(y_true[subjects == s], y_pred[subjects == s],
                 average="macro", zero_division=0)
        for s in unique_subjs
    ]
    overall = f1_score(y_true, y_pred, average="macro", zero_division=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_col = ["#E63946" if f < 0.5 else "#1D3557" for f in f1_scores]
    ax.bar(unique_subjs, f1_scores, color=bar_col, edgecolor="white")
    ax.axhline(overall, color="black", linestyle="--", linewidth=1.5,
               label=f"Overall Macro F1 = {overall:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Subject", fontsize=11)
    ax.set_ylabel("Macro F1-Score", fontsize=11)
    ax.set_title(
        f"Per-Subject LOSO Macro F1 — KNN "
        f"(k={params.get('n_neighbors')}, "
        f"metric={params.get('metric')})",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = out_dir / "knn_per_subject_f1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    k_sensitivity_plot(X, y, subjects, FIGURES_DIR)

    best_params = tune_hyperparameters(X, y, subjects)

    n_subj = len(np.unique(subjects))
    print(f"\nRunning LOSO CV ({n_subj} folds) …")
    y_true, y_pred = loso_cv(X, y, subjects, best_params)

    print("\n=== KNN — LOSO Results ===")
    print(classification_report(
        y_true, y_pred,
        target_names=RPE_LABELS, zero_division=0
    ))
    print(f"Overall macro-F1 : "
          f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
    print(f"Overall accuracy : {(y_true == y_pred).mean():.3f}")

    save_confusion_matrix(y_true, y_pred, FIGURES_DIR)
    save_per_subject_f1(y_true, y_pred, subjects, best_params, FIGURES_DIR)

    print("\nDone.")