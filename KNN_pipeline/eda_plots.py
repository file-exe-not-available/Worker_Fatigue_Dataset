"""
eda_plots.py — EDA visualisations for the NIOSH order-picking dataset
======================================================================
SELF-CONTAINED: no imports from pipeline.py or data_utils.py

Produces
  Plot 1  — Sensor signal vs elapsed time  (2x2, one panel per session)
  Plot 2  — RPE vs elapsed time            (2x2, one panel per session)
  Table 3 — 5-number summary + IQR        (printed + CSV)
  Plot 4  — Feature–RPE correlation bar chart

Run:
    python eda_plots.py
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these if needed
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH        = "NIOSH_Combined_Dataset.pkl"
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

HALF_WINDOW_SEC = 30    # ±30 s around interior RPE points
EDGE_WINDOW_SEC = 60    # 60 s one-sided for t=0 and t=45

# Representative signal for Plot 1
SIGNAL_COL = "upper_arm_Accelerometer.Z"


# ─────────────────────────────────────────────────────────────────────────────
# 0.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
FIGURES_DIR.mkdir(exist_ok=True)

print(f"Loading {DATA_PATH} …")
with open(DATA_PATH, "rb") as f:
    raw = pickle.load(f)

ts_data      = raw["ts_data"]
anthro_clean = raw["anthro_clean"]

# filtered, sorted subject list
all_subjects = sorted({k[0] for k in ts_data.keys()})
subjects     = [s for s in all_subjects if s not in EXCLUDE_SUBJECTS]
print(f"  Subjects (after exclusion): {subjects}\n")

colors         = plt.cm.viridis(np.linspace(0, 1, len(subjects)))
subject_colors = dict(zip(subjects, colors))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PLOT 1 — SENSOR SIGNAL vs ELAPSED TIME
#     2×2 grid, one panel per session (Session1–4)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Plot 1 — Sensor vs Time …")

fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=False)
axes1 = axes1.flatten()

for idx, sess in enumerate(SESSIONS_LIST):
    ax      = axes1[idx]
    plotted = 0

    for sub in subjects:
        key = (sub, sess)
        if key not in ts_data:
            continue

        df = ts_data[key].copy()

        # keep only rows whose Session column matches
        sess_mask = df["Session"].astype(str).str.strip() == sess
        df_s = df[sess_mask].copy()
        if df_s.empty:
            continue

        if SIGNAL_COL not in df_s.columns:
            continue

        t0 = df_s["Timestamp"].min()
        df_s["Elapsed_Min"] = (df_s["Timestamp"] - t0) / 60.0

        # keep first 45 min
        df_45 = df_s[
            (df_s["Elapsed_Min"] >= 0) &
            (df_s["Elapsed_Min"] <= 45)
        ].copy()

        if df_45.empty:
            continue

        # downsample to mean per 10-sec block so lines are clean
        df_45["TimeBlock"] = (df_45["Elapsed_Min"] * 6).round() / 6.0
        ds = (df_45.groupby("TimeBlock")[SIGNAL_COL]
                   .mean()
                   .reset_index())

        ax.plot(
            ds["TimeBlock"], ds[SIGNAL_COL],
            color=subject_colors[sub], alpha=0.80,
            linewidth=1.3, label=sub
        )
        plotted += 1

    ax.set_title(f"Session {idx + 1}", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(0, 46, 5))
    ax.set_xlim(-0.5, 45.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    print(f"  Session {idx + 1}: {plotted} subject(s) plotted")

# axis labels
axes1[0].set_ylabel("Upper-Arm Acc Z  (g)", fontsize=10)
axes1[2].set_ylabel("Upper-Arm Acc Z  (g)", fontsize=10)
axes1[2].set_xlabel("Elapsed Time (min)", fontsize=10)
axes1[3].set_xlabel("Elapsed Time (min)", fontsize=10)

# shared legend
handles, lbs = axes1[0].get_legend_handles_labels()
by_lbl = {l: h for l, h in zip(lbs, handles)}
fig1.legend(
    [by_lbl[l] for l in sorted(by_lbl)],
    sorted(by_lbl),
    loc="center right", bbox_to_anchor=(1.09, 0.5),
    title="Subjects", fontsize=9
)
fig1.suptitle(
    f"Sensor Signal Over Time — {SIGNAL_COL}\n(magnetometer excluded)",
    fontsize=14, fontweight="bold", y=0.98
)
plt.tight_layout()
plt.subplots_adjust(wspace=0.22, hspace=0.28, top=0.90)
out1 = FIGURES_DIR / "plot1_sensor_vs_time.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved: {out1}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PLOT 2 — RPE vs ELAPSED TIME
#     2×2 grid, one panel per session
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Plot 2 — RPE vs Time …")

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
axes2 = axes2.flatten()

for idx, sess in enumerate(SESSIONS_LIST):
    ax      = axes2[idx]
    plotted = 0

    for sub in subjects:
        key = (sub, sess)
        if key not in ts_data:
            continue

        df   = ts_data[key].copy()
        df_s = df[df["Session"].astype(str).str.strip() == sess].copy()
        if df_s.empty:
            continue

        t0 = df_s["Timestamp"].min()
        df_s["Elapsed_Min"] = (df_s["Timestamp"] - t0) / 60.0

        df_45  = df_s[
            (df_s["Elapsed_Min"] >= 0) &
            (df_s["Elapsed_Min"] <= 45)
        ].copy()

        df_rpe = df_45.dropna(subset=["RPE_Val"]).copy()
        if df_rpe.empty:
            continue

        # snap to nearest 5-min mark
        df_rpe["Interval_Min"] = (df_rpe["Elapsed_Min"] / 5).round() * 5
        grp = (
            df_rpe.groupby("Interval_Min")["RPE_Val"]
                  .mean()
                  .reset_index()
        )
        grp = grp[grp["Interval_Min"] <= 45]

        # known artifact rows from original notebook
        if (sub == "Sub13" and sess == "Session2") or \
           (sub == "Sub03" and sess == "Session3"):
            grp = grp.iloc[:-1]

        if grp.empty:
            continue

        ax.plot(
            grp["Interval_Min"], grp["RPE_Val"],
            "-o", color=subject_colors[sub],
            alpha=0.90, markersize=4, linewidth=1.75, label=sub
        )
        plotted += 1

    ax.set_title(f"Session {idx + 1}", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(0, 46, 5))
    ax.set_xlim(-1, 46)
    ax.set_ylim(-0.3, 11)
    ax.grid(True, linestyle="--", alpha=0.4)
    print(f"  Session {idx + 1}: {plotted} subject(s) plotted")

axes2[0].set_ylabel("Perceived Exertion (Borg CR-10)", fontsize=10)
axes2[2].set_ylabel("Perceived Exertion (Borg CR-10)", fontsize=10)
axes2[2].set_xlabel("Elapsed Time (min)", fontsize=10)
axes2[3].set_xlabel("Elapsed Time (min)", fontsize=10)

handles, lbs = axes2[0].get_legend_handles_labels()
by_lbl = {l: h for l, h in zip(lbs, handles)}
fig2.legend(
    [by_lbl[l] for l in sorted(by_lbl)],
    sorted(by_lbl),
    loc="center right", bbox_to_anchor=(1.09, 0.5),
    title="Subjects", fontsize=9
)
fig2.suptitle(
    "RPE Progression Over Time  (5-min intervals, one panel per session)",
    fontsize=14, fontweight="bold", y=0.98
)
plt.tight_layout()
plt.subplots_adjust(wspace=0.18, hspace=0.28, top=0.90)
out2 = FIGURES_DIR / "plot2_rpe_vs_time.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved: {out2}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE EXTRACTION  (for summary statistics and correlation plot)
#     ±30 sec window around every RPE timestamp; edges get 60 sec
# ─────────────────────────────────────────────────────────────────────────────
print("Extracting windowed features for summary statistics …")

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

    rpe_pts = df_s.dropna(subset=["RPE_Val"])

    for _, r in rpe_pts.iterrows():
        el  = r["_elapsed_min"]
        ts  = r["Timestamp"]
        rpe = r["RPE_Val"]

        is_edge = (el <= 0) or (el >= 45)
        half    = EDGE_WINDOW_SEC if is_edge else HALF_WINDOW_SEC
        win     = df_s[
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

        rows.append(row)

feat_df = pd.DataFrame(rows)
print(f"  {len(feat_df)} feature rows from "
      f"{feat_df['Subject'].nunique()} subjects\n")

sensor_feat_cols = [
    c for c in feat_df.columns
    if any(f"{seg}_{st}" in c
           for seg in BODY_SEGMENTS
           for st  in SENSOR_TYPES)
    and c.endswith(("_Mean", "_Std", "_RMS"))
]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  5-NUMBER SUMMARY + IQR
# ─────────────────────────────────────────────────────────────────────────────
print("── 5-Number Summary + IQR (sensor features, no magnetometer) ──\n")

summary_records = []
for col in sensor_feat_cols:
    vals = feat_df[col].dropna()
    q1, q3 = np.percentile(vals, [25, 75])
    summary_records.append({
        "Feature": col,
        "Min":     round(vals.min(),    4),
        "Q1":      round(q1,            4),
        "Median":  round(vals.median(), 4),
        "Q3":      round(q3,            4),
        "Max":     round(vals.max(),    4),
        "IQR":     round(q3 - q1,      4),
        "Mean":    round(vals.mean(),   4),
        "Std":     round(vals.std(),    4),
    })

summary_df = pd.DataFrame(summary_records)
pd.set_option("display.max_rows",  None)
pd.set_option("display.width",     220)
pd.set_option("display.max_colwidth", 55)
print(summary_df.to_string(index=False))

csv_path = FIGURES_DIR / "sensor_5num_summary.csv"
summary_df.to_csv(csv_path, index=False)
print(f"\n  → Saved: {csv_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOT 4 — FEATURE CORRELATION WITH RPE
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Plot 4 — Feature correlation with RPE …")

feat_df["RPE_Target"] = pd.to_numeric(feat_df["RPE_Target"], errors="coerce")
corr_series = (
    feat_df[sensor_feat_cols]
    .corrwith(feat_df["RPE_Target"])
    .sort_values(ascending=False)
)

top_pos  = corr_series.head(10)
top_neg  = corr_series.tail(10)
combined = pd.concat([top_pos, top_neg])

fig4, ax4 = plt.subplots(figsize=(10, 7))
bar_colors = ["#E63946" if v > 0 else "#1D3557" for v in combined.values]
ax4.barh(combined.index[::-1], combined.values[::-1],
         color=bar_colors[::-1])
ax4.axvline(0, color="black", linewidth=0.8)
ax4.set_xlabel("Pearson Correlation with RPE", fontsize=11)
ax4.set_title(
    "Top 10 Positive & Negative Feature Correlations with RPE\n"
    "(±30 sec window, magnetometer excluded)",
    fontsize=12, fontweight="bold"
)
ax4.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
out4 = FIGURES_DIR / "plot4_feature_correlation.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved: {out4}")

# save feature matrix so knn_model.py can reuse it
feat_df.to_csv("features_extracted.csv", index=False)
print("\n  Feature matrix saved → features_extracted.csv")
print("\nEDA complete.")