"""
Fatigue Dataset Visualization Script
Dataset: Human Fatigue and Recovery Modeling During a Controlled Order-Picking Experiment

Usage:
    python visualize_fatigue_dataset.py --data /path/to/dataset.pkl
"""

import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

_COLOR_PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0",
                  "#00BCD4", "#795548"]


def _build_rpe_maps(all_rpe_series):
    """Build tick-label and color dicts from unique RPE values in the data."""
    values = sorted(all_rpe_series.dropna().unique().astype(int))
    labels = {v: str(v) for v in values}
    colors = {v: _COLOR_PALETTE[i % len(_COLOR_PALETTE)] for i, v in enumerate(values)}
    return labels, colors


def _get_condition(subj, sess, settings):
    """Return (weight, pace) tuple for a given subject/session, or None."""
    if subj in settings:
        s = settings[subj]
        try:
            return (s.loc[sess, 'Weight'], s.loc[sess, 'Pace'])
        except (KeyError, AttributeError):
            pass
    return None


def _group_by_condition(ts_data, settings):
    """Return {(weight, pace): [(subj, sess), ...]} mapping."""
    cond_map = defaultdict(list)
    for subj, sess in ts_data.keys():
        cond = _get_condition(subj, sess, settings)
        if cond:
            cond_map[cond].append((subj, sess))
    return cond_map


def _cond_label(weight, pace):
    return f"Load {weight} kg, Pace {int(pace)} picks/min"


def load_data(path):
    print(f"Loading dataset from {path} ...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def clean_data(ts_data, anthro_df):
    """
    Remove excluded subjects and fix the non-monotone RPE tail in Sub13/Session2.

    Sub11, Sub12, Sub14 are excluded (no valid data / per protocol).
    Sub13/Session2: RPE drops from its peak back to 4 in the final block —
    this is a recording artefact, so everything after the last peak-RPE row
    is discarded.
    """
    exclude = {'Sub11', 'Sub12', 'Sub14'}
    ts_data = {k: v for k, v in ts_data.items() if k[0] not in exclude}
    anthro_df = anthro_df[~anthro_df['Subject'].isin(exclude)].reset_index(drop=True)

    key = ('Sub13', 'Session2')
    if key in ts_data:
        df = ts_data[key]
        rpe_nonnull = df[df['RPE_Val'].notna()]
        max_rpe = rpe_nonnull['RPE_Val'].max()
        last_max_row = rpe_nonnull[rpe_nonnull['RPE_Val'] == max_rpe].index[-1]
        ts_data[key] = df.loc[:last_max_row].copy()
        print(f"  Sub13/Session2: trimmed to row {last_max_row} (removed trailing RPE drop)")

    return ts_data, anthro_df


def describe_structure(data):
    ts = data["ts_data"]
    print("\n=== Dataset Structure (raw, before cleaning) ===")
    print(f"Total participant-session pairs: {len(ts)}")
    subjects = sorted(set(k[0] for k in ts.keys()))
    sessions = sorted(set(k[1] for k in ts.keys()))
    print(f"Subjects ({len(subjects)}): {subjects}")
    print(f"Sessions ({len(sessions)}): {sessions}")

    example_key = list(ts.keys())[0]
    df = ts[example_key]
    rpe_nonnull = df['RPE_Val'].dropna()
    print(f"\nExample key: {example_key}")
    print(f"  Total rows: {len(df):,}")
    print(f"  RPE rows: {len(rpe_nonnull):,}")
    print(f"  RPE unique values: {sorted(rpe_nonnull.unique())}")
    print(f"  RPE value counts:\n{rpe_nonnull.value_counts().sort_index()}")

    anthro = data["anthro_clean"]
    print(f"\nanthro_clean ({len(anthro)} participants):")
    print(anthro.to_string())


# ── Plot 1: RPE trajectories — four separate figures, one per condition ───────

def plot_rpe_trajectories(ts_data, experiment_settings, out_dir):
    """
    One PNG per experimental condition showing RPE vs elapsed time for every
    subject assigned to that condition.
    """
    cond_map = _group_by_condition(ts_data, experiment_settings)

    for cond in sorted(cond_map.keys()):
        weight, pace = cond
        sessions = sorted(cond_map[cond])

        n_cols = 4
        n_rows = int(np.ceil(len(sessions) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.4),
                                  sharey=True, squeeze=False)
        axes_flat = axes.flatten()

        for i, (subj, sess) in enumerate(sessions):
            ax = axes_flat[i]
            df = ts_data[(subj, sess)]
            rpe_df = df[df['RPE_Val'].notna()].copy()
            if rpe_df.empty:
                ax.set_visible(False)
                continue

            t_min = (rpe_df['Timestamp'] - df['Timestamp'].iloc[0]) / 60

            ax.plot(t_min, rpe_df['RPE_Val'], linewidth=1.8,
                    marker='o', markersize=3, alpha=0.75, color='#2196F3')
            ax.set_title(subj, fontsize=9, fontweight='bold')
            ax.set_ylim(-0.5, 10.5)
            ax.set_yticks(range(0, 11, 2))
            ax.set_xlabel("Time (min)", fontsize=8)
            ax.grid(True, alpha=0.3)

        for row in range(n_rows):
            axes[row, 0].set_ylabel("RPE", fontsize=9)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"RPE Trajectories — {_cond_label(weight, pace)}\n"
            "(each panel = one participant)",
            fontsize=12, y=1.01
        )
        plt.tight_layout()

        fname = f"01_rpe_traj_load{weight}kg_pace{int(pace)}.png"
        path = out_dir / fname
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


# ── Plot 2: RPE distribution ──────────────────────────────────────────────────

def plot_rpe_distribution(ts_data, out_dir):
    """Bar chart of RPE distribution overall and per subject."""
    all_rpe = pd.concat(
        [df['RPE_Val'].dropna() for df in ts_data.values()],
        ignore_index=True
    )
    all_counts = all_rpe.value_counts().sort_index()
    rpe_labels, rpe_colors = _build_rpe_maps(all_rpe)

    subjects = sorted(set(k[0] for k in ts_data.keys()))
    subj_data = []
    for subj in subjects:
        vals = pd.concat(
            [ts_data[k]['RPE_Val'].dropna() for k in ts_data if k[0] == subj],
            ignore_index=True
        )
        counts = vals.value_counts().sort_index()
        total = counts.sum()
        row = {"subject": subj}
        for v in sorted(rpe_labels.keys()):
            row[rpe_labels[v]] = counts.get(v, 0) / total * 100
        subj_data.append(row)
    df_subj = pd.DataFrame(subj_data).set_index("subject")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = [rpe_colors[int(v)] for v in all_counts.index]
    bars = ax1.bar(
        [rpe_labels[int(v)] for v in all_counts.index],
        all_counts.values,
        color=colors, edgecolor='white', linewidth=1.2
    )
    for bar, cnt in zip(bars, all_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{cnt:,}", ha='center', va='bottom', fontsize=9)
    ax1.set_title("Overall RPE Distribution", fontsize=11)
    ax1.set_ylabel("Number of RPE readings")
    ax1.set_xlabel("RPE")

    class_cols = list(rpe_labels.values())
    available_cols = [c for c in class_cols if c in df_subj.columns]
    bar_colors = [rpe_colors[k] for k, v in rpe_labels.items() if v in available_cols]
    df_subj[available_cols].plot(
        kind='bar', stacked=True, ax=ax2,
        color=bar_colors, edgecolor='white', linewidth=0.8
    )
    ax2.set_title("RPE Distribution per Subject (%)", fontsize=11)
    ax2.set_ylabel("% of RPE readings")
    ax2.set_xlabel("Subject")
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title="RPE", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    path = out_dir / "02_rpe_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 3: Between-subject RPE variability, separated by condition ───────────

def plot_between_subject_rpe_variability(ts_data, experiment_settings, out_dir):
    """
    Violin + box plot of RPE per subject, with one subplot per condition.
    Separating by condition reveals whether individual differences are
    consistent across task demands.
    """
    cond_map = _group_by_condition(ts_data, experiment_settings)
    sorted_conds = sorted(cond_map.keys())
    n_conds = len(sorted_conds)

    n_cols_layout = 2
    n_rows_layout = int(np.ceil(n_conds / n_cols_layout))
    fig, axes = plt.subplots(
        n_rows_layout, n_cols_layout,
        figsize=(7 * n_cols_layout, 5 * n_rows_layout),
        sharey=True
    )
    axes_flat = np.array(axes).flatten()

    all_rpe_pool = pd.concat(
        [df['RPE_Val'].dropna() for df in ts_data.values()], ignore_index=True
    )
    rpe_labels, _ = _build_rpe_maps(all_rpe_pool)

    for ax, cond in zip(axes_flat, sorted_conds):
        weight, pace = cond
        sessions = sorted(cond_map[cond])
        subjects = sorted(set(s[0] for s in sessions))

        rpe_per_subject = []
        valid_subjects = []
        for subj in subjects:
            vals = pd.concat(
                [ts_data[(s, se)]['RPE_Val'].dropna()
                 for (s, se) in sessions if s == subj],
                ignore_index=True
            ).values
            if len(vals) > 0:
                rpe_per_subject.append(vals)
                valid_subjects.append(subj)

        if not rpe_per_subject:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(rpe_per_subject, positions=range(len(valid_subjects)),
                              showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#90CAF9')
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('navy')

        ax.boxplot(rpe_per_subject, positions=range(len(valid_subjects)),
                   widths=0.15, patch_artist=False,
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(linewidth=1),
                   capprops=dict(linewidth=1),
                   flierprops=dict(marker='.', markersize=3, alpha=0.4))

        ax.set_xticks(range(len(valid_subjects)))
        ax.set_xticklabels(valid_subjects, rotation=45, fontsize=8)
        ax.set_title(_cond_label(weight, pace), fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

    for ax in axes_flat[:n_rows_layout * n_cols_layout:n_cols_layout]:
        ax.set_ylabel("RPE", fontsize=10)
        ax.set_yticks(sorted(rpe_labels.keys()))
        ax.set_yticklabels([rpe_labels[t] for t in sorted(rpe_labels.keys())])

    for j in range(n_conds, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Between-Subject RPE Variability by Condition\n"
        "(evidence for individual heterogeneity across task demands)",
        fontsize=12
    )
    plt.tight_layout()
    path = out_dir / "03_between_subject_rpe_variability.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 4: Anthropometrics ───────────────────────────────────────────────────

def plot_anthropometrics(anthro_df, out_dir):
    feature_candidates = ['Age', 'Height (cm)', 'Weight (kg)', 'Height', 'Weight']
    available = [f for f in feature_candidates if f in anthro_df.columns]
    # Deduplicate (prefer the '(cm)'/'(kg)' variants)
    seen_bases = set()
    features = []
    for f in available:
        base = f.split(' ')[0]
        if base not in seen_bases:
            features.append(f)
            seen_bases.add(base)

    gender_col = next((c for c in ['Gender', 'gender', 'Sex'] if c in anthro_df.columns), None)

    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 4))
    if len(features) == 1:
        axes = [axes]

    gender_colors = {'M': '#2196F3', 'F': '#E91E63', 'Male': '#2196F3', 'Female': '#E91E63'}

    for ax, feat in zip(axes, features):
        if gender_col:
            for gender, grp in anthro_df.groupby(gender_col):
                color = gender_colors.get(str(gender), '#9C27B0')
                ax.hist(grp[feat].dropna(), bins=8, alpha=0.7, color=color,
                        label=str(gender), edgecolor='white')
            ax.legend(title='Gender')
        else:
            ax.hist(anthro_df[feat].dropna(), bins=8, color='#2196F3',
                    alpha=0.8, edgecolor='white')
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        ax.set_title(f"{feat} Distribution")

    fig.suptitle("Participant Anthropometric Profiles", fontsize=11)
    plt.tight_layout()
    path = out_dir / "04_anthropometrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 5: Sample IMU signals ────────────────────────────────────────────────

def plot_sample_imu_signals(ts_data, out_dir):
    """Raw accelerometer signals for one subject/session with time in minutes."""
    key = list(ts_data.keys())[0]
    subj, sess = key
    df = ts_data[key]

    regions = ['trunk', 'upper_arm', 'wrist']
    region_labels = ['Trunk', 'Upper Arm', 'Wrist']
    axes_labels = ['X', 'Y', 'Z']

    fig, plot_axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    n_points = min(3000, len(df))
    idx = np.linspace(0, len(df) - 1, n_points, dtype=int)
    df_sub = df.iloc[idx]
    t_min = (df_sub['Timestamp'].values - df['Timestamp'].iloc[0]) / 60

    for ax, region, label in zip(plot_axes, regions, region_labels):
        for axis in axes_labels:
            col = f'{region}_Accelerometer.{axis}'
            if col not in df.columns:
                col = f'{region}_Acc_{axis}'
            if col in df.columns:
                ax.plot(t_min, df_sub[col].values, linewidth=0.6, alpha=0.85, label=axis)
        ax.set_ylabel(f'{label}\nAccel (g)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right', ncol=3, title='Axis')
        ax.grid(True, alpha=0.3)

    plot_axes[-1].set_xlabel("Time (min)")
    fig.suptitle(f"Raw IMU Accelerometer Signals — {subj}, {sess}", fontsize=11)
    plt.tight_layout()
    path = out_dir / "05_sample_imu_signals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 6: PS-PFL similarity matrix ─────────────────────────────────────────

def plot_profile_similarity_heatmap(anthro_df, out_dir):
    feature_candidates = ['Age', 'Height (cm)', 'Weight (kg)', 'Height', 'Weight']
    available = [f for f in feature_candidates if f in anthro_df.columns]
    seen_bases = set()
    features = []
    for f in available:
        base = f.split(' ')[0]
        if base not in seen_bases:
            features.append(f)
            seen_bases.add(base)

    gender_col = next((c for c in ['Gender', 'gender'] if c in anthro_df.columns), None)

    df = anthro_df.copy().reset_index(drop=True)
    if gender_col:
        df['Gender_bin'] = (df[gender_col].astype(str).str.upper().str.startswith('M')).astype(float)
        use_cols = ['Gender_bin'] + features
    else:
        use_cols = features

    profile = df[use_cols].dropna()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)
    norms = np.linalg.norm(profile_norm.values, axis=1, keepdims=True)
    p = profile_norm.values / (norms + 1e-8)
    sim = p @ p.T

    subj_col = next((c for c in df.columns if c == 'Subject' or 'sub' in c.lower()), None)
    labels = df[subj_col].values[:len(profile)] if subj_col else range(len(profile))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha='center', va='center',
                    fontsize=5.5, color='black' if sim[i, j] < 0.75 else 'white')

    ax.set_title(
        "Participant Profile Similarity Matrix\n"
        "(cosine similarity on normalised gender, age, height, weight)",
        fontsize=10
    )
    plt.tight_layout()
    path = out_dir / "06_profile_similarity_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 7: Mean RPE over elapsed time per condition ─────────────────────────

def plot_rpe_over_time_by_condition(ts_data, experiment_settings, out_dir):
    """
    Mean RPE (± 1 SD across participants) vs elapsed session time, one line
    per experimental condition.  Subjects are aligned on a normalised 0–1 time
    grid then converted back to minutes using the average session duration.
    """
    cond_map = _group_by_condition(ts_data, experiment_settings)
    palette = plt.cm.tab10(np.linspace(0, 1, len(cond_map)))
    N_GRID = 200

    fig, ax = plt.subplots(figsize=(10, 5))

    for color, cond in zip(palette, sorted(cond_map.keys())):
        weight, pace = cond
        traces = []
        session_durations = []

        for subj, sess in cond_map[cond]:
            df = ts_data[(subj, sess)]
            rpe_df = df[df['RPE_Val'].notna()].copy()
            if rpe_df.empty:
                continue
            t = (rpe_df['Timestamp'].values - df['Timestamp'].iloc[0]) / 60
            rpe = rpe_df['RPE_Val'].values
            t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
            grid = np.linspace(0, 1, N_GRID)
            traces.append(np.interp(grid, t_norm, rpe))
            session_durations.append(t.max())

        if not traces:
            continue

        avg_duration = float(np.mean(session_durations))
        t_axis = np.linspace(0, avg_duration, N_GRID)
        arr = np.array(traces)
        mean_rpe = arr.mean(axis=0)
        std_rpe = arr.std(axis=0)

        ax.plot(t_axis, mean_rpe, color=color,
                label=_cond_label(weight, pace), linewidth=2.2)
        ax.fill_between(t_axis,
                        np.clip(mean_rpe - std_rpe, 0, 10),
                        np.clip(mean_rpe + std_rpe, 0, 10),
                        alpha=0.18, color=color)

    ax.set_xlabel("Elapsed time (min)", fontsize=11)
    ax.set_ylabel("Mean RPE", fontsize=11)
    ax.set_ylim(-0.5, 10.5)
    ax.set_yticks(range(0, 11, 2))
    ax.set_title(
        "Mean RPE Over Time by Experimental Condition\n"
        "(shaded band = ±1 SD across participants)",
        fontsize=11
    )
    ax.legend(fontsize=9, title="Condition")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "07_rpe_over_time_by_condition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to .pkl dataset file')
    parser.add_argument('--out', default='./figures', help='Output directory')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data)
    describe_structure(data)

    ts = data['ts_data']
    anthro = data['anthro_clean']
    settings = data['experiment_settings']

    print("\nCleaning data...")
    ts, anthro = clean_data(ts, anthro)

    print("\nGenerating visualisations...")
    plot_rpe_trajectories(ts, settings, out_dir)
    plot_rpe_distribution(ts, out_dir)
    plot_between_subject_rpe_variability(ts, settings, out_dir)
    plot_anthropometrics(anthro, out_dir)
    plot_sample_imu_signals(ts, out_dir)
    plot_profile_similarity_heatmap(anthro, out_dir)
    plot_rpe_over_time_by_condition(ts, settings, out_dir)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == '__main__':
    main()
