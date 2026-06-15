"""
Centralized XGBoost baseline — Leave-One-Subject-Out evaluation.

Usage:
    python centralized_xgb.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RandomizedSearchCV, GroupKFold

from pipeline import build_dataset, load_and_clean, RPE_LABELS

DATA_PATH = 'data.pkl'
FIGURES_DIR = Path('figures')


def tune_hyperparameters(X, y, subjects):
    print("Tuning XGBoost hyperparameters (GroupKFold, 5-fold, 20 iterations)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sample_weights = compute_sample_weight('balanced', y)

    param_dist = {
        'n_estimators': [200, 300],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3],
    }
    search = RandomizedSearchCV(
        XGBClassifier(
            objective='multi:softmax', num_class=3,
            eval_metric='mlogloss', random_state=42,
            n_jobs=1, verbosity=0,  # n_jobs=1 here so the search can parallelize
        ),
        param_dist,
        n_iter=10,
        scoring='f1_macro',
        cv=GroupKFold(n_splits=5),
        random_state=42,
        n_jobs=-1,  # search parallelizes across combinations
        verbose=0,
    )
    search.fit(X_scaled, y, groups=subjects, sample_weight=sample_weights)
    print(f"Best params : {search.best_params_}")
    print(f"Best CV F1  : {search.best_score_:.3f}")
    return search.best_params_


def loso_cv(X, y, subjects, params):
    unique_subjs = np.unique(subjects)
    all_true, all_pred = [], []

    for test_subj in unique_subjs:
        test_mask = subjects == test_subj
        X_train, y_train = X[~test_mask], y[~test_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        sample_weights = compute_sample_weight('balanced', y_train)
        xgb = XGBClassifier(
            objective='multi:softmax', num_class=3,
            eval_metric='mlogloss', random_state=42,
            n_jobs=-1, verbosity=0, **params  # n_jobs=-1 fine for single fits
        )
        xgb.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = xgb.predict(X_test)

        fold_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"  {test_subj}: macro-F1 = {fold_f1:.3f}  (n={test_mask.sum()})")
        all_true.extend(y_test)
        all_pred.extend(y_pred)

    return np.array(all_true), np.array(all_pred)


def save_confusion_matrix(y_true, y_pred, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=RPE_LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Centralized XGBoost — LOSO Confusion Matrix')
    plt.tight_layout()
    path = out_dir / 'xgb_confusion_matrix.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def save_feature_importance(X, y, feat_names, params, out_dir, top_n=20):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sample_weights = compute_sample_weight('balanced', y)
    xgb = XGBClassifier(
        objective='multi:softmax', num_class=3,
        eval_metric='mlogloss', random_state=42,
        n_jobs=-1, verbosity=0, **params
    )
    xgb.fit(X_scaled, y, sample_weight=sample_weights)
    importances = xgb.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feat_names[i] for i in idx[::-1]], importances[idx[::-1]], color='darkorange')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances (all-data XGBoost, inspection only)')
    plt.tight_layout()
    path = out_dir / 'xgb_feature_importance.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    ts_data, _, _ = load_and_clean(DATA_PATH)
    X, y, subjects, feat_names = build_dataset(ts_data, cache_path='dataset_cache.npz')

    best_params = tune_hyperparameters(X, y, subjects)

    print(f"\nRunning LOSO CV ({len(np.unique(subjects))} folds) ...")
    y_true, y_pred = loso_cv(X, y, subjects, best_params)

    print('\n=== Centralized XGBoost — LOSO Results ===')
    print(classification_report(y_true, y_pred, target_names=RPE_LABELS))
    print(f"Overall macro-F1 : {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"Overall accuracy : {(y_true == y_pred).mean():.3f}")

    save_confusion_matrix(y_true, y_pred, FIGURES_DIR)
    save_feature_importance(X, y, feat_names, best_params, FIGURES_DIR)
    print('\nDone.')


if __name__ == '__main__':
    main()
