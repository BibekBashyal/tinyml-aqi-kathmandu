"""
train_baselines.py
==================
Trains and evaluates three baseline classifiers for 6-class AQI prediction:
  - Logistic Regression
  - Random Forest
  - XGBoost

For each model reports:
  - Accuracy and Macro-F1 on validation and test sets
  - Per-class precision, recall, F1
  - Confusion matrix (saved as PNG)
  - Training time

All results saved to results/baseline_comparison.csv
All figures saved to results/figures/

Usage:
    pip install scikit-learn xgboost matplotlib seaborn --break-system-packages
    python code/03_baseline_models/train_baselines.py

Author: Bibek Bashyal
Project: TinyML-Based AQI Classification on ESP32
Institution: Kathmandu University, 2026
"""

import os
import time
import warnings
import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES   = ["PM2.5", "PM10", "temperature", "humidity"]
TARGET     = "aqi_class"
RANDOM_STATE = 42

AQI_LABELS = [
    "Good",
    "Moderate",
    "Unhealthy\nfor Sensitive",
    "Unhealthy",
    "Very\nUnhealthy",
    "Hazardous",
]

PATHS = {
    "train":   "data/processed/train.csv",
    "val":     "data/processed/val.csv",
    "test":    "data/processed/test.csv",
    "results": "results",
    "figures": "results/figures/confusion_matrices",
}

# ── Directory setup ───────────────────────────────────────────────────────────
for path in [PATHS["results"], PATHS["figures"]]:
    os.makedirs(path, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    log.info("Loading datasets...")
    train = pd.read_csv(PATHS["train"])
    val   = pd.read_csv(PATHS["val"])
    test  = pd.read_csv(PATHS["test"])

    X_train = train[FEATURES].values
    y_train = train[TARGET].values.astype(int)
    X_val   = val[FEATURES].values
    y_val   = val[TARGET].values.astype(int)
    X_test  = test[FEATURES].values
    y_test  = test[TARGET].values.astype(int)

    log.info(f"  Train : {X_train.shape[0]:,} rows (SMOTE balanced)")
    log.info(f"  Val   : {X_val.shape[0]:,} rows")
    log.info(f"  Test  : {X_test.shape[0]:,} rows")
    log.info(f"  Features: {FEATURES}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X, y, split_name):
    y_pred = model.predict(X)
    acc    = accuracy_score(y, y_pred)
    f1     = f1_score(y, y_pred, average="macro", zero_division=0)
    return {
        "split":    split_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "y_pred":   y_pred,
    }

def print_report(model_name, val_metrics, test_metrics, y_test, y_pred_test):
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  {'Split':<10} {'Accuracy':>10} {'Macro-F1':>10}")
    print(f"  {'-'*32}")
    print(f"  {'Val':<10} {val_metrics['accuracy']:>10.4f} {val_metrics['macro_f1']:>10.4f}")
    print(f"  {'Test':<10} {test_metrics['accuracy']:>10.4f} {test_metrics['macro_f1']:>10.4f}")
    print(f"\n  Per-class report (test set):")
    labels = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]
    print(classification_report(
        y_test, y_pred_test,
        target_names=labels,
        zero_division=0,
        digits=4
    ))

def save_confusion_matrix(model_name, y_true, y_pred, filename):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{model_name} — Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=AQI_LABELS, yticklabels=AQI_LABELS,
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Percentage
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=AQI_LABELS, yticklabels=AQI_LABELS,
        ax=axes[1], linewidths=0.5
    )
    axes[1].set_title("Row % (Recall per class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    out_path = os.path.join(PATHS["figures"], filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Confusion matrix saved → {out_path}")

# ── Model definitions ─────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale": True,   # LR needs feature scaling
            "filename": "confusion_matrix_lr.png",
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale": False,
            "filename": "confusion_matrix_rf.png",
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
            "scale": False,
            "filename": "confusion_matrix_xgb.png",
        },
    }

# ── Feature importance plot (RF only) ────────────────────────────────────────
def plot_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(len(feature_names)),
        importances[indices],
        color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], fontsize=12)
    ax.set_ylabel("Importance Score", fontsize=12)
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(importances) * 1.2)

    for bar, imp in zip(bars, importances[indices]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{imp:.4f}", ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    out_path = os.path.join(PATHS["figures"], "feature_importance_rf.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Feature importance saved → {out_path}")

# ── Summary comparison plot ───────────────────────────────────────────────────
def plot_comparison(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Baseline Model Comparison — Test Set", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    models = results_df["model"].tolist()
    x      = np.arange(len(models))
    width  = 0.35

    for ax, metric, title in zip(
        axes,
        ["test_accuracy", "test_macro_f1"],
        ["Accuracy", "Macro-F1 Score"]
    ):
        vals = results_df[metric].values
        bars = ax.bar(x, vals, width=0.5, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1, label="Target (0.80)")
        ax.legend(fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

    plt.tight_layout()
    out_path = os.path.join(PATHS["results"], "baseline_comparison_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Comparison plot saved → {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  AQI Baseline Model Training")
    log.info("  Bibek Bashyal — Kathmandu University 2026")
    log.info("=" * 60)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Store all results
    all_results = []
    models      = get_models()
    rf_model    = None

    for model_name, config in models.items():
        log.info(f"\nTraining: {model_name}")

        # Feature scaling for Logistic Regression
        if config["scale"]:
            scaler  = StandardScaler()
            X_tr    = scaler.fit_transform(X_train)
            X_v     = scaler.transform(X_val)
            X_te    = scaler.transform(X_test)
        else:
            X_tr, X_v, X_te = X_train, X_val, X_test

        # Train
        t0 = time.time()
        config["model"].fit(X_tr, y_train)
        train_time = round(time.time() - t0, 1)
        log.info(f"  Training time: {train_time}s")

        # Evaluate
        val_metrics  = evaluate(config["model"], X_v,  y_val,  "val")
        test_metrics = evaluate(config["model"], X_te, y_test, "test")

        # Print report
        print_report(
            model_name,
            val_metrics,
            test_metrics,
            y_test,
            test_metrics["y_pred"]
        )

        # Save confusion matrix
        save_confusion_matrix(
            model_name,
            y_test,
            test_metrics["y_pred"],
            config["filename"]
        )

        # Store results
        all_results.append({
            "model":          model_name,
            "val_accuracy":   val_metrics["accuracy"],
            "val_macro_f1":   val_metrics["macro_f1"],
            "test_accuracy":  test_metrics["accuracy"],
            "test_macro_f1":  test_metrics["macro_f1"],
            "train_time_sec": train_time,
        })

        # Keep RF for feature importance
        if model_name == "Random Forest":
            rf_model = config["model"]

    # Feature importance
    if rf_model is not None:
        log.info("\nPlotting Random Forest feature importance...")
        plot_feature_importance(rf_model, FEATURES)

    # Summary table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(PATHS["results"], "baseline_comparison.csv"),
        index=False
    )

    # Comparison plot
    plot_comparison(results_df)

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY — Test Set")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Time(s)':>8}")
    print(f"  {'-'*55}")
    for _, row in results_df.iterrows():
        print(f"  {row['model']:<25} {row['test_accuracy']:>10.4f} {row['test_macro_f1']:>10.4f} {row['train_time_sec']:>8.1f}")
    print(f"{'='*60}")
    print(f"\n  Target macro-F1: > 0.80")
    best = results_df.loc[results_df["test_macro_f1"].idxmax()]
    print(f"  Best model: {best['model']} (macro-F1 = {best['test_macro_f1']:.4f})")
    print(f"\n  Results saved to: results/baseline_comparison.csv")
    print(f"  Figures saved to: results/figures/confusion_matrices/")
    print(f"\n  Next step: python code/04_tinyml_model/train_mlp.py")

if __name__ == "__main__":
    main()
