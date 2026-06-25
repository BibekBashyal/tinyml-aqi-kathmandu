"""
feature_ablation.py
====================
Systematically evaluates the contribution of each input feature
to AQI classification performance using two complementary methods:

  Method A — Leave-One-Out Ablation
    Removes one feature at a time and measures accuracy + macro-F1 drop.
    Answers: "Which feature hurts most when missing?"

  Method B — Single-Feature Ablation
    Trains on only one feature at a time.
    Answers: "How much does each feature contribute alone?"

  Method C — Humidity Noise Injection
    Adds synthetic Gaussian noise to PM2.5 (simulating high-humidity
    sensor bias) and measures degradation.
    Answers: "How fragile is the model to real-world sensor noise?"

Models used: XGBoost (best honest baseline, macro-F1 = 0.9852)

All results saved to:
  results/feature_ablation.csv
  results/figures/feature_ablation_looout.png
  results/figures/feature_ablation_single.png
  results/figures/feature_ablation_noise.png

Usage:
    python code/03_baseline_models/feature_ablation.py

Author: Bibek Bashyal
Project: TinyML-Based AQI Classification on ESP32
Institution: Kathmandu University, 2026
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES     = ["PM2.5", "PM10", "temperature", "humidity"]
TARGET       = "aqi_class"
RANDOM_STATE = 42

NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]   # std as fraction of feature std

AQI_LABELS = [
    "Good", "Moderate", "USG",
    "Unhealthy", "Very Unhealthy", "Hazardous"
]

PATHS = {
    "train":   "data/processed/train.csv",
    "val":     "data/processed/val.csv",
    "test":    "data/processed/test.csv",
    "results": "results",
    "figures": "results/figures",
}

COLORS = {
    "PM2.5":       "#2196F3",
    "PM10":        "#4CAF50",
    "temperature": "#FF9800",
    "humidity":    "#9C27B0",
    "baseline":    "#E53935",
}

# ── Directory setup ───────────────────────────────────────────────────────────
os.makedirs(PATHS["figures"], exist_ok=True)
os.makedirs(PATHS["results"], exist_ok=True)

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

    log.info(f"  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# ── Model factory ─────────────────────────────────────────────────────────────
def get_model():
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

def train_and_eval(X_tr, y_tr, X_te, y_te):
    model = get_model()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    return round(acc, 4), round(f1, 4)

# ── Method A: Leave-One-Out ablation ─────────────────────────────────────────
def method_a_leave_one_out(X_train, y_train, X_test, y_test):
    log.info("\n── Method A: Leave-One-Out Ablation ──────────────────────")

    # Baseline — all features
    base_acc, base_f1 = train_and_eval(X_train, y_train, X_test, y_test)
    log.info(f"  Baseline (all features) — Acc: {base_acc:.4f}  Macro-F1: {base_f1:.4f}")

    results = []
    for i, feat in enumerate(FEATURES):
        remaining = [j for j in range(len(FEATURES)) if j != i]
        feat_names = [FEATURES[j] for j in remaining]

        acc, f1 = train_and_eval(
            X_train[:, remaining], y_train,
            X_test[:, remaining],  y_test
        )

        acc_drop = round(base_acc - acc, 4)
        f1_drop  = round(base_f1 - f1, 4)

        log.info(
            f"  Remove '{feat:<12}' → Acc: {acc:.4f} (Δ{acc_drop:+.4f})  "
            f"Macro-F1: {f1:.4f} (Δ{f1_drop:+.4f})"
        )

        results.append({
            "method":         "leave_one_out",
            "removed_feature": feat,
            "remaining":      ", ".join(feat_names),
            "accuracy":       acc,
            "macro_f1":       f1,
            "accuracy_drop":  acc_drop,
            "f1_drop":        f1_drop,
        })

    return base_acc, base_f1, pd.DataFrame(results)

# ── Method B: Single-feature ablation ────────────────────────────────────────
def method_b_single_feature(X_train, y_train, X_test, y_test):
    log.info("\n── Method B: Single-Feature Ablation ─────────────────────")

    results = []
    for i, feat in enumerate(FEATURES):
        acc, f1 = train_and_eval(
            X_train[:, [i]], y_train,
            X_test[:, [i]],  y_test
        )
        log.info(f"  Only '{feat:<12}' → Acc: {acc:.4f}  Macro-F1: {f1:.4f}")
        results.append({
            "method":   "single_feature",
            "feature":  feat,
            "accuracy": acc,
            "macro_f1": f1,
        })

    return pd.DataFrame(results)

# ── Method C: Humidity noise injection on PM2.5 ───────────────────────────────
def method_c_noise_injection(X_train, y_train, X_test, y_test):
    log.info("\n── Method C: PM2.5 Noise Injection (Humidity Bias Simulation) ──")

    # Baseline — clean data
    base_acc, base_f1 = train_and_eval(X_train, y_train, X_test, y_test)

    pm25_std = X_train[:, 0].std()
    results  = [{"noise_level": 0.0, "noise_std": 0.0,
                 "accuracy": base_acc, "macro_f1": base_f1}]

    for noise_frac in NOISE_LEVELS:
        noise_std = noise_frac * pm25_std
        rng       = np.random.default_rng(RANDOM_STATE)

        # Add noise to PM2.5 (index 0) in test set only
        # Simulates what happens at inference time when RH is high
        X_test_noisy      = X_test.copy()
        X_test_noisy[:, 0] = X_test[:, 0] + rng.normal(0, noise_std, X_test.shape[0])
        X_test_noisy[:, 0] = np.clip(X_test_noisy[:, 0], 0, 500)

        model = get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test_noisy)

        acc = round(accuracy_score(y_test, y_pred), 4)
        f1  = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)

        log.info(
            f"  Noise {noise_frac*100:.0f}% of PM2.5 std ({noise_std:.1f} µg/m³) "
            f"→ Acc: {acc:.4f}  Macro-F1: {f1:.4f}"
        )
        results.append({
            "noise_level": noise_frac,
            "noise_std":   round(noise_std, 2),
            "accuracy":    acc,
            "macro_f1":    f1,
        })

    return pd.DataFrame(results)

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_leave_one_out(df, base_f1):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Method A — Leave-One-Out Feature Ablation (XGBoost, Test Set)",
        fontsize=13, fontweight="bold"
    )

    for ax, metric, label in zip(
        axes,
        ["macro_f1", "f1_drop"],
        ["Macro-F1 Score", "Macro-F1 Drop (Δ from baseline)"]
    ):
        colors = [COLORS.get(f, "#607D8B") for f in df["removed_feature"]]
        bars   = ax.bar(
            df["removed_feature"], df[metric],
            color=colors, edgecolor="white", linewidth=0.5, width=0.5
        )
        if metric == "macro_f1":
            ax.axhline(
                y=base_f1, color=COLORS["baseline"],
                linestyle="--", linewidth=1.5, label=f"Baseline ({base_f1:.4f})"
            )
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1.05)
        else:
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
            ax.set_ylim(min(df[metric].min() - 0.05, -0.02), max(df[metric].max() + 0.05, 0.02))

        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Removed Feature", fontsize=11)

        for bar, val in zip(bars, df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.003 if metric == "macro_f1" else 0.001),
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

    plt.tight_layout()
    out = os.path.join(PATHS["figures"], "feature_ablation_leaveout.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved → {out}")

def plot_single_feature(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "Method B — Single-Feature Classification (XGBoost, Test Set)",
        fontsize=13, fontweight="bold"
    )

    colors = [COLORS.get(f, "#607D8B") for f in df["feature"]]
    bars   = ax.bar(
        df["feature"], df["macro_f1"],
        color=colors, edgecolor="white", linewidth=0.5, width=0.5
    )
    ax.set_ylabel("Macro-F1 Score", fontsize=11)
    ax.set_xlabel("Single Feature Used", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.80, color=COLORS["baseline"],
               linestyle="--", linewidth=1.5, label="Target (0.80)")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, df["macro_f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    out = os.path.join(PATHS["figures"], "feature_ablation_single.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved → {out}")

def plot_noise_injection(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Method C — PM2.5 Noise Injection (Humidity Bias Simulation, Test Set)",
        fontsize=13, fontweight="bold"
    )

    noise_pct = (df["noise_level"] * 100).astype(int).astype(str) + "%"
    x         = np.arange(len(df))

    ax.plot(
        x, df["macro_f1"],
        marker="o", linewidth=2, markersize=8,
        color=COLORS["PM2.5"], label="Macro-F1"
    )
    ax.fill_between(x, df["macro_f1"], alpha=0.15, color=COLORS["PM2.5"])
    ax.axhline(
        y=df["macro_f1"].iloc[0], color=COLORS["baseline"],
        linestyle="--", linewidth=1.5, label=f"Baseline ({df['macro_f1'].iloc[0]:.4f})"
    )
    ax.axhline(y=0.80, color="orange",
               linestyle=":", linewidth=1.5, label="Target threshold (0.80)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n(±{s:.1f} µg/m³)" for p, s in zip(noise_pct, df["noise_std"])],
        fontsize=9
    )
    ax.set_xlabel("PM2.5 Noise Level (% of training std)", fontsize=11)
    ax.set_ylabel("Macro-F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    for xi, val in zip(x, df["macro_f1"]):
        ax.annotate(
            f"{val:.4f}",
            (xi, val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center", fontsize=9, fontweight="bold"
        )

    plt.tight_layout()
    out = os.path.join(PATHS["figures"], "feature_ablation_noise.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved → {out}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  Feature Ablation Study — XGBoost (macro-F1 = 0.9852)")
    log.info("  Bibek Bashyal — Kathmandu University 2026")
    log.info("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Run all three methods
    base_acc, base_f1, df_loo    = method_a_leave_one_out(X_train, y_train, X_test, y_test)
    df_single                    = method_b_single_feature(X_train, y_train, X_test, y_test)
    df_noise                     = method_c_noise_injection(X_train, y_train, X_test, y_test)

    # Save plots
    log.info("\nSaving figures...")
    plot_leave_one_out(df_loo, base_f1)
    plot_single_feature(df_single)
    plot_noise_injection(df_noise)

    # Save combined results CSV
    df_loo["experiment"]    = "A_leave_one_out"
    df_single["experiment"] = "B_single_feature"
    df_noise["experiment"]  = "C_noise_injection"

    df_all = pd.concat([df_loo, df_single, df_noise], ignore_index=True)
    out_csv = os.path.join(PATHS["results"], "feature_ablation.csv")
    df_all.to_csv(out_csv, index=False)

    # Final summary
    print(f"\n{'='*60}")
    print("  FEATURE ABLATION SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Method A — Leave-One-Out (baseline macro-F1 = {base_f1:.4f})")
    print(f"  {'Removed':<15} {'Macro-F1':>10} {'F1 Drop':>10}")
    print(f"  {'-'*37}")
    for _, row in df_loo.sort_values("f1_drop", ascending=False).iterrows():
        bar = "▓" * int(abs(row["f1_drop"]) * 200)
        print(f"  {row['removed_feature']:<15} {row['macro_f1']:>10.4f} {row['f1_drop']:>+10.4f}  {bar}")

    print(f"\n  Method B — Single Feature Only")
    print(f"  {'Feature':<15} {'Macro-F1':>10}")
    print(f"  {'-'*27}")
    for _, row in df_single.sort_values("macro_f1", ascending=False).iterrows():
        bar = "▓" * int(row["macro_f1"] * 30)
        print(f"  {row['feature']:<15} {row['macro_f1']:>10.4f}  {bar}")

    print(f"\n  Method C — PM2.5 Noise Injection")
    print(f"  {'Noise Level':<15} {'PM2.5 Std (µg)':<16} {'Macro-F1':>10}")
    print(f"  {'-'*43}")
    for _, row in df_noise.iterrows():
        print(f"  {row['noise_level']*100:>4.0f}%{'':>10} ±{row['noise_std']:<14.1f} {row['macro_f1']:>10.4f}")

    print(f"\n  Results saved → {out_csv}")
    print(f"  Figures saved → {PATHS['figures']}/")
    print(f"\n  Thesis insight:")
    most_important = df_loo.loc[df_loo["f1_drop"].idxmax(), "removed_feature"]
    print(f"  → Most important feature: '{most_important}' (largest F1 drop when removed)")
    print(f"  → This confirms the labeling relationship and motivates")
    print(f"     humidity correction on-device for real sensor readings.")
    print(f"\n  Next: python code/04_tinyml_model/train_mlp.py")

if __name__ == "__main__":
    main()
