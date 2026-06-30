"""
boundary_robustness.py
=======================
Tests how the XGBoost baseline (macro-F1 = 0.9852) performs specifically on
samples that sit close to an AQI class breakpoint, where misclassification
has the greatest public-health consequence. Overall macro-F1 is dominated by
"easy" interior points far from any class edge; this script isolates the
harder, more honest, boundary-only evaluation.

Five analyses, all on the same XGBoost model / test set used elsewhere in
this project:

  1. Headline comparison
     Overall macro-F1/accuracy vs. macro-F1/accuracy restricted to samples
     within MARGIN of an AQI breakpoint.

  2. Direction-of-error analysis
     Among boundary misclassifications: under-predictions (model says
     healthier than truth -- the dangerous direction for alerting) vs.
     over-predictions, overall and per true class.

  3. Margin sensitivity sweep
     Re-runs (1) across several margins (1, 2, 5, 10 µg/m3) to produce a
     macro-F1-vs-margin-width curve instead of one point estimate.

  4. Bootstrap confidence intervals
     Resamples the boundary subset (and a chosen class within it, default
     class 5 / Hazardous) to put a CI around small-sample F1 estimates.

  5. Method D -- boundary-restricted noise injection
     Extends Method C from feature_ablation.py (Gaussian noise added to
     PM2.5 in the test set, simulating humidity-induced sensor bias) by
     scoring the noised predictions separately on the boundary subset vs.
     the interior subset vs. the full test set. This directly tests whether
     boundary samples are more fragile to humidity-induced sensor bias than
     interior samples, which is the empirical case for on-device humidity
     correction specifically where it matters most.

All results saved to:
  results/boundary_robustness.json
  results/boundary_noise_injection.csv
  results/figures/boundary_margin_sweep.png
  results/figures/boundary_noise_injection.png

Usage:
    python code/06_analysis/boundary_robustness.py

Author: Bibek Bashyal
Project: TinyML-Based AQI Classification on ESP32
Institution: Kathmandu University, 2026
"""

import os
import json
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# ── Logging setup (matches feature_ablation.py / train_baselines.py) ──────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Constants (matches feature_ablation.py exactly) ───────────────────────────
FEATURES     = ["PM2.5", "PM10", "temperature", "humidity"]
TARGET       = "aqi_class"
RANDOM_STATE = 42

BREAKPOINTS  = [12.0, 35.4, 55.4, 150.4, 250.4]   # docs/aqi_breakpoints.md
MARGIN       = 2.0                                 # headline margin, µg/m3
MARGINS      = [1.0, 2.0, 5.0, 10.0]               # sweep
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]      # matches Method C exactly
N_BOOT       = 1000
BOOT_CLASS   = 5                                   # Hazardous

AQI_LABELS = [
    "Good", "Moderate", "USG",
    "Unhealthy", "Very Unhealthy", "Hazardous"
]

PATHS = {
    "train":      "data/processed/train.csv",
    "val":        "data/processed/val.csv",
    "test":       "data/processed/test.csv",
    "results":    "results",
    "figures":    "results/figures",
    "json_out":   "results/boundary_robustness.json",
    "noise_csv":  "results/boundary_noise_injection.csv",
    "sweep_plot": "results/figures/boundary_margin_sweep.png",
    "noise_plot": "results/figures/boundary_noise_injection.png",
}

for path in [PATHS["results"], PATHS["figures"]]:
    os.makedirs(path, exist_ok=True)


# ── Data + model (matches feature_ablation.py exactly) ────────────────────────

def load_data():
    log.info("Loading datasets...")
    train = pd.read_csv(PATHS["train"])
    test  = pd.read_csv(PATHS["test"])

    X_train = train[FEATURES].values
    y_train = train[TARGET].values.astype(int)
    X_test  = test[FEATURES].values
    y_test  = test[TARGET].values.astype(int)

    log.info(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, y_train, X_test, y_test


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


# ── Boundary helpers ───────────────────────────────────────────────────────────

def near_boundary_mask(pm25, breakpoints, margin):
    """True for any sample within `margin` µg/m3 of any class edge. pm25 is a
    1-D numpy array (FEATURES[0], i.e. X[:, 0])."""
    pm25 = np.asarray(pm25)
    mask = np.zeros_like(pm25, dtype=bool)
    for bp in breakpoints:
        mask |= np.abs(pm25 - bp) <= margin
    return mask


def macro_f1_acc(y_true, y_pred):
    if len(y_true) == 0:
        return None, None
    return (f1_score(y_true, y_pred, average="macro", zero_division=0),
            accuracy_score(y_true, y_pred))


# ── 2. Direction-of-error analysis ─────────────────────────────────────────────

def direction_of_error(y_true, y_pred):
    """Ordinal class labels (0=Good ... 5=Hazardous). Under-prediction = model
    output is a *lower* (healthier-looking) class than truth -- the dangerous
    direction for an alerting system."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    wrong = y_true != y_pred
    n_errors = int(wrong.sum())
    if n_errors == 0:
        return {"n_errors": 0, "n_under": 0, "n_over": 0,
                "pct_under": None, "pct_over": None, "by_true_class": {}}

    under = y_pred[wrong] < y_true[wrong]
    over  = y_pred[wrong] > y_true[wrong]
    n_under, n_over = int(under.sum()), int(over.sum())

    by_class = {}
    for c in sorted(np.unique(y_true)):
        c_wrong = wrong & (y_true == c)
        n_c = int(c_wrong.sum())
        if n_c == 0:
            continue
        by_class[AQI_LABELS[c]] = {
            "n_errors": n_c,
            "n_under":  int((y_pred[c_wrong] < c).sum()),
            "n_over":   int((y_pred[c_wrong] > c).sum()),
        }

    return {
        "n_errors": n_errors, "n_under": n_under, "n_over": n_over,
        "pct_under": round(100 * n_under / n_errors, 1),
        "pct_over":  round(100 * n_over  / n_errors, 1),
        "by_true_class": by_class,
    }


# ── 3. Margin sensitivity sweep ────────────────────────────────────────────────

def margin_sweep(pm25, y_true, y_pred, margins):
    rows, n_total = [], len(y_true)
    for m in margins:
        mask = near_boundary_mask(pm25, BREAKPOINTS, m)
        n_b = int(mask.sum())
        f1, acc = macro_f1_acc(y_true[mask], y_pred[mask]) if n_b > 0 else (None, None)
        rows.append({
            "margin": m, "n_boundary": n_b,
            "pct_boundary": round(100 * n_b / n_total, 2),
            "macro_f1": round(f1, 4) if f1 is not None else None,
            "accuracy": round(acc, 4) if acc is not None else None,
        })
    return pd.DataFrame(rows)


def plot_margin_sweep(sweep_df, overall_f1, target=0.80):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep_df["margin"], sweep_df["macro_f1"], marker="o", color="#1976D2",
            label="Boundary-subset macro-F1")
    ax.axhline(overall_f1, color="gray", linestyle="--",
               label=f"Overall macro-F1 ({overall_f1:.4f})")
    ax.axhline(target, color="#D32F2F", linestyle=":",
               label=f"Target threshold ({target:.2f})")
    ax.set_xlabel("Boundary margin (±µg/m³ from nearest breakpoint)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Classifier degradation as boundary margin narrows")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PATHS["sweep_plot"], dpi=150)
    plt.close(fig)
    log.info(f"  Saved {PATHS['sweep_plot']}")


# ── 4. Bootstrap confidence intervals ──────────────────────────────────────────

def bootstrap_f1_ci(y_true, y_pred, target_class=None, n_boot=N_BOOT, ci=95, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return None

    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if target_class is None:
            scores.append(f1_score(yt, yp, average="macro", zero_division=0))
        else:
            scores.append(f1_score(yt, yp, labels=[target_class], average="macro",
                                    zero_division=0))
    scores = np.array(scores)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "n_boot": n_boot,
        "point_estimate": round(float(np.mean(scores)), 4),
        f"ci{ci}_low":  round(float(np.percentile(scores, lo)), 4),
        f"ci{ci}_high": round(float(np.percentile(scores, hi)), 4),
    }


# ── 5. Method D: boundary-restricted noise injection ───────────────────────────

def method_d_boundary_noise_injection(X_train, y_train, X_test, y_test, margin=MARGIN):
    """Same noise mechanism as Method C in feature_ablation.py (Gaussian noise
    on PM2.5, simulating humidity-induced sensor bias), but scored separately
    on boundary vs. interior vs. full test set at each noise level."""
    log.info("\n── Method D: Boundary-Restricted Noise Injection ──────────")

    pm25_clean = X_test[:, 0].copy()
    boundary   = near_boundary_mask(pm25_clean, BREAKPOINTS, margin)
    interior   = ~boundary
    log.info(f"  Boundary: {boundary.sum():,} | Interior: {interior.sum():,} "
              f"(margin ±{margin} µg/m3)")

    model = get_model()
    model.fit(X_train, y_train)

    pm25_std = X_train[:, 0].std()
    rows = []

    def score(mask, y_pred):
        if mask.sum() == 0:
            return None
        return round(f1_score(y_test[mask], y_pred[mask], average="macro",
                               zero_division=0), 4)

    # noise level 0.0 = clean baseline
    y_pred_clean = model.predict(X_test)
    rows.append({
        "noise_level": 0.0, "noise_std": 0.0,
        "f1_overall": score(np.ones_like(boundary), y_pred_clean),
        "f1_boundary": score(boundary, y_pred_clean),
        "f1_interior": score(interior, y_pred_clean),
    })

    for noise_frac in NOISE_LEVELS:
        noise_std = noise_frac * pm25_std
        rng = np.random.default_rng(RANDOM_STATE)

        X_test_noisy = X_test.copy()
        X_test_noisy[:, 0] = X_test[:, 0] + rng.normal(0, noise_std, X_test.shape[0])
        X_test_noisy[:, 0] = np.clip(X_test_noisy[:, 0], 0, 500)

        y_pred = model.predict(X_test_noisy)

        f1_overall  = score(np.ones_like(boundary), y_pred)
        f1_boundary = score(boundary, y_pred)
        f1_interior = score(interior, y_pred)

        log.info(
            f"  Noise {noise_frac*100:.0f}% ({noise_std:.1f} µg/m³) -> "
            f"overall={f1_overall:.4f}  boundary={f1_boundary:.4f}  "
            f"interior={f1_interior:.4f}"
        )
        rows.append({
            "noise_level": noise_frac, "noise_std": round(noise_std, 2),
            "f1_overall": f1_overall, "f1_boundary": f1_boundary,
            "f1_interior": f1_interior,
        })

    return pd.DataFrame(rows)


def plot_boundary_noise(df, target=0.80):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df["noise_level"] * 100, df["f1_overall"], marker="o",
            label="Full test set", color="#607D8B")
    ax.plot(df["noise_level"] * 100, df["f1_interior"], marker="s",
            label="Interior (far from breakpoints)", color="#43A047")
    ax.plot(df["noise_level"] * 100, df["f1_boundary"], marker="^",
            label="Boundary (±2 µg/m³ of a breakpoint)", color="#D32F2F")
    ax.axhline(target, color="black", linestyle=":", linewidth=1,
               label=f"Target threshold ({target:.2f})")
    ax.set_xlabel("PM2.5 noise level (% of training std, simulating humidity bias)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Boundary samples degrade faster under simulated humidity bias")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PATHS["noise_plot"], dpi=150)
    plt.close(fig)
    log.info(f"  Saved {PATHS['noise_plot']}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    X_train, y_train, X_test, y_test = load_data()
    pm25_test = X_test[:, 0]

    # 1. Headline comparison (clean data, single model fit)
    model = get_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    boundary = near_boundary_mask(pm25_test, BREAKPOINTS, MARGIN)
    n_boundary, n_total = int(boundary.sum()), len(y_test)

    overall_f1, overall_acc = macro_f1_acc(y_test, y_pred)
    boundary_f1, boundary_acc = macro_f1_acc(y_test[boundary], y_pred[boundary])

    log.info("\n── 1. Headline Boundary Robustness ─────────────────────────")
    log.info(f"  Margin ±{MARGIN} µg/m3 | Boundary: {n_boundary:,}/{n_total:,} "
              f"({100*n_boundary/n_total:.2f}%)")
    log.info(f"  Overall  -> Macro-F1: {overall_f1:.4f}  Acc: {overall_acc:.4f}")
    log.info(f"  Boundary -> Macro-F1: {boundary_f1:.4f}  Acc: {boundary_acc:.4f}")
    print("\n--- Classification report (boundary subset) ---")
    print(classification_report(y_test[boundary], y_pred[boundary],
                                 target_names=AQI_LABELS, zero_division=0))

    # 2. Direction of error
    doe = direction_of_error(y_test[boundary], y_pred[boundary])
    log.info("\n── 2. Direction of Boundary Errors ─────────────────────────")
    if doe["n_errors"] > 0:
        log.info(f"  Total errors: {doe['n_errors']} | "
                  f"Under: {doe['n_under']} ({doe['pct_under']}%) | "
                  f"Over: {doe['n_over']} ({doe['pct_over']}%)")
        for cls, d in doe["by_true_class"].items():
            log.info(f"    {cls:<24} errors={d['n_errors']:>3}  "
                      f"under={d['n_under']:>3}  over={d['n_over']:>3}")

    # 3. Margin sweep
    sweep_df = margin_sweep(pm25_test, y_test, y_pred, MARGINS)
    log.info("\n── 3. Margin Sensitivity Sweep ─────────────────────────────")
    log.info("\n" + sweep_df.to_string(index=False))
    plot_margin_sweep(sweep_df, overall_f1)

    # 4. Bootstrap CIs
    log.info("\n── 4. Bootstrap Confidence Intervals (boundary subset) ────")
    boot_overall = bootstrap_f1_ci(y_test[boundary], y_pred[boundary], target_class=None)
    boot_class   = bootstrap_f1_ci(y_test[boundary], y_pred[boundary], target_class=BOOT_CLASS)
    log.info(f"  Boundary macro-F1: {boot_overall['point_estimate']} "
              f"[95% CI {boot_overall['ci95_low']}, {boot_overall['ci95_high']}]")
    log.info(f"  '{AQI_LABELS[BOOT_CLASS]}' F1: {boot_class['point_estimate']} "
              f"[95% CI {boot_class['ci95_low']}, {boot_class['ci95_high']}]")

    # 5. Method D: boundary-restricted noise injection
    noise_df = method_d_boundary_noise_injection(X_train, y_train, X_test, y_test, margin=MARGIN)
    plot_boundary_noise(noise_df)
    noise_df.to_csv(PATHS["noise_csv"], index=False)
    log.info(f"  Saved {PATHS['noise_csv']}")

    # Save full JSON
    results = {
        "margin_ugm3": MARGIN,
        "n_total": n_total,
        "n_boundary": n_boundary,
        "pct_boundary": round(100 * n_boundary / n_total, 2),
        "overall_macro_f1": round(float(overall_f1), 4),
        "overall_accuracy": round(float(overall_acc), 4),
        "boundary_macro_f1": round(float(boundary_f1), 4),
        "boundary_accuracy": round(float(boundary_acc), 4),
        "direction_of_error": doe,
        "margin_sweep": sweep_df.to_dict(orient="records"),
        "bootstrap_boundary_macro_f1": boot_overall,
        "bootstrap_class": {"class": AQI_LABELS[BOOT_CLASS], **boot_class},
        "method_d_boundary_noise_injection": noise_df.to_dict(orient="records"),
    }
    with open(PATHS["json_out"], "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved full results to {PATHS['json_out']}")


if __name__ == "__main__":
    main()
