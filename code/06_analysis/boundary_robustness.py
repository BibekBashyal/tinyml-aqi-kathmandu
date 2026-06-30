"""
code/06_analysis/boundary_robustness.py

Three analyses, all operating on the same trained model + test set:

  1. Headline comparison: overall macro-F1/accuracy vs. macro-F1/accuracy
     restricted to samples within --margin of an AQI breakpoint.
  2. Direction-of-error analysis: among boundary misclassifications, how many
     are under-predictions (model says healthier than truth -- the dangerous
     direction for public-health alerting) vs over-predictions.
  3. Margin sensitivity sweep: re-run (1) across several margins to produce a
     macro-F1-vs-margin-width curve instead of a single point estimate.
  4. Bootstrap confidence intervals: resample the boundary subset (and,
     optionally, a single class within it, e.g. class 5 / Hazardous) to get
     a CI around the boundary macro-F1 / per-class F1, since boundary subsets
     are small and a single point estimate invites exactly that question
     from a committee.

Usage:
    python code/06_analysis/boundary_robustness.py \
        --model models/xgboost_baseline.pkl \
        --test data/processed/test.csv \
        --pm25-col PM2.5 \
        --label-col aqi_class \
        --margin 2.0 \
        --margins 1 2 5 10 \
        --bootstrap --bootstrap-class 5

Adjust paths/column names (or the DEFAULTS below) to match your actual
01_data_prep / 03_baseline_models outputs.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Nepal DoE / US EPA breakpoints, from docs/aqi_breakpoints.md
BREAKPOINTS = [12.0, 35.4, 55.4, 150.4, 250.4]
CLASS_NAMES = ["Good", "Moderate", "Unhealthy (Sensitive)", "Unhealthy",
               "Very Unhealthy", "Hazardous"]

DEFAULTS = dict(
    model="models/xgboost_baseline.pkl",
    test="data/processed/test.csv",
    pm25_col="PM2.5",
    label_col="aqi_class",
    pred_col=None,
    feature_cols=["PM2.5", "PM10", "temperature", "humidity"],
    margin=2.0,
    margins=[1.0, 2.0, 5.0, 10.0],
    out="results/boundary_robustness.json",
    plot_out="results/figures/boundary_margin_sweep.png",
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def near_boundary_mask(pm25: pd.Series, breakpoints, margin: float) -> np.ndarray:
    """True for any sample within `margin` µg/m3 of any class edge."""
    mask = pd.Series(False, index=pm25.index)
    for bp in breakpoints:
        mask |= (pm25 - bp).abs() <= margin
    return mask.to_numpy()


def macro_f1_acc(y_true, y_pred):
    if len(y_true) == 0:
        return None, None
    return (f1_score(y_true, y_pred, average="macro"),
            accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# 1. Direction-of-error analysis (item 1)
# ---------------------------------------------------------------------------

def direction_of_error(y_true, y_pred):
    """
    Assumes ordinal integer class labels (0=Good ... 5=Hazardous), which your
    aqi_breakpoints.md labeling already gives you. Under-prediction = model
    output is a *lower* (healthier-looking) class than the truth -- the
    dangerous direction for an alerting system.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    wrong = y_true != y_pred
    n_errors = int(wrong.sum())
    if n_errors == 0:
        return {"n_errors": 0, "n_under": 0, "n_over": 0,
                "pct_under": None, "pct_over": None, "by_true_class": {}}

    under = (y_pred[wrong] < y_true[wrong])
    over = (y_pred[wrong] > y_true[wrong])
    n_under, n_over = int(under.sum()), int(over.sum())

    by_class = {}
    for c in sorted(np.unique(y_true)):
        c_wrong = wrong & (y_true == c)
        n_c = int(c_wrong.sum())
        if n_c == 0:
            continue
        c_under = int((y_pred[c_wrong] < c).sum())
        c_over = int((y_pred[c_wrong] > c).sum())
        by_class[CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)] = {
            "n_errors": n_c, "n_under": c_under, "n_over": c_over,
        }

    return {
        "n_errors": n_errors,
        "n_under": n_under,
        "n_over": n_over,
        "pct_under": round(100 * n_under / n_errors, 1),
        "pct_over": round(100 * n_over / n_errors, 1),
        "by_true_class": by_class,
    }


# ---------------------------------------------------------------------------
# 2. Margin sensitivity sweep (item 2)
# ---------------------------------------------------------------------------

def margin_sweep(pm25, y_true, y_pred, margins):
    rows = []
    n_total = len(y_true)
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


def plot_margin_sweep(sweep_df, overall_f1, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep_df["margin"], sweep_df["macro_f1"], marker="o",
            label="Boundary-subset macro-F1")
    ax.axhline(overall_f1, color="gray", linestyle="--",
               label=f"Overall macro-F1 ({overall_f1:.4f})")
    ax.set_xlabel("Boundary margin (±µg/m³ from nearest breakpoint)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Classifier degradation as boundary margin narrows")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Bootstrap confidence intervals (item 5)
# ---------------------------------------------------------------------------

def bootstrap_f1_ci(y_true, y_pred, target_class=None, n_boot=1000, ci=95, seed=42):
    """
    Resamples (y_true, y_pred) pairs with replacement n_boot times.
    If target_class is None: bootstraps macro-F1 over all classes present.
    If target_class is given: bootstraps F1 for just that class (one-vs-rest),
    which is what you want for a small/rare class like Hazardous.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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
            scores.append(f1_score(yt, yp, labels=[target_class],
                                    average="macro", zero_division=0))
    scores = np.array(scores)
    lo = (100 - ci) / 2
    hi = 100 - lo
    return {
        "n_boot": n_boot,
        "point_estimate": round(float(np.mean(scores)), 4),
        f"ci{ci}_low": round(float(np.percentile(scores, lo)), 4),
        f"ci{ci}_high": round(float(np.percentile(scores, hi)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--test", default=DEFAULTS["test"])
    p.add_argument("--pm25-col", default=DEFAULTS["pm25_col"])
    p.add_argument("--label-col", default=DEFAULTS["label_col"])
    p.add_argument("--pred-col", default=DEFAULTS["pred_col"])
    p.add_argument("--feature-cols", nargs="+", default=DEFAULTS["feature_cols"])
    p.add_argument("--margin", type=float, default=DEFAULTS["margin"])
    p.add_argument("--margins", nargs="+", type=float, default=DEFAULTS["margins"])
    p.add_argument("--out", default=DEFAULTS["out"])
    p.add_argument("--plot-out", default=DEFAULTS["plot_out"])
    p.add_argument("--bootstrap", action="store_true",
                    help="Run bootstrap CIs on the boundary subset")
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--bootstrap-class", type=int, default=None,
                    help="Class index (e.g. 5 for Hazardous) for a per-class bootstrap CI")
    args = p.parse_args()

    df = pd.read_csv(args.test)
    y_true = df[args.label_col].to_numpy()

    if args.pred_col and args.pred_col in df.columns:
        y_pred = df[args.pred_col].to_numpy()
    else:
        model = joblib.load(args.model)
        y_pred = model.predict(df[args.feature_cols])

    pm25 = df[args.pm25_col]
    n_total = len(df)

    # --- 1. Headline comparison at the chosen margin -----------------------
    boundary = near_boundary_mask(pm25, BREAKPOINTS, args.margin)
    n_boundary = int(boundary.sum())
    overall_f1, overall_acc = macro_f1_acc(y_true, y_pred)
    boundary_f1, boundary_acc = (macro_f1_acc(y_true[boundary], y_pred[boundary])
                                  if n_boundary > 0 else (None, None))

    print("\n=== 1. Headline Boundary Robustness ===")
    print(f"Margin: ±{args.margin} µg/m3 | Boundary samples: {n_boundary}/{n_total} "
          f"({100*n_boundary/n_total:.1f}%)")
    print(f"{'Metric':<12}{'Overall':>10}{'Boundary':>12}")
    print(f"{'Macro-F1':<12}{overall_f1:>10.4f}{(boundary_f1 or float('nan')):>12.4f}")
    print(f"{'Accuracy':<12}{overall_acc:>10.4f}{(boundary_acc or float('nan')):>12.4f}")
    if n_boundary > 0:
        print("\n--- Classification report (boundary subset) ---")
        print(classification_report(y_true[boundary], y_pred[boundary], zero_division=0))

    # --- 2. Direction-of-error analysis (item 1) ----------------------------
    doe = direction_of_error(y_true[boundary], y_pred[boundary]) if n_boundary > 0 else None
    if doe and doe["n_errors"] > 0:
        print("\n=== 2. Direction of Boundary Errors ===")
        print(f"Total boundary errors: {doe['n_errors']}")
        print(f"  Under-predicted (model says healthier than truth): "
              f"{doe['n_under']} ({doe['pct_under']}%)  <- the dangerous direction")
        print(f"  Over-predicted  (model says worse than truth):     "
              f"{doe['n_over']} ({doe['pct_over']}%)")
        print("  By true class:")
        for cls, d in doe["by_true_class"].items():
            print(f"    {cls:<24} errors={d['n_errors']:>3}  "
                  f"under={d['n_under']:>3}  over={d['n_over']:>3}")

    # --- 3. Margin sensitivity sweep (item 2) -------------------------------
    sweep_df = margin_sweep(pm25, y_true, y_pred, args.margins)
    print("\n=== 3. Margin Sensitivity Sweep ===")
    print(sweep_df.to_string(index=False))
    plot_margin_sweep(sweep_df, overall_f1, args.plot_out)
    print(f"Saved sweep plot to {args.plot_out}")

    # --- 4. Bootstrap CIs (item 5) -------------------------------------------
    boot_overall = boot_class = None
    if args.bootstrap and n_boundary > 0:
        print("\n=== 4. Bootstrap Confidence Intervals (boundary subset) ===")
        boot_overall = bootstrap_f1_ci(y_true[boundary], y_pred[boundary],
                                        target_class=None, n_boot=args.bootstrap_n)
        print(f"Boundary macro-F1: {boot_overall['point_estimate']} "
              f"[95% CI {boot_overall['ci95_low']}, {boot_overall['ci95_high']}] "
              f"(n_boot={boot_overall['n_boot']})")
        if args.bootstrap_class is not None:
            boot_class = bootstrap_f1_ci(y_true[boundary], y_pred[boundary],
                                          target_class=args.bootstrap_class,
                                          n_boot=args.bootstrap_n)
            cname = (CLASS_NAMES[args.bootstrap_class]
                     if args.bootstrap_class < len(CLASS_NAMES)
                     else str(args.bootstrap_class))
            print(f"Class '{cname}' F1: {boot_class['point_estimate']} "
                  f"[95% CI {boot_class['ci95_low']}, {boot_class['ci95_high']}]")

    # --- Save everything -----------------------------------------------------
    results = {
        "margin_ugm3": args.margin,
        "n_total": n_total,
        "n_boundary": n_boundary,
        "pct_boundary": round(100 * n_boundary / n_total, 2),
        "overall_macro_f1": round(float(overall_f1), 4),
        "overall_accuracy": round(float(overall_acc), 4),
        "boundary_macro_f1": round(float(boundary_f1), 4) if boundary_f1 is not None else None,
        "boundary_accuracy": round(float(boundary_acc), 4) if boundary_acc is not None else None,
        "direction_of_error": doe,
        "margin_sweep": sweep_df.to_dict(orient="records"),
        "bootstrap_boundary_macro_f1": boot_overall,
        "bootstrap_class": boot_class,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    main()
