"""
code/06_analysis/boundary_robustness.py

Compares overall macro-F1 against macro-F1 restricted to test samples that sit
close to an AQI class breakpoint (PM2.5 boundary). The idea: overall accuracy
is dominated by "easy" interior points far from any class edge. Performance
near the edges is what actually matters for public-health alerting, and is a
much harder, more honest test of the classifier.

Usage:
    python code/06_analysis/boundary_robustness.py \
        --model models/xgboost_baseline.pkl \
        --test data/processed/test.csv \
        --pm25-col PM2.5 \
        --label-col aqi_class \
        --margin 2.0

Adjust the four CLI args (or the DEFAULTS below) to match your actual file
names once 01_data_prep / 03_baseline_models are committed.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Nepal DoE / US EPA breakpoints, from docs/aqi_breakpoints.md
# (upper edge of each class, except the last which is open-ended)
BREAKPOINTS = [12.0, 35.4, 55.4, 150.4, 250.4]

DEFAULTS = dict(
    model="models/xgboost_baseline.pkl",
    test="data/processed/test.csv",
    pm25_col="PM2.5",
    label_col="aqi_class",
    pred_col=None,       # if your test CSV already has model predictions cached
    feature_cols=["PM2.5", "PM10", "temperature", "humidity"],
    margin=2.0,           # µg/m3 on either side of a breakpoint counts as "boundary"
    out="results/boundary_robustness.json",
)


def near_boundary_mask(pm25: pd.Series, breakpoints, margin: float) -> pd.Series:
    """True for any sample within `margin` µg/m3 of any class edge."""
    mask = pd.Series(False, index=pm25.index)
    for bp in breakpoints:
        mask |= (pm25 - bp).abs() <= margin
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--test", default=DEFAULTS["test"])
    p.add_argument("--pm25-col", default=DEFAULTS["pm25_col"])
    p.add_argument("--label-col", default=DEFAULTS["label_col"])
    p.add_argument("--pred-col", default=DEFAULTS["pred_col"])
    p.add_argument("--feature-cols", nargs="+", default=DEFAULTS["feature_cols"])
    p.add_argument("--margin", type=float, default=DEFAULTS["margin"])
    p.add_argument("--out", default=DEFAULTS["out"])
    args = p.parse_args()

    df = pd.read_csv(args.test)

    y_true = df[args.label_col].to_numpy()

    if args.pred_col and args.pred_col in df.columns:
        # Predictions already cached in the CSV (e.g. from 03_baseline_models)
        y_pred = df[args.pred_col].to_numpy()
    else:
        # Otherwise load the saved model and predict fresh
        model = joblib.load(args.model)
        X = df[args.feature_cols]
        y_pred = model.predict(X)

    boundary = near_boundary_mask(df[args.pm25_col], BREAKPOINTS, args.margin)
    n_boundary = int(boundary.sum())
    n_total = len(df)

    overall_f1 = f1_score(y_true, y_pred, average="macro")
    overall_acc = accuracy_score(y_true, y_pred)

    if n_boundary == 0:
        print(f"No test samples fall within ±{args.margin} of a breakpoint. "
              f"Try a larger --margin.")
        boundary_f1 = boundary_acc = None
    else:
        boundary_f1 = f1_score(y_true[boundary], y_pred[boundary], average="macro")
        boundary_acc = accuracy_score(y_true[boundary], y_pred[boundary])

    results = {
        "margin_ugm3": args.margin,
        "n_total": n_total,
        "n_boundary": n_boundary,
        "pct_boundary": round(100 * n_boundary / n_total, 2),
        "overall_macro_f1": round(float(overall_f1), 4),
        "overall_accuracy": round(float(overall_acc), 4),
        "boundary_macro_f1": round(float(boundary_f1), 4) if boundary_f1 is not None else None,
        "boundary_accuracy": round(float(boundary_acc), 4) if boundary_acc is not None else None,
    }

    print("\n=== Boundary Robustness Analysis ===")
    print(f"Breakpoints checked (µg/m3): {BREAKPOINTS}")
    print(f"Margin: ±{args.margin} µg/m3")
    print(f"Boundary samples: {n_boundary} / {n_total} ({results['pct_boundary']}%)\n")
    print(f"{'Metric':<20}{'Overall':>12}{'Near boundary':>16}")
    print(f"{'Macro-F1':<20}{overall_f1:>12.4f}"
          f"{(boundary_f1 if boundary_f1 is not None else float('nan')):>16.4f}")
    print(f"{'Accuracy':<20}{overall_acc:>12.4f}"
          f"{(boundary_acc if boundary_acc is not None else float('nan')):>16.4f}")

    if n_boundary > 0:
        print("\n--- Classification report (boundary subset only) ---")
        print(classification_report(y_true[boundary], y_pred[boundary], zero_division=0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
