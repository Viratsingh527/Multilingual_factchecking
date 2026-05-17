# significance_and_error_analysis.py

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import chi2


# -----------------------------
# Basic metrics
# -----------------------------
def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

import re

def extract_claim_from_input(text):
    text = str(text)

    match = re.search(
        r"Claim:\s*(.*?)\s*Evidences:",
        text,
        flags=re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return None

# -----------------------------
# Paired Bootstrap Test
# -----------------------------
def paired_bootstrap_test(
    y_true,
    pred_base,
    pred_ours,
    metric_fn=macro_f1,
    n_bootstrap=10000,
    seed=42
):
    """
    Paired bootstrap significance test.

    Null idea:
    If the improvement is not reliable, bootstrap samples
    should often show baseline >= ours.

    Returns:
    observed_diff, p_value, confidence interval
    """

    rng = np.random.default_rng(seed)
    n = len(y_true)

    y_true = np.array(y_true)
    pred_base = np.array(pred_base)
    pred_ours = np.array(pred_ours)

    observed_base = metric_fn(y_true, pred_base)
    observed_ours = metric_fn(y_true, pred_ours)
    observed_diff = observed_ours - observed_base

    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)

        score_base = metric_fn(y_true[indices], pred_base[indices])
        score_ours = metric_fn(y_true[indices], pred_ours[indices])

        bootstrap_diffs.append(score_ours - score_base)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # one-sided p-value: probability that improvement <= 0
    p_value = np.mean(bootstrap_diffs <= 0)

    ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])

    return {
        "baseline_score": observed_base,
        "ours_score": observed_ours,
        "observed_diff": observed_diff,
        "relative_improvement_percent": (observed_diff / observed_base) * 100 if observed_base != 0 else np.nan,
        "p_value": p_value,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }


# -----------------------------
# McNemar Test
# -----------------------------
def mcnemar_test(y_true, pred_base, pred_ours):
    """
    McNemar's test using paired correctness.

    b = baseline correct, ours wrong
    c = ours correct, baseline wrong

    Test checks whether b and c are significantly different.
    """

    y_true = np.array(y_true)
    pred_base = np.array(pred_base)
    pred_ours = np.array(pred_ours)

    base_correct = pred_base == y_true
    ours_correct = pred_ours == y_true

    both_correct = np.sum(base_correct & ours_correct)
    both_wrong = np.sum(~base_correct & ~ours_correct)

    b = np.sum(base_correct & ~ours_correct)
    c = np.sum(~base_correct & ours_correct)

    # continuity-corrected McNemar statistic
    if b + c == 0:
        chi_square = 0.0
        p_value = 1.0
    else:
        chi_square = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi_square, df=1)

    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_correct_ours_wrong": b,
        "ours_correct_baseline_wrong": c,
        "chi_square": chi_square,
        "p_value": p_value,
    }


# -----------------------------
# Error Analysis
# -----------------------------
def build_error_analysis_table(df, true_col, pred_col):
    """
    Creates label-level error analysis:
    - true label
    - predicted label
    - frequency
    """

    error_df = df[df[true_col] != df[pred_col]].copy()

    table = (
        error_df
        .groupby([true_col, pred_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    total_errors = len(error_df)
    table["percentage"] = (table["count"] / total_errors * 100).round(2)

    return table


def build_correctness_comparison(df, true_col, base_pred_col, ours_pred_col):
    """
    Creates instance-level comparison between baseline and ours.
    """

    df = df.copy()

    df["baseline_correct"] = df[base_pred_col] == df[true_col]
    df["ours_correct"] = df[ours_pred_col] == df[true_col]

    def category(row):
        if row["baseline_correct"] and row["ours_correct"]:
            return "Both Correct"
        elif (not row["baseline_correct"]) and row["ours_correct"]:
            return "Ours Correct, Baseline Wrong"
        elif row["baseline_correct"] and (not row["ours_correct"]):
            return "Baseline Correct, Ours Wrong"
        else:
            return "Both Wrong"

    df["comparison_category"] = df.apply(category, axis=1)

    summary = (
        df["comparison_category"]
        .value_counts()
        .reset_index()
    )
    summary.columns = ["category", "count"]
    summary["percentage"] = (summary["count"] / len(df) * 100).round(2)

    return df, summary


# -----------------------------
# Optional Manual Error Category Template
# -----------------------------
def create_manual_error_template(df, true_col, pred_col, output_path):
    error_df = df[df[true_col] != df[pred_col]].copy()
    error_df["error_category"] = ""

    useful_cols = [
        col for col in [
            "claim_id_extracted",
            "input_base",
            "input_ours",
            true_col,
            pred_col,
            "error_category"
        ] if col in error_df.columns
    ]

    error_df[useful_cols].to_csv(output_path, index=False)
    print(f"Manual error analysis template saved to: {output_path}")

import json
import os

def load_prediction_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Case 1: file is a list of dictionaries
        if isinstance(data, list):
            return pd.DataFrame(data)

        # Case 2: file is dictionary containing predictions
        elif isinstance(data, dict):
            for key in ["predictions", "data", "results", "outputs"]:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])

            # fallback: single dictionary
            return pd.DataFrame([data])

    elif ext == ".jsonl":
        return pd.read_json(path, lines=True)

    else:
        raise ValueError(f"Unsupported file format: {ext}")
# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline_file", required=True)
    parser.add_argument("--ours_file", required=True)

    parser.add_argument("--true_col", default="true_label")
    parser.add_argument("--pred_col", default="predicted_label")
    parser.add_argument("--id_col", default="claim_id_extracted")

    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--output_prefix", default="analysis_results")

    args = parser.parse_args()

    base_df = load_prediction_file(args.baseline_file)
    ours_df = load_prediction_file(args.ours_file)

    # Extract claim from input
    base_df["claim_id_extracted"] = base_df["input"].apply(extract_claim_from_input)
    ours_df["claim_id_extracted"] = ours_df["input"].apply(extract_claim_from_input)

    # Remove failed extraction
    base_df = base_df.dropna(subset=["claim_id_extracted"])
    ours_df = ours_df.dropna(subset=["claim_id_extracted"])

    # Check duplicate claims
    base_dup = base_df["claim_id_extracted"].duplicated().sum()
    ours_dup = ours_df["claim_id_extracted"].duplicated().sum()

    print(f"Duplicate extracted claims in baseline: {base_dup}")
    print(f"Duplicate extracted claims in ours: {ours_dup}")

    if base_dup > 0 or ours_dup > 0:
        print("Warning: duplicate claims found. Merge may create repeated rows.")

    # Merge using extracted claim
    merged = base_df.merge(
        ours_df,
        on=args.id_col,
        suffixes=("_base", "_ours")
    )

    print(f"Baseline rows: {len(base_df)}")
    print(f"Ours rows: {len(ours_df)}")
    print(f"Merged rows: {len(merged)}")

    true_col = args.true_col + "_base"
    base_pred_col = args.pred_col + "_base"
    ours_pred_col = args.pred_col + "_ours"

    y_true = merged[true_col].astype(str).str.lower().str.strip().values
    pred_base = merged[base_pred_col].astype(str).str.lower().str.strip().values
    pred_ours = merged[ours_pred_col].astype(str).str.lower().str.strip().values

    bootstrap_result = paired_bootstrap_test(
        y_true,
        pred_base,
        pred_ours,
        metric_fn=macro_f1,
        n_bootstrap=args.n_bootstrap
    )

    bootstrap_df = pd.DataFrame([bootstrap_result])
    bootstrap_df.to_csv(f"{args.output_prefix}_paired_bootstrap.csv", index=False)

    mcnemar_result = mcnemar_test(y_true, pred_base, pred_ours)
    mcnemar_df = pd.DataFrame([mcnemar_result])
    mcnemar_df.to_csv(f"{args.output_prefix}_mcnemar.csv", index=False)

    ours_error_table = build_error_analysis_table(
        merged,
        true_col=true_col,
        pred_col=ours_pred_col
    )
    ours_error_table.to_csv(f"{args.output_prefix}_ours_label_errors.csv", index=False)

    comparison_df, comparison_summary = build_correctness_comparison(
        merged,
        true_col=true_col,
        base_pred_col=base_pred_col,
        ours_pred_col=ours_pred_col
    )

    comparison_df.to_csv(f"{args.output_prefix}_instance_comparison.csv", index=False)
    comparison_summary.to_csv(f"{args.output_prefix}_correctness_summary.csv", index=False)

    create_manual_error_template(
        comparison_df,
        true_col=true_col,
        pred_col=ours_pred_col,
        output_path=f"{args.output_prefix}_manual_error_template.csv"
    )

    print("\n===== Paired Bootstrap Test =====")
    print(bootstrap_df.to_string(index=False))

    print("\n===== McNemar Test =====")
    print(mcnemar_df.to_string(index=False))

    print("\n===== Ours Label Error Table =====")
    print(ours_error_table.to_string(index=False))

    print("\n===== Correctness Comparison =====")
    print(comparison_summary.to_string(index=False))

if __name__ == "__main__":
    main()