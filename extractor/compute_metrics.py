#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np

def safe_div(n, d):
    return float(n) / d if d else 0.0

def add_row_metrics(df: pd.DataFrame, total_tools: int) -> pd.DataFrame:
    """
    Expects columns:
      - 'true_matches' (TP)
      - 'only_in_input' (FN)
      - 'only_in_extracted' (FP)
      - 'number_of_tools' (P)  # ground-truth positives per row
    """
    # Rename to internal variables for clarity
    TP = df["true_matches"].astype(int)
    FN = df["only_in_input"].astype(int)
    FP = df["only_in_extracted"].astype(int)
    P  = df["number of tools"].astype(int)

    TN = (total_tools - P) - FP
    TN = TN.clip(lower=0)  # just in case of noisy data

    # Ensure numeric float Series
    TP = pd.to_numeric(TP, errors="coerce").astype(float)
    FP = pd.to_numeric(FP, errors="coerce").astype(float)
    FN = pd.to_numeric(FN, errors="coerce").astype(float)
    TN = pd.to_numeric(TN, errors="coerce").astype(float)
    
    # denominators
    pos_pred = TP + FP          # predicted positives
    pos_true = TP + FN          # actual positives
    union = (TP + FP + FN)

    # per-row metrics
    df = df.copy()
    df["TP"] = TP
    df["FP"] = FP
    df["FN"] = FN
    df["TN"] = TN

    # precision = TP / (TP + FP)
    df["precision"] = (TP / pos_pred.where(pos_pred != 0, np.nan)).fillna(0.0)

    # recall = TP / (TP + FN)
    df["recall"] = (TP / pos_true.where(pos_true != 0, np.nan)).fillna(0.0)

    # miss rate = FN / (TP + FN) = 1 - recall
    df["miss_rate"] = (FN / pos_true.where(pos_true != 0, np.nan)).fillna(0.0)

    # F1 = 2PR / (P+R)
    _den = (df["precision"] + df["recall"])
    df["F1"] = (2 * df["precision"] * df["recall"] / _den.where(_den != 0, np.nan)).fillna(0.0)

    # accuracy over tool universe size T (per row)
    df["accuracy"] = ((TP + TN) / float(total_tools)).clip(upper=1.0)

    # hallucinations = FP / (TP + FN)  (your FP per actual positive)
    df["hallucinations"] = (FP / pos_true.where(pos_true != 0, np.nan)).fillna(0.0)



    return df

def per_run_aggregate(df: pd.DataFrame, total_tools: int) -> dict:
    """
    Run-level summaries:
      (1) avg number_of_tools across rows
      (2) ratios: sum(TP)/sum(P), sum(FN)/sum(P), sum(FP)/sum(P)
      plus a few useful overall metrics.
    """
    TP_sum = df["TP"].sum()
    FN_sum = df["FN"].sum()
    FP_sum = df["FP"].sum()

    # TN summed across rows as defined (sum over rows of TN)
    TN_sum = df["TN"].sum()

    # macro = mean of per-row values
    precision_macro = df["precision"].mean()
    recall_macro    = df["recall"].mean()
    miss_rate_macro = df["miss_rate"].mean()
    f1_macro        = df["F1"].mean()
    accuracy_macro  = df["accuracy"].mean()
    halluc_macro    = df["hallucinations"].mean()

    # micro = sum-based
    precision_micro = safe_div(TP_sum, TP_sum + FP_sum)
    recall_micro    = safe_div(TP_sum, TP_sum + FN_sum)
    miss_rate_micro = safe_div(FN_sum, TP_sum + FN_sum)
    f1_micro        = safe_div(2 * precision_micro * recall_micro, precision_micro + recall_micro)
    accuracy_micro  = safe_div(TP_sum + TN_sum, total_tools * len(df))
    halluc_micro    = safe_div(FP_sum, TP_sum + FN_sum)


    return {
        "TP_sum": TP_sum, "FN_sum": FN_sum, "FP_sum": FP_sum, "TN_sum": TN_sum,

        # macro metrics
        "precision_mean": precision_macro,
        "recall_mean": recall_macro,
        "miss_rate_mean": miss_rate_macro,
        "f1_mean": f1_macro,
        "accuracy_mean": accuracy_macro,
        "hallucinations_mean": halluc_macro,

        # micro metrics
        "precision_all": precision_micro,
        "recall_all": recall_micro,
        "miss_rate_all": miss_rate_micro,
        "f1_all": f1_micro,
        "accuracy_all": accuracy_micro,
        "hallucinations_all": halluc_micro
    }


def process_root(root_dir: str,
                 comparison_filename: str = "comparison_summary.csv",
                 total_tools: int = 20,
                 write_augmented: bool = True,
                 augmented_name: str = "comparison_with_metrics.csv",
                 unified_out: str = "all_runs_metrics.csv"):

    rows = []
    for entry in sorted(os.listdir(root_dir)):
        run_dir = os.path.join(root_dir, entry)
        comp_path = os.path.join(run_dir, comparison_filename)
        if not os.path.isdir(run_dir) or not os.path.isfile(comp_path):
            continue

        try:
            df = pd.read_csv(comp_path)
        except Exception as e:
            print(f"[WARN] Skipping {comp_path}: {e}")
            continue

        # Add row metrics and save per-run augmented CSV
        df_aug = add_row_metrics(df, total_tools=total_tools)
        if write_augmented:
            df_aug.to_csv(os.path.join(run_dir, augmented_name), index=False)

        # Per-run aggregates (using df_aug since it contains derived cols)
        agg = per_run_aggregate(df_aug, total_tools=total_tools)
        agg["run"] = entry
        rows.append(agg)

    if not rows:
        print(f"[INFO] No runs found in {root_dir} with '{comparison_filename}'.")
        return

    unified = pd.DataFrame(rows).set_index("run")
    unified.to_csv(os.path.join(root_dir, unified_out))
    print(f"[OK] Wrote unified metrics: {os.path.join(root_dir, unified_out)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute per-row and per-run metrics across many runs.")
    ap.add_argument("--root_dir", type=str, default="extractor_eval", help="Root directory containing many run subfolders.")
    ap.add_argument("--comparison_filename", type=str, default="comparison_summary.csv")
    ap.add_argument("--total_tools", type=int, default=20, help="Size of the tool universe (e.g., 20).")
    ap.add_argument("--no_write_augmented", action="store_true",
                    help="Do not write per-run augmented CSVs.")
    ap.add_argument("--augmented_name", type=str, default="comparison_with_metrics.csv")
    ap.add_argument("--unified_out", type=str, default="all_runs_metrics.csv")
    args = ap.parse_args()

    process_root(
        root_dir=args.root_dir,
        comparison_filename=args.comparison_filename,
        total_tools=args.total_tools,
        write_augmented=not args.no_write_augmented,
        augmented_name=args.augmented_name,
        unified_out=args.unified_out,
    )
