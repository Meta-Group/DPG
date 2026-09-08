#!/usr/bin/env python3
"""Export seed-level metrics, paired pivots, and corrected tests for the journal audit."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DPG_METRIC_COLUMNS = [
    "local_matches_model_rate",
    "local_accuracy",
    "avg_evidence_margin_pred_vs_competitor",
    "avg_edge_precision",
    "avg_edge_recall",
    "avg_recombination_rate",
    "avg_explanation_confidence",
]

BASELINE_METRIC_COLUMNS = [
    "local_matches_model_rate",
    "local_accuracy",
    "avg_score_margin_pred_vs_competitor",
]

PAIRED_METRICS = ["fidelity", "local_acc", "margin"]
SEED_METRICS = [
    "fidelity",
    "local_acc",
    "margin",
    "edge_precision",
    "edge_recall",
    "recombination",
    "explanation_confidence",
]


def strip_seed(config_id: object) -> str:
    text = "" if pd.isna(config_id) else str(config_id)
    return re.sub(r"_s\d+(?=$|__)", "", text)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def collect_dataset_csvs(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        path = ds_dir / filename
        if path.exists():
            df = read_csv(path)
            if not df.empty:
                df["dataset"] = ds_dir.name
                frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def best_baseline_configs(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(
            ["dataset", "method", "local_matches_model_rate", "local_accuracy", "avg_score_margin_pred_vs_competitor"],
            ascending=[True, True, False, False, False],
        )
        .drop_duplicates(subset=["dataset", "method"], keep="first")
        .reset_index(drop=True)
    )


def canonical_dpg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method"] = out["graph_construction_mode"]
    out["config_family"] = out["config_id"].map(strip_seed)
    rename = {
        "local_matches_model_rate": "fidelity",
        "local_accuracy": "local_acc",
        "avg_evidence_margin_pred_vs_competitor": "margin",
        "avg_edge_precision": "edge_precision",
        "avg_edge_recall": "edge_recall",
        "avg_recombination_rate": "recombination",
        "avg_explanation_confidence": "explanation_confidence",
    }
    return out.rename(columns=rename)


def canonical_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["config_family"] = out["config_id"].map(strip_seed)
    rename = {
        "local_matches_model_rate": "fidelity",
        "local_accuracy": "local_acc",
        "avg_score_margin_pred_vs_competitor": "margin",
    }
    return out.rename(columns=rename)


def selected_seed_rows(all_rows: pd.DataFrame, selected_rows: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    keys = selected_rows[key_cols].drop_duplicates()
    return all_rows.merge(keys, on=key_cols, how="inner")


def t_ci95_half_width(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    n = len(values)
    if n < 2:
        return float("nan")
    sem = values.std(ddof=1) / math.sqrt(n)
    return float(stats.t.ppf(0.975, n - 1) * sem)


def seed_ci_table(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), group in selected.groupby(["dataset", "method"], dropna=False):
        row = {
            "dataset": dataset,
            "method": method,
            "n_seeds": int(group["seed"].nunique()) if "seed" in group.columns else 0,
            "seeds": ",".join(str(int(s)) for s in sorted(group["seed"].dropna().unique())) if "seed" in group.columns else "",
        }
        for metric in SEED_METRICS:
            if metric not in group.columns:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_ci95_half_width"] = np.nan
                continue
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_ci95_half_width"] = t_ci95_half_width(values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: (math.inf if pd.isna(pvalues[i]) else pvalues[i]))
    adjusted = [np.nan] * m
    running = 0.0
    for rank, idx in enumerate(order):
        p = pvalues[idx]
        if pd.isna(p):
            adjusted[idx] = np.nan
            continue
        running = max(running, (m - rank) * p)
        adjusted[idx] = min(1.0, running)
    return adjusted


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: (math.inf if pd.isna(pvalues[i]) else pvalues[i]), reverse=True)
    adjusted = [np.nan] * m
    running = 1.0
    for reverse_rank, idx in enumerate(order):
        p = pvalues[idx]
        if pd.isna(p):
            adjusted[idx] = np.nan
            continue
        rank = m - reverse_rank
        running = min(running, p * m / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_wilcoxon_tables(paired_quant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stat_rows = []
    friedman_rows = []
    methods = [m for m in ["execution_trace", "aggregated_transitions", "shap", "ice", "anchors", "tree_path", "lime"] if m in paired_quant["method"].unique()]

    for metric in PAIRED_METRICS:
        pivot = paired_quant.pivot(index="dataset", columns="method", values=metric)
        available = [m for m in methods if m in pivot.columns]
        complete = pivot[available].dropna()
        if len(available) >= 3 and len(complete) >= 3:
            friedman = stats.friedmanchisquare(*[complete[m].values for m in available])
            friedman_rows.append(
                {
                    "metric": metric,
                    "methods": ",".join(available),
                    "n_datasets": int(len(complete)),
                    "statistic": float(friedman.statistic),
                    "pvalue": float(friedman.pvalue),
                }
            )
        for comparator in [m for m in available if m != "execution_trace"]:
            paired = pivot[["execution_trace", comparator]].dropna()
            if len(paired) < 3:
                stat = np.nan
                pvalue = np.nan
            else:
                result = stats.wilcoxon(paired["execution_trace"], paired[comparator], alternative="two-sided", zero_method="wilcox")
                stat = float(result.statistic)
                pvalue = float(result.pvalue)
            stat_rows.append(
                {
                    "metric": metric,
                    "left": "execution_trace",
                    "right": comparator,
                    "n_datasets": int(len(paired)),
                    "statistic": stat,
                    "pvalue_raw": pvalue,
                }
            )

    stat_df = pd.DataFrame(stat_rows)
    for metric, idx in stat_df.groupby("metric").groups.items():
        pvalues = stat_df.loc[idx, "pvalue_raw"].tolist()
        stat_df.loc[idx, "pvalue_holm"] = holm(pvalues)
        stat_df.loc[idx, "pvalue_bh"] = benjamini_hochberg(pvalues)
    return stat_df, pd.DataFrame(friedman_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpg_root", type=Path, default=Path("experiments_local_explanation/experiment_dpg2_next_phase"))
    parser.add_argument("--baseline_root", type=Path, default=Path("experiments_local_explanation/results_baselines_by_dataset"))
    parser.add_argument("--out_dir", type=Path, default=Path("JOURNAL_paper/statistical_audit_2026-07-20"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_configs = read_csv(args.dpg_root / "_analysis" / "best_configs.csv")
    all_dpg = collect_dataset_csvs(args.dpg_root, "summary.csv")
    all_baselines = collect_dataset_csvs(args.baseline_root, "summary_baselines.csv")

    datasets = sorted(best_configs["dataset"].unique())
    all_dpg = all_dpg[all_dpg["dataset"].isin(datasets)].copy()
    all_baselines = all_baselines[all_baselines["dataset"].isin(datasets)].copy()

    best_dpg = canonical_dpg(best_configs)
    all_dpg_canon = canonical_dpg(all_dpg)
    baseline_best = best_baseline_configs(all_baselines)
    all_baseline_canon = canonical_baseline(all_baselines)
    best_baseline_canon = canonical_baseline(baseline_best)

    dpg_metric_cols = ["dataset", "method", "graph_construction_mode", "config_id", "config_family", "seed", "n_estimators", "max_depth", "perc_var", "decimal_threshold"] + SEED_METRICS
    baseline_metric_cols = ["dataset", "method", "config_id", "config_family", "seed", "n_estimators", "max_depth", "perc_var", "decimal_threshold", "fidelity", "local_acc", "margin"]

    all_dpg_canon[[c for c in dpg_metric_cols if c in all_dpg_canon.columns]].to_csv(args.out_dir / "dpg_all_seed_config_metrics.csv", index=False)
    all_baseline_canon[[c for c in baseline_metric_cols if c in all_baseline_canon.columns]].to_csv(args.out_dir / "baseline_all_seed_config_metrics.csv", index=False)

    selected_dpg_seeds = selected_seed_rows(all_dpg_canon, best_dpg, ["dataset", "method", "config_family"])
    selected_baseline_seeds = selected_seed_rows(all_baseline_canon, best_baseline_canon, ["dataset", "method", "config_family"])
    selected_seed_metrics = pd.concat([selected_dpg_seeds, selected_baseline_seeds], ignore_index=True, sort=False)
    selected_seed_metrics[[c for c in dpg_metric_cols if c in selected_seed_metrics.columns]].to_csv(args.out_dir / "selected_per_seed_metrics.csv", index=False)
    seed_ci_table(selected_seed_metrics).to_csv(args.out_dir / "selected_seed_ci_by_dataset_method.csv", index=False)

    best_dpg_table = best_dpg[
        ["dataset", "method", "fidelity", "local_acc", "margin", "edge_precision", "edge_recall", "recombination", "explanation_confidence"]
    ].copy()
    best_baseline_table = best_baseline_canon[["dataset", "method", "fidelity", "local_acc", "margin"]].copy()
    paired_quant = pd.concat([best_dpg_table, best_baseline_table], ignore_index=True, sort=False)
    paired_quant.to_csv(args.out_dir / "paired_quant_selected_long.csv", index=False)

    for metric in PAIRED_METRICS:
        paired_quant.pivot(index="dataset", columns="method", values=metric).sort_index().to_csv(args.out_dir / f"paired_quant_{metric}_pivot.csv")

    wilcoxon, friedman = paired_wilcoxon_tables(paired_quant)
    wilcoxon.to_csv(args.out_dir / "paired_wilcoxon_corrected.csv", index=False)
    friedman.to_csv(args.out_dir / "friedman_tests.csv", index=False)

    summary = pd.DataFrame(
        [
            {"item": "datasets", "value": len(datasets)},
            {"item": "dpg_all_seed_config_rows", "value": len(all_dpg_canon)},
            {"item": "baseline_all_seed_config_rows", "value": len(all_baseline_canon)},
            {"item": "selected_per_seed_rows", "value": len(selected_seed_metrics)},
            {"item": "paired_quant_rows", "value": len(paired_quant)},
        ]
    )
    summary.to_csv(args.out_dir / "export_summary.csv", index=False)

    print(f"Wrote statistical audit exports to {args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
