#!/usr/bin/env python3
"""Summarize the focused DPG depth/ensemble scalability experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_METRICS = [
    "model_accuracy",
    "local_matches_model_rate",
    "local_accuracy",
    "avg_explanation_confidence",
    "avg_support_margin",
    "avg_competitor_exposure",
    "avg_edge_recall",
    "avg_edge_precision",
    "avg_recombination_rate",
    "avg_num_paths",
    "avg_num_active_nodes",
    "avg_num_active_edges_filtered",
    "avg_runtime_ms",
]


def _read_summaries(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/summary.csv")):
        df = pd.read_csv(path, low_memory=False)
        df["source_summary_path"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No dataset summary.csv files found under {root}")
    out = pd.concat(frames, ignore_index=True)
    out["max_depth_label"] = out["max_depth"].astype(str).replace({"nan": "None", "": "None"})
    return out


def _mean_table(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    agg = {"dataset": "nunique", "config_id": "count", "n_test_evaluated": "sum"}
    for metric in KEY_METRICS:
        if metric in df.columns:
            agg[metric] = "mean"
    table = df.groupby(list(group_cols), dropna=False).agg(agg).reset_index()
    table = table.rename(
        columns={
            "dataset": "n_datasets",
            "config_id": "n_runs",
            "n_test_evaluated": "total_evaluated_samples",
        }
    )
    return table


def _stress_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_deep"] = pd.to_numeric(out["max_depth"], errors="coerce").fillna(9999) >= 8
    out["is_large_forest"] = pd.to_numeric(out["n_estimators"], errors="coerce") >= 50
    return out


def _worst_cases(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "method",
        "graph_construction_mode",
        "n_estimators",
        "max_depth_label",
        "seed",
        "model_accuracy",
        "local_matches_model_rate",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_num_active_nodes",
        "avg_num_active_edges_filtered",
        "avg_runtime_ms",
        "local_failure_rate",
    ]
    available = [c for c in cols if c in df.columns]
    return (
        df[available]
        .sort_values(["local_matches_model_rate", "avg_runtime_ms", "avg_num_active_nodes"], ascending=[True, False, False])
        .head(25)
        .reset_index(drop=True)
    )


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    if df.empty:
        return ""
    cols = list(df.columns)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value)

    rows = [[fmt(row[col]) for col in cols] for _, row in df.iterrows()]
    widths = [max(len(str(col)), *(len(row[idx]) for row in rows)) for idx, col in enumerate(cols)]
    header = "| " + " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(cols))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(cols))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _compact(table: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "graph_construction_mode",
        "method",
        "n_estimators",
        "max_depth_label",
        "n_datasets",
        "n_runs",
        "local_matches_model_rate",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_edge_recall",
        "avg_recombination_rate",
        "avg_num_active_nodes",
        "avg_runtime_ms",
    ]
    return table[[c for c in keep if c in table.columns]].copy()


def _write_report(out_dir: Path, by_depth: pd.DataFrame, by_depth_est: pd.DataFrame, worst: pd.DataFrame) -> None:
    lines = [
        "# DPG Depth And Ensemble Scalability",
        "",
        "This focused stress experiment evaluates whether DPG-local remains usable beyond the shallow depth-4 random forests used in the main ECML submission.",
        "",
        "The goal is not to open a new broad benchmark, but to bound the journal claim about random forests with deeper trees and larger ensembles.",
        "",
        "## Summary By Depth",
        "",
        _markdown_table(_compact(by_depth)),
        "",
        "## Summary By Depth And Ensemble Size",
        "",
        _markdown_table(_compact(by_depth_est)),
        "",
        "## Lowest-Match / Highest-Cost Runs",
        "",
        _markdown_table(worst),
        "",
        "## Writing Guidance",
        "",
        "- Use this experiment to answer the reviewer concern about shallow random forests.",
        "- If graph size or runtime grows sharply, narrow the claim to small-to-moderate random forests or report pruning as a deployment requirement.",
        "- If trace recovery and diagnostic scores remain stable, report this as robustness evidence, not as a new main contribution.",
        "- Keep output-fidelity metrics as context; the primary scalability quantities are graph size, runtime, edge recall, and recombination.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DPG depth/ensemble scalability outputs.")
    parser.add_argument(
        "--run_root",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/dpg_scalability_depth_by_dataset"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/dpg_scalability_depth_report"),
    )
    args = parser.parse_args()

    raw = _stress_flags(_read_summaries(args.run_root))
    by_depth = _mean_table(raw, ["graph_construction_mode", "method", "max_depth_label"])
    by_depth_est = _mean_table(raw, ["graph_construction_mode", "method", "n_estimators", "max_depth_label"])
    worst = _worst_cases(raw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_dir / "scalability_all_runs.csv", index=False)
    by_depth.to_csv(args.out_dir / "scalability_by_depth.csv", index=False)
    by_depth_est.to_csv(args.out_dir / "scalability_by_depth_estimators.csv", index=False)
    worst.to_csv(args.out_dir / "scalability_worst_cases.csv", index=False)
    _write_report(args.out_dir, by_depth=by_depth, by_depth_est=by_depth_est, worst=worst)

    print(f"Saved scalability report to {args.out_dir}")
    print(f"  - scalability_all_runs.csv ({raw.shape[0]} rows)")
    print("  - scalability_by_depth.csv")
    print("  - scalability_by_depth_estimators.csv")
    print("  - scalability_worst_cases.csv")
    print("  - summary.md")


if __name__ == "__main__":
    main()
