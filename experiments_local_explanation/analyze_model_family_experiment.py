#!/usr/bin/env python3
"""Summarize DPG-supported sklearn classification model-family experiments."""

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
    "avg_model_vote_agreement",
    "avg_edge_recall",
    "avg_edge_precision",
    "avg_recombination_rate",
    "avg_num_paths",
    "avg_num_active_nodes",
    "avg_num_active_edges_filtered",
    "avg_runtime_ms",
    "local_failure_rate",
]


def _read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if "model_family" not in df.columns:
        raise ValueError(f"{path} does not contain a model_family column")
    if "dataset" not in df.columns:
        raise ValueError(f"{path} does not contain a dataset column")
    return df


def _read_summary_tree(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No dataset summary.csv files found under {root}")
    frames = [_read_summary(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _mean_table(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    group_cols = list(group_cols)
    agg = {"config_id": "count", "n_test_evaluated": "sum"}
    if "dataset" not in group_cols:
        agg["dataset"] = "nunique"
    for metric in KEY_METRICS:
        if metric in df.columns:
            agg[metric] = "mean"
    table = df.groupby(list(group_cols), dropna=False).agg(agg).reset_index()
    table = table.rename(
        columns={
            "config_id": "n_runs",
            "n_test_evaluated": "total_evaluated_samples",
        }
    )
    if "dataset" not in group_cols and "dataset" in table.columns:
        table = table.rename(columns={"dataset": "n_datasets"})
    return table


def _compact(table: pd.DataFrame, include_dataset: bool = False) -> pd.DataFrame:
    keep = [
        "dataset" if include_dataset else "",
        "model_family",
        "method",
        "n_datasets",
        "n_runs",
        "total_evaluated_samples",
        "model_accuracy",
        "local_matches_model_rate",
        "local_accuracy",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_model_vote_agreement",
        "avg_edge_recall",
        "avg_recombination_rate",
        "avg_num_active_nodes",
        "avg_runtime_ms",
        "local_failure_rate",
    ]
    return table[[c for c in keep if c and c in table.columns]].copy()


def _worst_cases(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "model_family",
        "method",
        "model_accuracy",
        "local_matches_model_rate",
        "local_accuracy",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_edge_recall",
        "avg_recombination_rate",
        "avg_num_active_nodes",
        "avg_runtime_ms",
        "local_failure_rate",
    ]
    available = [c for c in cols if c in df.columns]
    return (
        df[available]
        .sort_values(
            ["local_matches_model_rate", "avg_competitor_exposure", "avg_runtime_ms"],
            ascending=[True, False, False],
        )
        .head(15)
        .reset_index(drop=True)
    )


def _best_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "model_family",
        "model_accuracy",
        "local_matches_model_rate",
        "local_accuracy",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_runtime_ms",
    ]
    available = [c for c in cols if c in df.columns]
    return (
        df[available]
        .sort_values(
            ["dataset", "local_matches_model_rate", "avg_explanation_confidence", "avg_runtime_ms"],
            ascending=[True, False, False, True],
        )
        .groupby("dataset", as_index=False, dropna=False)
        .head(2)
        .reset_index(drop=True)
    )


def _dataset_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "model_family": "nunique",
        "local_matches_model_rate": "mean",
        "avg_explanation_confidence": "mean",
        "avg_support_margin": "mean",
        "avg_competitor_exposure": "mean",
        "avg_num_active_nodes": "mean",
        "avg_runtime_ms": "mean",
    }
    table = df.groupby("dataset", dropna=False).agg(agg).reset_index()
    table = table.rename(columns={"model_family": "n_model_families"})
    return table.sort_values(
        ["local_matches_model_rate", "avg_competitor_exposure"],
        ascending=[True, False],
    ).reset_index(drop=True)


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


def _write_report(
    out_dir: Path,
    source: Path,
    by_family: pd.DataFrame,
    by_dataset_family: pd.DataFrame,
    difficulty: pd.DataFrame,
    best: pd.DataFrame,
    worst: pd.DataFrame,
) -> None:
    lines = [
        "# DPG-Supported Model-Family Report",
        "",
        f"Source: `{source}`",
        "",
        "This focused experiment tests whether DPG-local execution-trace diagnostics remain usable across sklearn classification ensemble families currently handled by the local experiment runner.",
        "",
        "Included families: RandomForest, ExtraTrees, AdaBoost with tree base learners, and Bagging with tree base learners. GradientBoosting is intentionally excluded until a dedicated adapter handles its 2D estimator layout and additive class-contribution semantics.",
        "",
        "## Summary By Model Family",
        "",
        _markdown_table(_compact(by_family)),
        "",
        "## Summary By Dataset And Model Family",
        "",
        _markdown_table(_compact(by_dataset_family, include_dataset=True)),
        "",
        "## Dataset Difficulty Across Families",
        "",
        _markdown_table(difficulty),
        "",
        "## Top Families Per Dataset",
        "",
        _markdown_table(best),
        "",
        "## Lowest-Match / Highest-Competition Runs",
        "",
        _markdown_table(worst),
        "",
        "## Writing Guidance",
        "",
        "- Use this experiment to broaden the paper from RandomForest-only evidence to DPG-compatible sklearn ensemble classifiers.",
        "- The consistent edge recall and zero recombination should be framed as construction robustness across model families, not as the primary utility result.",
        "- Model-family behavior is not uniform: Bagging often gives stronger local match on this subset, ExtraTrees can be weaker on some datasets, and high-dimensional/many-class settings remain difficult.",
        "- `isolet` remains a clear hard regime across families, supporting the failure-mode discussion.",
        "- Avoid claiming GradientBoosting support in the journal experiment until the adapter and its additive semantics are validated.",
        "- Preferred claim: DPG-local provides a common predicate-transition diagnostic representation for several DPG-compatible sklearn classification ensembles, with model-family-dependent diagnostic behavior and computational cost.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DPG model-family experiment outputs.")
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset/summary.csv"),
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=None,
        help="Optional per-dataset output root containing */summary.csv files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset_report"),
    )
    args = parser.parse_args()

    source = args.input_root if args.input_root is not None else args.summary_csv
    raw = _read_summary_tree(args.input_root) if args.input_root is not None else _read_summary(args.summary_csv)
    by_family = _mean_table(raw, ["model_family", "method"])
    by_dataset_family = _mean_table(raw, ["dataset", "model_family", "method"])
    difficulty = _dataset_difficulty(raw)
    best = _best_by_dataset(raw)
    worst = _worst_cases(raw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_dir / "model_family_all_runs.csv", index=False)
    by_family.to_csv(args.out_dir / "model_family_by_family.csv", index=False)
    by_dataset_family.to_csv(args.out_dir / "model_family_by_dataset_family.csv", index=False)
    difficulty.to_csv(args.out_dir / "model_family_dataset_difficulty.csv", index=False)
    best.to_csv(args.out_dir / "model_family_top_by_dataset.csv", index=False)
    worst.to_csv(args.out_dir / "model_family_worst_cases.csv", index=False)
    _write_report(args.out_dir, source, by_family, by_dataset_family, difficulty, best, worst)

    print(f"Saved model-family report to {args.out_dir}")
    print(f"  - model_family_all_runs.csv ({raw.shape[0]} rows)")
    print("  - model_family_by_family.csv")
    print("  - model_family_by_dataset_family.csv")
    print("  - model_family_dataset_difficulty.csv")
    print("  - model_family_top_by_dataset.csv")
    print("  - model_family_worst_cases.csv")
    print("  - summary.md")


if __name__ == "__main__":
    main()
