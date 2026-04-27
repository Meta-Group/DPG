"""Analyze lightweight local explanation experiment outputs."""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Sequence, Tuple

import pandas as pd


REQUIRED_SUMMARY_COLUMNS = {
    "dataset",
    "graph_construction_mode",
    "model_accuracy",
    "local_matches_model_rate",
    "local_accuracy",
    "avg_vote_confidence",
    "avg_evidence_score_pred",
    "avg_trace_coverage_score",
    "avg_recombination_rate",
    "avg_num_paths",
}

REQUIRED_PER_SAMPLE_COLUMNS = {
    "dataset",
    "graph_construction_mode",
    "local_matches_model",
    "local_correct",
    "vote_confidence",
    "evidence_score_pred",
    "trace_coverage_score",
    "recombination_rate",
    "num_paths",
}

SUMMARY_METRICS = [
    "model_accuracy",
    "local_matches_model_rate",
    "local_accuracy",
    "avg_vote_confidence",
    "avg_evidence_score_pred",
    "avg_trace_coverage_score",
    "avg_recombination_rate",
    "avg_num_paths",
]

COHORT_METRICS = [
    "vote_confidence",
    "evidence_score_pred",
    "trace_coverage_score",
    "recombination_rate",
    "num_paths",
]


def load_results(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = os.path.join(results_dir, "summary.csv")
    per_sample_path = os.path.join(results_dir, "per_sample.csv")

    if not os.path.exists(summary_path):
        raise ValueError(f"summary.csv not found in results_dir: {results_dir}")
    if not os.path.exists(per_sample_path):
        raise ValueError(f"per_sample.csv not found in results_dir: {results_dir}")

    summary_df = pd.read_csv(summary_path)
    per_sample_df = pd.read_csv(per_sample_path)

    if summary_df.empty:
        raise ValueError("summary.csv is empty.")
    if per_sample_df.empty:
        raise ValueError("per_sample.csv is empty.")

    missing_summary = REQUIRED_SUMMARY_COLUMNS - set(summary_df.columns)
    missing_per_sample = REQUIRED_PER_SAMPLE_COLUMNS - set(per_sample_df.columns)
    if missing_summary:
        raise ValueError(f"summary.csv is missing required columns: {sorted(missing_summary)}")
    if missing_per_sample:
        raise ValueError(f"per_sample.csv is missing required columns: {sorted(missing_per_sample)}")

    return summary_df, per_sample_df


def aggregate_summary(summary_df: pd.DataFrame, group_by: Sequence[str]) -> pd.DataFrame:
    grouped = summary_df.groupby(list(group_by), dropna=False)
    aggregated = grouped[SUMMARY_METRICS].agg(["mean", "std", "count"]).reset_index()
    aggregated.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in aggregated.columns.to_flat_index()
    ]
    return aggregated


def build_cohort_summary(per_sample_df: pd.DataFrame, group_by: Sequence[str]) -> pd.DataFrame:
    def assign_cohort(row: pd.Series) -> str:
        correct = bool(row["local_correct"])
        matches_model = bool(row["local_matches_model"])
        if correct and matches_model:
            return "correct_and_matches_model"
        if correct and not matches_model:
            return "correct_but_disagrees_model"
        if not correct and matches_model:
            return "wrong_but_matches_model"
        return "wrong_and_disagrees_model"

    enriched = per_sample_df.copy()
    enriched["cohort"] = enriched.apply(assign_cohort, axis=1)
    grouped = enriched.groupby(list(group_by) + ["cohort"], dropna=False)
    aggregated = grouped[COHORT_METRICS].mean().reset_index()
    counts = grouped.size().reset_index(name="n_samples")
    aggregated = aggregated.merge(counts, on=list(group_by) + ["cohort"], how="left")
    aggregated = aggregated.rename(
        columns={
            "vote_confidence": "mean_vote_confidence",
            "evidence_score_pred": "mean_evidence_score_pred",
            "trace_coverage_score": "mean_trace_coverage_score",
            "recombination_rate": "mean_recombination_rate",
            "num_paths": "mean_num_paths",
        }
    )
    return aggregated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze local explanation experiment CSV outputs.")
    parser.add_argument("--results_dir", required=True, help="Directory containing summary.csv and per_sample.csv")
    parser.add_argument("--out_dir", required=True, help="Directory where aggregate CSVs are written")
    parser.add_argument(
        "--group_by",
        default="dataset,graph_construction_mode",
        help="Comma-separated grouping columns. Default: dataset,graph_construction_mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    group_by = [item.strip() for item in args.group_by.split(",") if item.strip()]
    if not group_by:
        raise ValueError("group_by must include at least one column.")

    summary_df, per_sample_df = load_results(args.results_dir)
    for col in group_by:
        if col not in summary_df.columns:
            raise ValueError(f"group_by column '{col}' not found in summary.csv")
        if col not in per_sample_df.columns:
            raise ValueError(f"group_by column '{col}' not found in per_sample.csv")

    os.makedirs(args.out_dir, exist_ok=True)
    aggregate_df = aggregate_summary(summary_df, group_by)
    cohort_df = build_cohort_summary(per_sample_df, group_by)
    aggregate_df.to_csv(os.path.join(args.out_dir, "aggregate_summary.csv"), index=False)
    cohort_df.to_csv(os.path.join(args.out_dir, "cohort_summary.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
