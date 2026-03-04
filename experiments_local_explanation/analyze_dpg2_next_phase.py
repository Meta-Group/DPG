#!/usr/bin/env python3
"""Aggregate DPG 2.0 next-phase results across dataset folders.

Expected layout:
    <out_root>/<dataset>/summary.csv
    <out_root>/<dataset>/per_sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _collect_frames(out_root: Path, filename: str) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for dataset_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        path = dataset_dir / filename
        if not path.exists():
            continue
        frames.append(pd.read_csv(path))
    return frames


def _best_configs(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    ranked = summary_df.sort_values(
        [
            "dataset",
            "graph_construction_mode",
            "local_matches_model_rate",
            "avg_edge_precision",
            "avg_node_precision",
            "avg_recombination_rate",
            "avg_explanation_confidence",
        ],
        ascending=[True, True, False, False, False, True, False],
    )
    return (
        ranked.groupby(["dataset", "graph_construction_mode"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _cohort_summary(per_sample_df: pd.DataFrame) -> pd.DataFrame:
    if per_sample_df.empty:
        return pd.DataFrame()
    total_counts = (
        per_sample_df.groupby(["dataset", "graph_construction_mode"], as_index=False)
        .agg(total_samples=("sample_idx", "count"))
    )
    cohort = (
        per_sample_df.groupby(["dataset", "graph_construction_mode", "cohort_label"], as_index=False)
        .agg(
            samples=("sample_idx", "count"),
            mean_explanation_confidence=("explanation_confidence", "mean"),
            median_explanation_confidence=("explanation_confidence", "median"),
            mean_node_recall=("node_recall", "mean"),
            mean_edge_precision=("edge_precision", "mean"),
            mean_recombination_rate=("recombination_rate", "mean"),
            mean_path_purity=("path_purity", "mean"),
            mean_competitor_exposure=("competitor_exposure", "mean"),
            mean_critical_split_depth=("critical_split_depth", "mean"),
            mean_critical_node_contrast=("critical_node_contrast", "mean"),
        )
    )
    cohort = cohort.merge(total_counts, on=["dataset", "graph_construction_mode"], how="left")
    cohort["cohort_rate"] = cohort["samples"] / cohort["total_samples"]
    return cohort.sort_values(["dataset", "graph_construction_mode", "cohort_label"]).reset_index(drop=True)


def _agreement_summary(per_sample_df: pd.DataFrame) -> pd.DataFrame:
    if per_sample_df.empty:
        return pd.DataFrame()
    grouped = (
        per_sample_df.groupby(["dataset", "graph_construction_mode", "disagree_with_model"], as_index=False)
        .agg(
            samples=("sample_idx", "count"),
            mean_explanation_confidence=("explanation_confidence", "mean"),
            mean_trace_coverage_score=("trace_coverage_score", "mean"),
            mean_recombination_rate=("recombination_rate", "mean"),
            mean_support_margin=("support_margin", "mean"),
            mean_model_vote_agreement=("model_vote_agreement", "mean"),
        )
        .sort_values(["dataset", "graph_construction_mode", "disagree_with_model"])
        .reset_index(drop=True)
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate DPG 2.0 next-phase experiment outputs.")
    parser.add_argument(
        "--out_root",
        type=str,
        default="experiments_local_explanation/experiment_dpg2_next_phase",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="experiments_local_explanation/experiment_dpg2_next_phase/_analysis",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    analysis_dir = Path(args.analysis_dir).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_frames = _collect_frames(out_root, "summary.csv")
    per_sample_frames = _collect_frames(out_root, "per_sample.csv")

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    per_sample_df = pd.concat(per_sample_frames, ignore_index=True) if per_sample_frames else pd.DataFrame()

    summary_df.to_csv(analysis_dir / "combined_summary.csv", index=False)
    per_sample_df.to_csv(analysis_dir / "combined_per_sample.csv", index=False)

    best_df = _best_configs(summary_df)
    best_df.to_csv(analysis_dir / "best_configs.csv", index=False)

    if not best_df.empty and not per_sample_df.empty:
        best_keys = best_df[["dataset", "config_id", "seed"]].drop_duplicates()
        best_per_sample_df = per_sample_df.merge(best_keys, on=["dataset", "config_id", "seed"], how="inner")
    else:
        best_per_sample_df = pd.DataFrame()
    best_per_sample_df.to_csv(analysis_dir / "best_per_sample.csv", index=False)

    cohort_df = _cohort_summary(best_per_sample_df)
    cohort_df.to_csv(analysis_dir / "best_cohort_summary.csv", index=False)

    agreement_df = _agreement_summary(best_per_sample_df)
    agreement_df.to_csv(analysis_dir / "best_agreement_summary.csv", index=False)

    print(f"Saved: {analysis_dir / 'combined_summary.csv'}")
    print(f"Saved: {analysis_dir / 'combined_per_sample.csv'}")
    print(f"Saved: {analysis_dir / 'best_configs.csv'}")
    print(f"Saved: {analysis_dir / 'best_per_sample.csv'}")
    print(f"Saved: {analysis_dir / 'best_cohort_summary.csv'}")
    print(f"Saved: {analysis_dir / 'best_agreement_summary.csv'}")


if __name__ == "__main__":
    main()
