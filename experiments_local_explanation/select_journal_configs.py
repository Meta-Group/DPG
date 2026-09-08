#!/usr/bin/env python3
"""Select journal experiment configurations from validation summaries.

This script is the bridge between validation-grid runs and final test runs.
It reads DPG and/or baseline validation summary CSVs, ranks configurations by
registered journal tie-breakers, and writes selected configuration manifests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


DPG_SORT_COLUMNS = [
    "local_matches_model_rate",
    "avg_edge_recall",
    "avg_edge_precision",
    "avg_recombination_rate",
    "avg_explanation_confidence",
    "local_accuracy",
]
DPG_ASCENDING = [False, False, False, True, False, False]

BASELINE_SORT_COLUMNS = [
    "local_matches_model_rate",
    "local_accuracy",
    "avg_score_margin_pred_vs_competitor",
    "avg_runtime_ms",
]
BASELINE_ASCENDING = [False, False, False, True]


def _parse_paths(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        for raw in str(value).split(","):
            item = raw.strip()
            if item:
                paths.append(Path(item).resolve())
    return paths


def _load_csvs(paths: Iterable[Path], required_name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for path in paths:
        if path.is_dir():
            path = path / required_name
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        df["source_summary_path"] = str(path)
        frames.append(df)
    if missing:
        raise FileNotFoundError("Missing summary files:\n" + "\n".join(missing))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_method_filter(text: str) -> set[str]:
    return {item.strip().lower() for item in text.split(",") if item.strip()}


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


def _rank_and_select(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    sort_cols: Sequence[str],
    ascending: Sequence[bool],
    selection_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ranked = _ensure_columns(df, sort_cols)
    ranked = ranked.copy()
    ranked["selection_kind"] = selection_kind
    ranked["selection_metric_primary"] = sort_cols[0]
    ranked["selection_tie_breakers"] = ",".join(sort_cols[1:])

    full_sort_cols = list(group_cols) + list(sort_cols) + ["config_id", "seed"]
    full_ascending = [True] * len(group_cols) + list(ascending) + [True, True]
    ranked = ranked.sort_values(full_sort_cols, ascending=full_ascending, na_position="last").reset_index(drop=True)
    ranked["selection_rank"] = ranked.groupby(list(group_cols)).cumcount() + 1

    selected = ranked[ranked["selection_rank"] == 1].reset_index(drop=True)
    return selected, ranked


def _write_if_any(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select journal configs from validation summary CSVs.")
    parser.add_argument(
        "--dpg_summaries",
        nargs="*",
        default=[],
        help="DPG summary.csv paths or directories. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--baseline_summaries",
        nargs="*",
        default=[],
        help="Baseline summary_baselines.csv paths or directories. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/selection"),
    )
    parser.add_argument(
        "--include_baseline_methods",
        type=str,
        default="",
        help="Optional comma-separated baseline methods to keep before ranking.",
    )
    parser.add_argument(
        "--exclude_baseline_methods",
        type=str,
        default="",
        help="Optional comma-separated baseline methods to remove before ranking.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dpg_paths = _parse_paths(args.dpg_summaries)
    baseline_paths = _parse_paths(args.baseline_summaries)

    dpg_df = _load_csvs(dpg_paths, "summary.csv") if dpg_paths else pd.DataFrame()
    baseline_df = _load_csvs(baseline_paths, "summary_baselines.csv") if baseline_paths else pd.DataFrame()
    if not baseline_df.empty:
        include_methods = _parse_method_filter(args.include_baseline_methods)
        exclude_methods = _parse_method_filter(args.exclude_baseline_methods)
        methods = baseline_df["method"].astype(str).str.lower()
        if include_methods:
            baseline_df = baseline_df[methods.isin(include_methods)].copy()
            methods = baseline_df["method"].astype(str).str.lower()
        if exclude_methods:
            baseline_df = baseline_df[~methods.isin(exclude_methods)].copy()

    selected_frames: List[pd.DataFrame] = []
    ranked_frames: List[pd.DataFrame] = []

    if not dpg_df.empty:
        selected_dpg, ranked_dpg = _rank_and_select(
            dpg_df,
            group_cols=["dataset", "method"],
            sort_cols=DPG_SORT_COLUMNS,
            ascending=DPG_ASCENDING,
            selection_kind="dpg_validation",
        )
        _write_if_any(selected_dpg, out_dir / "selected_dpg_configs.csv")
        _write_if_any(ranked_dpg, out_dir / "ranked_dpg_configs.csv")
        selected_frames.append(selected_dpg)
        ranked_frames.append(ranked_dpg)

    if not baseline_df.empty:
        selected_baselines, ranked_baselines = _rank_and_select(
            baseline_df,
            group_cols=["dataset", "method"],
            sort_cols=BASELINE_SORT_COLUMNS,
            ascending=BASELINE_ASCENDING,
            selection_kind="baseline_validation",
        )
        _write_if_any(selected_baselines, out_dir / "selected_baseline_configs.csv")
        _write_if_any(ranked_baselines, out_dir / "ranked_baseline_configs.csv")
        selected_frames.append(selected_baselines)
        ranked_frames.append(ranked_baselines)

    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    ranked_all = pd.concat(ranked_frames, ignore_index=True) if ranked_frames else pd.DataFrame()
    _write_if_any(selected_all, out_dir / "selected_configs.csv")
    _write_if_any(ranked_all, out_dir / "ranked_configs.csv")

    print(f"Saved selected configs: {out_dir / 'selected_configs.csv'} ({len(selected_all)} rows)")
    print(f"Saved ranked configs: {out_dir / 'ranked_configs.csv'} ({len(ranked_all)} rows)")
    if not dpg_df.empty:
        print(f"Saved DPG selections: {out_dir / 'selected_dpg_configs.csv'}")
    if not baseline_df.empty:
        print(f"Saved baseline selections: {out_dir / 'selected_baseline_configs.csv'}")


if __name__ == "__main__":
    main()
