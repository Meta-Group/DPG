#!/usr/bin/env python3
"""Analyze readability/cost trade-offs for journal DPG-local experiments.

This report reuses locked final-test per-sample outputs. It addresses the
reviewer concern that path/graph explanations may be hard to read by measuring
the size and runtime cost of each explanation object and by comparing DPG-local
with simple path controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "experiments_local_explanation/results_journal_v1/"
    "final_test_main_with_lore_path_controls/per_sample_selected_test.csv"
)
DEFAULT_OUT = Path("experiments_local_explanation/results_journal_v1/readability_cost_analysis")

METHOD_GROUPS = {
    "dpg": "DPG",
    "dpg_execution_trace": "DPG",
    "tree_path": "path-feature",
    "raw_path_union": "path-control",
    "path_bag": "path-control",
    "random_same_size_path": "path-control",
    "shap": "feature-attribution",
    "lime": "feature-attribution",
    "anchors": "rule",
    "lore": "local-surrogate-rule",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    required = {"dataset", "method", "sample_idx"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    numeric_cols = [
        "num_active_nodes",
        "num_active_edges_filtered",
        "num_paths",
        "runtime_ms",
        "local_matches_model",
        "local_correct",
        "explanation_confidence",
        "support_margin",
        "competitor_exposure",
        "path_control_predicate_count",
        "path_control_unique_predicate_count",
        "path_control_unique_feature_count",
        "path_control_mean_path_length",
        "contrib_nnz",
        "anchor_coverage",
        "anchor_precision",
        "lore_rule_length",
        "lore_fidelity_neighborhood",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["local_matches_model", "local_correct"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df["method_group"] = df["method"].map(METHOD_GROUPS).fillna("other")
    df["display_size"] = _display_size(df)
    df["diagnostic_signal"] = _diagnostic_signal(df)
    df["signal_per_100_nodes"] = 100.0 * df["diagnostic_signal"] / df["display_size"].replace(0, np.nan)
    return df


def _first_available(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in cols:
        if col in df.columns:
            out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out


def _display_size(df: pd.DataFrame) -> pd.Series:
    path_control_size = _first_available(
        df,
        [
            "path_control_unique_predicate_count",
            "path_control_predicate_count",
        ],
    )
    graph_size = _first_available(df, ["num_active_nodes"])
    feature_size = _first_available(df, ["contrib_nnz", "lore_rule_length"])
    anchor_size = pd.Series(np.nan, index=df.index, dtype=float)
    if "anchor_rule" in df.columns:
        anchor_size = df["anchor_rule"].fillna("").astype(str).map(
            lambda value: np.nan if value.strip() in {"", "nan", "None"} else float(value.count(" AND ") + 1)
        )
    out = pd.Series(np.nan, index=df.index, dtype=float)
    methods = df["method"].astype(str)
    out = out.mask(methods.isin({"raw_path_union", "path_bag", "random_same_size_path"}), path_control_size)
    out = out.mask(methods.isin({"dpg", "dpg_execution_trace", "tree_path", "lore"}), graph_size.fillna(feature_size))
    out = out.mask(methods.isin({"shap", "lime"}), feature_size)
    out = out.mask(methods.eq("anchors"), anchor_size)
    return out


def _diagnostic_signal(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ["explanation_confidence", "support_margin"]:
        if col in df.columns:
            parts.append(pd.to_numeric(df[col], errors="coerce"))
    if "competitor_exposure" in df.columns:
        parts.append(1.0 - pd.to_numeric(df["competitor_exposure"], errors="coerce"))
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1)


def _q(series: pd.Series, q: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.quantile(q))


def _mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else np.nan


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_samples": int(sub.shape[0]),
                "mean_display_size": _mean(sub["display_size"]),
                "median_display_size": _q(sub["display_size"], 0.50),
                "p90_display_size": _q(sub["display_size"], 0.90),
                "mean_active_edges": _mean(sub.get("num_active_edges_filtered", pd.Series(dtype=float))),
                "mean_num_paths": _mean(sub.get("num_paths", pd.Series(dtype=float))),
                "mean_runtime_ms": _mean(sub.get("runtime_ms", pd.Series(dtype=float))),
                "p90_runtime_ms": _q(sub.get("runtime_ms", pd.Series(dtype=float)), 0.90),
                "mean_local_match": _mean(sub.get("local_matches_model", pd.Series(dtype=float))),
                "mean_local_accuracy": _mean(sub.get("local_correct", pd.Series(dtype=float))),
                "mean_confidence": _mean(sub.get("explanation_confidence", pd.Series(dtype=float))),
                "mean_support_margin": _mean(sub.get("support_margin", pd.Series(dtype=float))),
                "mean_competitor_exposure": _mean(sub.get("competitor_exposure", pd.Series(dtype=float))),
                "mean_signal_per_100_nodes": _mean(sub["signal_per_100_nodes"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_ratios(df: pd.DataFrame) -> pd.DataFrame:
    method_summary = _summary(df, ["method"])
    index = method_summary.set_index("method")
    rows = []
    base = "dpg_execution_trace"
    comparators = ["raw_path_union", "path_bag", "random_same_size_path", "tree_path", "shap", "lime", "anchors", "lore"]
    if base not in index.index:
        return pd.DataFrame()
    for comp in comparators:
        if comp not in index.index:
            continue
        base_row = index.loc[base]
        comp_row = index.loc[comp]
        rows.append(
            {
                "base_method": base,
                "comparator": comp,
                "display_size_ratio_comparator_over_dpg": _ratio(
                    comp_row["mean_display_size"], base_row["mean_display_size"]
                ),
                "runtime_ratio_comparator_over_dpg": _ratio(comp_row["mean_runtime_ms"], base_row["mean_runtime_ms"]),
                "local_match_delta_dpg_minus_comparator": base_row["mean_local_match"] - comp_row["mean_local_match"],
                "local_accuracy_delta_dpg_minus_comparator": base_row["mean_local_accuracy"]
                - comp_row["mean_local_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def _ratio(num: float, denom: float) -> float:
    if pd.isna(num) or pd.isna(denom) or denom == 0:
        return np.nan
    return float(num / denom)


def _dataset_pressure(df: pd.DataFrame) -> pd.DataFrame:
    dpg = df[df["method"].eq("dpg_execution_trace")].copy()
    if dpg.empty:
        return pd.DataFrame()
    cols = [
        "dataset",
        "n_samples",
        "mean_display_size",
        "p90_display_size",
        "mean_active_edges",
        "mean_runtime_ms",
        "p90_runtime_ms",
        "mean_local_match",
        "mean_confidence",
        "mean_competitor_exposure",
    ]
    table = _summary(dpg, ["dataset"])
    return table[[c for c in cols if c in table.columns]].sort_values(
        ["mean_display_size", "mean_runtime_ms"], ascending=[False, False]
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


def _write_report(
    out_dir: Path,
    source: Path,
    by_method: pd.DataFrame,
    by_group: pd.DataFrame,
    ratios: pd.DataFrame,
    pressure: pd.DataFrame,
) -> None:
    compact_cols = [
        "method",
        "method_group",
        "n_samples",
        "mean_display_size",
        "p90_display_size",
        "mean_runtime_ms",
        "p90_runtime_ms",
        "mean_local_match",
        "mean_local_accuracy",
        "mean_confidence",
        "mean_support_margin",
        "mean_competitor_exposure",
        "mean_signal_per_100_nodes",
    ]
    method_compact = by_method[[c for c in compact_cols if c in by_method.columns]].copy()
    if "mean_display_size" in method_compact:
        method_compact = method_compact.sort_values("mean_display_size")

    lines = [
        "# Readability And Cost Analysis",
        "",
        f"Source: `{source}`",
        "",
        "This analysis measures the size and runtime cost of explanation objects in the locked final-test output. The `display_size` proxy is the number of graph nodes for DPG, the number of unique predicates for path controls, nonzero feature contributions for attribution methods, and rule length for rule/surrogate methods when available.",
        "",
        "## Summary By Method",
        "",
        _markdown_table(method_compact),
        "",
        "## Summary By Explanation Family",
        "",
        _markdown_table(by_group.sort_values("mean_display_size")),
        "",
        "## DPG-Local Versus Comparators",
        "",
        _markdown_table(ratios),
        "",
        "## Highest-Cost DPG Datasets",
        "",
        _markdown_table(pressure.head(15)),
        "",
        "## Writing Guidance",
        "",
        "- Use this as a readability/cost supplement, not as a new fidelity benchmark.",
        "- Path controls can be output-faithful but often require substantially larger predicate sets, supporting the claim that DPG-local adds a structured diagnostic view rather than merely dumping all paths.",
        "- Tree-path summaries are compact, but they discard predicate-transition topology and predicted-versus-competitor branch structure.",
        "- High-cost datasets should be discussed as readability/scalability boundaries and motivate top-k, support-threshold, or sampled graph views.",
        "- Do not claim a universal readability advantage: DPG-local is a graph object and needs focused views for large or high-dimensional cases.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze explanation readability/cost trade-offs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = _read(args.input)
    by_method = _summary(df, ["method", "method_group"])
    by_group = _summary(df, ["method_group"])
    ratios = _pairwise_ratios(df)
    pressure = _dataset_pressure(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "readability_per_sample.csv", index=False)
    by_method.to_csv(args.out_dir / "readability_by_method.csv", index=False)
    by_group.to_csv(args.out_dir / "readability_by_group.csv", index=False)
    ratios.to_csv(args.out_dir / "readability_dpg_ratios.csv", index=False)
    pressure.to_csv(args.out_dir / "readability_dpg_dataset_pressure.csv", index=False)
    _write_report(args.out_dir, args.input, by_method, by_group, ratios, pressure)

    print(f"Saved readability/cost analysis to {args.out_dir}")
    print(f"  - readability_per_sample.csv ({df.shape[0]} rows)")
    print("  - readability_by_method.csv")
    print("  - readability_by_group.csv")
    print("  - readability_dpg_ratios.csv")
    print("  - readability_dpg_dataset_pressure.csv")
    print("  - summary.md")


if __name__ == "__main__":
    main()
