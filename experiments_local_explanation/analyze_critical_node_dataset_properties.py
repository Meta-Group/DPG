#!/usr/bin/env python3
"""Analyze which dataset properties make critical nodes more useful.

This is intentionally dataset-level: it tests whether global data properties
such as imbalance, dimensionality, class count, and model difficulty explain
critical-node occurrence or intervention usefulness.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _counts_from_json(text: object) -> dict[str, int]:
    if pd.isna(text):
        return {}
    raw = json.loads(str(text))
    return {str(k): int(v) for k, v in raw.items()}


def _imbalance_features(counts: dict[str, int], prefix: str) -> dict[str, float]:
    values = np.asarray(list(counts.values()), dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return {
            f"{prefix}_total": np.nan,
            f"{prefix}_minority_fraction": np.nan,
            f"{prefix}_majority_fraction": np.nan,
            f"{prefix}_imbalance_ratio": np.nan,
            f"{prefix}_normalized_entropy": np.nan,
            f"{prefix}_gini_impurity": np.nan,
        }
    total = float(values.sum())
    probs = values / total
    entropy = -float(np.sum(probs * np.log(probs)))
    normalized_entropy = entropy / math.log(values.size) if values.size > 1 else 0.0
    return {
        f"{prefix}_total": total,
        f"{prefix}_minority_fraction": float(probs.min()),
        f"{prefix}_majority_fraction": float(probs.max()),
        f"{prefix}_imbalance_ratio": float(values.max() / values.min()),
        f"{prefix}_normalized_entropy": normalized_entropy,
        f"{prefix}_gini_impurity": float(1.0 - np.sum(probs**2)),
    }


def _dataset_properties(split_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in split_summary.iterrows():
        out = {
            "dataset": row["dataset"],
            "n_existing_train": row.get("n_existing_train"),
            "n_train_core": row.get("n_train_core"),
            "n_validation": row.get("n_validation"),
            "n_test": row.get("n_test"),
            "n_features": row.get("n_features"),
            "n_classes_train": row.get("n_classes_train"),
            "features_per_train_sample": (
                float(row.get("n_features")) / float(row.get("n_train_core"))
                if float(row.get("n_train_core", np.nan)) > 0
                else np.nan
            ),
            "log_n_train_core": math.log(float(row.get("n_train_core"))) if float(row.get("n_train_core", 0)) > 0 else np.nan,
            "log_n_features": math.log(float(row.get("n_features"))) if float(row.get("n_features", 0)) > 0 else np.nan,
        }
        for split in ["train_core", "validation", "test"]:
            col = f"class_counts_{split}"
            if col in row:
                out.update(_imbalance_features(_counts_from_json(row[col]), split))
        rows.append(out)
    return pd.DataFrame(rows)


def _class_prevalence(split_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in split_summary.iterrows():
        counts = _counts_from_json(row["class_counts_train_core"])
        total = sum(counts.values())
        if total <= 0:
            continue
        values = np.asarray(list(counts.values()), dtype=float)
        minority = float(values.min())
        majority = float(values.max())
        for label, count in counts.items():
            count_f = float(count)
            rows.append(
                {
                    "dataset": row["dataset"],
                    "class_label": int(label),
                    "train_class_count": count_f,
                    "train_class_fraction": count_f / float(total),
                    "train_class_imbalance_vs_majority": majority / count_f if count_f > 0 else np.nan,
                    "is_train_minority_class": bool(count_f == minority),
                    "is_train_majority_class": bool(count_f == majority),
                }
            )
    return pd.DataFrame(rows)


def _dpg_summary(final_summary: pd.DataFrame) -> pd.DataFrame:
    dpg = final_summary[final_summary["method"].astype(str).eq("dpg")].copy()
    keep = [
        "dataset",
        "model_accuracy",
        "local_matches_model_rate",
        "local_accuracy",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_recombination_rate",
        "avg_num_active_nodes",
        "avg_num_paths",
    ]
    return dpg[[c for c in keep if c in dpg.columns]].drop_duplicates("dataset")


def _critical_summary(critical_dataset: pd.DataFrame) -> pd.DataFrame:
    dpg = critical_dataset[critical_dataset["method"].astype(str).eq("dpg")].copy()
    dpg["critical_changed_advantage_vs_random"] = (
        dpg["critical_changed_to_competitor_rate"] - dpg["control_random_path_changed_to_competitor_rate"]
    )
    dpg["critical_changed_advantage_vs_same_depth"] = (
        dpg["critical_changed_to_competitor_rate"] - dpg["control_same_depth_changed_to_competitor_rate"]
    )
    dpg["critical_competitor_delta_advantage_vs_random"] = dpg["mean_critical_vs_random_path_competitor_delta"]
    dpg["critical_competitor_delta_advantage_vs_same_depth"] = dpg["mean_critical_vs_same_depth_competitor_delta"]
    return dpg.drop(columns=["method", "graph_construction_mode"], errors="ignore")


def _class_level_summary(critical_per_sample: pd.DataFrame, class_prev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dpg = critical_per_sample[critical_per_sample["method"].astype(str).eq("dpg")].copy()
    target_prev = class_prev.rename(
        columns={
            "class_label": "target_label",
            "train_class_fraction": "target_train_fraction",
            "train_class_imbalance_vs_majority": "target_imbalance_vs_majority",
            "is_train_minority_class": "target_is_minority",
            "is_train_majority_class": "target_is_majority",
        }
    )[["dataset", "target_label", "target_train_fraction", "target_imbalance_vs_majority", "target_is_minority", "target_is_majority"]]
    dpg["target_label"] = pd.to_numeric(dpg["target_label"], errors="coerce").astype("Int64")
    dpg = dpg.merge(target_prev, on=["dataset", "target_label"], how="left")
    dpg["critical_changed_advantage_vs_random_sample"] = (
        dpg["critical_comp_branch_changed_to_competitor"].astype(float)
        - dpg["control_random_path_changed_to_competitor"].astype(float)
    )
    dpg["critical_changed_advantage_vs_same_depth_sample"] = (
        dpg["critical_comp_branch_changed_to_competitor"].astype(float)
        - dpg["control_same_depth_changed_to_competitor"].astype(float)
    )
    class_summary = (
        dpg.groupby(
            [
                "dataset",
                "target_label",
                "target_train_fraction",
                "target_imbalance_vs_majority",
                "target_is_minority",
                "target_is_majority",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            samples=("sample_idx", "count"),
            critical_present_rate=("critical_node_present", "mean"),
            critical_eligible_rate=("critical_eligible", "mean"),
            critical_changed_to_competitor_rate=("critical_comp_branch_changed_to_competitor", "mean"),
            control_random_changed_to_competitor_rate=("control_random_path_changed_to_competitor", "mean"),
            control_same_depth_changed_to_competitor_rate=("control_same_depth_changed_to_competitor", "mean"),
            critical_advantage_vs_random=("critical_changed_advantage_vs_random_sample", "mean"),
            critical_advantage_vs_same_depth=("critical_changed_advantage_vs_same_depth_sample", "mean"),
        )
        .sort_values(["critical_advantage_vs_random", "critical_changed_to_competitor_rate"], ascending=[False, False])
        .reset_index(drop=True)
    )
    cohort_summary = (
        dpg.groupby(["target_is_minority", "target_is_majority"], dropna=False, as_index=False)
        .agg(
            samples=("sample_idx", "count"),
            datasets=("dataset", "nunique"),
            mean_target_train_fraction=("target_train_fraction", "mean"),
            critical_present_rate=("critical_node_present", "mean"),
            critical_eligible_rate=("critical_eligible", "mean"),
            critical_changed_to_competitor_rate=("critical_comp_branch_changed_to_competitor", "mean"),
            control_random_changed_to_competitor_rate=("control_random_path_changed_to_competitor", "mean"),
            critical_advantage_vs_random=("critical_changed_advantage_vs_random_sample", "mean"),
        )
        .sort_values(["target_is_minority", "target_is_majority"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return class_summary, cohort_summary


def _spearman_table(df: pd.DataFrame, predictors: Iterable[str], outcomes: Iterable[str]) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        for predictor in predictors:
            if predictor not in df.columns:
                continue
            valid = df[[predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
            if valid.shape[0] < 4 or valid[predictor].nunique() < 2 or valid[outcome].nunique() < 2:
                corr = np.nan
            else:
                corr = float(valid[predictor].corr(valid[outcome], method="spearman"))
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_datasets": int(valid.shape[0]),
                    "spearman_rho": corr,
                    "abs_spearman_rho": abs(corr) if pd.notna(corr) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["outcome", "abs_spearman_rho"], ascending=[True, False]).reset_index(drop=True)


def _top_contexts(merged: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "n_train_core",
        "n_features",
        "n_classes_train",
        "train_core_imbalance_ratio",
        "train_core_minority_fraction",
        "model_accuracy",
        "avg_support_margin",
        "avg_competitor_exposure",
        "critical_node_present_rate",
        "critical_eligible_rate",
        "critical_changed_to_competitor_rate",
        "control_random_path_changed_to_competitor_rate",
        "critical_changed_advantage_vs_random",
        "critical_competitor_delta_advantage_vs_random",
    ]
    available = [c for c in cols if c in merged.columns]
    return (
        merged[available]
        .sort_values(
            ["critical_changed_advantage_vs_random", "critical_changed_to_competitor_rate", "critical_eligible_rate"],
            ascending=[False, False, False],
        )
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


def _write_report(
    out_dir: Path,
    merged: pd.DataFrame,
    correlations: pd.DataFrame,
    top_contexts: pd.DataFrame,
    class_cohorts: pd.DataFrame,
    class_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Critical Nodes And Dataset Properties",
        "",
        "This report asks whether global dataset properties explain when critical nodes become more useful.",
        "",
        "Important caution: this is a dataset-level exploratory analysis over 15 datasets, so correlations are hypothesis-generating rather than confirmatory.",
        "",
        "## Strongest Associations",
        "",
    ]
    focus_outcomes = [
        "critical_node_present_rate",
        "critical_eligible_rate",
        "critical_changed_to_competitor_rate",
        "critical_changed_advantage_vs_random",
        "critical_competitor_delta_advantage_vs_random",
    ]
    for outcome in focus_outcomes:
        subset = correlations[correlations["outcome"].eq(outcome)].head(6)
        if subset.empty:
            continue
        lines.append(f"### {outcome}")
        lines.append("")
        lines.append(_markdown_table(subset[["predictor", "n_datasets", "spearman_rho"]]))
        lines.append("")
    lines.extend(
        [
            "## Most Favorable Contexts",
            "",
            _markdown_table(top_contexts.head(10)),
            "",
            "## Target-Class Minority/Majority Cohorts",
            "",
            _markdown_table(class_cohorts),
            "",
            "## Most Favorable Target Classes",
            "",
            _markdown_table(class_summary.head(12)),
            "",
            "## Working Interpretation",
            "",
            "- Critical-node usefulness should be treated as conditional and dataset-dependent.",
            "- Imbalance-related columns are included explicitly; if they are not among the strongest associations, the paper should avoid claiming imbalance is the main driver.",
            "- The class-level table tests whether minority target classes behave differently from majority target classes.",
            "- Favorable contexts are better identified by the actual advantage over controls, not just by critical-node occurrence.",
            "- Because n=15 datasets, any property-level conclusion should be framed as exploratory and moved to discussion or appendix unless the signal is very strong.",
            "",
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dataset properties associated with critical-node usefulness.")
    parser.add_argument(
        "--split_summary",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/splits/split_summary.csv"),
    )
    parser.add_argument(
        "--critical_dataset_summary",
        type=Path,
        default=Path(
            "experiments_local_explanation/results_journal_v1/critical_node_validation_final_fixed/critical_node_dataset_summary.csv"
        ),
    )
    parser.add_argument(
        "--critical_per_sample",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/critical_node_validation_final_fixed/critical_node_per_sample.csv"),
    )
    parser.add_argument(
        "--final_summary",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/summary_selected_test.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/critical_node_property_analysis"),
    )
    args = parser.parse_args()

    props = _dataset_properties(_read_csv(args.split_summary))
    critical = _critical_summary(_read_csv(args.critical_dataset_summary))
    dpg = _dpg_summary(_read_csv(args.final_summary))
    merged = props.merge(dpg, on="dataset", how="left").merge(critical, on="dataset", how="inner")

    predictors = [
        "n_train_core",
        "n_test",
        "n_features",
        "n_classes_train",
        "features_per_train_sample",
        "log_n_train_core",
        "log_n_features",
        "train_core_minority_fraction",
        "train_core_majority_fraction",
        "train_core_imbalance_ratio",
        "train_core_normalized_entropy",
        "train_core_gini_impurity",
        "test_minority_fraction",
        "test_majority_fraction",
        "test_imbalance_ratio",
        "test_normalized_entropy",
        "test_gini_impurity",
        "model_accuracy",
        "local_matches_model_rate",
        "local_accuracy",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_recombination_rate",
        "avg_num_active_nodes",
        "avg_num_paths",
    ]
    outcomes = [
        "critical_node_present_rate",
        "critical_eligible_rate",
        "critical_comp_label_changed_rate",
        "critical_changed_to_competitor_rate",
        "critical_changed_advantage_vs_same_depth",
        "critical_changed_advantage_vs_random",
        "mean_critical_comp_competitor_delta",
        "critical_competitor_delta_advantage_vs_same_depth",
        "critical_competitor_delta_advantage_vs_random",
    ]
    correlations = _spearman_table(merged, predictors=predictors, outcomes=outcomes)
    top_contexts = _top_contexts(merged)
    class_summary, class_cohorts = _class_level_summary(
        _read_csv(args.critical_per_sample),
        class_prev=_class_prevalence(_read_csv(args.split_summary)),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "critical_node_dataset_property_table.csv", index=False)
    correlations.to_csv(args.out_dir / "critical_node_property_correlations.csv", index=False)
    top_contexts.to_csv(args.out_dir / "critical_node_favorable_contexts.csv", index=False)
    class_summary.to_csv(args.out_dir / "critical_node_class_level_summary.csv", index=False)
    class_cohorts.to_csv(args.out_dir / "critical_node_minority_majority_cohorts.csv", index=False)
    _write_report(
        args.out_dir,
        merged=merged,
        correlations=correlations,
        top_contexts=top_contexts,
        class_cohorts=class_cohorts,
        class_summary=class_summary,
    )

    print(f"Saved critical-node property analysis to {args.out_dir}")
    print(f"  - critical_node_dataset_property_table.csv ({merged.shape[0]} datasets)")
    print("  - critical_node_property_correlations.csv")
    print("  - critical_node_favorable_contexts.csv")
    print("  - critical_node_class_level_summary.csv")
    print("  - critical_node_minority_majority_cohorts.csv")
    print("  - summary.md")


if __name__ == "__main__":
    main()
