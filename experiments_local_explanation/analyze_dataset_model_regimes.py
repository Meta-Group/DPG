#!/usr/bin/env python3
"""Synthesize dataset/model regimes for DPG-local journal discussion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _main_dpg_dataset_table(per_sample: pd.DataFrame, method: str) -> pd.DataFrame:
    df = per_sample[per_sample["method"].astype(str).eq(method)].copy()
    if df.empty:
        raise ValueError(f"No rows for method={method}")
    for col in [
        "model_correct",
        "local_matches_model",
        "local_correct",
        "disagree_with_model",
    ]:
        df[col] = df[col].astype(bool)
    agg = (
        df.groupby("dataset", as_index=False, dropna=False)
        .agg(
            samples=("sample_idx", "count"),
            n_model_classes=("n_model_classes", "max"),
            n_features=("n_features", "max"),
            model_accuracy=("model_correct", "mean"),
            local_matches_model_rate=("local_matches_model", "mean"),
            local_accuracy=("local_correct", "mean"),
            disagree_rate=("disagree_with_model", "mean"),
            avg_explanation_confidence=("explanation_confidence", "mean"),
            avg_support_margin=("support_margin", "mean"),
            avg_competitor_exposure=("competitor_exposure", "mean"),
            avg_model_vote_agreement=("model_vote_agreement", "mean"),
            avg_edge_recall=("edge_recall", "mean"),
            avg_recombination_rate=("recombination_rate", "mean"),
            avg_num_active_nodes=("num_active_nodes", "mean"),
            avg_runtime_ms=("runtime_ms", "mean"),
        )
        .reset_index(drop=True)
    )
    return agg


def _confidence_per_dataset(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    keep_scores = {"risk_low_reported", "risk_low_vote_agreement_only", "risk_competitor_exposure"}
    sub = df[
        df["target"].astype(str).eq("local_disagreement")
        & df["score"].astype(str).isin(keep_scores)
        & df["method"].astype(str).eq("dpg_execution_trace")
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(
        index="dataset",
        columns="score",
        values="auroc",
        aggfunc="mean",
    ).reset_index()
    return pivot.rename(
        columns={
            "risk_low_reported": "auroc_low_confidence_disagreement",
            "risk_low_vote_agreement_only": "auroc_low_vote_agreement_disagreement",
            "risk_competitor_exposure": "auroc_competitor_exposure_disagreement",
        }
    )


def _model_family_difficulty(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    if "dataset" not in df.columns:
        return pd.DataFrame()
    cols = [
        "dataset",
        "n_model_families",
        "local_matches_model_rate",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_num_active_nodes",
        "avg_runtime_ms",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    return out.rename(
        columns={
            "local_matches_model_rate": "model_family_mean_local_match",
            "avg_explanation_confidence": "model_family_mean_confidence",
            "avg_support_margin": "model_family_mean_margin",
            "avg_competitor_exposure": "model_family_mean_competitor_exposure",
            "avg_num_active_nodes": "model_family_mean_active_nodes",
            "avg_runtime_ms": "model_family_mean_runtime_ms",
        }
    )


def _scalability_by_dataset(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame()
    for col in ["max_depth", "n_estimators"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = (
        df.groupby("dataset", as_index=False, dropna=False)
        .agg(
            scalability_runs=("config_id", "count"),
            depth4_runtime_ms=("avg_runtime_ms", lambda s: np.nan),
            max_depth_tested=("max_depth", "max"),
            max_estimators_tested=("n_estimators", "max"),
            max_runtime_ms=("avg_runtime_ms", "max"),
            max_active_nodes=("avg_num_active_nodes", "max"),
            min_edge_recall=("avg_edge_recall", "min"),
            max_local_match=("local_matches_model_rate", "max"),
        )
        .reset_index(drop=True)
    )
    depth4 = df[df["max_depth"].eq(4)].groupby("dataset")["avg_runtime_ms"].mean()
    grouped["depth4_runtime_ms"] = grouped["dataset"].map(depth4)
    return grouped


def _add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dimension_regime"] = np.select(
        [
            out["n_features"].ge(100),
            out["n_features"].le(10),
        ],
        ["high_dimensional", "low_dimensional"],
        default="moderate_dimensional",
    )
    out["class_regime"] = np.select(
        [
            out["n_model_classes"].ge(10),
            out["n_model_classes"].eq(2),
        ],
        ["many_class", "binary"],
        default="multiclass",
    )
    out["diagnostic_regime"] = np.select(
        [
            out["local_matches_model_rate"].lt(0.7) | out["avg_competitor_exposure"].gt(0.4),
            out["local_matches_model_rate"].ge(0.85) & out["avg_explanation_confidence"].ge(0.6),
        ],
        ["hard_or_contested", "strong"],
        default="intermediate",
    )
    out["cost_regime"] = np.select(
        [
            out.get("max_runtime_ms", pd.Series(np.nan, index=out.index)).gt(10000),
            out["avg_num_active_nodes"].gt(80),
        ],
        ["high_cost_under_depth_stress", "large_graph"],
        default="moderate_cost",
    )
    return out


def _markdown_table(df: pd.DataFrame, max_rows: int = 30, floatfmt: str = ".4g") -> str:
    if df.empty:
        return ""
    table = df.head(max_rows).copy()
    cols = list(table.columns)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value)

    rows = [[fmt(row[col]) for col in cols] for _, row in table.iterrows()]
    widths = [max(len(str(col)), *(len(row[idx]) for row in rows)) for idx, col in enumerate(cols)]
    header = "| " + " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(cols))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx, col in enumerate(cols)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_report(out_dir: Path, regime: pd.DataFrame) -> None:
    cols = [
        "dataset",
        "dimension_regime",
        "class_regime",
        "diagnostic_regime",
        "cost_regime",
        "n_model_classes",
        "n_features",
        "local_matches_model_rate",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "disagree_rate",
        "auroc_low_confidence_disagreement",
        "model_family_mean_local_match",
        "max_runtime_ms",
    ]
    compact = regime[[c for c in cols if c in regime.columns]].sort_values(
        ["diagnostic_regime", "avg_competitor_exposure", "local_matches_model_rate"],
        ascending=[True, False, True],
    )
    hard = compact[compact["diagnostic_regime"].eq("hard_or_contested")]
    strong = compact[compact["diagnostic_regime"].eq("strong")]

    lines = [
        "# Dataset And Model Regime Synthesis",
        "",
        "This report combines final-test DPG execution-trace diagnostics, confidence sensitivity, depth scalability, and the supported model-family subset to identify where DPG-local diagnostics are strongest or weakest.",
        "",
        "## Regime Table",
        "",
        _markdown_table(compact, max_rows=40),
        "",
        "## Strong Regimes",
        "",
        _markdown_table(strong, max_rows=20),
        "",
        "## Hard Or Contested Regimes",
        "",
        _markdown_table(hard, max_rows=20),
        "",
        "## Writing Guidance",
        "",
        "- DPG diagnostics are strongest on low- to moderate-dimensional datasets where model evidence is concentrated and competitor exposure is low.",
        "- High-dimensional and many-class settings, especially `isolet`, remain difficult: local agreement is low and competitor exposure is high even though trace recovery remains strong.",
        "- `madelon` is a high-dimensional binary boundary case: it is less severe than `isolet`, but still shows lower confidence and higher cost than simple low-dimensional datasets.",
        "- Model family matters. The supported-family subset shows that Bagging can yield stronger local agreement on several datasets, while ExtraTrees can be more contested despite stable trace recovery.",
        "- The paper should separate construction robustness from diagnostic utility: edge recall and zero recombination remain strong across regimes, but confidence, support margin, competitor exposure, runtime, and graph size determine practical usefulness.",
        "- Use this synthesis to support a bounded contribution: DPGs enable local diagnostic analysis, and the diagnostics reveal when local path evidence is concentrated, contested, expensive, or weak.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize dataset/model regimes for DPG-local.")
    parser.add_argument(
        "--per_sample",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/per_sample_selected_test.csv"),
    )
    parser.add_argument(
        "--confidence_per_dataset",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/confidence_sensitivity/confidence_sensitivity_per_dataset.csv"),
    )
    parser.add_argument(
        "--model_family_difficulty",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset_report/model_family_dataset_difficulty.csv"),
    )
    parser.add_argument(
        "--scalability_all_runs",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_report/scalability_all_runs.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/dataset_model_regime_synthesis"),
    )
    args = parser.parse_args()

    base = _main_dpg_dataset_table(_read_csv(args.per_sample), method="dpg_execution_trace")
    confidence = _confidence_per_dataset(args.confidence_per_dataset)
    model_family = _model_family_difficulty(args.model_family_difficulty)
    scalability = _scalability_by_dataset(args.scalability_all_runs)

    regime = base.merge(confidence, on="dataset", how="left")
    regime = regime.merge(model_family, on="dataset", how="left")
    regime = regime.merge(scalability, on="dataset", how="left")
    regime = _add_regime_labels(regime)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    regime.to_csv(args.out_dir / "dataset_model_regime_table.csv", index=False)
    regime.groupby(["dimension_regime", "class_regime", "diagnostic_regime"], dropna=False).agg(
        datasets=("dataset", "nunique"),
        mean_local_match=("local_matches_model_rate", "mean"),
        mean_confidence=("avg_explanation_confidence", "mean"),
        mean_competitor_exposure=("avg_competitor_exposure", "mean"),
        mean_active_nodes=("avg_num_active_nodes", "mean"),
    ).reset_index().to_csv(args.out_dir / "regime_group_summary.csv", index=False)
    _write_report(args.out_dir, regime)

    print(f"Saved dataset/model regime synthesis to {args.out_dir}")
    print("  - dataset_model_regime_table.csv")
    print("  - regime_group_summary.csv")
    print("  - summary.md")


if __name__ == "__main__":
    main()
