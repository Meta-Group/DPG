#!/usr/bin/env python3
"""Generate journal-ready summaries from the locked final-test run.

The report intentionally separates construction checks from diagnostic-value
evidence, mirroring the ECML reviewer concerns.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - optional dependency guard
    wilcoxon = None


DPG_METHODS = {"dpg", "dpg_execution_trace"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _method_group(method: object) -> str:
    text = str(method)
    if text in DPG_METHODS:
        return "structure-aware"
    if text in {"anchors", "lore"}:
        return "rule-surrogate"
    if text in {"raw_path_union", "path_bag", "random_same_size_path"}:
        return "path-control"
    if text in {"shap", "lime", "tree_path"}:
        return "feature/output-aligned"
    if text == "ice":
        return "response-profile"
    return "other"


def _metric_available(df: pd.DataFrame, metric: str) -> bool:
    return metric in df.columns and df[metric].notna().any()


def _bootstrap_ci(values: Sequence[float], rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    arr = np.asarray([float(x) for x in values if pd.notna(x)], dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(arr.mean())
    if arr.size == 1 or n_boot <= 0:
        return (mean, np.nan, np.nan)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (mean, float(lo), float(hi))


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray([np.nan if pd.isna(x) else float(x) for x in p_values], dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.where(~np.isnan(p))[0]
    if valid.size == 0:
        return adjusted.tolist()
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * p[idx], 1.0)
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def _rank_biserial_from_pairs(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    n = diff.size
    if n == 0:
        return np.nan
    abs_diff = np.abs(diff)
    order = np.argsort(abs_diff, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs_diff[order[j]] == abs_diff[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    pos = float(ranks[diff > 0].sum())
    neg = float(ranks[diff < 0].sum())
    denom = n * (n + 1) / 2.0
    return (pos - neg) / denom if denom else np.nan


def _paired_tests(summary: pd.DataFrame, metrics: Sequence[str], reference: str) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if not _metric_available(summary, metric):
            continue
        pivot = summary.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")
        if reference not in pivot.columns:
            continue
        for method in sorted(c for c in pivot.columns if c != reference):
            paired = pivot[[reference, method]].dropna()
            if paired.shape[0] < 2:
                stat = np.nan
                p_value = np.nan
            elif wilcoxon is None:
                stat = np.nan
                p_value = np.nan
            else:
                diff = paired[reference].to_numpy(dtype=float) - paired[method].to_numpy(dtype=float)
                if np.allclose(diff, 0):
                    stat = 0.0
                    p_value = 1.0
                else:
                    res = wilcoxon(paired[reference], paired[method], zero_method="wilcox", alternative="two-sided")
                    stat = float(res.statistic)
                    p_value = float(res.pvalue)
            rows.append(
                {
                    "metric": metric,
                    "reference_method": reference,
                    "comparison_method": method,
                    "n_datasets": int(paired.shape[0]),
                    "reference_mean": float(paired[reference].mean()) if not paired.empty else np.nan,
                    "comparison_mean": float(paired[method].mean()) if not paired.empty else np.nan,
                    "mean_difference_ref_minus_comparison": (
                        float((paired[reference] - paired[method]).mean()) if not paired.empty else np.nan
                    ),
                    "wilcoxon_statistic": stat,
                    "p_value": p_value,
                    "rank_biserial_ref_minus_comparison": _rank_biserial_from_pairs(
                        paired[reference].to_numpy(dtype=float), paired[method].to_numpy(dtype=float)
                    )
                    if not paired.empty
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = np.nan
        for metric, idx in out.groupby("metric").groups.items():
            out.loc[idx, "p_holm"] = _holm_adjust(out.loc[idx, "p_value"].tolist())
    return out


def _native_size(row: pd.Series) -> float:
    method = str(row.get("method", ""))
    if method in DPG_METHODS:
        return _float_or_nan(row.get("num_active_nodes"))
    if method == "anchors":
        rule = row.get("anchor_rule")
        if pd.isna(rule) or str(rule).strip() == "":
            return np.nan
        return float(len([part for part in str(rule).split(" AND ") if part.strip()]))
    if method == "lore":
        return _float_or_nan(row.get("lore_rule_length"))
    if method in {"shap", "lime", "tree_path"}:
        value = _float_or_nan(row.get("contrib_nnz"))
        return value if pd.notna(value) else _float_or_nan(row.get("n_features"))
    if method in {"raw_path_union", "path_bag", "random_same_size_path"}:
        value = _float_or_nan(row.get("path_control_predicate_count"))
        return value if pd.notna(value) else _float_or_nan(row.get("num_active_nodes"))
    return np.nan


def _float_or_nan(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _add_sample_derived_columns(per_sample: pd.DataFrame) -> pd.DataFrame:
    df = per_sample.copy()
    df["method_group"] = df["method"].map(_method_group)
    df["native_size"] = df.apply(_native_size, axis=1)
    if "model_correct" in df.columns:
        df["model_error"] = ~df["model_correct"].fillna(False).astype(bool)
    if "local_matches_model" in df.columns:
        df["local_disagreement"] = ~df["local_matches_model"].fillna(True).astype(bool)
    if "support_margin" in df.columns:
        df["dpg_uncertainty_score"] = 1.0 - pd.to_numeric(df["support_margin"], errors="coerce")
    if "explanation_confidence" in df.columns:
        df["dpg_low_confidence_score"] = 1.0 - pd.to_numeric(df["explanation_confidence"], errors="coerce")
    if "competitor_exposure" in df.columns:
        df["dpg_competitor_exposure_score"] = pd.to_numeric(df["competitor_exposure"], errors="coerce")
    return df


def _summarize_methods(summary: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    metrics = [
        "model_accuracy",
        "local_matches_model_rate",
        "local_accuracy",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_path_purity",
        "avg_recombination_rate",
        "avg_runtime_ms",
        "avg_lore_fidelity_neighborhood",
        "avg_anchor_precision",
        "avg_anchor_coverage",
    ]
    rows = []
    for method, group in summary.groupby("method", dropna=False):
        row = {
            "method": method,
            "method_group": _method_group(method),
            "n_datasets": int(group["dataset"].nunique()),
        }
        for metric in metrics:
            if not _metric_available(group, metric):
                continue
            mean, lo, hi = _bootstrap_ci(group[metric].dropna().to_numpy(dtype=float), rng, n_boot=n_boot)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method_group", "method"]).reset_index(drop=True)


def _summarize_native_size(per_sample: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    dataset_method = (
        per_sample.groupby(["dataset", "method"], as_index=False)
        .agg(
            native_size_mean=("native_size", "mean"),
            runtime_ms_mean=("runtime_ms", "mean"),
            n_samples=("sample_idx", "count"),
        )
        .dropna(subset=["native_size_mean"], how="all")
    )
    rows = []
    for method, group in dataset_method.groupby("method", dropna=False):
        mean, lo, hi = _bootstrap_ci(group["native_size_mean"].dropna(), rng, n_boot=n_boot)
        r_mean, r_lo, r_hi = _bootstrap_ci(group["runtime_ms_mean"].dropna(), rng, n_boot=n_boot)
        rows.append(
            {
                "method": method,
                "method_group": _method_group(method),
                "n_datasets": int(group["dataset"].nunique()),
                "native_size_mean": mean,
                "native_size_ci95_low": lo,
                "native_size_ci95_high": hi,
                "runtime_ms_mean": r_mean,
                "runtime_ms_ci95_low": r_lo,
                "runtime_ms_ci95_high": r_hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["method_group", "method"]).reset_index(drop=True)


def _diagnostic_value(per_sample: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = per_sample[per_sample["method"].isin(DPG_METHODS)].copy()
    score_cols = [
        "dpg_low_confidence_score",
        "dpg_uncertainty_score",
        "dpg_competitor_exposure_score",
        "critical_node_contrast",
        "recombination_rate",
        "num_active_nodes",
    ]
    targets = ["model_error", "local_disagreement"]
    rows = []
    per_dataset_rows = []
    for method, method_df in df.groupby("method"):
        for score_col in score_cols:
            if not _metric_available(method_df, score_col):
                continue
            for target in targets:
                if target not in method_df.columns:
                    continue
                for dataset, group in method_df.groupby("dataset"):
                    valid = group[[target, score_col]].dropna()
                    if valid[target].nunique() < 2:
                        continue
                    y = valid[target].astype(int).to_numpy()
                    score = valid[score_col].astype(float).to_numpy()
                    try:
                        auroc = float(roc_auc_score(y, score))
                        auprc = float(average_precision_score(y, score))
                    except ValueError:
                        continue
                    per_dataset_rows.append(
                        {
                            "method": method,
                            "dataset": dataset,
                            "target": target,
                            "score": score_col,
                            "n_samples": int(valid.shape[0]),
                            "positive_rate": float(y.mean()),
                            "auroc": auroc,
                            "auprc": auprc,
                        }
                    )
    per_dataset = pd.DataFrame(per_dataset_rows)
    if per_dataset.empty:
        return pd.DataFrame(), per_dataset
    for keys, group in per_dataset.groupby(["method", "target", "score"]):
        method, target, score_col = keys
        auroc_mean, auroc_lo, auroc_hi = _bootstrap_ci(group["auroc"], rng, n_boot=n_boot)
        auprc_mean, auprc_lo, auprc_hi = _bootstrap_ci(group["auprc"], rng, n_boot=n_boot)
        rows.append(
            {
                "method": method,
                "target": target,
                "score": score_col,
                "n_datasets": int(group["dataset"].nunique()),
                "mean_positive_rate": float(group["positive_rate"].mean()),
                "auroc_mean": auroc_mean,
                "auroc_ci95_low": auroc_lo,
                "auroc_ci95_high": auroc_hi,
                "auprc_mean": auprc_mean,
                "auprc_ci95_low": auprc_lo,
                "auprc_ci95_high": auprc_hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "method", "score"]).reset_index(drop=True), per_dataset


def _failure_analysis(summary: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "dataset",
        "method",
        "n_model_classes",
        "n_features",
        "n_test_evaluated",
        "model_accuracy",
        "local_matches_model_rate",
        "local_accuracy",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_num_active_nodes",
        "avg_num_paths",
        "avg_runtime_ms",
    ]
    cols = [c for c in wanted if c in summary.columns]
    df = summary[cols].copy()
    focus = {"isolet", "madelon"}
    df["failure_focus"] = df["dataset"].isin(focus)
    sort_cols = [c for c in ["failure_focus", "dataset", "method"] if c in df.columns]
    return df.sort_values(sort_cols, ascending=[False, True, True]).reset_index(drop=True)


def _common_interface_table() -> pd.DataFrame:
    rows = [
        {
            "method": "dpg",
            "native_object": "local DPG subgraph from executed forest paths",
            "main_role": "structure-aware diagnostic",
            "explained_class": "support-predicted/model class from graph evidence",
            "score_field": "explanation_confidence, support_margin, competitor_exposure",
            "size_field": "active graph nodes/edges",
            "unsupported_or_limited": "not optimized for output fidelity alone",
        },
        {
            "method": "dpg_execution_trace",
            "native_object": "exact executed trace graph",
            "main_role": "trace-faithful ablation",
            "explained_class": "support-predicted/model class from executed trace evidence",
            "score_field": "explanation_confidence, support_margin, competitor_exposure",
            "size_field": "active trace nodes/edges",
            "unsupported_or_limited": "larger/readability cost than aggregated DPG",
        },
        {
            "method": "shap",
            "native_object": "per-class feature attribution vector",
            "main_role": "output-aligned feature attribution",
            "explained_class": "model-predicted class",
            "score_field": "score_margin_pred_vs_competitor, contribution norms",
            "size_field": "non-zero attribution count",
            "unsupported_or_limited": "no path or transition object",
        },
        {
            "method": "lime",
            "native_object": "local linear surrogate coefficients",
            "main_role": "output-aligned local surrogate",
            "explained_class": "model-predicted class",
            "score_field": "score_margin_pred_vs_competitor, contribution norms",
            "size_field": "non-zero coefficient count",
            "unsupported_or_limited": "no executed-path semantics",
        },
        {
            "method": "anchors",
            "native_object": "local precision rule",
            "main_role": "rule-surrogate baseline",
            "explained_class": "model-predicted class",
            "score_field": "anchor_precision, anchor_coverage",
            "size_field": "rule predicate count",
            "unsupported_or_limited": "rule is perturbation-induced, not forest trace",
        },
        {
            "method": "lore",
            "native_object": "LORE-style local decision-tree rule and nearest counterfactual",
            "main_role": "rule/counterfactual surrogate baseline",
            "explained_class": "model-predicted class",
            "score_field": "lore_fidelity_neighborhood, counterfactual distance",
            "size_field": "local rule predicate count",
            "unsupported_or_limited": "lightweight LORE-style implementation, not full LOREM package",
        },
        {
            "method": "tree_path",
            "native_object": "feature ranking from executed tree path probability deltas",
            "main_role": "simple path-feature baseline",
            "explained_class": "model-predicted class",
            "score_field": "score_margin_pred_vs_competitor",
            "size_field": "non-zero path feature count",
            "unsupported_or_limited": "feature-level only; loses transition graph",
        },
        {
            "method": "raw_path_union",
            "native_object": "set of unique predicates executed by the forest for the sample",
            "main_role": "simple path-control baseline",
            "explained_class": "model-predicted class",
            "score_field": "score_margin_pred_vs_competitor",
            "size_field": "unique executed predicate count",
            "unsupported_or_limited": "no DPG canonicalization, class support graph, competitor exposure, or critical node",
        },
        {
            "method": "path_bag",
            "native_object": "multiset of all executed predicates across trees",
            "main_role": "simple path-control baseline",
            "explained_class": "model-predicted class",
            "score_field": "score_margin_pred_vs_competitor",
            "size_field": "executed predicate count with duplicates",
            "unsupported_or_limited": "keeps path volume but loses graph transitions and semantic diagnostics",
        },
        {
            "method": "random_same_size_path",
            "native_object": "random forest predicates matched to raw-path-union size",
            "main_role": "negative path-control baseline",
            "explained_class": "model-predicted class",
            "score_field": "score_margin_pred_vs_competitor",
            "size_field": "sampled predicate count",
            "unsupported_or_limited": "size control only; no local execution semantics",
        },
        {
            "method": "ice",
            "native_object": "single-feature response profile",
            "main_role": "diagnostic response-profile only",
            "explained_class": "model-predicted class",
            "score_field": "ice_prob_range, ice_slope",
            "size_field": "one varied feature",
            "unsupported_or_limited": "not a local route explainer; excluded from main ranking",
        },
    ]
    return pd.DataFrame(rows)


def _format_ci(row: pd.Series, base: str) -> str:
    mean = row.get(f"{base}_mean")
    lo = row.get(f"{base}_ci95_low")
    hi = row.get(f"{base}_ci95_high")
    if pd.isna(mean):
        return "n/a"
    if pd.isna(lo) or pd.isna(hi):
        return f"{mean:.3f}"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


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
    widths = [
        max(len(str(col)), *(len(row[idx]) for row in rows))
        for idx, col in enumerate(cols)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(cols))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(cols))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _critical_node_display(method_summary: pd.DataFrame) -> pd.DataFrame:
    if method_summary.empty:
        return pd.DataFrame()
    cols = [
        "method",
        "graph_construction_mode",
        "datasets",
        "total_samples",
        "mean_critical_node_present_rate",
        "mean_critical_eligible_rate",
        "mean_critical_comp_label_changed_rate",
        "mean_critical_changed_to_competitor_rate",
        "mean_control_same_depth_changed_to_competitor_rate",
        "mean_control_random_path_changed_to_competitor_rate",
        "mean_critical_comp_competitor_delta",
        "mean_critical_vs_same_depth_competitor_delta",
        "mean_critical_vs_random_path_competitor_delta",
    ]
    return method_summary[[c for c in cols if c in method_summary.columns]].copy()


def _critical_node_notes(method_summary: pd.DataFrame) -> list[str]:
    if method_summary.empty:
        return ["No critical-node validation summary was available."]
    notes = []
    dpg = method_summary[method_summary["method"].astype(str).eq("dpg")]
    trace = method_summary[method_summary["method"].astype(str).eq("dpg_execution_trace")]
    if not dpg.empty:
        row = dpg.iloc[0]
        present = _float_or_nan(row.get("mean_critical_node_present_rate"))
        eligible = _float_or_nan(row.get("mean_critical_eligible_rate"))
        changed = _float_or_nan(row.get("mean_critical_changed_to_competitor_rate"))
        control_random = _float_or_nan(row.get("mean_control_random_path_changed_to_competitor_rate"))
        if pd.notna(present) and pd.notna(eligible):
            notes.append(
                f"Aggregated DPG exposes critical nodes in {present:.1%} of held-out cases on average, "
                f"but only {eligible:.1%} are intervention-eligible under the current non-root successor test."
            )
        if pd.notna(changed) and pd.notna(control_random):
            notes.append(
                f"The changed-to-competitor rate is {changed:.1%}, compared with {control_random:.1%} "
                "for the random-path control; this is weak evidence for a causal intervention claim."
            )
    if not trace.empty:
        trace_present = _float_or_nan(trace.iloc[0].get("mean_critical_node_present_rate"))
        if pd.notna(trace_present) and trace_present == 0.0:
            notes.append(
                "Execution-trace DPG has no critical nodes under this definition, because the trace variant "
                "does not create the same recombined divergence structure."
            )
    notes.append(
        "Paper framing: report critical nodes as a conditional descriptive diagnostic, while using confidence, "
        "support margin, and competitor exposure as the main diagnostic-value evidence."
    )
    return notes


def _write_markdown(
    out_path: Path,
    method_summary: pd.DataFrame,
    size_runtime: pd.DataFrame,
    paired: pd.DataFrame,
    diagnostic: pd.DataFrame,
    failure: pd.DataFrame,
    critical_node_summary: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Journal Final-Test Report",
        "",
        "Generated from the validation-selected, locked final-test outputs.",
        "",
        "## Main Method Summary",
        "",
    ]
    display_metrics = ["local_matches_model_rate", "local_accuracy", "avg_explanation_confidence", "avg_runtime_ms"]
    compact_rows = []
    for _, row in method_summary.iterrows():
        compact = {"method": row["method"], "group": row["method_group"], "datasets": row["n_datasets"]}
        for metric in display_metrics:
            compact[metric] = _format_ci(row, metric)
        compact_rows.append(compact)
    lines.append(_markdown_table(pd.DataFrame(compact_rows)))
    lines.extend(["", "## Size And Runtime", ""])
    compact_rows = []
    for _, row in size_runtime.iterrows():
        compact_rows.append(
            {
                "method": row["method"],
                "group": row["method_group"],
                "native_size": _format_ci(row, "native_size"),
                "runtime_ms": _format_ci(row, "runtime_ms"),
            }
        )
    lines.append(_markdown_table(pd.DataFrame(compact_rows)))
    lines.extend(["", "## Corrected Paired Tests", ""])
    if paired.empty:
        lines.append("No paired tests were available.")
    else:
        cols = [
            "metric",
            "reference_method",
            "comparison_method",
            "n_datasets",
            "mean_difference_ref_minus_comparison",
            "p_value",
            "p_holm",
            "rank_biserial_ref_minus_comparison",
        ]
        lines.append(_markdown_table(paired[cols], floatfmt=".4g"))
    lines.extend(["", "## Diagnostic Value", ""])
    if diagnostic.empty:
        lines.append("No diagnostic AUROC/AUPRC rows were available.")
    else:
        lines.append(_markdown_table(diagnostic, floatfmt=".4g"))
    if critical_node_summary is not None:
        lines.extend(["", "## Critical Node Conditional Validation", ""])
        critical_display = _critical_node_display(critical_node_summary)
        if critical_display.empty:
            lines.append("No critical-node validation summary was available.")
        else:
            lines.append(_markdown_table(critical_display, floatfmt=".4g"))
            lines.extend(["", "Interpretation:", ""])
            for note in _critical_node_notes(critical_node_summary):
                lines.append(f"- {note}")
    lines.extend(["", "## Isolet And Madelon Failure Focus", ""])
    focus = failure[failure["failure_focus"]].copy()
    if focus.empty:
        lines.append("No isolet/madelon rows were found.")
    else:
        keep = [
            c
            for c in [
                "dataset",
                "method",
                "n_model_classes",
                "n_features",
                "model_accuracy",
                "local_matches_model_rate",
                "local_accuracy",
                "avg_explanation_confidence",
                "avg_support_margin",
                "avg_competitor_exposure",
                "avg_num_active_nodes",
                "avg_num_paths",
                "avg_runtime_ms",
            ]
            if c in focus.columns
        ]
        lines.append(_markdown_table(focus[keep], floatfmt=".4g"))
    lines.extend(
        [
            "",
            "## Reviewer-Facing Notes",
            "",
            "- ICE is documented as a response-profile diagnostic, not a main route/path explainer.",
            "- Structural trace recovery should be presented as a construction check; diagnostic AUROC/AUPRC rows are the stronger value evidence.",
            "- Path-control baselines directly address the basic path-union critique; interpret their output-fidelity rows as controls, not as semantic diagnostics.",
            "- Critical-node intervention is now included as a conditional diagnostic check; the current evidence is mixed and should not be overclaimed.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize locked final-test outputs for the journal revision.")
    parser.add_argument(
        "--final_test_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/final_test_main_with_lore_noice"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/journal_report_final_test"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--reference_method", type=str, default="dpg_execution_trace")
    parser.add_argument("--critical_node_dir", type=Path, default=None)
    args = parser.parse_args()

    summary = _read_csv(args.final_test_dir / "summary_selected_test.csv")
    per_sample = _read_csv(args.final_test_dir / "per_sample_selected_test.csv")
    summary = summary.copy()
    summary["method_group"] = summary["method"].map(_method_group)
    per_sample = _add_sample_derived_columns(per_sample)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    method_summary = _summarize_methods(summary, rng, n_boot=int(args.bootstrap))
    size_runtime = _summarize_native_size(per_sample, rng, n_boot=int(args.bootstrap))
    paired_metrics = [
        "local_matches_model_rate",
        "local_accuracy",
        "avg_explanation_confidence",
        "avg_support_margin",
        "avg_competitor_exposure",
        "avg_path_purity",
        "avg_recombination_rate",
    ]
    paired = _paired_tests(summary, metrics=paired_metrics, reference=str(args.reference_method))
    diagnostic, diagnostic_per_dataset = _diagnostic_value(per_sample, rng, n_boot=int(args.bootstrap))
    failure = _failure_analysis(summary)
    interface = _common_interface_table()
    critical_node_summary = None
    critical_node_dataset_summary = None
    if args.critical_node_dir is not None:
        critical_node_summary = _read_csv(args.critical_node_dir / "critical_node_method_summary.csv")
        dataset_path = args.critical_node_dir / "critical_node_dataset_summary.csv"
        if dataset_path.exists():
            critical_node_dataset_summary = _read_csv(dataset_path)

    outputs = {
        "method_summary_ci.csv": method_summary,
        "size_runtime_summary_ci.csv": size_runtime,
        "paired_tests_holm.csv": paired,
        "diagnostic_value_summary.csv": diagnostic,
        "diagnostic_value_per_dataset.csv": diagnostic_per_dataset,
        "failure_focus_dataset_method.csv": failure,
        "common_explainer_interface.csv": interface,
    }
    if critical_node_summary is not None:
        outputs["critical_node_method_summary.csv"] = critical_node_summary
    if critical_node_dataset_summary is not None:
        outputs["critical_node_dataset_summary.csv"] = critical_node_dataset_summary
    for name, df in outputs.items():
        df.to_csv(args.out_dir / name, index=False)

    _write_markdown(
        args.out_dir / "summary.md",
        method_summary=method_summary,
        size_runtime=size_runtime,
        paired=paired,
        diagnostic=diagnostic,
        failure=failure,
        critical_node_summary=critical_node_summary,
    )
    print(f"Saved journal report artifacts to {args.out_dir}")
    for name in outputs:
        print(f"  - {name}")
    print("  - summary.md")


if __name__ == "__main__":
    main()
