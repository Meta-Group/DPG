#!/usr/bin/env python3
"""Complete the remaining DPG 2.0 comparison-story analysis.

Outputs:
- consolidated comparison tables across legacy DPG, pruning, DPG 2.0, and baselines
- cohort-separation effect sizes
- exemplar case-study visualizations
- a compact markdown summary for DPG2.0.md updates
"""

from __future__ import annotations

import json
import sys
import argparse
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from collections import Counter
from graphviz import Digraph
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpg import DPGExplainer


TARGET_DATASETS = [
    "banknote-authentication",
    "breast_cancer",
    "diabetes",
    "digits",
    "ionosphere",
    "iris",
    "isolet",
    "madelon",
    "phoneme",
    "qsar-biodeg",
    "segment",
    "spambase",
    "vehicle",
    "wdbc",
    "wine",
]


@dataclass
class DatasetBundle:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def _load_bundle(data_dir: Path, dataset: str) -> DatasetBundle:
    dataset_dir = data_dir / dataset
    with open(dataset_dir / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return DatasetBundle(
        name=dataset,
        X_train=np.load(dataset_dir / "X_train.npy"),
        y_train=np.load(dataset_dir / "y_train.npy"),
        X_test=np.load(dataset_dir / "X_test.npy"),
        y_test=np.load(dataset_dir / "y_test.npy"),
        feature_names=list(feature_names),
    )


def _read_csv_tree(root: Path, filename: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if not root.exists():
        return pd.DataFrame()
    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        path = dataset_dir / filename
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _cohen_d(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(pd.Series(list(a)).dropna(), dtype=float)
    b = np.asarray(pd.Series(list(b)).dropna(), dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    mean_diff = float(a.mean() - b.mean())
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    pooled_denom = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / max(len(a) + len(b) - 2, 1)
    if pooled_denom <= 0:
        return np.nan
    return float(mean_diff / np.sqrt(pooled_denom))


def _cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(pd.Series(list(a)).dropna(), dtype=float)
    b = np.asarray(pd.Series(list(b)).dropna(), dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = 0
    lt = 0
    for av in a:
        gt += int(np.sum(av > b))
        lt += int(np.sum(av < b))
    return float((gt - lt) / (len(a) * len(b)))


def _best_next_phase_tables(next_analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_configs = pd.read_csv(next_analysis_dir / "best_configs.csv", low_memory=False)
    best_per_sample = pd.read_csv(next_analysis_dir / "best_per_sample.csv", low_memory=False)
    return best_configs, best_per_sample


def _legacy_best(legacy_root: Path) -> pd.DataFrame:
    df = _read_csv_tree(legacy_root, "summary.csv")
    if df.empty:
        return df
    df = df[df["dataset"].isin(TARGET_DATASETS)].copy()
    best = (
        df.sort_values(
            ["dataset", "local_matches_model_rate", "local_accuracy", "avg_evidence_margin_pred_vs_competitor"],
            ascending=[True, False, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best["comparison_family"] = "legacy_dpg"
    best["comparison_method"] = "legacy_dpg_best"
    return best


def _pruning_best(pruning_root: Path) -> pd.DataFrame:
    df = _read_csv_tree(pruning_root, "summary.csv")
    if df.empty:
        return df
    df = df[df["dataset"].isin(TARGET_DATASETS)].copy()
    best = (
        df.sort_values(
            [
                "dataset",
                "local_matches_model_rate",
                "local_accuracy",
                "avg_evidence_margin_pred_vs_competitor",
                "avg_num_paths",
            ],
            ascending=[True, False, False, False, True],
        )
        .groupby("dataset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best["comparison_family"] = "pruning"
    best["comparison_method"] = "pruning_best"
    return best


def _next_phase_best(next_best: pd.DataFrame) -> pd.DataFrame:
    best = next_best.copy()
    best["comparison_family"] = "dpg2_next_phase"
    best["comparison_method"] = best["graph_construction_mode"].map(
        {
            "aggregated_transitions": "dpg2_aggregated_transitions",
            "execution_trace": "dpg2_execution_trace",
        }
    )
    return best


def _baseline_best(baseline_root: Path) -> pd.DataFrame:
    df = _read_csv_tree(baseline_root, "summary_baselines.csv")
    if df.empty:
        return df
    df = df[df["dataset"].isin(TARGET_DATASETS) & df["method"].isin(["shap", "lime", "ice"])].copy()
    best = (
        df.sort_values(
            ["dataset", "method", "local_matches_model_rate", "local_accuracy", "avg_score_margin_pred_vs_competitor"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["dataset", "method"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best["comparison_family"] = "baseline"
    best["comparison_method"] = best["method"]
    return best


def _consolidated_tables(
    legacy_best: pd.DataFrame,
    pruning_best: pd.DataFrame,
    next_best: pd.DataFrame,
    baseline_best: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []

    if not legacy_best.empty:
        frames.append(
            legacy_best.assign(
                fidelity=legacy_best["local_matches_model_rate"],
                local_acc=legacy_best["local_accuracy"],
                margin=legacy_best["avg_evidence_margin_pred_vs_competitor"],
                edge_precision=np.nan,
                edge_recall=np.nan,
                recombination_rate=np.nan,
                explanation_confidence=np.nan,
            )[
                [
                    "dataset",
                    "comparison_family",
                    "comparison_method",
                    "config_id",
                    "seed",
                    "fidelity",
                    "local_acc",
                    "margin",
                    "avg_num_paths",
                    "avg_num_active_nodes",
                    "edge_precision",
                    "edge_recall",
                    "recombination_rate",
                    "explanation_confidence",
                ]
            ]
        )
    if not pruning_best.empty:
        frames.append(
            pruning_best.assign(
                fidelity=pruning_best["local_matches_model_rate"],
                local_acc=pruning_best["local_accuracy"],
                margin=pruning_best["avg_evidence_margin_pred_vs_competitor"],
                edge_precision=np.nan,
                edge_recall=np.nan,
                recombination_rate=np.nan,
                explanation_confidence=np.nan,
            )[
                [
                    "dataset",
                    "comparison_family",
                    "comparison_method",
                    "config_id",
                    "seed",
                    "fidelity",
                    "local_acc",
                    "margin",
                    "avg_num_paths",
                    "avg_num_active_nodes",
                    "edge_precision",
                    "edge_recall",
                    "recombination_rate",
                    "explanation_confidence",
                ]
            ]
        )
    if not next_best.empty:
        frames.append(
            next_best.assign(
                fidelity=next_best["local_matches_model_rate"],
                local_acc=next_best["local_accuracy"],
                margin=next_best["avg_evidence_margin_pred_vs_competitor"],
                edge_precision=next_best["avg_edge_precision"],
                edge_recall=next_best["avg_edge_recall"],
                recombination_rate=next_best["avg_recombination_rate"],
                explanation_confidence=next_best["avg_explanation_confidence"],
            )[
                [
                    "dataset",
                    "comparison_family",
                    "comparison_method",
                    "config_id",
                    "seed",
                    "fidelity",
                    "local_acc",
                    "margin",
                    "avg_num_paths",
                    "avg_num_active_nodes",
                    "edge_precision",
                    "edge_recall",
                    "recombination_rate",
                    "explanation_confidence",
                ]
            ]
        )
    if not baseline_best.empty:
        frames.append(
            baseline_best.assign(
                fidelity=baseline_best["local_matches_model_rate"],
                local_acc=baseline_best["local_accuracy"],
                margin=baseline_best["avg_score_margin_pred_vs_competitor"],
                edge_precision=np.nan,
                edge_recall=np.nan,
                recombination_rate=np.nan,
                explanation_confidence=np.nan,
                avg_num_paths=baseline_best["avg_num_paths"],
                avg_num_active_nodes=baseline_best["avg_num_active_nodes"],
            )[
                [
                    "dataset",
                    "comparison_family",
                    "comparison_method",
                    "config_id",
                    "seed",
                    "fidelity",
                    "local_acc",
                    "margin",
                    "avg_num_paths",
                    "avg_num_active_nodes",
                    "edge_precision",
                    "edge_recall",
                    "recombination_rate",
                    "explanation_confidence",
                ]
            ]
        )

    dataset_level = pd.concat(frames, ignore_index=True)
    summary = (
        dataset_level.groupby(["comparison_family", "comparison_method"], as_index=False)
        .agg(
            datasets=("dataset", "count"),
            mean_fidelity=("fidelity", "mean"),
            mean_local_accuracy=("local_acc", "mean"),
            mean_margin=("margin", "mean"),
            mean_num_paths=("avg_num_paths", "mean"),
            mean_num_active_nodes=("avg_num_active_nodes", "mean"),
            mean_edge_precision=("edge_precision", "mean"),
            mean_edge_recall=("edge_recall", "mean"),
            mean_recombination_rate=("recombination_rate", "mean"),
            mean_explanation_confidence=("explanation_confidence", "mean"),
        )
        .sort_values(["mean_fidelity", "mean_local_accuracy", "mean_margin"], ascending=False)
        .reset_index(drop=True)
    )
    return dataset_level, summary


def _effect_sizes(best_per_sample: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "explanation_confidence",
        "support_margin",
        "predicted_class_concentration_top3",
        "model_vote_agreement",
        "trace_coverage_score",
        "node_precision",
        "edge_precision",
        "recombination_rate",
        "path_purity",
        "competitor_exposure",
        "critical_split_depth",
        "critical_node_contrast",
    ]
    rows: List[Dict[str, Any]] = []
    for mode, sub in best_per_sample.groupby("graph_construction_mode"):
        mc_ec = sub[sub["cohort_label"] == "MC-EC"]
        mw_em = sub[sub["cohort_label"] == "MW-EM"]
        agree = sub[~sub["disagree_with_model"]]
        disagree = sub[sub["disagree_with_model"]]
        for metric in metrics:
            rows.append(
                {
                    "graph_construction_mode": mode,
                    "comparison": "MW-EM_vs_MC-EC",
                    "metric": metric,
                    "group_a": "MW-EM",
                    "group_b": "MC-EC",
                    "mean_a": pd.to_numeric(mw_em[metric], errors="coerce").mean(),
                    "mean_b": pd.to_numeric(mc_ec[metric], errors="coerce").mean(),
                    "delta_mean": pd.to_numeric(mw_em[metric], errors="coerce").mean()
                    - pd.to_numeric(mc_ec[metric], errors="coerce").mean(),
                    "cohen_d": _cohen_d(mw_em[metric], mc_ec[metric]),
                    "cliffs_delta": _cliffs_delta(mw_em[metric], mc_ec[metric]),
                    "n_a": int(pd.to_numeric(mw_em[metric], errors="coerce").notna().sum()),
                    "n_b": int(pd.to_numeric(mc_ec[metric], errors="coerce").notna().sum()),
                }
            )
            rows.append(
                {
                    "graph_construction_mode": mode,
                    "comparison": "DISAGREE_vs_AGREE",
                    "metric": metric,
                    "group_a": "DISAGREE",
                    "group_b": "AGREE",
                    "mean_a": pd.to_numeric(disagree[metric], errors="coerce").mean(),
                    "mean_b": pd.to_numeric(agree[metric], errors="coerce").mean(),
                    "delta_mean": pd.to_numeric(disagree[metric], errors="coerce").mean()
                    - pd.to_numeric(agree[metric], errors="coerce").mean(),
                    "cohen_d": _cohen_d(disagree[metric], agree[metric]),
                    "cliffs_delta": _cliffs_delta(disagree[metric], agree[metric]),
                    "n_a": int(pd.to_numeric(disagree[metric], errors="coerce").notna().sum()),
                    "n_b": int(pd.to_numeric(agree[metric], errors="coerce").notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def _top_path_index(local_explanation: Any, cls: str) -> Optional[int]:
    best_idx: Optional[int] = None
    best_score = -1.0
    for idx, path in enumerate(local_explanation.tree_paths):
        labels = list(path.labels or [])
        if not labels:
            continue
        leaf = labels[-1]
        if not str(leaf).startswith("Class "):
            continue
        leaf_cls = str(leaf).replace("Class ", "", 1)
        if leaf_cls != cls:
            continue
        score = float(path.path_confidence or 0.0)
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx


def _build_explainer(bundle: DatasetBundle, row: pd.Series) -> tuple[DPGExplainer, np.ndarray]:
    model = RandomForestClassifier(
        n_estimators=int(row["n_estimators"]),
        max_depth=None if pd.isna(row["max_depth"]) else int(row["max_depth"]),
        random_state=int(row["seed"]),
        n_jobs=1,
    )
    model.fit(bundle.X_train, bundle.y_train)
    class_names = [str(c) for c in model.classes_]
    explainer = DPGExplainer(
        model=model,
        feature_names=bundle.feature_names,
        target_names=class_names,
        dpg_config={
            "dpg": {
                "default": {
                    "perc_var": float(row["perc_var"]),
                    "decimal_threshold": int(row["decimal_threshold"]),
                    "n_jobs": 1,
                },
                "graph_construction": {
                    "mode": str(row["graph_construction_mode"]),
                },
                "local_evidence": {
                    "variant": "top_competitor",
                    "base_lambda": 0.8,
                },
            }
        },
    )
    explainer.fit(bundle.X_train)
    return explainer, model.predict(bundle.X_test)


def _select_case_rows(
    best_per_sample: pd.DataFrame,
    max_case_dataset_samples: int = 600,
    max_case_features: int = 40,
) -> pd.DataFrame:
    trace = best_per_sample[best_per_sample["graph_construction_mode"] == "execution_trace"].copy()
    dataset_sizes = trace.groupby("dataset")["sample_idx"].nunique().to_dict()
    dataset_features = trace.groupby("dataset")["n_features"].median().to_dict()
    if max_case_dataset_samples > 0:
        allowed = {ds for ds, n in dataset_sizes.items() if int(n) <= int(max_case_dataset_samples)}
        filtered = trace[trace["dataset"].isin(allowed)].copy()
        if not filtered.empty:
            trace = filtered
    if max_case_features > 0:
        allowed = {ds for ds, n in dataset_features.items() if float(n) <= float(max_case_features)}
        filtered = trace[trace["dataset"].isin(allowed)].copy()
        if not filtered.empty:
            trace = filtered
    chosen: List[pd.Series] = []
    used: set[tuple[str, int]] = set()

    def pick(df: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        nonlocal chosen
        if df.empty:
            return
        ordered = df.sort_values(sort_cols, ascending=ascending)
        for _, row in ordered.iterrows():
            key = (str(row["dataset"]), int(row["sample_idx"]))
            if key in used:
                continue
            chosen.append(row)
            used.add(key)
            return

    pick(
        trace[trace["cohort_label"] == "MC-EC"],
        ["explanation_confidence", "support_margin"],
        [False, False],
    )
    pick(
        trace[trace["cohort_label"] == "MC-EC"],
        ["explanation_confidence", "support_margin"],
        [True, True],
    )
    pick(
        trace[~trace["model_correct"]],
        ["competitor_exposure", "explanation_confidence"],
        [False, False],
    )
    prefer_disagree = trace[trace["disagree_with_model"] & trace["local_correct"]]
    if prefer_disagree.empty:
        prefer_disagree = trace[trace["disagree_with_model"]]
    pick(
        prefer_disagree,
        ["explanation_confidence", "support_margin"],
        [False, False],
    )

    selected = pd.DataFrame(chosen).reset_index(drop=True)
    selected.insert(
        0,
        "case_label",
        [
            "high_confidence_correct",
            "low_confidence_ambiguous_correct",
            "model_misclassification_high_competitor_exposure",
            "explanation_model_disagreement",
        ][: len(selected)],
    )
    return selected


def _normalized_node_path(path: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for node_id, label in zip(path.node_ids or [], path.labels or []):
        if node_id is None:
            continue
        pairs.append((str(node_id), str(label)))
    return pairs


def _critical_path_structure(
    local_explanation: Any,
    pred_path_idx: Optional[int],
    comp_path_idx: Optional[int],
) -> dict[str, Optional[str]]:
    result = {
        "critical_node_id": None,
        "critical_successor_pred_id": None,
        "critical_successor_comp_id": None,
        "critical_node_label": None,
        "critical_successor_pred_label": None,
        "critical_successor_comp_label": None,
    }
    if pred_path_idx is None or comp_path_idx is None:
        return result

    pred_path = _normalized_node_path(local_explanation.tree_paths[int(pred_path_idx)])
    comp_path = _normalized_node_path(local_explanation.tree_paths[int(comp_path_idx)])
    shared = 0
    while shared < min(len(pred_path), len(comp_path)) and pred_path[shared][0] == comp_path[shared][0]:
        shared += 1

    # Do not treat the root predicate as a valid critical node.
    if shared <= 1:
        return result

    crit_id, crit_label = pred_path[shared - 1]
    result["critical_node_id"] = crit_id
    result["critical_node_label"] = crit_label

    if shared < len(pred_path):
        result["critical_successor_pred_id"] = pred_path[shared][0]
        result["critical_successor_pred_label"] = pred_path[shared][1]
    if shared < len(comp_path):
        result["critical_successor_comp_id"] = comp_path[shared][0]
        result["critical_successor_comp_label"] = comp_path[shared][1]
    return result


def _render_local_dpg_subgraph_case(
    local_explanation: Any,
    plot_name: str,
    save_dir: Path,
    true_label: str,
    mode_label: str,
    focus_path_idx: Optional[int] = None,
    critical_structure: Optional[dict[str, Optional[str]]] = None,
) -> str:
    context_paths = [_normalized_node_path(path) for path in local_explanation.tree_paths]
    context_paths = [path for path in context_paths if path]
    if not context_paths:
        raise ValueError("No valid local explanation paths to render.")

    node_labels: dict[str, str] = {}
    node_counter: Counter[str] = Counter()
    edge_counter: Counter[tuple[str, str]] = Counter()
    focus_nodes: set[str] = set()
    focus_edges: set[tuple[str, str]] = set()

    for idx, path in enumerate(context_paths):
        node_ids = [node_id for node_id, _ in path]
        for node_id, label in path:
            node_labels.setdefault(node_id, label)
            node_counter[node_id] += 1
        for src, dst in zip(node_ids, node_ids[1:]):
            edge_counter[(src, dst)] += 1
        if focus_path_idx is not None and idx == int(focus_path_idx):
            focus_nodes.update(node_ids)
            focus_edges.update(zip(node_ids, node_ids[1:]))

    roots = sorted({path[0][0] for path in context_paths if path})
    max_node_visits = max(node_counter.values()) if node_counter else 1
    max_edge_visits = max(edge_counter.values()) if edge_counter else 1
    critical_structure = critical_structure or {}
    critical_node_id = critical_structure.get("critical_node_id")
    critical_pred_id = critical_structure.get("critical_successor_pred_id")
    critical_comp_id = critical_structure.get("critical_successor_comp_id")

    dot = Digraph(name=f"{plot_name}_sid_{local_explanation.sample_id}", format="png")
    title_bits = [f"sample_id={local_explanation.sample_id}", f"mode={mode_label}"]
    if critical_structure.get("critical_node_label"):
        title_bits.append(f"critical={critical_structure['critical_node_label']}")
    dot.attr(
        rankdir="LR",
        bgcolor="white",
        nodesep="0.45",
        ranksep="0.55",
        pad="0.15",
        margin="0.03",
        splines="polyline",
        concentrate="true",
        label=" | ".join(title_bits),
        labelloc="t",
        fontsize="18",
        fontname="Helvetica",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        color="#5f6b7a",
        penwidth="1.2",
        fontname="Helvetica",
        fontsize="12",
        margin="0.08,0.05",
    )
    dot.attr("edge", color="#9aa3ad", penwidth="1.4", arrowsize="0.7")

    for root in roots:
        dot.node(root, node_labels[root])

    for node_id in sorted(node_labels):
        raw_label = node_labels[node_id]
        wrapped_label = textwrap.fill(raw_label, width=18)
        visits = node_counter[node_id]
        intensity = visits / max_node_visits if max_node_visits else 0.0

        if node_id == critical_node_id:
            fill = "#ffd966"
            penwidth = "2.6"
            border = "#7f6000"
        elif node_id == critical_pred_id:
            fill = "#b6d7a8"
            penwidth = "2.2"
            border = "#38761d"
        elif node_id == critical_comp_id:
            fill = "#f4cccc"
            penwidth = "2.2"
            border = "#990000"
        elif raw_label.startswith("Class "):
            leaf_cls = raw_label.replace("Class ", "", 1)
            fill = "#93c47d" if leaf_cls == str(true_label) else "#e6b8af"
            penwidth = "1.4"
            border = "#5f6b7a"
        elif focus_path_idx is not None and node_id in focus_nodes:
            fill = "#9fc5e8"
            penwidth = "2.1"
            border = "#0b5394"
        else:
            if focus_path_idx is None:
                if intensity >= 0.75:
                    fill = "#bdd7ee"
                elif intensity >= 0.4:
                    fill = "#ddebf7"
                else:
                    fill = "#ffffff"
            else:
                fill = "#f8f9fb"
            penwidth = "1.1"
            border = "#7f8c99"
        dot.node(node_id, wrapped_label, fillcolor=fill, penwidth=penwidth, color=border)

    for (src, dst), visits in edge_counter.items():
        frac = visits / max_edge_visits if max_edge_visits else 0.0
        if critical_node_id is not None and (src, dst) == (critical_node_id, critical_pred_id):
            color = "#38761d"
            width = "3.2"
        elif critical_node_id is not None and (src, dst) == (critical_node_id, critical_comp_id):
            color = "#990000"
            width = "3.2"
        elif focus_path_idx is not None and (src, dst) in focus_edges:
            color = "#0b5394"
            width = "3.2"
        elif focus_path_idx is None:
            color = "#6fa8dc" if frac >= 0.75 else "#9fc5e8" if frac >= 0.4 else "#cfe2f3"
            width = f"{1.2 + 1.8 * frac:.2f}"
        else:
            color = "#c6cbd2"
            width = "1.0"
        dot.edge(src, dst, color=color, penwidth=width)

    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(
            label="Legend",
            labelloc="t",
            fontsize="14",
            fontname="Helvetica",
            color="#c9d2dc",
            style="rounded",
            margin="12",
        )
        legend.attr(rankdir="TB")
        legend.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fontname="Helvetica",
            fontsize="11",
            margin="0.07,0.04",
        )
        legend.attr("edge", arrowsize="0.7")

        legend.node(
            "legend_critical_node",
            "Critical node",
            fillcolor="#ffd966",
            color="#7f6000",
            penwidth="2.6",
        )
        legend.node(
            "legend_pred_successor",
            "Predicted branch",
            fillcolor="#b6d7a8",
            color="#38761d",
            penwidth="2.2",
        )
        legend.node(
            "legend_comp_successor",
            "Competitor branch",
            fillcolor="#f4cccc",
            color="#990000",
            penwidth="2.2",
        )
        legend.node(
            "legend_focus_node",
            "Focused path node",
            fillcolor="#9fc5e8",
            color="#0b5394",
            penwidth="2.1",
        )
        legend.node(
            "legend_context_node",
            "Context node",
            fillcolor="#f8f9fb" if focus_path_idx is not None else "#ddebf7",
            color="#7f8c99",
            penwidth="1.1",
        )
        legend.node(
            "legend_true_leaf",
            "True class leaf",
            fillcolor="#93c47d",
            color="#5f6b7a",
            penwidth="1.4",
        )
        legend.node(
            "legend_other_leaf",
            "Other class leaf",
            fillcolor="#e6b8af",
            color="#5f6b7a",
            penwidth="1.4",
        )

        legend.node("legend_edge_pred_src", "", shape="point", width="0.03", color="#ffffff")
        legend.node("legend_edge_pred_dst", "Critical predicted edge", fillcolor="#ffffff", color="#ffffff")
        legend.edge("legend_edge_pred_src", "legend_edge_pred_dst", color="#38761d", penwidth="3.2")

        legend.node("legend_edge_comp_src", "", shape="point", width="0.03", color="#ffffff")
        legend.node("legend_edge_comp_dst", "Critical competitor edge", fillcolor="#ffffff", color="#ffffff")
        legend.edge("legend_edge_comp_src", "legend_edge_comp_dst", color="#990000", penwidth="3.2")

        legend.node("legend_edge_focus_src", "", shape="point", width="0.03", color="#ffffff")
        legend.node("legend_edge_focus_dst", "Focused path edge", fillcolor="#ffffff", color="#ffffff")
        legend.edge("legend_edge_focus_src", "legend_edge_focus_dst", color="#0b5394", penwidth="3.2")

        legend.node("legend_edge_high_src", "", shape="point", width="0.03", color="#ffffff")
        legend.node("legend_edge_high_dst", "High-support edge", fillcolor="#ffffff", color="#ffffff")
        legend.edge("legend_edge_high_src", "legend_edge_high_dst", color="#6fa8dc", penwidth="3.0")

        legend.node("legend_edge_low_src", "", shape="point", width="0.03", color="#ffffff")
        legend.node("legend_edge_low_dst", "Low-support edge", fillcolor="#ffffff", color="#ffffff")
        legend.edge("legend_edge_low_src", "legend_edge_low_dst", color="#cfe2f3", penwidth="1.2")

        legend.node(
            "legend_edge_note",
            "Edge width encodes shared support.\nThicker edges appear more often in the local subgraph.",
            shape="note",
            fillcolor="#ffffff",
            color="#d0d7de",
            penwidth="1.0",
        )

        legend.edge("legend_critical_node", "legend_pred_successor", color="#38761d", penwidth="3.2")
        legend.edge("legend_critical_node", "legend_comp_successor", color="#990000", penwidth="3.2")

    if roots:
        dot.edge(roots[0], "legend_critical_node", style="invis", weight="0")

    png_path = save_dir / f"{plot_name}_sid_{local_explanation.sample_id}.png"
    png_bytes = dot.pipe(format="png")
    Image.open(BytesIO(png_bytes)).save(png_path)
    return str(png_path.resolve())


def _render_case_studies(
    case_rows: pd.DataFrame,
    best_configs: pd.DataFrame,
    data_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    bundles: Dict[str, DatasetBundle] = {}

    for _, case_row in case_rows.iterrows():
        dataset = str(case_row["dataset"])
        sample_idx = int(case_row["sample_idx"])
        case_label = str(case_row["case_label"])
        if dataset not in bundles:
            bundles[dataset] = _load_bundle(data_dir, dataset)
        bundle = bundles[dataset]
        sample = bundle.X_test[sample_idx]
        true_label = str(bundle.y_test[sample_idx])

        for mode in ["aggregated_transitions", "execution_trace"]:
            cfg_row = best_configs[
                (best_configs["dataset"] == dataset)
                & (best_configs["graph_construction_mode"] == mode)
            ]
            if cfg_row.empty:
                continue
            cfg_row = cfg_row.iloc[0]
            explainer, model_preds = _build_explainer(bundle, cfg_row)
            local = explainer.explain_local(sample=sample, sample_id=sample_idx, validate_graph=True)

            pred_cls = local.sample_confidence.get("support_pred_class") or local.majority_vote
            comp_cls = local.sample_confidence.get("support_top_competitor_class")
            pred_path_idx = _top_path_index(local, str(pred_cls)) if pred_cls is not None else None
            comp_path_idx = _top_path_index(local, str(comp_cls)) if comp_cls is not None else None
            critical_structure = _critical_path_structure(local, pred_path_idx, comp_path_idx)

            plot_prefix = f"{case_label}_{dataset}_{mode}"
            case_mode_dir = out_dir / case_label / mode
            case_mode_dir.mkdir(parents=True, exist_ok=True)
            aggregate_png = _render_local_dpg_subgraph_case(
                local_explanation=local,
                plot_name=f"{plot_prefix}_aggregate",
                save_dir=case_mode_dir,
                true_label=true_label,
                mode_label="aggregate_local_subgraph",
                critical_structure=critical_structure,
            )
            pred_path_png = None
            if pred_path_idx is not None:
                pred_path_png = _render_local_dpg_subgraph_case(
                    local_explanation=local,
                    plot_name=f"{plot_prefix}_pred_path",
                    save_dir=case_mode_dir,
                    true_label=true_label,
                    mode_label="focused_pred_path",
                    focus_path_idx=int(pred_path_idx),
                    critical_structure=critical_structure,
                )
            comp_path_png = None
            if comp_path_idx is not None:
                comp_path_png = _render_local_dpg_subgraph_case(
                    local_explanation=local,
                    plot_name=f"{plot_prefix}_competitor_path",
                    save_dir=case_mode_dir,
                    true_label=true_label,
                    mode_label="focused_competitor_path",
                    focus_path_idx=int(comp_path_idx),
                    critical_structure=critical_structure,
                )

            rows.append(
                {
                    "case_label": case_label,
                    "dataset": dataset,
                    "sample_idx": sample_idx,
                    "graph_construction_mode": mode,
                    "true_label": true_label,
                    "model_pred": str(model_preds[sample_idx]),
                    "local_pred": local.majority_vote,
                    "support_pred_class": local.sample_confidence.get("support_pred_class"),
                    "top_competitor_class": local.sample_confidence.get("support_top_competitor_class"),
                    "explanation_confidence": local.sample_confidence.get("explanation_confidence"),
                    "support_margin": local.sample_confidence.get("support_margin"),
                    "path_purity": local.sample_confidence.get("path_purity"),
                    "competitor_exposure": local.sample_confidence.get("competitor_exposure"),
                    "edge_precision": local.sample_confidence.get("edge_precision"),
                    "recombination_rate": local.sample_confidence.get("recombination_rate"),
                    "critical_node_label": local.sample_confidence.get("critical_node_label"),
                    "critical_split_depth": local.sample_confidence.get("critical_split_depth"),
                    "critical_node_contrast": local.sample_confidence.get("critical_node_contrast"),
                    "critical_successor_pred": local.sample_confidence.get("critical_successor_pred"),
                    "critical_successor_comp": local.sample_confidence.get("critical_successor_comp"),
                    "critical_node_id": critical_structure.get("critical_node_id"),
                    "critical_successor_pred_label": critical_structure.get("critical_successor_pred_label"),
                    "critical_successor_comp_label": critical_structure.get("critical_successor_comp_label"),
                    "aggregate_png": aggregate_png,
                    "pred_path_png": pred_path_png,
                    "competitor_path_png": comp_path_png,
                }
            )

    return pd.DataFrame(rows)


def _write_markdown_summary(
    out_path: Path,
    consolidated_summary: pd.DataFrame,
    effect_sizes: pd.DataFrame,
    case_metadata: pd.DataFrame,
) -> None:
    def as_markdown(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
        try:
            return df.to_markdown(index=False, floatfmt=floatfmt)
        except ImportError:
            return df.to_string(index=False, float_format=lambda x: format(x, floatfmt))

    lines: List[str] = []
    lines.append("# DPG 2.0 Remaining Analysis Results")
    lines.append("")
    lines.append("## Consolidated Comparison")
    lines.append("")
    top_summary = consolidated_summary.copy()
    lines.append(as_markdown(top_summary, floatfmt=".4f"))
    lines.append("")

    lines.append("## Key Cohort Effect Sizes")
    lines.append("")
    filt = effect_sizes[
        effect_sizes["metric"].isin(
            [
                "explanation_confidence",
                "path_purity",
                "competitor_exposure",
                "edge_precision",
                "recombination_rate",
            ]
        )
    ].copy()
    lines.append(as_markdown(filt.sort_values(["graph_construction_mode", "comparison", "metric"]), floatfmt=".4f"))
    lines.append("")

    lines.append("## Case Studies")
    lines.append("")
    cols = [
        "case_label",
        "dataset",
        "sample_idx",
        "graph_construction_mode",
        "true_label",
        "model_pred",
        "local_pred",
        "explanation_confidence",
        "support_margin",
        "edge_precision",
        "recombination_rate",
        "critical_node_label",
        "critical_split_depth",
        "critical_successor_pred",
        "critical_successor_comp",
        "aggregate_png",
    ]
    lines.append(as_markdown(case_metadata[cols], floatfmt=".4f"))
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the DPG 2.0 comparison story analysis.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="experiments_local_explanation/experiment_dpg2_next_phase/_analysis/comparison_story",
    )
    parser.add_argument(
        "--max_case_dataset_samples",
        type=int,
        default=600,
        help="Prefer exemplar cases from datasets with at most this many test samples. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max_case_features",
        type=int,
        default=40,
        help="Prefer exemplar cases from datasets with at most this many features. Set <=0 to disable.",
    )
    parser.add_argument(
        "--selected_case_rows_path",
        type=str,
        default="",
        help="Optional CSV path with preselected case rows to rerender instead of reselecting cases.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    next_phase_root = repo_root / "experiments_local_explanation" / "experiment_dpg2_next_phase"
    next_phase_analysis_dir = next_phase_root / "_analysis"
    baseline_root = repo_root / "experiments_local_explanation" / "results_baselines_by_dataset"
    legacy_root = repo_root / "experiments_local_explanation" / "results_by_dataset"
    pruning_root = repo_root / "experiments_local_explanation" / "experiment_pruning"
    data_dir = repo_root / "experiments_local_explanation" / "data_numeric"
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_configs, best_per_sample = _best_next_phase_tables(next_phase_analysis_dir)
    legacy_best = _legacy_best(legacy_root)
    pruning_best = _pruning_best(pruning_root)
    next_best = _next_phase_best(best_configs)
    baseline_best = _baseline_best(baseline_root)

    dataset_level, consolidated_summary = _consolidated_tables(
        legacy_best=legacy_best,
        pruning_best=pruning_best,
        next_best=next_best,
        baseline_best=baseline_best,
    )
    dataset_level.to_csv(out_dir / "consolidated_comparison_dataset_level.csv", index=False)
    consolidated_summary.to_csv(out_dir / "consolidated_comparison_summary.csv", index=False)

    effect_sizes = _effect_sizes(best_per_sample)
    effect_sizes.to_csv(out_dir / "cohort_effect_sizes.csv", index=False)

    if args.selected_case_rows_path:
        case_rows = pd.read_csv((repo_root / args.selected_case_rows_path).resolve(), low_memory=False)
    else:
        case_rows = _select_case_rows(
            best_per_sample=best_per_sample,
            max_case_dataset_samples=int(args.max_case_dataset_samples),
            max_case_features=int(args.max_case_features),
        )
    case_rows.to_csv(out_dir / "selected_case_rows.csv", index=False)
    case_metadata = _render_case_studies(
        case_rows=case_rows,
        best_configs=best_configs,
        data_dir=data_dir,
        out_dir=out_dir / "case_studies",
    )
    case_metadata.to_csv(out_dir / "case_study_metadata.csv", index=False)

    _write_markdown_summary(
        out_path=out_dir / "comparison_story_summary.md",
        consolidated_summary=consolidated_summary,
        effect_sizes=effect_sizes,
        case_metadata=case_metadata,
    )

    print(f"Saved: {out_dir / 'consolidated_comparison_dataset_level.csv'}")
    print(f"Saved: {out_dir / 'consolidated_comparison_summary.csv'}")
    print(f"Saved: {out_dir / 'cohort_effect_sizes.csv'}")
    print(f"Saved: {out_dir / 'selected_case_rows.csv'}")
    print(f"Saved: {out_dir / 'case_study_metadata.csv'}")
    print(f"Saved: {out_dir / 'comparison_story_summary.md'}")


if __name__ == "__main__":
    main()
