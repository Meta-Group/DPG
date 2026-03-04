#!/usr/bin/env python3
"""Run local-explanation research experiments for DPG across prepared datasets.

Expected dataset layout (created by prepare_datasets.py):
    <data_dir>/<dataset>/X_train.npy
    <data_dir>/<dataset>/y_train.npy
    <data_dir>/<dataset>/X_test.npy
    <data_dir>/<dataset>/y_test.npy
    <data_dir>/<dataset>/feature_names.json
    <data_dir>/<dataset>/meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure we import DPG from this repository (not an older installed package).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpg import DPGExplainer

EXCLUDED_DATASETS = {"fashion_mnist_784", "mnist_784"}


@dataclass
class ExperimentConfig:
    n_estimators: int
    max_depth: int | None
    perc_var: float
    decimal_threshold: int
    random_state: int
    graph_construction_mode: str


@dataclass
class DatasetBundle:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


@dataclass(frozen=True)
class RunIdentity:
    dataset: str
    config_id: str

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.config_id}"


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_depth_list(text: str) -> List[int | None]:
    out: List[int | None] = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"none", "null"}:
            out.append(None)
        else:
            out.append(int(item))
    return out


def _load_bundle(dataset_dir: Path) -> DatasetBundle:
    with open(dataset_dir / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return DatasetBundle(
        name=dataset_dir.name,
        X_train=np.load(dataset_dir / "X_train.npy"),
        y_train=np.load(dataset_dir / "y_train.npy"),
        X_test=np.load(dataset_dir / "X_test.npy"),
        y_test=np.load(dataset_dir / "y_test.npy"),
        feature_names=list(feature_names),
    )


def _dataset_dirs(data_dir: Path, only: Sequence[str] | None = None) -> List[Path]:
    all_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name not in EXCLUDED_DATASETS])
    if not only:
        return all_dirs
    selected = set(only)
    return [p for p in all_dirs if p.name in selected and p.name not in EXCLUDED_DATASETS]


def _resolve_cli_path(raw_path: str, script_dir: Path) -> Path:
    """
    Resolve relative CLI paths robustly.

    Priority:
    1) current working directory
    2) script directory (experiments_local_explanation/)
    """
    p = Path(raw_path)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    script_candidate = script_dir / p
    if script_candidate.exists():
        return script_candidate
    return cwd_candidate


def _make_configs(args: argparse.Namespace) -> List[ExperimentConfig]:
    n_estimators_list = _parse_int_list(args.n_estimators)
    max_depth_list = _parse_depth_list(args.max_depth)
    perc_var_list = _parse_float_list(args.perc_var)
    decimal_threshold_list = _parse_int_list(args.decimal_threshold)
    seeds = _parse_int_list(args.seeds)
    graph_construction_modes = [
        x.strip().lower() for x in args.graph_construction_modes.split(",") if x.strip()
    ]

    configs: List[ExperimentConfig] = []
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for perc_var in perc_var_list:
                for decimal_threshold in decimal_threshold_list:
                    for seed in seeds:
                        for graph_construction_mode in graph_construction_modes:
                            configs.append(
                                ExperimentConfig(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    perc_var=perc_var,
                                    decimal_threshold=decimal_threshold,
                                    random_state=seed,
                                    graph_construction_mode=graph_construction_mode,
                                )
                            )
    return configs


def _config_id(cfg: ExperimentConfig) -> str:
    # max_depth=None in sklearn means unlimited tree depth.
    depth = "unlimited" if cfg.max_depth is None else str(cfg.max_depth)
    return (
        f"rf{cfg.n_estimators}_d{depth}_pv{cfg.perc_var:g}"
        f"_dt{cfg.decimal_threshold}_gm{cfg.graph_construction_mode}_s{cfg.random_state}"
    )


def _run_identity(dataset: str, cfg: ExperimentConfig) -> RunIdentity:
    return RunIdentity(dataset=dataset, config_id=_config_id(cfg))


def _checkpoint_dir(out_dir: Path) -> Path:
    return out_dir / "checkpoints"


def _checkpoint_summary_path(out_dir: Path, run_id: RunIdentity) -> Path:
    return _checkpoint_dir(out_dir) / "summary" / f"{run_id.key}.csv"


def _checkpoint_detail_path(out_dir: Path, run_id: RunIdentity) -> Path:
    return _checkpoint_dir(out_dir) / "per_sample" / f"{run_id.key}.csv"


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _mode_or_default(series: pd.Series, default: str | int | None = None) -> str | int | None:
    clean = series.dropna()
    if clean.empty:
        return default
    mode = clean.mode()
    if mode.empty:
        return default
    return mode.iloc[0]


def _string_or_none(value: str | int | None) -> str | None:
    return None if value is None else str(value)


def _cohort_label(y_true: str, y_model_pred: str, y_local_pred: str | None) -> str:
    if y_local_pred is None:
        return "LOCAL_FAILED"
    model_correct = y_model_pred == y_true
    local_correct = y_local_pred == y_true
    if model_correct and local_correct:
        return "MC-EC"
    if model_correct and not local_correct:
        return "MC-EW"
    if not model_correct and y_local_pred == y_model_pred:
        return "MW-EM"
    if not model_correct and local_correct:
        return "MW-EC"
    return "MW-EW"


def _save_run_checkpoint(
    out_dir: Path,
    run_id: RunIdentity,
    summary: Dict[str, float | int | str | None],
    detail_df: pd.DataFrame,
) -> None:
    summary_df = pd.DataFrame([summary])
    _atomic_write_csv(summary_df, _checkpoint_summary_path(out_dir, run_id))
    _atomic_write_csv(detail_df, _checkpoint_detail_path(out_dir, run_id))


def _completed_run_keys(out_dir: Path) -> set[str]:
    sum_dir = _checkpoint_dir(out_dir) / "summary"
    det_dir = _checkpoint_dir(out_dir) / "per_sample"
    if not sum_dir.exists() or not det_dir.exists():
        return set()
    summary_keys = {p.stem for p in sum_dir.glob("*.csv")}
    detail_keys = {p.stem for p in det_dir.glob("*.csv")}
    return summary_keys & detail_keys


def _read_checkpoints(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sum_dir = _checkpoint_dir(out_dir) / "summary"
    det_dir = _checkpoint_dir(out_dir) / "per_sample"
    summary_frames: List[pd.DataFrame] = []
    detail_frames: List[pd.DataFrame] = []

    if sum_dir.exists():
        for p in sorted(sum_dir.glob("*.csv")):
            summary_frames.append(pd.read_csv(p))
    if det_dir.exists():
        for p in sorted(det_dir.glob("*.csv")):
            detail_frames.append(pd.read_csv(p))

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    if len(summary_df):
        summary_sort_cols = [c for c in ["dataset", "method", "config_id"] if c in summary_df.columns]
        summary_df = (
            summary_df.drop_duplicates(subset=["dataset", "config_id", "seed"], keep="last")
            .sort_values(summary_sort_cols)
            .reset_index(drop=True)
        )
    if len(detail_df):
        detail_sort_cols = [c for c in ["dataset", "method", "config_id", "sample_idx"] if c in detail_df.columns]
        detail_df = (
            detail_df.drop_duplicates(subset=["dataset", "config_id", "seed", "sample_idx"], keep="last")
            .sort_values(detail_sort_cols)
            .reset_index(drop=True)
        )
    return summary_df, detail_df


def _run_one_dataset(
    bundle: DatasetBundle,
    cfg: ExperimentConfig,
    max_test_samples: int,
    progress_every: int,
    rf_n_jobs: int,
) -> tuple[pd.DataFrame, Dict[str, float | int | str | None]]:
    if not hasattr(DPGExplainer, "explain_local"):
        loaded_path = sys.modules[DPGExplainer.__module__].__file__
        raise RuntimeError(
            "Loaded DPGExplainer does not provide explain_local(). "
            f"Loaded module path: {loaded_path}"
        )

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=rf_n_jobs,
    )
    model.fit(bundle.X_train, bundle.y_train)

    y_pred = model.predict(bundle.X_test)
    model_acc = float(accuracy_score(bundle.y_test, y_pred))

    class_names = [str(c) for c in model.classes_]
    explainer = DPGExplainer(
        model=model,
        feature_names=bundle.feature_names,
        target_names=class_names,
        dpg_config={
            "dpg": {
                "default": {
                    "perc_var": cfg.perc_var,
                    "decimal_threshold": cfg.decimal_threshold,
                    "n_jobs": 1,
                },
                "graph_construction": {
                    "mode": cfg.graph_construction_mode,
                },
                "local_evidence": {
                    "variant": "top_competitor",
                    "base_lambda": 0.8,
                },
            }
        },
    )
    explainer.fit(bundle.X_train)

    n_test = bundle.X_test.shape[0]
    n_eval = min(max_test_samples, n_test) if max_test_samples > 0 else n_test
    sample_rows: List[Dict[str, float | int | str | bool | None]] = []
    local_failures = 0

    for idx in range(n_eval):
        if progress_every > 0 and (idx % progress_every == 0 or idx == n_eval - 1):
            print(
                f"  explain_local progress: {idx + 1}/{n_eval} "
                f"({bundle.name} | {_config_id(cfg)})",
                flush=True,
            )
        sample = bundle.X_test[idx]
        y_true = str(bundle.y_test[idx])
        y_hat = str(y_pred[idx])
        local_status = "ok"
        local_error = ""
        try:
            local = explainer.explain_local(sample=sample, sample_id=idx, validate_graph=True)
            scores = local.sample_confidence.get("evidence_scores", {}) or {}
            support = local.sample_confidence.get("class_support", {}) or {}

            local_pred = str(local.majority_vote) if local.majority_vote is not None else None
            local_pred_score = float(scores.get(local_pred, np.nan)) if local_pred is not None else np.nan
            true_score = float(scores.get(y_true, np.nan))
            top_comp = local.sample_confidence.get("top_competitor_class_pred")
            top_comp_score = local.sample_confidence.get("evidence_score_competitor_pred")
            margin = local.sample_confidence.get("evidence_margin_pred_vs_competitor")
            num_paths = int(local.sample_confidence.get("num_paths") or 0)
            num_paths_raw = int(local.sample_confidence.get("num_paths_raw") or 0)
            num_paths_pruned = int(local.sample_confidence.get("num_paths_pruned") or 0)
            num_active_nodes = int(local.sample_confidence.get("num_active_nodes") or 0)
            num_active_edges_raw = int(local.sample_confidence.get("num_active_edges_raw") or 0)
            num_active_edges_filtered = int(local.sample_confidence.get("num_active_edges_filtered") or 0)
            path_confidence_max = local.sample_confidence.get("path_confidence_max")
            path_confidence_min_kept = local.sample_confidence.get("path_confidence_min_kept")
            mean_lrc_active_nodes = local.sample_confidence.get("mean_lrc_active_nodes")
            mean_bc_active_nodes = local.sample_confidence.get("mean_bc_active_nodes")
            class_support_total = float(sum(float(v) for v in support.values())) if support else 0.0
            n_class_support = int(len(support))
            evidence_lambda = local.sample_confidence.get("evidence_lambda")
            evidence_lambda_rule = local.sample_confidence.get("evidence_lambda_rule")
            evidence_score_rule = local.sample_confidence.get("evidence_score_rule")
            n_model_classes = local.sample_confidence.get("n_model_classes")
            n_features = local.sample_confidence.get("n_features")
            graph_construction_mode = local.sample_confidence.get("graph_construction_mode")
            num_executed_paths = local.sample_confidence.get("num_executed_paths")
            num_executed_predicates = local.sample_confidence.get("num_executed_predicates")
            num_executed_edges = local.sample_confidence.get("num_executed_edges")
            num_trace_predicates_missing_from_dpg = local.sample_confidence.get(
                "num_trace_predicates_missing_from_dpg"
            )
            num_trace_edges_missing_from_dpg = local.sample_confidence.get(
                "num_trace_edges_missing_from_dpg"
            )
            trace_node_coverage = local.sample_confidence.get("trace_node_coverage")
            trace_edge_coverage = local.sample_confidence.get("trace_edge_coverage")
            support_pred_class = local.sample_confidence.get("support_pred_class")
            support_top_competitor_class = local.sample_confidence.get("support_top_competitor_class")
            support_pred_score = local.sample_confidence.get("support_pred_score")
            support_top_competitor_score = local.sample_confidence.get("support_top_competitor_score")
            support_margin = local.sample_confidence.get("support_margin")
            predicted_class_concentration_top3 = local.sample_confidence.get(
                "predicted_class_concentration_top3"
            )
            model_vote_agreement = local.sample_confidence.get("model_vote_agreement")
            trace_coverage_score = local.sample_confidence.get("trace_coverage_score")
            explanation_confidence = local.sample_confidence.get("explanation_confidence")
            node_recall = local.sample_confidence.get("node_recall")
            node_precision = local.sample_confidence.get("node_precision")
            edge_recall = local.sample_confidence.get("edge_recall")
            edge_precision = local.sample_confidence.get("edge_precision")
            path_purity = local.sample_confidence.get("path_purity")
            competitor_exposure = local.sample_confidence.get("competitor_exposure")
            recombination_rate = local.sample_confidence.get("recombination_rate")
            critical_node_label = local.sample_confidence.get("critical_node_label")
            critical_split_depth = local.sample_confidence.get("critical_split_depth")
            critical_successor_pred = local.sample_confidence.get("critical_successor_pred")
            critical_successor_comp = local.sample_confidence.get("critical_successor_comp")
            critical_node_contrast = local.sample_confidence.get("critical_node_contrast")
            trace_node_count_unique = local.sample_confidence.get("trace_node_count_unique")
            trace_edge_count_unique = local.sample_confidence.get("trace_edge_count_unique")
            explanation_node_count_unique = local.sample_confidence.get("explanation_node_count_unique")
            explanation_edge_count_unique = local.sample_confidence.get("explanation_edge_count_unique")
        except Exception as exc:
            local_failures += 1
            local_status = "failed"
            local_error = f"{type(exc).__name__}: {exc}"
            print(
                f"  explain_local failed on sample {idx + 1}/{n_eval} "
                f"({bundle.name} | {_config_id(cfg)}): {local_error}",
                flush=True,
            )
            local_pred = None
            local_pred_score = np.nan
            true_score = np.nan
            top_comp = None
            top_comp_score = np.nan
            margin = np.nan
            num_paths = 0
            num_paths_raw = 0
            num_paths_pruned = 0
            num_active_nodes = 0
            num_active_edges_raw = 0
            num_active_edges_filtered = 0
            path_confidence_max = np.nan
            path_confidence_min_kept = np.nan
            mean_lrc_active_nodes = np.nan
            mean_bc_active_nodes = np.nan
            class_support_total = 0.0
            n_class_support = 0
            evidence_lambda = np.nan
            evidence_lambda_rule = None
            evidence_score_rule = None
            n_model_classes = np.nan
            n_features = np.nan
            graph_construction_mode = cfg.graph_construction_mode
            num_executed_paths = np.nan
            num_executed_predicates = np.nan
            num_executed_edges = np.nan
            num_trace_predicates_missing_from_dpg = np.nan
            num_trace_edges_missing_from_dpg = np.nan
            trace_node_coverage = np.nan
            trace_edge_coverage = np.nan
            support_pred_class = None
            support_top_competitor_class = None
            support_pred_score = np.nan
            support_top_competitor_score = np.nan
            support_margin = np.nan
            predicted_class_concentration_top3 = np.nan
            model_vote_agreement = np.nan
            trace_coverage_score = np.nan
            explanation_confidence = np.nan
            node_recall = np.nan
            node_precision = np.nan
            edge_recall = np.nan
            edge_precision = np.nan
            path_purity = np.nan
            competitor_exposure = np.nan
            recombination_rate = np.nan
            critical_node_label = None
            critical_split_depth = np.nan
            critical_successor_pred = None
            critical_successor_comp = None
            critical_node_contrast = np.nan
            trace_node_count_unique = np.nan
            trace_edge_count_unique = np.nan
            explanation_node_count_unique = np.nan
            explanation_edge_count_unique = np.nan

        method = "dpg" if cfg.graph_construction_mode == "aggregated_transitions" else "dpg_execution_trace"
        cohort_label = _cohort_label(y_true=y_true, y_model_pred=y_hat, y_local_pred=local_pred)
        sample_rows.append(
            {
                "dataset": bundle.name,
                "method": method,
                "graph_construction_mode": graph_construction_mode,
                "config_id": _config_id(cfg),
                "seed": cfg.random_state,
                "sample_idx": idx,
                "y_true": y_true,
                "y_model_pred": y_hat,
                "y_local_pred": local_pred,
                "y_support_pred": support_pred_class,
                "local_status": local_status,
                "local_error": local_error,
                "model_correct": y_hat == y_true,
                "local_matches_model": local_pred == y_hat if local_pred is not None else False,
                "local_correct": local_pred == y_true if local_pred is not None else False,
                "support_pred_matches_model": support_pred_class == y_hat if support_pred_class is not None else False,
                "support_pred_correct": support_pred_class == y_true if support_pred_class is not None else False,
                "support_pred_matches_local": (
                    support_pred_class == local_pred if support_pred_class is not None and local_pred is not None else False
                ),
                "cohort_label": cohort_label,
                "disagree_with_model": local_pred != y_hat if local_pred is not None else False,
                "num_paths_raw": num_paths_raw,
                "num_paths_pruned": num_paths_pruned,
                "num_paths": num_paths,
                "num_active_nodes": num_active_nodes,
                "num_active_edges_raw": num_active_edges_raw,
                "num_active_edges_filtered": num_active_edges_filtered,
                "path_confidence_max": path_confidence_max,
                "path_confidence_min_kept": path_confidence_min_kept,
                "mean_lrc_active_nodes": mean_lrc_active_nodes,
                "mean_bc_active_nodes": mean_bc_active_nodes,
                "evidence_score_local_pred": local_pred_score,
                "evidence_score_true_class": true_score,
                "evidence_margin_pred_vs_competitor": margin,
                "top_competitor_class_pred": top_comp,
                "evidence_score_competitor_pred": top_comp_score,
                "evidence_lambda": evidence_lambda,
                "evidence_lambda_rule": evidence_lambda_rule,
                "evidence_score_rule": evidence_score_rule,
                "n_model_classes": n_model_classes,
                "n_features": n_features,
                "num_executed_paths": num_executed_paths,
                "num_executed_predicates": num_executed_predicates,
                "num_executed_edges": num_executed_edges,
                "num_trace_predicates_missing_from_dpg": num_trace_predicates_missing_from_dpg,
                "num_trace_edges_missing_from_dpg": num_trace_edges_missing_from_dpg,
                "trace_node_coverage": trace_node_coverage,
                "trace_edge_coverage": trace_edge_coverage,
                "support_top_competitor_class": support_top_competitor_class,
                "support_pred_score": support_pred_score,
                "support_top_competitor_score": support_top_competitor_score,
                "support_margin": support_margin,
                "predicted_class_concentration_top3": predicted_class_concentration_top3,
                "model_vote_agreement": model_vote_agreement,
                "trace_coverage_score": trace_coverage_score,
                "explanation_confidence": explanation_confidence,
                "node_recall": node_recall,
                "node_precision": node_precision,
                "edge_recall": edge_recall,
                "edge_precision": edge_precision,
                "path_purity": path_purity,
                "competitor_exposure": competitor_exposure,
                "recombination_rate": recombination_rate,
                "critical_node_label": critical_node_label,
                "critical_split_depth": critical_split_depth,
                "critical_successor_pred": critical_successor_pred,
                "critical_successor_comp": critical_successor_comp,
                "critical_node_contrast": critical_node_contrast,
                "trace_node_count_unique": trace_node_count_unique,
                "trace_edge_count_unique": trace_edge_count_unique,
                "explanation_node_count_unique": explanation_node_count_unique,
                "explanation_edge_count_unique": explanation_edge_count_unique,
                "class_support_total": class_support_total,
                "n_class_support": n_class_support,
            }
        )

    df = pd.DataFrame(sample_rows)

    method = "dpg" if cfg.graph_construction_mode == "aggregated_transitions" else "dpg_execution_trace"
    summary: Dict[str, float | int | str | None] = {
        "dataset": bundle.name,
        "method": method,
        "graph_construction_mode": cfg.graph_construction_mode,
        "config_id": _config_id(cfg),
        "seed": cfg.random_state,
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "perc_var": cfg.perc_var,
        "decimal_threshold": cfg.decimal_threshold,
        "evidence_lambda_rule": _string_or_none(_mode_or_default(df["evidence_lambda_rule"])) if len(df) else None,
        "evidence_score_rule": _string_or_none(_mode_or_default(df["evidence_score_rule"])) if len(df) else None,
        "avg_effective_lambda": float(df["evidence_lambda"].mean()) if len(df) else np.nan,
        "n_model_classes": int(_mode_or_default(df["n_model_classes"], 0)) if len(df) else 0,
        "n_features": int(_mode_or_default(df["n_features"], 0)) if len(df) else 0,
        "avg_num_executed_paths": float(df["num_executed_paths"].mean()) if len(df) else np.nan,
        "avg_num_executed_predicates": float(df["num_executed_predicates"].mean()) if len(df) else np.nan,
        "avg_num_executed_edges": float(df["num_executed_edges"].mean()) if len(df) else np.nan,
        "avg_num_trace_predicates_missing_from_dpg": (
            float(df["num_trace_predicates_missing_from_dpg"].mean()) if len(df) else np.nan
        ),
        "avg_num_trace_edges_missing_from_dpg": (
            float(df["num_trace_edges_missing_from_dpg"].mean()) if len(df) else np.nan
        ),
        "avg_trace_node_coverage": float(df["trace_node_coverage"].mean()) if len(df) else np.nan,
        "avg_trace_edge_coverage": float(df["trace_edge_coverage"].mean()) if len(df) else np.nan,
        "n_test_total": int(n_test),
        "n_test_evaluated": int(n_eval),
        "model_accuracy": model_acc,
        "local_matches_model_rate": float(df["local_matches_model"].mean()) if len(df) else np.nan,
        "local_accuracy": float(df["local_correct"].mean()) if len(df) else np.nan,
        "avg_num_paths_raw": float(df["num_paths_raw"].mean()) if len(df) else np.nan,
        "avg_num_paths_pruned": float(df["num_paths_pruned"].mean()) if len(df) else np.nan,
        "avg_num_paths": float(df["num_paths"].mean()) if len(df) else np.nan,
        "avg_num_active_nodes": float(df["num_active_nodes"].mean()) if len(df) else np.nan,
        "avg_num_active_edges_raw": float(df["num_active_edges_raw"].mean()) if len(df) else np.nan,
        "avg_num_active_edges_filtered": float(df["num_active_edges_filtered"].mean()) if len(df) else np.nan,
        "avg_node_recall": float(df["node_recall"].mean()) if len(df) else np.nan,
        "avg_node_precision": float(df["node_precision"].mean()) if len(df) else np.nan,
        "avg_edge_recall": float(df["edge_recall"].mean()) if len(df) else np.nan,
        "avg_edge_precision": float(df["edge_precision"].mean()) if len(df) else np.nan,
        "avg_recombination_rate": float(df["recombination_rate"].mean()) if len(df) else np.nan,
        "avg_path_purity": float(df["path_purity"].mean()) if len(df) else np.nan,
        "avg_competitor_exposure": float(df["competitor_exposure"].mean()) if len(df) else np.nan,
        "avg_support_margin": float(df["support_margin"].mean()) if len(df) else np.nan,
        "avg_predicted_class_concentration_top3": (
            float(df["predicted_class_concentration_top3"].mean()) if len(df) else np.nan
        ),
        "avg_model_vote_agreement": float(df["model_vote_agreement"].mean()) if len(df) else np.nan,
        "avg_trace_coverage_score": float(df["trace_coverage_score"].mean()) if len(df) else np.nan,
        "avg_explanation_confidence": float(df["explanation_confidence"].mean()) if len(df) else np.nan,
        "avg_critical_split_depth": float(df["critical_split_depth"].mean()) if len(df) else np.nan,
        "avg_critical_node_contrast": float(df["critical_node_contrast"].mean()) if len(df) else np.nan,
        "avg_path_confidence_max": float(df["path_confidence_max"].mean()) if len(df) else np.nan,
        "avg_path_confidence_min_kept": float(df["path_confidence_min_kept"].mean()) if len(df) else np.nan,
        "avg_evidence_score_local_pred": float(df["evidence_score_local_pred"].mean()) if len(df) else np.nan,
        "avg_evidence_margin_pred_vs_competitor": float(df["evidence_margin_pred_vs_competitor"].mean())
        if len(df)
        else np.nan,
        "local_failures": int(local_failures),
        "local_failure_rate": float(local_failures / n_eval) if n_eval > 0 else np.nan,
    }

    if len(df):
        correct_df = df[df["model_correct"]]
        wrong_df = df[~df["model_correct"]]
        disagree_df = df[df["disagree_with_model"]]
        agree_df = df[~df["disagree_with_model"]]
        summary["avg_margin_model_correct"] = (
            float(correct_df["evidence_margin_pred_vs_competitor"].mean())
            if len(correct_df)
            else np.nan
        )
        summary["avg_margin_model_wrong"] = (
            float(wrong_df["evidence_margin_pred_vs_competitor"].mean())
            if len(wrong_df)
            else np.nan
        )
        summary["disagree_rate"] = float(df["disagree_with_model"].mean())
        for cohort in ["MC-EC", "MC-EW", "MW-EM", "MW-EC", "MW-EW", "LOCAL_FAILED"]:
            cohort_df = df[df["cohort_label"] == cohort]
            key_prefix = cohort.lower().replace("-", "_")
            summary[f"{key_prefix}_count"] = int(len(cohort_df))
            summary[f"{key_prefix}_rate"] = float(len(cohort_df) / len(df))
            summary[f"{key_prefix}_avg_explanation_confidence"] = (
                float(cohort_df["explanation_confidence"].mean()) if len(cohort_df) else np.nan
            )
            summary[f"{key_prefix}_avg_recombination_rate"] = (
                float(cohort_df["recombination_rate"].mean()) if len(cohort_df) else np.nan
            )
            summary[f"{key_prefix}_avg_path_purity"] = (
                float(cohort_df["path_purity"].mean()) if len(cohort_df) else np.nan
            )
            summary[f"{key_prefix}_avg_critical_split_depth"] = (
                float(cohort_df["critical_split_depth"].mean()) if len(cohort_df) else np.nan
            )
        summary["agree_avg_explanation_confidence"] = (
            float(agree_df["explanation_confidence"].mean()) if len(agree_df) else np.nan
        )
        summary["disagree_avg_explanation_confidence"] = (
            float(disagree_df["explanation_confidence"].mean()) if len(disagree_df) else np.nan
        )
    else:
        summary["avg_margin_model_correct"] = np.nan
        summary["avg_margin_model_wrong"] = np.nan
        summary["disagree_rate"] = np.nan

    return df, summary


def _write_report(
    out_dir: Path,
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    detail_df.to_csv(out_dir / "per_sample.csv", index=False)

    grouped = pd.DataFrame()
    if len(summary_df):
        group_cols = ["dataset", "method"] if "method" in summary_df.columns else ["dataset"]
        grouped = (
            summary_df.groupby(group_cols, as_index=False)
            .agg(
                runs=("config_id", "count"),
                best_model_accuracy=("model_accuracy", "max"),
                best_local_match_rate=("local_matches_model_rate", "max"),
                best_local_accuracy=("local_accuracy", "max"),
                min_avg_paths=("avg_num_paths", "min"),
                max_avg_paths=("avg_num_paths", "max"),
            )
            .sort_values("dataset")
        )
    grouped.to_csv(out_dir / "dataset_overview.csv", index=False)



def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run DPG local explanation experiments over prepared numeric datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_numeric",
        help="Directory that contains prepared datasets.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory for CSV reports.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names. Empty means all subdirs in data_dir.",
    )
    parser.add_argument(
        "--n_estimators",
        type=str,
        default="10,20",
        help="Comma-separated RandomForest n_estimators values.",
    )
    parser.add_argument(
        "--rf_n_jobs",
        type=int,
        default=-1,
        help="RandomForest n_jobs per process. Use with --parallel to control total CPU usage.",
    )
    parser.add_argument(
        "--max_depth",
        type=str,
        default="2,4,None",
        help="Comma-separated max_depth values (use None for unlimited).",
    )
    parser.add_argument(
        "--perc_var",
        type=str,
        default="0.0,0.001",
        help="Comma-separated DPG perc_var values.",
    )
    parser.add_argument(
        "--decimal_threshold",
        type=str,
        default="4,6",
        help="Comma-separated DPG decimal_threshold values.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="27,42",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--graph_construction_modes",
        type=str,
        default="aggregated_transitions",
        help="Comma-separated DPG graph construction modes.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=0,
        help="Max test samples to explain per dataset/config. 0 means all test samples.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=25,
        help="Print local-explanation progress every N samples (0 disables).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from per-run checkpoints in out_dir/checkpoints (default: enabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore checkpoints and recompute all runs.",
    )
    args = parser.parse_args()

    data_dir = _resolve_cli_path(args.data_dir, script_dir=script_dir)
    out_dir = _resolve_cli_path(args.out_dir, script_dir=script_dir)
    only = [x.strip() for x in args.datasets.split(",") if x.strip()]

    dataset_paths = _dataset_dirs(data_dir, only=only)
    if not dataset_paths:
        raise ValueError(f"No dataset directories found in: {data_dir}")

    configs = _make_configs(args)
    if not configs:
        raise ValueError("No experiment configurations generated from CLI arguments")

    already_done = _completed_run_keys(out_dir) if args.resume and not args.overwrite else set()
    if already_done:
        print(f"Found {len(already_done)} completed checkpoints in {_checkpoint_dir(out_dir)}")

    total_runs = len(dataset_paths) * len(configs)
    run_idx = 0
    executed_runs = 0
    skipped_runs = 0
    for ds_path in dataset_paths:
        bundle = _load_bundle(ds_path)
        for cfg in configs:
            run_idx += 1
            run_id = _run_identity(bundle.name, cfg)
            if run_id.key in already_done:
                skipped_runs += 1
                print(f"[{run_idx}/{total_runs}] skip (checkpoint) {bundle.name} | {_config_id(cfg)}")
                continue
            print(f"[{run_idx}/{total_runs}] {bundle.name} | {_config_id(cfg)}")
            detail_df, summary = _run_one_dataset(
                bundle=bundle,
                cfg=cfg,
                max_test_samples=args.max_test_samples,
                progress_every=args.progress_every,
                rf_n_jobs=int(args.rf_n_jobs),
            )
            _save_run_checkpoint(out_dir=out_dir, run_id=run_id, summary=summary, detail_df=detail_df)
            executed_runs += 1
            print(f"  checkpoint saved: {run_id.key}")

    summary_df, detail_df = _read_checkpoints(out_dir)

    _write_report(out_dir=out_dir, summary_df=summary_df, detail_df=detail_df)
    print(f"Executed runs: {executed_runs}, skipped runs: {skipped_runs}")
    print(f"Saved: {out_dir / 'summary.csv'}")
    print(f"Saved: {out_dir / 'per_sample.csv'}")
    print(f"Saved: {out_dir / 'dataset_overview.csv'}")


if __name__ == "__main__":
    main()
