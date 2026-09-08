#!/usr/bin/env python3
"""Run local-explanation baseline experiments (DPG, SHAP, LIME, ICE, Anchors, tree-native).

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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Ensure we import DPG from this repository (not an older installed package).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpg import DPGExplainer
from experiments_local_explanation.journal_splits import apply_split_registry, load_split_registry

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None


EXCLUDED_DATASETS = {"fashion_mnist_784", "mnist_784"}
METHODS_DEFAULT = (
    "dpg",
    "shap",
    "lime",
    "ice",
    "anchors",
    "tree_path",
    "lore",
    "raw_path_union",
    "path_bag",
    "random_same_size_path",
)


@dataclass
class ExperimentConfig:
    n_estimators: int
    max_depth: int | None
    perc_var: float
    decimal_threshold: int
    random_state: int


@dataclass
class DatasetBundle:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    train_split: str = "existing_train"
    eval_split: str = "existing_test"
    split_registry_path: str | None = None


@dataclass(frozen=True)
class RunIdentity:
    dataset: str
    config_id: str
    method: str
    train_split: str = "existing_train"
    eval_split: str = "existing_test"

    @property
    def key(self) -> str:
        if self.train_split == "existing_train" and self.eval_split in {"existing_test", "test"}:
            return f"{self.dataset}__{self.config_id}__{self.method}"
        return f"{self.dataset}__tr{self.train_split}__ev{self.eval_split}__{self.config_id}__{self.method}"


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


def _parse_method_list(text: str) -> List[str]:
    methods = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not methods:
        methods = list(METHODS_DEFAULT)
    invalid = sorted([m for m in methods if m not in METHODS_DEFAULT])
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Allowed: {METHODS_DEFAULT}")
    return methods


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

    configs: List[ExperimentConfig] = []
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for perc_var in perc_var_list:
                for decimal_threshold in decimal_threshold_list:
                    for seed in seeds:
                        configs.append(
                            ExperimentConfig(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                perc_var=perc_var,
                                decimal_threshold=decimal_threshold,
                                random_state=seed,
                            )
                        )
    return configs


def _config_id(cfg: ExperimentConfig) -> str:
    depth = "unlimited" if cfg.max_depth is None else str(cfg.max_depth)
    return (
        f"rf{cfg.n_estimators}_d{depth}_pv{cfg.perc_var:g}"
        f"_dt{cfg.decimal_threshold}_s{cfg.random_state}"
    )


def _run_identity(
    dataset: str,
    cfg: ExperimentConfig,
    method: str,
    train_split: str = "existing_train",
    eval_split: str = "existing_test",
) -> RunIdentity:
    return RunIdentity(
        dataset=dataset,
        config_id=_config_id(cfg),
        method=method,
        train_split=train_split,
        eval_split=eval_split,
    )


def _checkpoint_dir(out_dir: Path) -> Path:
    return out_dir / "checkpoints_baselines"


def _checkpoint_summary_path(out_dir: Path, run_id: RunIdentity) -> Path:
    return _checkpoint_dir(out_dir) / "summary" / f"{run_id.key}.csv"


def _checkpoint_detail_path(out_dir: Path, run_id: RunIdentity) -> Path:
    return _checkpoint_dir(out_dir) / "per_sample" / f"{run_id.key}.csv"


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


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
        summary_unique_cols = [
            c
            for c in ["dataset", "method", "train_split", "eval_split", "config_id", "seed"]
            if c in summary_df.columns
        ]
        summary_df = (
            summary_df.drop_duplicates(subset=summary_unique_cols, keep="last")
            .sort_values(["dataset", "method", "config_id"])
            .reset_index(drop=True)
        )
    if len(detail_df):
        detail_unique_cols = [
            c
            for c in ["dataset", "method", "train_split", "eval_split", "config_id", "seed", "sample_idx"]
            if c in detail_df.columns
        ]
        detail_df = (
            detail_df.drop_duplicates(subset=detail_unique_cols, keep="last")
            .sort_values(["dataset", "method", "config_id", "sample_idx"])
            .reset_index(drop=True)
        )
    return summary_df, detail_df


def _score_margin_from_probs(prob: np.ndarray, pred_idx: int) -> float:
    if prob.size <= 1:
        return float("nan")
    sorted_idx = np.argsort(prob)
    competitor = int(sorted_idx[-2]) if prob.size > 1 else pred_idx
    return float(prob[pred_idx] - prob[competitor])


def _vector_stats(v: np.ndarray) -> tuple[float, float, int]:
    if v.size == 0:
        return float("nan"), float("nan"), 0
    l1 = float(np.sum(np.abs(v)))
    l2 = float(np.linalg.norm(v))
    nnz = int(np.sum(np.abs(v) > 1e-12))
    return l1, l2, nnz


def _tree_node_prob_vector(tree: Any, node_id: int) -> np.ndarray:
    counts = np.asarray(tree.tree_.value[node_id][0], dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total


def _tree_path_nodes_and_predicates(
    estimator: Any,
    sample: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[List[int], List[Tuple[int, str, str, float]]]:
    tree = estimator.tree_
    node_id = 0
    path_nodes = [int(node_id)]
    predicates: List[Tuple[int, str, str, float]] = []

    while tree.feature[node_id] >= 0:
        feat_idx = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        if float(sample[feat_idx]) <= threshold:
            op = "<="
            child = int(tree.children_left[node_id])
        else:
            op = ">"
            child = int(tree.children_right[node_id])
        predicates.append((feat_idx, str(feature_names[feat_idx]), op, threshold))
        node_id = child
        path_nodes.append(int(node_id))
    return path_nodes, predicates


def _explain_with_tree_path(
    model: RandomForestClassifier,
    sample: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    prob = model.predict_proba(sample.reshape(1, -1))[0]
    pred_idx = int(np.argmax(prob))
    pred_label = str(model.classes_[pred_idx])
    margin = _score_margin_from_probs(prob, pred_idx)

    contrib_vec = np.zeros(sample.shape[0], dtype=float)
    path_lengths: List[int] = []
    for estimator in model.estimators_:
        path_nodes, predicates = _tree_path_nodes_and_predicates(estimator, sample, feature_names)
        path_lengths.append(len(predicates))
        for parent, child in zip(path_nodes, path_nodes[1:]):
            delta = _tree_node_prob_vector(estimator, child)[pred_idx] - _tree_node_prob_vector(estimator, parent)[pred_idx]
            feat_idx = int(estimator.tree_.feature[parent])
            if feat_idx >= 0:
                contrib_vec[feat_idx] += float(delta) / max(len(model.estimators_), 1)

    abs_vec = np.abs(contrib_vec)
    top_idx = int(np.argmax(abs_vec)) if abs_vec.size else -1
    top_abs = float(abs_vec[top_idx]) if top_idx >= 0 else np.nan
    l1, l2, nnz = _vector_stats(contrib_vec)
    competitor_idx = int(np.argsort(prob)[-2]) if prob.size > 1 else pred_idx

    return {
        "y_local_pred": pred_label,
        "evidence_score_local_pred": float(prob[pred_idx]),
        "evidence_score_true_class": np.nan,
        "top_competitor_class_pred": str(model.classes_[competitor_idx]) if prob.size > 1 else None,
        "evidence_score_competitor_pred": float(prob[competitor_idx]) if prob.size > 1 else np.nan,
        "score_margin_pred_vs_competitor": margin,
        "num_paths": np.nan,
        "num_active_nodes": float(np.mean(path_lengths)) if path_lengths else np.nan,
        "mean_lrc_active_nodes": np.nan,
        "mean_bc_active_nodes": np.nan,
        "class_support_total": np.nan,
        "n_class_support": np.nan,
        "top_feature_idx": top_idx if top_idx >= 0 else np.nan,
        "top_feature_abs_contrib": top_abs,
        "contrib_l1": l1,
        "contrib_l2": l2,
        "contrib_nnz": nnz,
        "ice_feature_idx": np.nan,
        "ice_prob_min": np.nan,
        "ice_prob_max": np.nan,
        "ice_prob_range": np.nan,
        "ice_slope": np.nan,
        "anchor_rule": None,
        "anchor_precision": np.nan,
        "anchor_coverage": np.nan,
    }


def _rule_mask(X: np.ndarray, rule: Sequence[Tuple[int, str, str, float]]) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(X.shape[0], dtype=bool)
    for feat_idx, _feat_name, op, threshold in rule:
        if op == "<=":
            mask &= X[:, feat_idx] <= float(threshold)
        else:
            mask &= X[:, feat_idx] > float(threshold)
    return mask


def _format_anchor_rule(rule: Sequence[Tuple[int, str, str, float]]) -> str:
    return " AND ".join(f"{feat_name} {op} {threshold:.6f}" for _, feat_name, op, threshold in rule)


def _predicate_key(predicate: Tuple[int, str, str, float]) -> Tuple[int, str, str, float]:
    feat_idx, feat_name, op, threshold = predicate
    return (int(feat_idx), str(feat_name), str(op), round(float(threshold), 6))


def _forest_predicate_vocabulary(
    model: RandomForestClassifier,
    feature_names: Sequence[str],
) -> List[Tuple[int, str, str, float]]:
    vocab: List[Tuple[int, str, str, float]] = []
    seen = set()
    for estimator in model.estimators_:
        tree = estimator.tree_
        for node_id in range(tree.node_count):
            feat_idx = int(tree.feature[node_id])
            if feat_idx < 0:
                continue
            key = (feat_idx, str(feature_names[feat_idx]), "<=", round(float(tree.threshold[node_id]), 6))
            if key not in seen:
                vocab.append(key)
                seen.add(key)
            mirror = (feat_idx, str(feature_names[feat_idx]), ">", round(float(tree.threshold[node_id]), 6))
            if mirror not in seen:
                vocab.append(mirror)
                seen.add(mirror)
    return vocab


def _path_control_output(
    model: RandomForestClassifier,
    sample: np.ndarray,
    feature_names: Sequence[str],
    method: str,
    random_state: int,
) -> Dict[str, Any]:
    prob = model.predict_proba(sample.reshape(1, -1))[0]
    pred_idx = int(np.argmax(prob))
    pred_label = str(model.classes_[pred_idx])
    competitor_idx = int(np.argsort(prob)[-2]) if prob.size > 1 else pred_idx
    margin = _score_margin_from_probs(prob, pred_idx)

    path_predicates: List[Tuple[int, str, str, float]] = []
    path_lengths: List[int] = []
    for estimator in model.estimators_:
        _, predicates = _tree_path_nodes_and_predicates(estimator, sample, feature_names)
        path_lengths.append(len(predicates))
        path_predicates.extend(_predicate_key(p) for p in predicates)

    unique_predicates = sorted(set(path_predicates), key=lambda p: (p[0], p[1], p[2], p[3]))
    rng = np.random.default_rng(int(random_state))

    if method == "raw_path_union":
        selected = unique_predicates
        denominator = max(len(unique_predicates), 1)
        description = "union of unique executed predicates"
    elif method == "path_bag":
        selected = list(path_predicates)
        denominator = max(len(path_predicates), 1)
        description = "multiset of executed predicates"
    elif method == "random_same_size_path":
        vocab = _forest_predicate_vocabulary(model, feature_names)
        size = min(len(unique_predicates), len(vocab))
        if size > 0:
            take = rng.choice(np.arange(len(vocab)), size=size, replace=False)
            selected = [vocab[int(i)] for i in take]
        else:
            selected = []
        denominator = max(len(selected), 1)
        description = "random same-size forest predicate control"
    else:
        raise ValueError(f"Unsupported path control: {method}")

    contrib_vec = np.zeros(sample.shape[0], dtype=float)
    for feat_idx, _feat_name, _op, _threshold in selected:
        if 0 <= int(feat_idx) < contrib_vec.size:
            contrib_vec[int(feat_idx)] += 1.0 / denominator

    abs_vec = np.abs(contrib_vec)
    top_idx = int(np.argmax(abs_vec)) if abs_vec.size and np.any(abs_vec > 0) else -1
    top_abs = float(abs_vec[top_idx]) if top_idx >= 0 else np.nan
    l1, l2, nnz = _vector_stats(contrib_vec)
    unique_features = sorted({int(p[0]) for p in selected})

    return {
        "y_local_pred": pred_label,
        "evidence_score_local_pred": float(prob[pred_idx]),
        "evidence_score_true_class": np.nan,
        "top_competitor_class_pred": str(model.classes_[competitor_idx]) if prob.size > 1 else None,
        "evidence_score_competitor_pred": float(prob[competitor_idx]) if prob.size > 1 else np.nan,
        "score_margin_pred_vs_competitor": margin,
        "num_paths": float(len(model.estimators_)),
        "num_active_nodes": float(len(selected)),
        "mean_lrc_active_nodes": np.nan,
        "mean_bc_active_nodes": np.nan,
        "class_support_total": np.nan,
        "n_class_support": np.nan,
        "top_feature_idx": top_idx if top_idx >= 0 else np.nan,
        "top_feature_abs_contrib": top_abs,
        "contrib_l1": l1,
        "contrib_l2": l2,
        "contrib_nnz": nnz,
        "ice_feature_idx": np.nan,
        "ice_prob_min": np.nan,
        "ice_prob_max": np.nan,
        "ice_prob_range": np.nan,
        "ice_slope": np.nan,
        "anchor_rule": None,
        "anchor_precision": np.nan,
        "anchor_coverage": np.nan,
        "lore_rule": None,
        "lore_rule_length": np.nan,
        "lore_fidelity_neighborhood": np.nan,
        "lore_counterfactual_distance": np.nan,
        "lore_counterfactual_features": "",
        "path_control_type": description,
        "path_control_predicate_count": float(len(selected)),
        "path_control_unique_predicate_count": float(len(set(selected))),
        "path_control_unique_feature_count": float(len(unique_features)),
        "path_control_mean_path_length": float(np.mean(path_lengths)) if path_lengths else np.nan,
    }


def _best_anchor_rule(
    model: RandomForestClassifier,
    X_reference: np.ndarray,
    sample: np.ndarray,
    feature_names: Sequence[str],
    target_label: str,
    precision_target: float,
    max_rule_len: int,
    max_candidates: int,
) -> tuple[List[Tuple[int, str, str, float]], float, float, str | None, float]:
    pred_supporting: List[List[Tuple[int, str, str, float]]] = []
    for estimator in model.estimators_:
        tree_pred = str(estimator.predict(sample.reshape(1, -1))[0])
        if tree_pred != target_label:
            continue
        _, predicates = _tree_path_nodes_and_predicates(estimator, sample, feature_names)
        pred_supporting.append(predicates)

    predicate_counts: Dict[Tuple[int, str, str, float], int] = {}
    for predicates in pred_supporting:
        seen = set()
        for feat_idx, feat_name, op, threshold in predicates:
            key = (int(feat_idx), str(feat_name), str(op), round(float(threshold), 6))
            if key in seen:
                continue
            predicate_counts[key] = predicate_counts.get(key, 0) + 1
            seen.add(key)

    ranked_candidates = sorted(
        predicate_counts.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][2], kv[0][3]),
    )[: max(int(max_candidates), 1)]

    best_rule: List[Tuple[int, str, str, float]] = []
    best_precision = 0.0
    best_coverage = 0.0
    best_competitor = None
    best_competitor_score = 0.0

    current_rule: List[Tuple[int, str, str, float]] = []
    for (feat_idx, feat_name, op, threshold), _count in ranked_candidates:
        candidate = current_rule + [(feat_idx, feat_name, op, float(threshold))]
        mask = _rule_mask(X_reference, candidate)
        covered = int(mask.sum())
        if covered <= 0:
            continue
        covered_preds = model.predict(X_reference[mask])
        pred_share = float(np.mean(covered_preds.astype(str) == str(target_label)))
        values, counts = np.unique(covered_preds.astype(str), return_counts=True)
        shares = sorted(
            ((str(v), float(c / covered)) for v, c in zip(values, counts) if str(v) != str(target_label)),
            key=lambda kv: kv[1],
            reverse=True,
        )
        competitor = shares[0][0] if shares else None
        competitor_score = shares[0][1] if shares else 0.0
        coverage = float(covered / X_reference.shape[0])

        improved = (
            pred_share > best_precision
            or (abs(pred_share - best_precision) <= 1e-12 and coverage > best_coverage)
        )
        if improved:
            best_rule = list(candidate)
            best_precision = pred_share
            best_coverage = coverage
            best_competitor = competitor
            best_competitor_score = competitor_score

        current_rule = candidate
        if pred_share >= precision_target or len(current_rule) >= int(max_rule_len):
            best_rule = list(candidate)
            best_precision = pred_share
            best_coverage = coverage
            best_competitor = competitor
            best_competitor_score = competitor_score
            break

    return best_rule, best_precision, best_coverage, best_competitor, best_competitor_score


def _explain_with_anchors(
    model: RandomForestClassifier,
    bundle: DatasetBundle,
    sample: np.ndarray,
    precision_target: float,
    max_rule_len: int,
    max_candidates: int,
) -> Dict[str, Any]:
    prob = model.predict_proba(sample.reshape(1, -1))[0]
    pred_idx = int(np.argmax(prob))
    target_label = str(model.classes_[pred_idx])
    rule, precision, coverage, competitor, competitor_score = _best_anchor_rule(
        model=model,
        X_reference=bundle.X_train,
        sample=sample,
        feature_names=bundle.feature_names,
        target_label=target_label,
        precision_target=float(precision_target),
        max_rule_len=int(max_rule_len),
        max_candidates=int(max_candidates),
    )

    contrib_vec = np.zeros(sample.shape[0], dtype=float)
    if rule:
        weight = float(precision / max(len(rule), 1))
        for feat_idx, _feat_name, _op, _threshold in rule:
            contrib_vec[int(feat_idx)] += weight

    abs_vec = np.abs(contrib_vec)
    top_idx = int(np.argmax(abs_vec)) if abs_vec.size and np.any(abs_vec > 0) else -1
    top_abs = float(abs_vec[top_idx]) if top_idx >= 0 else np.nan
    l1, l2, nnz = _vector_stats(contrib_vec)

    return {
        "y_local_pred": target_label,
        "evidence_score_local_pred": float(precision),
        "evidence_score_true_class": np.nan,
        "top_competitor_class_pred": competitor,
        "evidence_score_competitor_pred": float(competitor_score),
        "score_margin_pred_vs_competitor": float(precision - competitor_score),
        "num_paths": 1.0 if rule else np.nan,
        "num_active_nodes": float(len(rule)) if rule else np.nan,
        "mean_lrc_active_nodes": np.nan,
        "mean_bc_active_nodes": np.nan,
        "class_support_total": float(coverage),
        "n_class_support": float(len(rule)),
        "top_feature_idx": top_idx if top_idx >= 0 else np.nan,
        "top_feature_abs_contrib": top_abs,
        "contrib_l1": l1,
        "contrib_l2": l2,
        "contrib_nnz": nnz,
        "ice_feature_idx": np.nan,
        "ice_prob_min": np.nan,
        "ice_prob_max": np.nan,
        "ice_prob_range": np.nan,
        "ice_slope": np.nan,
        "anchor_rule": _format_anchor_rule(rule) if rule else None,
        "anchor_precision": float(precision),
        "anchor_coverage": float(coverage),
    }


def _explain_with_lore(
    model: RandomForestClassifier,
    bundle: DatasetBundle,
    sample: np.ndarray,
    random_state: int,
    neighborhood_size: int,
    max_depth: int,
    min_samples_leaf: int,
    neighborhood_scale: float,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(random_state))
    X_train = np.asarray(bundle.X_train, dtype=float)
    sample = np.asarray(sample, dtype=float)
    n_features = sample.shape[0]
    n_neighbors = max(int(neighborhood_size), 32)

    feature_std = np.nanstd(X_train, axis=0)
    positive_std = feature_std[feature_std > 1e-12]
    fallback_std = float(np.median(positive_std)) if positive_std.size else 1.0
    feature_std = np.where(feature_std > 1e-12, feature_std, fallback_std)
    noise = rng.normal(loc=0.0, scale=feature_std * float(neighborhood_scale), size=(n_neighbors, n_features))
    neighborhood = sample.reshape(1, -1) + noise
    neighborhood[0] = sample
    lo = np.nanmin(X_train, axis=0)
    hi = np.nanmax(X_train, axis=0)
    neighborhood = np.clip(neighborhood, lo, hi)

    model_labels = model.predict(neighborhood).astype(str)
    classes, counts = np.unique(model_labels, return_counts=True)
    if classes.size < 2:
        # Borrow nearest training points if pure local noise cannot cross a model boundary.
        train_labels = model.predict(X_train).astype(str)
        train_dist = np.linalg.norm((X_train - sample) / feature_std, axis=1)
        borrow_idx = np.argsort(train_dist)[: min(n_neighbors, X_train.shape[0])]
        neighborhood = np.vstack([neighborhood, X_train[borrow_idx]])
        model_labels = np.concatenate([model_labels, train_labels[borrow_idx]])

    local_tree = DecisionTreeClassifier(
        max_depth=max(int(max_depth), 1),
        min_samples_leaf=max(int(min_samples_leaf), 1),
        random_state=int(random_state),
    )
    local_tree.fit(neighborhood, model_labels)
    local_pred = str(local_tree.predict(sample.reshape(1, -1))[0])
    proba = local_tree.predict_proba(sample.reshape(1, -1))[0]
    class_labels = [str(c) for c in local_tree.classes_]
    local_idx = class_labels.index(local_pred)
    sorted_idx = np.argsort(proba)
    comp_idx = int(sorted_idx[-2]) if proba.size > 1 else local_idx

    path_nodes, predicates = _tree_path_nodes_and_predicates(local_tree, sample, bundle.feature_names)
    contrib_vec = np.zeros(n_features, dtype=float)
    for parent, child in zip(path_nodes, path_nodes[1:]):
        parent_probs = _tree_node_prob_vector(local_tree, parent)
        child_probs = _tree_node_prob_vector(local_tree, child)
        feat_idx = int(local_tree.tree_.feature[parent])
        if feat_idx >= 0 and local_idx < child_probs.size and local_idx < parent_probs.size:
            contrib_vec[feat_idx] += float(child_probs[local_idx] - parent_probs[local_idx])

    neighborhood_pred = local_tree.predict(neighborhood).astype(str)
    local_fidelity = float(np.mean(neighborhood_pred == model_labels)) if model_labels.size else np.nan

    opposite_mask = model_labels != local_pred
    counterfactual_distance = np.nan
    counterfactual_features = ""
    if np.any(opposite_mask):
        opposite = neighborhood[opposite_mask]
        distances = np.linalg.norm((opposite - sample) / feature_std, axis=1)
        best_idx = int(np.argmin(distances))
        cf = opposite[best_idx]
        counterfactual_distance = float(distances[best_idx])
        changed = np.argsort(np.abs((cf - sample) / feature_std))[::-1]
        changed = [int(i) for i in changed if abs(float(cf[i] - sample[i])) > 1e-12][:5]
        counterfactual_features = ",".join(str(bundle.feature_names[i]) for i in changed)

    abs_vec = np.abs(contrib_vec)
    top_idx = int(np.argmax(abs_vec)) if abs_vec.size else -1
    top_abs = float(abs_vec[top_idx]) if top_idx >= 0 else np.nan
    l1, l2, nnz = _vector_stats(contrib_vec)

    return {
        "y_local_pred": local_pred,
        "evidence_score_local_pred": float(proba[local_idx]),
        "evidence_score_true_class": np.nan,
        "top_competitor_class_pred": class_labels[comp_idx] if comp_idx < len(class_labels) else None,
        "evidence_score_competitor_pred": float(proba[comp_idx]) if comp_idx < proba.size else np.nan,
        "score_margin_pred_vs_competitor": float(proba[local_idx] - proba[comp_idx]) if proba.size > 1 else np.nan,
        "num_paths": 1.0,
        "num_active_nodes": float(len(predicates)),
        "mean_lrc_active_nodes": np.nan,
        "mean_bc_active_nodes": np.nan,
        "class_support_total": float(np.sum(model_labels == local_pred) / len(model_labels)) if len(model_labels) else np.nan,
        "n_class_support": float(len(np.unique(model_labels))) if len(model_labels) else np.nan,
        "top_feature_idx": top_idx if top_idx >= 0 else np.nan,
        "top_feature_abs_contrib": top_abs,
        "contrib_l1": l1,
        "contrib_l2": l2,
        "contrib_nnz": nnz,
        "ice_feature_idx": np.nan,
        "ice_prob_min": np.nan,
        "ice_prob_max": np.nan,
        "ice_prob_range": np.nan,
        "ice_slope": np.nan,
        "lore_rule": _format_anchor_rule(predicates) if predicates else None,
        "lore_rule_length": float(len(predicates)),
        "lore_fidelity_neighborhood": local_fidelity,
        "lore_counterfactual_distance": counterfactual_distance,
        "lore_counterfactual_features": counterfactual_features,
    }


def _explain_with_dpg(
    explainer: DPGExplainer,
    sample: np.ndarray,
    sample_idx: int,
    y_true: str,
    y_hat: str,
) -> Dict[str, Any]:
    local = explainer.explain_local(sample=sample, sample_id=sample_idx, validate_graph=True)
    conf = local.sample_confidence
    scores = conf.get("evidence_scores", {}) or {}
    support = conf.get("class_support", {}) or {}
    local_pred = str(local.majority_vote) if local.majority_vote is not None else None
    local_pred_score = float(scores.get(local_pred, np.nan)) if local_pred is not None else np.nan
    true_score = float(scores.get(y_true, np.nan))
    top_comp = conf.get("top_competitor_class_pred")
    top_comp_score = conf.get("evidence_score_competitor_pred")
    margin = conf.get("evidence_margin_pred_vs_competitor")
    num_paths = int(conf.get("num_paths") or 0)
    num_active_nodes = int(conf.get("num_active_nodes") or 0)
    mean_lrc_active_nodes = conf.get("mean_lrc_active_nodes")
    mean_bc_active_nodes = conf.get("mean_bc_active_nodes")
    class_support_total = float(sum(float(v) for v in support.values())) if support else 0.0
    n_class_support = int(len(support))

    contrib_vec = np.array([float(v) for v in scores.values()], dtype=float)
    l1, l2, nnz = _vector_stats(contrib_vec)

    return {
        "y_local_pred": local_pred,
        "evidence_score_local_pred": local_pred_score,
        "evidence_score_true_class": true_score,
        "top_competitor_class_pred": top_comp,
        "evidence_score_competitor_pred": top_comp_score,
        "score_margin_pred_vs_competitor": margin,
        "num_paths": num_paths,
        "num_active_nodes": num_active_nodes,
        "mean_lrc_active_nodes": mean_lrc_active_nodes,
        "mean_bc_active_nodes": mean_bc_active_nodes,
        "class_support_total": class_support_total,
        "n_class_support": n_class_support,
        "top_feature_idx": np.nan,
        "top_feature_abs_contrib": np.nan,
        "contrib_l1": l1,
        "contrib_l2": l2,
        "contrib_nnz": nnz,
        "ice_feature_idx": np.nan,
        "ice_prob_min": np.nan,
        "ice_prob_max": np.nan,
        "ice_prob_range": np.nan,
        "ice_slope": np.nan,
    }


def _explain_with_shap(
    explainer: Any,
    model: RandomForestClassifier,
    sample: np.ndarray,
) -> Dict[str, Any]:
    prob = model.predict_proba(sample.reshape(1, -1))[0]
    pred_idx = int(np.argmax(prob))
    y_local_pred = str(model.classes_[pred_idx])
    margin = _score_margin_from_probs(prob, pred_idx)

    exp = explainer(sample.reshape(1, -1), check_additivity=False)
    values = np.asarray(exp.values)

    # values shape can vary by SHAP version/model:
    # - (1, n_features) or (n_features,) for single-output
    # - (1, n_features, n_classes) for multiclass
    if values.ndim == 3:
        vec = np.asarray(values[0, :, pred_idx], dtype=float)
    elif values.ndim == 2:
        vec = np.asarray(values[0], dtype=float)
    else:
        vec = np.asarray(values.reshape(-1), dtype=float)

    abs_vec = np.abs(vec)
    top_idx = int(np.argmax(abs_vec)) if abs_vec.size else -1
    top_abs = float(abs_vec[top_idx]) if top_idx >= 0 else np.nan
    l1, l2, nnz = _vector_stats(vec)

    return {
        "y_local_pred": y_local_pred,
        "evidence_score_local_pred": float(prob[pred_idx]),
        "evidence_score_true_class": np.nan,
        "top_competitor_class_pred": str(model.classes_[int(np.argsort(prob)[-2])]) if prob.size > 1 else None,
        "evidence_score_competitor_pred": float(np.sort(prob)[-2]) if prob.size > 1 else np.nan,
        "score_margin_pred_vs_competitor": margin,
        "num_paths": np.nan,
        "num_active_nodes": np.nan,
        "mean_lrc_active_nodes": np.nan,
        "mean_bc_active_nodes": np.nan,
        "class_support_total": np.nan,
        "n_class_support": np.nan,
        "top_feature_idx": top_idx if top_idx >= 0 else np.nan,
        "top_feature_abs_contrib": top_abs,
        "contrib_l1": l1,
        "contrib_l2": l2,
        "contrib_nnz": nnz,
        "ice_feature_idx": np.nan,
        "ice_prob_min": np.nan,
        "ice_prob_max": np.nan,
        "ice_prob_range": np.nan,
        "ice_slope": np.nan,
    }


def _explain_with_lime(
    explainer: LimeTabularExplainer,
    model: RandomForestClassifier,
    sample: np.ndarray,
    num_features: int,
    num_samples: int,
) -> Dict[str, Any]:
    prob = model.predict_proba(sample.reshape(1, -1))[0]
    pred_idx = int(np.argmax(prob))
    pred_label = model.classes_[pred_idx]
    n_classes = len(model.classes_)
    class_names = [str(c) for c in model.classes_]

    exp = explainer.explain_instance(
        data_row=sample,
        predict_fn=model.predict_proba,
        labels=list(range(n_classes)),
        num_features=num_features,
        num_samples=num_samples,
    )

    contrib_vec = np.zeros(sample.shape[0], dtype=float)
    pred_exp_idx = pred_idx
    for feat_idx, w in exp.local_exp.get(pred_exp_idx, []):
        contrib_vec[int(feat_idx)] = float(w)

    approx_scores: Dict[int, float] = {}
    for cls_idx in range(n_classes):
        intercept = float(exp.intercept.get(cls_idx, 0.0))
        score = intercept + sum(float(w) for _, w in exp.local_exp.get(cls_idx, []))
        approx_scores[cls_idx] = score

    local_idx = int(max(approx_scores, key=approx_scores.get)) if approx_scores else pred_idx
    local_label = str(model.classes_[local_idx])
    sorted_scores = sorted(approx_scores.items(), key=lambda kv: kv[1])
    comp_idx = int(sorted_scores[-2][0]) if len(sorted_scores) > 1 else local_idx
    margin = float(approx_scores.get(local_idx, np.nan) - approx_scores.get(comp_idx, np.nan))

    abs_vec = np.abs(contrib_vec)
    top_idx = int(np.argmax(abs_vec)) if abs_vec.size else -1
    top_abs = float(abs_vec[top_idx]) if top_idx >= 0 else np.nan
    l1, l2, nnz = _vector_stats(contrib_vec)

    return {
        "y_local_pred": local_label,
        "evidence_score_local_pred": float(approx_scores.get(local_idx, np.nan)),
        "evidence_score_true_class": np.nan,
        "top_competitor_class_pred": class_names[comp_idx] if comp_idx < len(class_names) else None,
        "evidence_score_competitor_pred": float(approx_scores.get(comp_idx, np.nan)),
        "score_margin_pred_vs_competitor": margin,
        "num_paths": np.nan,
        "num_active_nodes": np.nan,
        "mean_lrc_active_nodes": np.nan,
        "mean_bc_active_nodes": np.nan,
        "class_support_total": np.nan,
        "n_class_support": np.nan,
        "top_feature_idx": top_idx if top_idx >= 0 else np.nan,
        "top_feature_abs_contrib": top_abs,
        "contrib_l1": l1,
        "contrib_l2": l2,
        "contrib_nnz": nnz,
        "ice_feature_idx": np.nan,
        "ice_prob_min": np.nan,
        "ice_prob_max": np.nan,
        "ice_prob_range": np.nan,
        "ice_slope": np.nan,
    }


def _explain_with_ice(
    model: RandomForestClassifier,
    sample: np.ndarray,
    feature_idx: int,
    grid_values: np.ndarray,
) -> Dict[str, Any]:
    tiled = np.repeat(sample.reshape(1, -1), repeats=grid_values.shape[0], axis=0)
    tiled[:, feature_idx] = grid_values
    probs = model.predict_proba(tiled)

    base_prob = model.predict_proba(sample.reshape(1, -1))[0]
    pred_idx = int(np.argmax(base_prob))
    mean_probs = probs.mean(axis=0)
    local_idx = int(np.argmax(mean_probs))
    sorted_idx = np.argsort(mean_probs)
    comp_idx = int(sorted_idx[-2]) if mean_probs.size > 1 else local_idx

    pred_curve = probs[:, pred_idx]
    prob_min = float(np.min(pred_curve))
    prob_max = float(np.max(pred_curve))
    prob_range = prob_max - prob_min
    denom = float(grid_values[-1] - grid_values[0]) if grid_values.size >= 2 else 0.0
    slope = float((pred_curve[-1] - pred_curve[0]) / denom) if abs(denom) > 1e-12 else np.nan

    return {
        "y_local_pred": str(model.classes_[local_idx]),
        "evidence_score_local_pred": float(mean_probs[local_idx]),
        "evidence_score_true_class": np.nan,
        "top_competitor_class_pred": str(model.classes_[comp_idx]),
        "evidence_score_competitor_pred": float(mean_probs[comp_idx]),
        "score_margin_pred_vs_competitor": float(mean_probs[local_idx] - mean_probs[comp_idx]),
        "num_paths": np.nan,
        "num_active_nodes": np.nan,
        "mean_lrc_active_nodes": np.nan,
        "mean_bc_active_nodes": np.nan,
        "class_support_total": np.nan,
        "n_class_support": np.nan,
        "top_feature_idx": feature_idx,
        "top_feature_abs_contrib": prob_range,
        "contrib_l1": np.nan,
        "contrib_l2": np.nan,
        "contrib_nnz": np.nan,
        "ice_feature_idx": feature_idx,
        "ice_prob_min": prob_min,
        "ice_prob_max": prob_max,
        "ice_prob_range": prob_range,
        "ice_slope": slope,
    }


def _run_one_dataset_method(
    bundle: DatasetBundle,
    cfg: ExperimentConfig,
    method: str,
    max_test_samples: int,
    progress_every: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, Dict[str, float | int | str | None]]:
    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=int(args.rf_n_jobs),
    )
    model.fit(bundle.X_train, bundle.y_train)

    y_pred = model.predict(bundle.X_test)
    model_acc = float(accuracy_score(bundle.y_test, y_pred))

    dpg_explainer = None
    shap_explainer = None
    lime_explainer = None
    ice_feature_idx = None
    ice_grid_values = None

    if method == "dpg":
        class_names = [str(c) for c in model.classes_]
        dpg_explainer = DPGExplainer(
            model=model,
            feature_names=bundle.feature_names,
            target_names=class_names,
            dpg_config={
                "dpg": {
                    "default": {
                        "perc_var": cfg.perc_var,
                        "decimal_threshold": cfg.decimal_threshold,
                        "n_jobs": 1,
                    }
                }
            },
        )
        dpg_explainer.fit(bundle.X_train)
    elif method == "shap":
        if shap is None:
            raise RuntimeError("Method 'shap' requested but 'shap' package is not installed.")
        bg_size = min(int(args.shap_background_size), bundle.X_train.shape[0])
        if bg_size <= 0:
            bg_size = min(200, bundle.X_train.shape[0])
        rng = np.random.default_rng(cfg.random_state)
        idx = rng.choice(bundle.X_train.shape[0], size=bg_size, replace=False)
        background = bundle.X_train[idx]
        shap_explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
            model_output="probability",
        )
    elif method == "lime":
        if LimeTabularExplainer is None:
            raise RuntimeError("Method 'lime' requested but 'lime' package is not installed.")
        lime_explainer = LimeTabularExplainer(
            training_data=np.asarray(bundle.X_train, dtype=float),
            feature_names=bundle.feature_names,
            class_names=[str(c) for c in model.classes_],
            mode="classification",
            discretize_continuous=False,
            random_state=cfg.random_state,
        )
    elif method == "ice":
        # ICE is a response-curve diagnostic. We vary the globally most important RF feature.
        ice_feature_idx = int(np.argmax(model.feature_importances_))
        q_low, q_high = np.quantile(bundle.X_train[:, ice_feature_idx], [0.05, 0.95])
        if abs(float(q_high - q_low)) <= 1e-12:
            q_low, q_high = np.min(bundle.X_train[:, ice_feature_idx]), np.max(
                bundle.X_train[:, ice_feature_idx]
            )
        if abs(float(q_high - q_low)) <= 1e-12:
            q_low, q_high = float(q_low) - 1.0, float(q_high) + 1.0
        ice_grid_values = np.linspace(q_low, q_high, num=int(args.ice_grid_points))
    elif method == "tree_path":
        pass
    elif method in {"raw_path_union", "path_bag", "random_same_size_path"}:
        pass
    elif method == "anchors":
        pass
    elif method == "lore":
        pass
    else:
        raise ValueError(f"Unsupported method: {method}")

    n_test = bundle.X_test.shape[0]
    n_eval = min(max_test_samples, n_test) if max_test_samples > 0 else n_test
    sample_rows: List[Dict[str, float | int | str | bool | None]] = []
    local_failures = 0

    for idx in range(n_eval):
        if progress_every > 0 and (idx % progress_every == 0 or idx == n_eval - 1):
            print(
                f"  baseline progress: {idx + 1}/{n_eval} "
                f"({bundle.name} | {_config_id(cfg)} | {method})",
                flush=True,
            )

        sample = bundle.X_test[idx]
        y_true = str(bundle.y_test[idx])
        y_hat = str(y_pred[idx])
        local_status = "ok"
        local_error = ""

        t0 = time.perf_counter()
        try:
            if method == "dpg":
                out = _explain_with_dpg(
                    explainer=dpg_explainer,
                    sample=sample,
                    sample_idx=idx,
                    y_true=y_true,
                    y_hat=y_hat,
                )
            elif method == "shap":
                out = _explain_with_shap(
                    explainer=shap_explainer,
                    model=model,
                    sample=sample,
                )
            elif method == "lime":
                out = _explain_with_lime(
                    explainer=lime_explainer,
                    model=model,
                    sample=sample,
                    num_features=min(int(args.lime_num_features), sample.shape[0]),
                    num_samples=int(args.lime_num_samples),
                )
            elif method == "ice":
                out = _explain_with_ice(
                    model=model,
                    sample=sample,
                    feature_idx=int(ice_feature_idx),
                    grid_values=np.asarray(ice_grid_values, dtype=float),
                )
            elif method == "tree_path":
                out = _explain_with_tree_path(
                    model=model,
                    sample=sample,
                    feature_names=bundle.feature_names,
                )
            elif method in {"raw_path_union", "path_bag", "random_same_size_path"}:
                out = _path_control_output(
                    model=model,
                    sample=sample,
                    feature_names=bundle.feature_names,
                    method=method,
                    random_state=int(cfg.random_state + idx),
                )
            elif method == "anchors":
                out = _explain_with_anchors(
                    model=model,
                    bundle=bundle,
                    sample=sample,
                    precision_target=float(args.anchor_precision_target),
                    max_rule_len=int(args.anchor_max_rule_len),
                    max_candidates=int(args.anchor_max_candidates),
                )
            elif method == "lore":
                out = _explain_with_lore(
                    model=model,
                    bundle=bundle,
                    sample=sample,
                    random_state=int(cfg.random_state + idx),
                    neighborhood_size=int(args.lore_neighborhood_size),
                    max_depth=int(args.lore_max_depth),
                    min_samples_leaf=int(args.lore_min_samples_leaf),
                    neighborhood_scale=float(args.lore_neighborhood_scale),
                )
            else:
                raise ValueError(method)
        except Exception as exc:
            local_failures += 1
            local_status = "failed"
            local_error = f"{type(exc).__name__}: {exc}"
            out = {
                "y_local_pred": None,
                "evidence_score_local_pred": np.nan,
                "evidence_score_true_class": np.nan,
                "top_competitor_class_pred": None,
                "evidence_score_competitor_pred": np.nan,
                "score_margin_pred_vs_competitor": np.nan,
                "num_paths": np.nan,
                "num_active_nodes": np.nan,
                "mean_lrc_active_nodes": np.nan,
                "mean_bc_active_nodes": np.nan,
                "class_support_total": np.nan,
                "n_class_support": np.nan,
                "top_feature_idx": np.nan,
                "top_feature_abs_contrib": np.nan,
                "contrib_l1": np.nan,
                "contrib_l2": np.nan,
                "contrib_nnz": np.nan,
                "ice_feature_idx": np.nan,
                "ice_prob_min": np.nan,
                "ice_prob_max": np.nan,
                "ice_prob_range": np.nan,
                "ice_slope": np.nan,
                "anchor_rule": None,
                "anchor_precision": np.nan,
                "anchor_coverage": np.nan,
                "lore_rule": None,
                "lore_rule_length": np.nan,
                "lore_fidelity_neighborhood": np.nan,
                "lore_counterfactual_distance": np.nan,
                "lore_counterfactual_features": "",
                "path_control_type": "",
                "path_control_predicate_count": np.nan,
                "path_control_unique_predicate_count": np.nan,
                "path_control_unique_feature_count": np.nan,
                "path_control_mean_path_length": np.nan,
            }
        runtime_ms = float((time.perf_counter() - t0) * 1000.0)

        local_pred = out["y_local_pred"]
        sample_rows.append(
            {
                "dataset": bundle.name,
                "train_split": bundle.train_split,
                "eval_split": bundle.eval_split,
                "split_registry_path": bundle.split_registry_path,
                "method": method,
                "config_id": _config_id(cfg),
                "seed": cfg.random_state,
                "sample_idx": idx,
                "y_true": y_true,
                "y_model_pred": y_hat,
                "y_local_pred": local_pred,
                "local_status": local_status,
                "local_error": local_error,
                "runtime_ms": runtime_ms,
                "model_correct": y_hat == y_true,
                "local_matches_model": local_pred == y_hat if local_pred is not None else False,
                "local_correct": local_pred == y_true if local_pred is not None else False,
                **out,
            }
        )

    df = pd.DataFrame(sample_rows)

    def _mean_or_nan(column: str) -> float:
        if not len(df) or column not in df.columns:
            return float(np.nan)
        return float(df[column].mean())

    summary: Dict[str, float | int | str | None] = {
        "dataset": bundle.name,
        "train_split": bundle.train_split,
        "eval_split": bundle.eval_split,
        "split_registry_path": bundle.split_registry_path,
        "method": method,
        "config_id": _config_id(cfg),
        "seed": cfg.random_state,
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "perc_var": cfg.perc_var,
        "decimal_threshold": cfg.decimal_threshold,
        "n_test_total": int(n_test),
        "n_test_evaluated": int(n_eval),
        "model_accuracy": model_acc,
        "local_matches_model_rate": _mean_or_nan("local_matches_model"),
        "local_accuracy": _mean_or_nan("local_correct"),
        "avg_runtime_ms": _mean_or_nan("runtime_ms"),
        "avg_score_margin_pred_vs_competitor": _mean_or_nan("score_margin_pred_vs_competitor"),
        "avg_top_feature_abs_contrib": _mean_or_nan("top_feature_abs_contrib"),
        "avg_contrib_l1": _mean_or_nan("contrib_l1"),
        "avg_contrib_l2": _mean_or_nan("contrib_l2"),
        "avg_num_paths": _mean_or_nan("num_paths"),
        "avg_num_active_nodes": _mean_or_nan("num_active_nodes"),
        "avg_evidence_score_local_pred": _mean_or_nan("evidence_score_local_pred"),
        "avg_ice_prob_range": _mean_or_nan("ice_prob_range"),
        "avg_ice_slope": _mean_or_nan("ice_slope"),
        "avg_anchor_precision": _mean_or_nan("anchor_precision"),
        "avg_anchor_coverage": _mean_or_nan("anchor_coverage"),
        "avg_lore_rule_length": _mean_or_nan("lore_rule_length"),
        "avg_lore_fidelity_neighborhood": _mean_or_nan("lore_fidelity_neighborhood"),
        "avg_lore_counterfactual_distance": _mean_or_nan("lore_counterfactual_distance"),
        "avg_path_control_predicate_count": _mean_or_nan("path_control_predicate_count"),
        "avg_path_control_unique_predicate_count": _mean_or_nan("path_control_unique_predicate_count"),
        "avg_path_control_unique_feature_count": _mean_or_nan("path_control_unique_feature_count"),
        "avg_path_control_mean_path_length": _mean_or_nan("path_control_mean_path_length"),
        "local_failures": int(local_failures),
        "local_failure_rate": float(local_failures / n_eval) if n_eval > 0 else np.nan,
    }
    return df, summary


def _write_report(
    out_dir: Path,
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary_baselines.csv", index=False)
    detail_df.to_csv(out_dir / "per_sample_baselines.csv", index=False)

    grouped = pd.DataFrame()
    if len(summary_df):
        grouped = (
            summary_df.groupby(["dataset", "method"], as_index=False)
            .agg(
                runs=("config_id", "count"),
                best_model_accuracy=("model_accuracy", "max"),
                best_local_match_rate=("local_matches_model_rate", "max"),
                best_local_accuracy=("local_accuracy", "max"),
                min_avg_runtime_ms=("avg_runtime_ms", "min"),
                max_avg_runtime_ms=("avg_runtime_ms", "max"),
            )
            .sort_values(["dataset", "method"])
        )
    grouped.to_csv(out_dir / "dataset_overview_baselines.csv", index=False)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run DPG/SHAP/LIME/ICE/LORE-style baseline experiments over prepared numeric datasets."
    )
    parser.add_argument("--data_dir", type=str, default="experiments_local_explanation/data_numeric")
    parser.add_argument("--out_dir", type=str, default="experiments_local_explanation/results_baselines")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names. Empty means all subdirs in data_dir.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="dpg,shap,lime,ice",
        help=(
            "Comma-separated list of methods in "
            "{dpg,shap,lime,ice,anchors,tree_path,lore,"
            "raw_path_union,path_bag,random_same_size_path}."
        ),
    )
    parser.add_argument("--n_estimators", type=str, default="10,20")
    parser.add_argument(
        "--rf_n_jobs",
        type=int,
        default=-1,
        help="RandomForest n_jobs per process. Use 1 when running many dataset workers in parallel.",
    )
    parser.add_argument("--max_depth", type=str, default="2,4,None")
    parser.add_argument("--perc_var", type=str, default="0.0,0.001")
    parser.add_argument("--decimal_threshold", type=str, default="4,6")
    parser.add_argument("--seeds", type=str, default="27,42")
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument(
        "--split_registry",
        type=str,
        default="",
        help="Optional split_registry.json for journal train/validation/test protocol.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="existing_train",
        help="Training split: existing_train, train_core, or train_core_validation.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="existing_test",
        help="Evaluation split: existing_test/test or validation.",
    )
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from per-run checkpoints in out_dir/checkpoints_baselines (default: enabled).",
    )
    parser.add_argument("--overwrite", action="store_true")

    # SHAP/LIME/ICE method controls
    parser.add_argument("--shap_background_size", type=int, default=200)
    parser.add_argument("--lime_num_features", type=int, default=20)
    parser.add_argument("--lime_num_samples", type=int, default=1000)
    parser.add_argument("--ice_grid_points", type=int, default=21)
    parser.add_argument("--anchor_precision_target", type=float, default=0.95)
    parser.add_argument("--anchor_max_rule_len", type=int, default=4)
    parser.add_argument("--anchor_max_candidates", type=int, default=12)
    parser.add_argument("--lore_neighborhood_size", type=int, default=512)
    parser.add_argument("--lore_max_depth", type=int, default=4)
    parser.add_argument("--lore_min_samples_leaf", type=int, default=10)
    parser.add_argument("--lore_neighborhood_scale", type=float, default=0.5)

    args = parser.parse_args()
    args.data_dir = _resolve_cli_path(args.data_dir, script_dir=script_dir)
    args.out_dir = _resolve_cli_path(args.out_dir, script_dir=script_dir)
    args.split_registry = (
        _resolve_cli_path(args.split_registry, script_dir=script_dir) if args.split_registry else None
    )
    return args


def main() -> None:
    args = parse_args()
    only = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = _parse_method_list(args.methods)
    configs = _make_configs(args)
    dataset_paths = _dataset_dirs(args.data_dir, only=only)
    split_registry = load_split_registry(args.split_registry)
    if not dataset_paths:
        raise ValueError(f"No dataset directories found in: {args.data_dir}")

    if "shap" in methods and shap is None:
        raise RuntimeError("Method 'shap' requested but package 'shap' is not installed.")
    if "lime" in methods and LimeTabularExplainer is None:
        raise RuntimeError("Method 'lime' requested but package 'lime' is not installed.")

    already_done = set()
    if args.resume and not args.overwrite:
        already_done = _completed_run_keys(args.out_dir)
        print(f"Found {len(already_done)} completed checkpoints in {_checkpoint_dir(args.out_dir)}")
    elif args.overwrite:
        already_done = set()
        print("Overwrite requested: ignoring existing baseline checkpoints.")
    else:
        print("Resume disabled: recomputing all baseline runs for this launch.")

    summary_rows: List[Dict[str, float | int | str | None]] = []
    detail_frames: List[pd.DataFrame] = []

    total_runs = len(dataset_paths) * len(configs) * len(methods)
    done = 0

    for ds_path in dataset_paths:
        bundle = apply_split_registry(
            bundle=_load_bundle(ds_path),
            registry=split_registry,
            registry_path=args.split_registry,
            train_split=args.train_split,
            eval_split=args.eval_split,
        )
        print(f"\n=== DATASET: {bundle.name} ===", flush=True)
        for cfg in configs:
            for method in methods:
                run_id = _run_identity(bundle.name, cfg, method, bundle.train_split, bundle.eval_split)
                done += 1
                print(f"[{done}/{total_runs}] {run_id.key}", flush=True)

                if run_id.key in already_done:
                    print("  -> checkpoint exists; skipping", flush=True)
                    continue

                detail_df, summary = _run_one_dataset_method(
                    bundle=bundle,
                    cfg=cfg,
                    method=method,
                    max_test_samples=args.max_test_samples,
                    progress_every=args.progress_every,
                    args=args,
                )
                _save_run_checkpoint(args.out_dir, run_id, summary, detail_df)
                summary_rows.append(summary)
                detail_frames.append(detail_df)

    sum_ckpt_df, det_ckpt_df = _read_checkpoints(args.out_dir)
    if summary_rows:
        sum_new_df = pd.DataFrame(summary_rows)
        sum_df = pd.concat([sum_ckpt_df, sum_new_df], ignore_index=True)
    else:
        sum_df = sum_ckpt_df

    if detail_frames:
        det_new_df = pd.concat(detail_frames, ignore_index=True)
        det_df = pd.concat([det_ckpt_df, det_new_df], ignore_index=True)
    else:
        det_df = det_ckpt_df

    if len(sum_df):
        sum_unique_cols = [
            c
            for c in ["dataset", "method", "train_split", "eval_split", "config_id", "seed"]
            if c in sum_df.columns
        ]
        sum_df = sum_df.drop_duplicates(
            subset=sum_unique_cols, keep="last"
        ).sort_values(["dataset", "method", "config_id"])
    if len(det_df):
        det_unique_cols = [
            c
            for c in ["dataset", "method", "train_split", "eval_split", "config_id", "seed", "sample_idx"]
            if c in det_df.columns
        ]
        det_df = det_df.drop_duplicates(
            subset=det_unique_cols, keep="last"
        ).sort_values(["dataset", "method", "config_id", "sample_idx"])

    _write_report(args.out_dir, summary_df=sum_df.reset_index(drop=True), detail_df=det_df.reset_index(drop=True))
    print(f"\nSaved: {args.out_dir / 'summary_baselines.csv'}")
    print(f"Saved: {args.out_dir / 'per_sample_baselines.csv'}")
    print(f"Saved: {args.out_dir / 'dataset_overview_baselines.csv'}")


if __name__ == "__main__":
    main()
