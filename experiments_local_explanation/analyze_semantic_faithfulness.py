#!/usr/bin/env python3
"""Evaluate semantic faithfulness for best DPG and baseline explanations.

Outputs:
    <out_dir>/semantic_faithfulness_per_sample.csv
    <out_dir>/semantic_faithfulness_summary.csv
    <out_dir>/critical_branch_flip_summary.csv
    <out_dir>/critical_case_reports.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpg import DPGExplainer
from dpg.visualizer import parse_predicate_parts
from experiments_local_explanation.run_baseline_experiments import (
    LimeTabularExplainer,
    _best_anchor_rule,
    _config_id as _baseline_config_id,
    _explain_with_ice,
    _score_margin_from_probs,
    _tree_path_nodes_and_predicates,
    _tree_node_prob_vector,
    shap,
)


EXCLUDED_DATASETS = {"fashion_mnist_784", "mnist_784"}


def _load_bundle(dataset_dir: Path) -> Dict[str, Any]:
    with open(dataset_dir / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return {
        "name": dataset_dir.name,
        "X_train": np.load(dataset_dir / "X_train.npy"),
        "y_train": np.load(dataset_dir / "y_train.npy"),
        "X_test": np.load(dataset_dir / "X_test.npy"),
        "y_test": np.load(dataset_dir / "y_test.npy"),
        "feature_names": list(feature_names),
    }


def _fit_model(cfg_row: pd.Series, X_train: np.ndarray, y_train: np.ndarray, rf_n_jobs: int) -> RandomForestClassifier:
    max_depth = cfg_row.get("max_depth")
    if pd.isna(max_depth):
        max_depth = None
    else:
        max_depth = int(max_depth)
    model = RandomForestClassifier(
        n_estimators=int(cfg_row["n_estimators"]),
        max_depth=max_depth,
        random_state=int(cfg_row["seed"]),
        n_jobs=int(rf_n_jobs),
    )
    model.fit(X_train, y_train)
    return model


def _build_dpg_explainer(
    cfg_row: pd.Series,
    bundle: Dict[str, Any],
    rf_n_jobs: int,
) -> tuple[RandomForestClassifier, DPGExplainer]:
    model = _fit_model(cfg_row, bundle["X_train"], bundle["y_train"], rf_n_jobs=rf_n_jobs)
    class_names = [str(c) for c in model.classes_]
    explainer = DPGExplainer(
        model=model,
        feature_names=bundle["feature_names"],
        target_names=class_names,
        dpg_config={
            "dpg": {
                "default": {
                    "perc_var": float(cfg_row["perc_var"]),
                    "decimal_threshold": int(cfg_row["decimal_threshold"]),
                    "n_jobs": 1,
                },
                "graph_construction": {
                    "mode": str(cfg_row["graph_construction_mode"]),
                },
                "local_evidence": {
                    "variant": "top_competitor",
                    "base_lambda": 0.8,
                },
            }
        },
    )
    explainer.fit(bundle["X_train"])
    return model, explainer


def _build_shap_explainer(model: RandomForestClassifier, X_train: np.ndarray, seed: int, bg_size: int) -> Any:
    if shap is None:
        raise RuntimeError("shap is not installed in the active environment.")
    size = min(int(bg_size), X_train.shape[0])
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(X_train.shape[0], size=size, replace=False)
    return shap.TreeExplainer(
        model,
        data=X_train[idx],
        feature_perturbation="interventional",
        model_output="probability",
    )


def _build_lime_explainer(model: RandomForestClassifier, bundle: Dict[str, Any], seed: int) -> LimeTabularExplainer:
    if LimeTabularExplainer is None:
        raise RuntimeError("lime is not installed in the active environment.")
    return LimeTabularExplainer(
        training_data=np.asarray(bundle["X_train"], dtype=float),
        feature_names=bundle["feature_names"],
        class_names=[str(c) for c in model.classes_],
        mode="classification",
        discretize_continuous=False,
        random_state=int(seed),
    )


def _feature_name_to_index(feature_names: Sequence[str]) -> Dict[str, int]:
    return {str(name): idx for idx, name in enumerate(feature_names)}


def _leaf_class(path: Any) -> Optional[str]:
    if not path.labels:
        return None
    label = str(path.labels[-1])
    if label.startswith("Class "):
        return label.replace("Class ", "", 1)
    return None


def _top_dpg_feature_indices(local: Any, feature_names: Sequence[str]) -> List[int]:
    support_pred_class = local.sample_confidence.get("support_pred_class") or local.majority_vote
    pred_paths = [p for p in local.tree_paths if _leaf_class(p) == str(support_pred_class)]
    if not pred_paths:
        pred_paths = list(local.tree_paths)
    if not pred_paths:
        return []
    top_path = max(pred_paths, key=lambda p: float(p.path_confidence or 0.0))
    feature_index = _feature_name_to_index(feature_names)
    selected: List[int] = []
    seen: set[int] = set()
    for label in top_path.labels[:-1]:
        parsed = parse_predicate_parts(str(label))
        if not parsed:
            continue
        feat_name = str(parsed[0])
        idx = feature_index.get(feat_name)
        if idx is None or idx in seen:
            continue
        selected.append(int(idx))
        seen.add(int(idx))
    return selected


def _shap_feature_ranking(explainer: Any, model: RandomForestClassifier, sample: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(sample.reshape(1, -1))[0]
    pred_idx = int(np.argmax(prob))
    exp = explainer(sample.reshape(1, -1), check_additivity=False)
    values = np.asarray(exp.values)
    if values.ndim == 3:
        vec = np.asarray(values[0, :, pred_idx], dtype=float)
    elif values.ndim == 2:
        vec = np.asarray(values[0], dtype=float)
    else:
        vec = np.asarray(values.reshape(-1), dtype=float)
    return np.argsort(np.abs(vec))[::-1]


def _lime_feature_ranking(
    explainer: LimeTabularExplainer,
    model: RandomForestClassifier,
    sample: np.ndarray,
    num_features: int,
    num_samples: int,
) -> np.ndarray:
    prob = model.predict_proba(sample.reshape(1, -1))[0]
    pred_idx = int(np.argmax(prob))
    exp = explainer.explain_instance(
        data_row=sample,
        predict_fn=model.predict_proba,
        labels=list(range(len(model.classes_))),
        num_features=min(int(num_features), sample.shape[0]),
        num_samples=int(num_samples),
    )
    contrib_vec = np.zeros(sample.shape[0], dtype=float)
    for feat_idx, weight in exp.local_exp.get(pred_idx, []):
        contrib_vec[int(feat_idx)] = float(weight)
    return np.argsort(np.abs(contrib_vec))[::-1]


def _tree_path_feature_ranking(model: RandomForestClassifier, sample: np.ndarray, n_features: int) -> np.ndarray:
    prob = model.predict_proba(sample.reshape(1, -1))[0]
    pred_idx = int(np.argmax(prob))
    contrib_vec = np.zeros(n_features, dtype=float)
    for estimator in model.estimators_:
        path_nodes, _predicates = _tree_path_nodes_and_predicates(estimator, sample, [str(i) for i in range(n_features)])
        for parent, child in zip(path_nodes, path_nodes[1:]):
            feat_idx = int(estimator.tree_.feature[parent])
            if feat_idx < 0:
                continue
            delta = _tree_node_prob_vector(estimator, child)[pred_idx] - _tree_node_prob_vector(estimator, parent)[pred_idx]
            contrib_vec[feat_idx] += float(delta) / max(len(model.estimators_), 1)
    return np.argsort(np.abs(contrib_vec))[::-1]


def _anchor_feature_indices(
    model: RandomForestClassifier,
    bundle: Dict[str, Any],
    sample: np.ndarray,
    precision_target: float,
    max_rule_len: int,
    max_candidates: int,
) -> List[int]:
    target_label = str(model.predict(sample.reshape(1, -1))[0])
    rule, _precision, _coverage, _competitor, _comp_score = _best_anchor_rule(
        model=model,
        X_reference=bundle["X_train"],
        sample=sample,
        feature_names=bundle["feature_names"],
        target_label=target_label,
        precision_target=float(precision_target),
        max_rule_len=int(max_rule_len),
        max_candidates=int(max_candidates),
    )
    seen: set[int] = set()
    selected: List[int] = []
    for feat_idx, _feat_name, _op, _threshold in rule:
        if int(feat_idx) not in seen:
            selected.append(int(feat_idx))
            seen.add(int(feat_idx))
    return selected


def _ice_feature_index(model: RandomForestClassifier) -> int:
    return int(np.argmax(model.feature_importances_))


def _target_probability(model: RandomForestClassifier, sample: np.ndarray, target_label: str) -> float:
    prob = model.predict_proba(sample.reshape(1, -1))[0]
    classes = [str(c) for c in model.classes_]
    if str(target_label) not in classes:
        return float("nan")
    idx = classes.index(str(target_label))
    return float(prob[idx])


def _reference_vector(X_train: np.ndarray) -> np.ndarray:
    return np.nanmedian(np.asarray(X_train, dtype=float), axis=0)


def _keep_only_features(sample: np.ndarray, keep_indices: Sequence[int], reference: np.ndarray) -> np.ndarray:
    out = np.asarray(reference, dtype=float).copy()
    for idx in keep_indices:
        out[int(idx)] = float(sample[int(idx)])
    return out


def _remove_features(sample: np.ndarray, remove_indices: Sequence[int], reference: np.ndarray) -> np.ndarray:
    out = np.asarray(sample, dtype=float).copy()
    for idx in remove_indices:
        out[int(idx)] = float(reference[int(idx)])
    return out


def _sufficiency_comprehensiveness(
    model: RandomForestClassifier,
    sample: np.ndarray,
    target_label: str,
    feature_indices: Sequence[int],
    reference: np.ndarray,
) -> Dict[str, float]:
    if not feature_indices:
        return {
            "target_prob": float("nan"),
            "sufficiency_prob": float("nan"),
            "sufficiency_drop": float("nan"),
            "comprehensiveness_prob": float("nan"),
            "comprehensiveness_drop": float("nan"),
        }
    target_prob = _target_probability(model, sample, target_label)
    keep_sample = _keep_only_features(sample, feature_indices, reference)
    remove_sample = _remove_features(sample, feature_indices, reference)
    suff_prob = _target_probability(model, keep_sample, target_label)
    comp_prob = _target_probability(model, remove_sample, target_label)
    return {
        "target_prob": float(target_prob),
        "sufficiency_prob": float(suff_prob),
        "sufficiency_drop": float(target_prob - suff_prob),
        "comprehensiveness_prob": float(comp_prob),
        "comprehensiveness_drop": float(target_prob - comp_prob),
    }


def _apply_predicate(sample: np.ndarray, predicate_label: Optional[str], feature_names: Sequence[str]) -> Optional[np.ndarray]:
    if not predicate_label:
        return None
    parsed = parse_predicate_parts(str(predicate_label))
    if not parsed:
        return None
    feat_name, op, threshold = parsed
    feature_index = _feature_name_to_index(feature_names)
    feat_idx = feature_index.get(str(feat_name))
    if feat_idx is None:
        return None
    out = np.asarray(sample, dtype=float).copy()
    eps = 1e-6
    if str(op) == "<=":
        out[int(feat_idx)] = min(float(out[int(feat_idx)]), float(threshold) - eps)
    else:
        out[int(feat_idx)] = max(float(out[int(feat_idx)]), float(threshold) + eps)
    return out


def _critical_branch_flip_metrics(
    model: RandomForestClassifier,
    sample: np.ndarray,
    local: Any,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    target_label = local.sample_confidence.get("support_pred_class") or local.majority_vote
    competitor = local.sample_confidence.get("support_top_competitor_class")
    pred_successor = local.sample_confidence.get("critical_successor_pred")
    comp_successor = local.sample_confidence.get("critical_successor_comp")
    pred_branch_sample = _apply_predicate(sample, pred_successor, feature_names)
    comp_branch_sample = _apply_predicate(sample, comp_successor, feature_names)
    base_prob = _target_probability(model, sample, str(target_label)) if target_label is not None else float("nan")

    pred_branch_prob = float("nan")
    pred_branch_label = None
    if pred_branch_sample is not None and target_label is not None:
        pred_branch_prob = _target_probability(model, pred_branch_sample, str(target_label))
        pred_branch_label = str(model.predict(pred_branch_sample.reshape(1, -1))[0])

    comp_branch_prob = float("nan")
    comp_branch_label = None
    comp_branch_competitor_prob = float("nan")
    if comp_branch_sample is not None and target_label is not None:
        comp_branch_prob = _target_probability(model, comp_branch_sample, str(target_label))
        comp_branch_label = str(model.predict(comp_branch_sample.reshape(1, -1))[0])
        if competitor is not None:
            comp_branch_competitor_prob = _target_probability(model, comp_branch_sample, str(competitor))

    return {
        "critical_target_prob": float(base_prob),
        "critical_pred_branch_prob": float(pred_branch_prob),
        "critical_comp_branch_prob": float(comp_branch_prob),
        "critical_comp_branch_competitor_prob": float(comp_branch_competitor_prob),
        "critical_branch_margin_delta": float(pred_branch_prob - comp_branch_prob)
        if pred_branch_prob == pred_branch_prob and comp_branch_prob == comp_branch_prob
        else float("nan"),
        "critical_comp_branch_label": comp_branch_label,
        "critical_pred_branch_label": pred_branch_label,
        "critical_flip_to_competitor": bool(comp_branch_label == str(competitor))
        if comp_branch_label is not None and competitor is not None
        else False,
    }


def _select_sample_indices(n_total: int, max_samples: int, must_include: Iterable[int]) -> List[int]:
    required = {int(i) for i in must_include if 0 <= int(i) < n_total}
    if max_samples <= 0 or max_samples >= n_total:
        return sorted(required | set(range(n_total)))
    rng = np.random.default_rng(42)
    remaining = [i for i in range(n_total) if i not in required]
    take = max(int(max_samples) - len(required), 0)
    chosen = set(required)
    if take > 0 and remaining:
        sampled = rng.choice(np.asarray(remaining), size=min(take, len(remaining)), replace=False)
        chosen.update(int(i) for i in np.asarray(sampled).tolist())
    return sorted(chosen)


def _best_baseline_configs(baseline_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for dataset_dir in sorted(p for p in baseline_root.iterdir() if p.is_dir()):
        path = dataset_dir / "summary_baselines.csv"
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                df["dataset"] = dataset_dir.name
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return (
        df.sort_values(
            ["dataset", "method", "local_matches_model_rate", "local_accuracy", "avg_score_margin_pred_vs_competitor"],
            ascending=[True, True, False, False, False],
        )
        .drop_duplicates(subset=["dataset", "method"], keep="first")
        .reset_index(drop=True)
    )


def _load_best_dpg(next_phase_analysis_dir: Path) -> pd.DataFrame:
    return pd.read_csv(next_phase_analysis_dir / "best_configs.csv", low_memory=False)


def _evaluate_dataset(
    dataset: str,
    bundle: Dict[str, Any],
    best_dpg: pd.DataFrame,
    best_baselines: pd.DataFrame,
    max_samples_per_dataset: int,
    rf_n_jobs: int,
    shap_background_size: int,
    lime_num_features: int,
    lime_num_samples: int,
    anchor_precision_target: float,
    anchor_max_rule_len: int,
    anchor_max_candidates: int,
    include_cases: Sequence[int],
) -> pd.DataFrame:
    dataset_rows: List[Dict[str, Any]] = []
    reference = _reference_vector(bundle["X_train"])
    sample_indices = _select_sample_indices(bundle["X_test"].shape[0], max_samples_per_dataset, include_cases)

    dpg_rows = best_dpg[best_dpg["dataset"] == dataset]
    if dpg_rows.empty:
        return pd.DataFrame()

    dpg_models: Dict[str, Any] = {}
    for _, row in dpg_rows.iterrows():
        model, explainer = _build_dpg_explainer(row, bundle, rf_n_jobs=rf_n_jobs)
        dpg_models[str(row["graph_construction_mode"])] = (row, model, explainer)

    baseline_rows = best_baselines[best_baselines["dataset"] == dataset]
    baseline_models: Dict[str, Any] = {}
    for _, row in baseline_rows.iterrows():
        model = _fit_model(row, bundle["X_train"], bundle["y_train"], rf_n_jobs=rf_n_jobs)
        method = str(row["method"])
        if method == "shap":
            aux = _build_shap_explainer(model, bundle["X_train"], seed=int(row["seed"]), bg_size=shap_background_size)
        elif method == "lime":
            aux = _build_lime_explainer(model, bundle, seed=int(row["seed"]))
        else:
            aux = None
        baseline_models[method] = (row, model, aux)

    for sample_idx in sample_indices:
        sample = np.asarray(bundle["X_test"][sample_idx], dtype=float)
        y_true = str(bundle["y_test"][sample_idx])

        exec_feature_budget = 0
        for mode in ["execution_trace", "aggregated_transitions"]:
            if mode not in dpg_models:
                continue
            cfg_row, model, explainer = dpg_models[mode]
            local = explainer.explain_local(sample=sample, sample_id=int(sample_idx), validate_graph=True)
            target_label = str(local.sample_confidence.get("support_pred_class") or local.majority_vote)
            feature_indices = _top_dpg_feature_indices(local, bundle["feature_names"])
            if mode == "execution_trace":
                exec_feature_budget = max(len(feature_indices), 1)
            pert = _sufficiency_comprehensiveness(model, sample, target_label, feature_indices, reference)
            row = {
                "dataset": dataset,
                "method_family": "dpg",
                "method": str(mode),
                "sample_idx": int(sample_idx),
                "y_true": y_true,
                "y_model_pred": str(model.predict(sample.reshape(1, -1))[0]),
                "y_explained_class": target_label,
                "feature_count": int(len(feature_indices)),
                "feature_indices": ",".join(str(i) for i in feature_indices),
                **pert,
                "critical_node_label": local.sample_confidence.get("critical_node_label"),
                "critical_split_depth": local.sample_confidence.get("critical_split_depth"),
                "critical_node_contrast": local.sample_confidence.get("critical_node_contrast"),
                "support_margin": local.sample_confidence.get("support_margin"),
                "competitor_exposure": local.sample_confidence.get("competitor_exposure"),
            }
            row.update(_critical_branch_flip_metrics(model, sample, local, bundle["feature_names"]))
            dataset_rows.append(row)

        budget = max(exec_feature_budget, 1)
        for method, payload in baseline_models.items():
            _cfg_row, model, aux = payload
            y_model_pred = str(model.predict(sample.reshape(1, -1))[0])
            if method == "shap":
                ranked = _shap_feature_ranking(aux, model, sample)
                feature_indices = [int(i) for i in ranked[:budget]]
            elif method == "lime":
                ranked = _lime_feature_ranking(aux, model, sample, lime_num_features, lime_num_samples)
                feature_indices = [int(i) for i in ranked[:budget]]
            elif method == "tree_path":
                ranked = _tree_path_feature_ranking(model, sample, sample.shape[0])
                feature_indices = [int(i) for i in ranked[:budget]]
            elif method == "anchors":
                feature_indices = _anchor_feature_indices(
                    model=model,
                    bundle=bundle,
                    sample=sample,
                    precision_target=anchor_precision_target,
                    max_rule_len=anchor_max_rule_len,
                    max_candidates=anchor_max_candidates,
                )
            elif method == "ice":
                feature_indices = [int(_ice_feature_index(model))]
            else:
                continue
            pert = _sufficiency_comprehensiveness(model, sample, y_model_pred, feature_indices, reference)
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "method_family": "baseline",
                    "method": method,
                    "sample_idx": int(sample_idx),
                    "y_true": y_true,
                    "y_model_pred": y_model_pred,
                    "y_explained_class": y_model_pred,
                    "feature_count": int(len(feature_indices)),
                    "feature_indices": ",".join(str(i) for i in feature_indices),
                    **pert,
                    "critical_node_label": None,
                    "critical_split_depth": np.nan,
                    "critical_node_contrast": np.nan,
                    "support_margin": np.nan,
                    "competitor_exposure": np.nan,
                    "critical_target_prob": np.nan,
                    "critical_pred_branch_prob": np.nan,
                    "critical_comp_branch_prob": np.nan,
                    "critical_comp_branch_competitor_prob": np.nan,
                    "critical_branch_margin_delta": np.nan,
                    "critical_comp_branch_label": None,
                    "critical_pred_branch_label": None,
                    "critical_flip_to_competitor": False,
                }
            )

    return pd.DataFrame(dataset_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic faithfulness for best DPG and baseline runs.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("experiments_local_explanation/data_numeric"),
    )
    parser.add_argument(
        "--next_phase_analysis_dir",
        type=Path,
        default=Path("experiments_local_explanation/experiment_dpg2_next_phase/_analysis"),
    )
    parser.add_argument(
        "--baseline_root",
        type=Path,
        default=Path("experiments_local_explanation/results_baselines_by_dataset"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/experiment_dpg2_next_phase/_analysis/semantic_faithfulness"),
    )
    parser.add_argument("--max_samples_per_dataset", type=int, default=300)
    parser.add_argument("--rf_n_jobs", type=int, default=1)
    parser.add_argument("--shap_background_size", type=int, default=200)
    parser.add_argument("--lime_num_features", type=int, default=20)
    parser.add_argument("--lime_num_samples", type=int, default=1000)
    parser.add_argument("--anchor_precision_target", type=float, default=0.95)
    parser.add_argument("--anchor_max_rule_len", type=int, default=4)
    parser.add_argument("--anchor_max_candidates", type=int, default=12)
    args = parser.parse_args()

    data_dir = (REPO_ROOT / args.data_dir).resolve()
    next_phase_analysis_dir = (REPO_ROOT / args.next_phase_analysis_dir).resolve()
    baseline_root = (REPO_ROOT / args.baseline_root).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_dpg = _load_best_dpg(next_phase_analysis_dir)
    best_baselines = _best_baseline_configs(baseline_root)
    datasets = sorted(
        ds for ds in best_dpg["dataset"].dropna().astype(str).unique().tolist() if ds not in EXCLUDED_DATASETS
    )

    must_include_by_dataset = {
        "spambase": [859],
        "vehicle": [56],
    }

    frames: List[pd.DataFrame] = []
    for dataset in datasets:
        bundle = _load_bundle(data_dir / dataset)
        frames.append(
            _evaluate_dataset(
                dataset=dataset,
                bundle=bundle,
                best_dpg=best_dpg,
                best_baselines=best_baselines,
                max_samples_per_dataset=int(args.max_samples_per_dataset),
                rf_n_jobs=int(args.rf_n_jobs),
                shap_background_size=int(args.shap_background_size),
                lime_num_features=int(args.lime_num_features),
                lime_num_samples=int(args.lime_num_samples),
                anchor_precision_target=float(args.anchor_precision_target),
                anchor_max_rule_len=int(args.anchor_max_rule_len),
                anchor_max_candidates=int(args.anchor_max_candidates),
                include_cases=must_include_by_dataset.get(dataset, []),
            )
        )

    per_sample = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    per_sample.to_csv(out_dir / "semantic_faithfulness_per_sample.csv", index=False)

    summary = pd.DataFrame()
    if not per_sample.empty:
        summary = (
            per_sample.groupby(["method_family", "method"], as_index=False)
            .agg(
                datasets=("dataset", "nunique"),
                samples=("sample_idx", "count"),
                mean_feature_count=("feature_count", "mean"),
                mean_target_prob=("target_prob", "mean"),
                mean_sufficiency_prob=("sufficiency_prob", "mean"),
                mean_sufficiency_drop=("sufficiency_drop", "mean"),
                mean_comprehensiveness_prob=("comprehensiveness_prob", "mean"),
                mean_comprehensiveness_drop=("comprehensiveness_drop", "mean"),
                mean_critical_branch_margin_delta=("critical_branch_margin_delta", "mean"),
                critical_flip_rate=("critical_flip_to_competitor", "mean"),
            )
            .sort_values(["method_family", "mean_sufficiency_prob", "mean_comprehensiveness_drop"], ascending=[True, False, False])
            .reset_index(drop=True)
        )
    summary.to_csv(out_dir / "semantic_faithfulness_summary.csv", index=False)

    critical_summary = pd.DataFrame()
    if not per_sample.empty:
        critical = per_sample[
            (per_sample["method_family"] == "dpg")
            & per_sample["critical_node_label"].notna()
        ].copy()
        if not critical.empty:
            critical_summary = (
                critical.groupby(["dataset", "method"], as_index=False)
                .agg(
                    cases=("sample_idx", "count"),
                    mean_critical_branch_margin_delta=("critical_branch_margin_delta", "mean"),
                    critical_flip_rate=("critical_flip_to_competitor", "mean"),
                    mean_support_margin=("support_margin", "mean"),
                    mean_competitor_exposure=("competitor_exposure", "mean"),
                )
                .sort_values(["dataset", "method"])
                .reset_index(drop=True)
            )
    critical_summary.to_csv(out_dir / "critical_branch_flip_summary.csv", index=False)

    case_reports = per_sample[
        ((per_sample["dataset"] == "spambase") & (per_sample["sample_idx"] == 859))
        | ((per_sample["dataset"] == "vehicle") & (per_sample["sample_idx"] == 56))
    ].copy()
    case_reports.to_csv(out_dir / "critical_case_reports.csv", index=False)

    print(f"Saved: {out_dir / 'semantic_faithfulness_per_sample.csv'}")
    print(f"Saved: {out_dir / 'semantic_faithfulness_summary.csv'}")
    print(f"Saved: {out_dir / 'critical_branch_flip_summary.csv'}")
    print(f"Saved: {out_dir / 'critical_case_reports.csv'}")


if __name__ == "__main__":
    main()
