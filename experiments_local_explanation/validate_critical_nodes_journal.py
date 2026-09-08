#!/usr/bin/env python3
"""Validate DPG critical nodes under the journal split protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpg import DPGExplainer
from dpg.visualizer import parse_predicate_parts
from experiments_local_explanation.journal_splits import apply_split_registry, load_split_registry
from experiments_local_explanation.run_local_experiments import DatasetBundle, _load_bundle


def _depth_text(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "nan", "null"}:
        return None
    return int(float(text))


def _fit_model(row: pd.Series, bundle: DatasetBundle, rf_n_jobs: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=int(row["n_estimators"]),
        max_depth=_depth_text(row.get("max_depth")),
        random_state=int(row["seed"]),
        n_jobs=int(rf_n_jobs),
    )
    model.fit(bundle.X_train, bundle.y_train)
    return model


def _build_explainer(row: pd.Series, bundle: DatasetBundle, rf_n_jobs: int) -> tuple[RandomForestClassifier, DPGExplainer]:
    model = _fit_model(row, bundle, rf_n_jobs=rf_n_jobs)
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
    return model, explainer


def _feature_name_to_index(feature_names: Sequence[str]) -> Dict[str, int]:
    return {str(name): idx for idx, name in enumerate(feature_names)}


def _apply_predicate(
    sample: np.ndarray,
    predicate_label: Optional[str],
    feature_names: Sequence[str],
) -> Optional[np.ndarray]:
    if not predicate_label:
        return None
    parsed = parse_predicate_parts(str(predicate_label))
    if not parsed:
        return None
    feat_name, op, threshold = parsed
    feat_idx = _feature_name_to_index(feature_names).get(str(feat_name))
    if feat_idx is None:
        return None
    out = np.asarray(sample, dtype=float).copy()
    eps = 1e-6
    if str(op) == "<=":
        out[int(feat_idx)] = float(threshold) - eps
    else:
        out[int(feat_idx)] = float(threshold) + eps
    return out


def _target_probability(model: RandomForestClassifier, sample: np.ndarray, label: str) -> float:
    probs = model.predict_proba(sample.reshape(1, -1))[0]
    classes = [str(c) for c in model.classes_]
    if str(label) not in classes:
        return float("nan")
    return float(probs[classes.index(str(label))])


def _all_path_sequences(local: Any) -> List[List[str]]:
    sequences: List[List[str]] = []
    for path in getattr(local, "tree_paths", []) or []:
        labels = [str(label) for label in getattr(path, "labels", [])[:-1]]
        if labels:
            sequences.append(labels)
    return sequences


def _random_same_depth_label(local: Any, depth: int, exclude: set[str], rng: np.random.Generator) -> Optional[str]:
    candidates = []
    for seq in _all_path_sequences(local):
        if len(seq) > depth and seq[depth] not in exclude:
            candidates.append(seq[depth])
    candidates = sorted(set(candidates))
    if not candidates:
        return None
    return str(candidates[int(rng.integers(0, len(candidates)))])


def _random_path_label(local: Any, exclude: set[str], rng: np.random.Generator) -> Optional[str]:
    candidates = sorted({label for seq in _all_path_sequences(local) for label in seq if label not in exclude})
    if not candidates:
        return None
    return str(candidates[int(rng.integers(0, len(candidates)))])


def _intervention_metrics(
    model: RandomForestClassifier,
    sample: np.ndarray,
    feature_names: Sequence[str],
    target_label: str,
    competitor_label: Optional[str],
    predicate_label: Optional[str],
    prefix: str,
) -> Dict[str, Any]:
    changed = _apply_predicate(sample, predicate_label, feature_names)
    base_label = str(model.predict(sample.reshape(1, -1))[0])
    if changed is None:
        return {
            f"{prefix}_predicate": predicate_label,
            f"{prefix}_eligible": False,
            f"{prefix}_feature_changed": False,
            f"{prefix}_base_predicted_label": base_label,
            f"{prefix}_target_prob": np.nan,
            f"{prefix}_competitor_prob": np.nan,
            f"{prefix}_target_delta": np.nan,
            f"{prefix}_competitor_delta": np.nan,
            f"{prefix}_predicted_label": None,
            f"{prefix}_already_competitor": bool(competitor_label is not None and base_label == str(competitor_label)),
            f"{prefix}_flip_to_competitor": False,
            f"{prefix}_changed_to_competitor": False,
            f"{prefix}_label_changed": False,
        }
    base_target = _target_probability(model, sample, target_label)
    new_target = _target_probability(model, changed, target_label)
    base_comp = _target_probability(model, sample, competitor_label) if competitor_label is not None else np.nan
    new_comp = _target_probability(model, changed, competitor_label) if competitor_label is not None else np.nan
    pred_label = str(model.predict(changed.reshape(1, -1))[0])
    already_competitor = bool(competitor_label is not None and base_label == str(competitor_label))
    flip_to_competitor = bool(competitor_label is not None and pred_label == str(competitor_label))
    return {
        f"{prefix}_predicate": predicate_label,
        f"{prefix}_eligible": True,
        f"{prefix}_feature_changed": bool(not np.allclose(sample, changed, equal_nan=True)),
        f"{prefix}_base_predicted_label": base_label,
        f"{prefix}_target_prob": new_target,
        f"{prefix}_competitor_prob": new_comp,
        f"{prefix}_target_delta": new_target - base_target if pd.notna(new_target) and pd.notna(base_target) else np.nan,
        f"{prefix}_competitor_delta": new_comp - base_comp if pd.notna(new_comp) and pd.notna(base_comp) else np.nan,
        f"{prefix}_predicted_label": pred_label,
        f"{prefix}_already_competitor": already_competitor,
        f"{prefix}_flip_to_competitor": flip_to_competitor,
        f"{prefix}_changed_to_competitor": bool(flip_to_competitor and not already_competitor),
        f"{prefix}_label_changed": bool(pred_label != base_label),
    }


def _sample_indices(n_total: int, max_samples: int, rng: np.random.Generator) -> List[int]:
    if max_samples <= 0 or max_samples >= n_total:
        return list(range(n_total))
    return sorted(int(i) for i in rng.choice(np.arange(n_total), size=int(max_samples), replace=False))


def _evaluate_config(
    row: pd.Series,
    data_dir: Path,
    registry: dict[str, Any],
    registry_path: Path,
    max_samples: int,
    rf_n_jobs: int,
    seed: int,
) -> pd.DataFrame:
    raw_bundle = _load_bundle(data_dir / str(row["dataset"]))
    bundle = apply_split_registry(
        raw_bundle,
        registry=registry,
        registry_path=registry_path,
        train_split="train_core_validation",
        eval_split="test",
    )
    model, explainer = _build_explainer(row, bundle, rf_n_jobs=rf_n_jobs)
    dataset_offset = sum(ord(ch) for ch in str(row["dataset"]))
    rng = np.random.default_rng(int(seed) + int(row["seed"]) + dataset_offset)
    indices = _sample_indices(bundle.X_test.shape[0], max_samples=max_samples, rng=rng)
    rows: List[Dict[str, Any]] = []
    for sample_idx in indices:
        sample = np.asarray(bundle.X_test[int(sample_idx)], dtype=float)
        y_true = str(bundle.y_test[int(sample_idx)])
        y_model_pred = str(model.predict(sample.reshape(1, -1))[0])
        local = explainer.explain_local(sample=sample, sample_id=int(sample_idx), validate_graph=True)
        conf = dict(local.sample_confidence or {})
        target = conf.get("support_pred_class") or local.majority_vote or y_model_pred
        competitor = conf.get("support_top_competitor_class")
        critical_node = conf.get("critical_node_label")
        pred_successor = conf.get("critical_successor_pred")
        comp_successor = conf.get("critical_successor_comp")
        depth_value = conf.get("critical_split_depth")
        try:
            depth = int(depth_value) if depth_value is not None and pd.notna(depth_value) else None
        except Exception:
            depth = None
        eligible = bool(critical_node and pred_successor and comp_successor and depth is not None)
        exclude = {str(x) for x in [critical_node, pred_successor, comp_successor] if x}
        same_depth = _random_same_depth_label(local, int(depth), exclude, rng) if depth is not None else None
        random_path = _random_path_label(local, exclude, rng)

        base_target = _target_probability(model, sample, str(target))
        base_comp = _target_probability(model, sample, str(competitor)) if competitor is not None else np.nan
        out: Dict[str, Any] = {
            "dataset": bundle.name,
            "method": str(row["method"]),
            "graph_construction_mode": str(row["graph_construction_mode"]),
            "config_id": str(row["config_id"]),
            "seed": int(row["seed"]),
            "sample_idx": int(sample_idx),
            "y_true": y_true,
            "y_model_pred": y_model_pred,
            "target_label": str(target),
            "competitor_label": str(competitor) if competitor is not None else None,
            "base_target_prob": base_target,
            "base_competitor_prob": base_comp,
            "critical_node_label": critical_node,
            "critical_split_depth": depth,
            "critical_successor_pred": pred_successor,
            "critical_successor_comp": comp_successor,
            "critical_node_contrast": conf.get("critical_node_contrast"),
            "support_margin": conf.get("support_margin"),
            "competitor_exposure": conf.get("competitor_exposure"),
            "critical_node_present": bool(critical_node),
            "critical_successors_present": bool(pred_successor and comp_successor),
            "critical_eligible": eligible,
            "critical_nonroot_eligible": bool(eligible and depth is not None and depth >= 2),
        }
        out.update(
            _intervention_metrics(
                model,
                sample,
                bundle.feature_names,
                str(target),
                str(competitor) if competitor is not None else None,
                str(comp_successor) if comp_successor is not None else None,
                "critical_comp_branch",
            )
        )
        out.update(
            _intervention_metrics(
                model,
                sample,
                bundle.feature_names,
                str(target),
                str(competitor) if competitor is not None else None,
                str(pred_successor) if pred_successor is not None else None,
                "critical_pred_branch",
            )
        )
        out.update(
            _intervention_metrics(
                model,
                sample,
                bundle.feature_names,
                str(target),
                str(competitor) if competitor is not None else None,
                same_depth,
                "control_same_depth",
            )
        )
        out.update(
            _intervention_metrics(
                model,
                sample,
                bundle.feature_names,
                str(target),
                str(competitor) if competitor is not None else None,
                random_path,
                "control_random_path",
            )
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _mean_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def _summarize(per_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows = []
    for keys, group in per_sample.groupby(["dataset", "method", "graph_construction_mode"], dropna=False):
        dataset, method, graph_mode = keys
        eligible = group[group["critical_eligible"].astype(bool)].copy()
        dataset_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "graph_construction_mode": graph_mode,
                "samples": int(group.shape[0]),
                "critical_node_present_count": int(group["critical_node_present"].sum()),
                "critical_node_present_rate": float(group["critical_node_present"].mean()),
                "critical_successors_present_count": int(group["critical_successors_present"].sum()),
                "critical_successors_present_rate": float(group["critical_successors_present"].mean()),
                "critical_eligible_count": int(group["critical_eligible"].sum()),
                "critical_eligible_rate": float(group["critical_eligible"].mean()),
                "critical_nonroot_eligible_rate": float(group["critical_nonroot_eligible"].mean()),
                "mean_support_margin_all": _mean_or_nan(group["support_margin"]),
                "mean_competitor_exposure_all": _mean_or_nan(group["competitor_exposure"]),
                "mean_critical_comp_target_delta": _mean_or_nan(eligible["critical_comp_branch_target_delta"]),
                "mean_critical_comp_competitor_delta": _mean_or_nan(eligible["critical_comp_branch_competitor_delta"]),
                "critical_comp_feature_changed_rate": _mean_or_nan(eligible["critical_comp_branch_feature_changed"]),
                "critical_comp_label_changed_rate": _mean_or_nan(eligible["critical_comp_branch_label_changed"]),
                "critical_flip_to_competitor_rate": _mean_or_nan(eligible["critical_comp_branch_flip_to_competitor"]),
                "critical_changed_to_competitor_rate": _mean_or_nan(
                    eligible["critical_comp_branch_changed_to_competitor"]
                ),
                "critical_already_competitor_rate": _mean_or_nan(eligible["critical_comp_branch_already_competitor"]),
                "mean_control_same_depth_target_delta": _mean_or_nan(eligible["control_same_depth_target_delta"]),
                "mean_control_same_depth_competitor_delta": _mean_or_nan(eligible["control_same_depth_competitor_delta"]),
                "control_same_depth_label_changed_rate": _mean_or_nan(
                    eligible["control_same_depth_label_changed"]
                ),
                "control_same_depth_flip_to_competitor_rate": _mean_or_nan(
                    eligible["control_same_depth_flip_to_competitor"]
                ),
                "control_same_depth_changed_to_competitor_rate": _mean_or_nan(
                    eligible["control_same_depth_changed_to_competitor"]
                ),
                "mean_control_random_path_target_delta": _mean_or_nan(eligible["control_random_path_target_delta"]),
                "mean_control_random_path_competitor_delta": _mean_or_nan(
                    eligible["control_random_path_competitor_delta"]
                ),
                "control_random_path_label_changed_rate": _mean_or_nan(
                    eligible["control_random_path_label_changed"]
                ),
                "control_random_path_flip_to_competitor_rate": _mean_or_nan(
                    eligible["control_random_path_flip_to_competitor"]
                ),
                "control_random_path_changed_to_competitor_rate": _mean_or_nan(
                    eligible["control_random_path_changed_to_competitor"]
                ),
                "mean_critical_vs_same_depth_competitor_delta": _mean_or_nan(
                    eligible["critical_comp_branch_competitor_delta"]
                    - eligible["control_same_depth_competitor_delta"]
                ),
                "mean_critical_vs_random_path_competitor_delta": _mean_or_nan(
                    eligible["critical_comp_branch_competitor_delta"]
                    - eligible["control_random_path_competitor_delta"]
                ),
            }
        )
    dataset_summary = pd.DataFrame(dataset_rows).sort_values(["dataset", "method"]).reset_index(drop=True)
    method_summary = (
        dataset_summary.groupby(["method", "graph_construction_mode"], as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            total_samples=("samples", "sum"),
            mean_critical_node_present_rate=("critical_node_present_rate", "mean"),
            mean_critical_successors_present_rate=("critical_successors_present_rate", "mean"),
            mean_critical_eligible_rate=("critical_eligible_rate", "mean"),
            mean_nonroot_eligible_rate=("critical_nonroot_eligible_rate", "mean"),
            mean_critical_comp_target_delta=("mean_critical_comp_target_delta", "mean"),
            mean_critical_comp_competitor_delta=("mean_critical_comp_competitor_delta", "mean"),
            mean_critical_comp_feature_changed_rate=("critical_comp_feature_changed_rate", "mean"),
            mean_critical_comp_label_changed_rate=("critical_comp_label_changed_rate", "mean"),
            mean_critical_flip_to_competitor_rate=("critical_flip_to_competitor_rate", "mean"),
            mean_critical_changed_to_competitor_rate=("critical_changed_to_competitor_rate", "mean"),
            mean_critical_already_competitor_rate=("critical_already_competitor_rate", "mean"),
            mean_control_same_depth_competitor_delta=("mean_control_same_depth_competitor_delta", "mean"),
            mean_control_random_path_competitor_delta=("mean_control_random_path_competitor_delta", "mean"),
            mean_control_same_depth_label_changed_rate=("control_same_depth_label_changed_rate", "mean"),
            mean_control_random_path_label_changed_rate=("control_random_path_label_changed_rate", "mean"),
            mean_control_same_depth_changed_to_competitor_rate=(
                "control_same_depth_changed_to_competitor_rate",
                "mean",
            ),
            mean_control_random_path_changed_to_competitor_rate=(
                "control_random_path_changed_to_competitor_rate",
                "mean",
            ),
            mean_critical_vs_same_depth_competitor_delta=("mean_critical_vs_same_depth_competitor_delta", "mean"),
            mean_critical_vs_random_path_competitor_delta=("mean_critical_vs_random_path_competitor_delta", "mean"),
        )
        .sort_values(["method", "graph_construction_mode"])
        .reset_index(drop=True)
    )
    return dataset_summary, method_summary


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        return str(value)

    rows = [[fmt(row[col]) for col in cols] for _, row in df.iterrows()]
    widths = [max(len(str(col)), *(len(row[i]) for row in rows)) for i, col in enumerate(cols)]
    header = "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_report(out_dir: Path, method_summary: pd.DataFrame, dataset_summary: pd.DataFrame) -> None:
    compact_cols = [
        "method",
        "graph_construction_mode",
        "datasets",
        "total_samples",
        "mean_critical_node_present_rate",
        "mean_critical_successors_present_rate",
        "mean_critical_eligible_rate",
        "mean_critical_comp_target_delta",
        "mean_critical_comp_competitor_delta",
        "mean_critical_comp_feature_changed_rate",
        "mean_critical_comp_label_changed_rate",
        "mean_critical_flip_to_competitor_rate",
        "mean_critical_changed_to_competitor_rate",
        "mean_critical_already_competitor_rate",
        "mean_control_same_depth_label_changed_rate",
        "mean_control_random_path_label_changed_rate",
        "mean_critical_vs_same_depth_competitor_delta",
        "mean_critical_vs_random_path_competitor_delta",
    ]
    lines = [
        "# Critical Node Conditional Validation",
        "",
        "Critical nodes are evaluated only when a non-root divergence between the predicted and strongest competitor path is available.",
        "",
        "## Method Summary",
        "",
        _markdown_table(method_summary[[c for c in compact_cols if c in method_summary.columns]]),
        "",
        "## Dataset Summary",
        "",
        _markdown_table(dataset_summary),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DPG critical nodes on the journal final-test split.")
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/validation_selection_main_with_lore_noice/selected_dpg_configs.csv"),
    )
    parser.add_argument("--data_dir", type=Path, default=Path("experiments_local_explanation/data_numeric"))
    parser.add_argument(
        "--split_registry",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/splits/split_registry.json"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/critical_node_validation"),
    )
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--max_samples_per_dataset", type=int, default=0)
    parser.add_argument("--rf_n_jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260618)
    args = parser.parse_args()

    selection = pd.read_csv(args.selection)
    if args.datasets.strip():
        wanted = {item.strip() for item in args.datasets.split(",") if item.strip()}
        selection = selection[selection["dataset"].astype(str).isin(wanted)].copy()
    if selection.empty:
        raise ValueError("No selected DPG configs to evaluate.")

    registry = load_split_registry(args.split_registry)
    data_dir = (REPO_ROOT / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    registry_path = (REPO_ROOT / args.split_registry).resolve() if not args.split_registry.is_absolute() else args.split_registry
    out_dir = (REPO_ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for _, row in selection.sort_values(["dataset", "method"]).iterrows():
        print(f"Evaluating {row['dataset']} {row['method']} {row['config_id']}", flush=True)
        frames.append(
            _evaluate_config(
                row=row,
                data_dir=data_dir,
                registry=registry,
                registry_path=registry_path,
                max_samples=int(args.max_samples_per_dataset),
                rf_n_jobs=int(args.rf_n_jobs),
                seed=int(args.seed),
            )
        )
    per_sample = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    dataset_summary, method_summary = _summarize(per_sample)
    per_sample.to_csv(out_dir / "critical_node_per_sample.csv", index=False)
    dataset_summary.to_csv(out_dir / "critical_node_dataset_summary.csv", index=False)
    method_summary.to_csv(out_dir / "critical_node_method_summary.csv", index=False)
    _write_report(out_dir, method_summary=method_summary, dataset_summary=dataset_summary)
    print(f"Saved: {out_dir / 'critical_node_per_sample.csv'}")
    print(f"Saved: {out_dir / 'critical_node_dataset_summary.csv'}")
    print(f"Saved: {out_dir / 'critical_node_method_summary.csv'}")
    print(f"Saved: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
