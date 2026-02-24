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
    # max_depth=None in sklearn means unlimited tree depth.
    depth = "unlimited" if cfg.max_depth is None else str(cfg.max_depth)
    return (
        f"rf{cfg.n_estimators}_d{depth}_pv{cfg.perc_var:g}"
        f"_dt{cfg.decimal_threshold}_s{cfg.random_state}"
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
        summary_df = (
            summary_df.drop_duplicates(subset=["dataset", "config_id", "seed"], keep="last")
            .sort_values(["dataset", "config_id"])
            .reset_index(drop=True)
        )
    if len(detail_df):
        detail_df = (
            detail_df.drop_duplicates(subset=["dataset", "config_id", "seed", "sample_idx"], keep="last")
            .sort_values(["dataset", "config_id", "sample_idx"])
            .reset_index(drop=True)
        )
    return summary_df, detail_df


def _run_one_dataset(
    bundle: DatasetBundle,
    cfg: ExperimentConfig,
    max_test_samples: int,
    progress_every: int,
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
        n_jobs=-1,
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
                }
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
            num_active_nodes = int(local.sample_confidence.get("num_active_nodes") or 0)
            mean_lrc_active_nodes = local.sample_confidence.get("mean_lrc_active_nodes")
            mean_bc_active_nodes = local.sample_confidence.get("mean_bc_active_nodes")
            class_support_total = float(sum(float(v) for v in support.values())) if support else 0.0
            n_class_support = int(len(support))
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
            num_active_nodes = 0
            mean_lrc_active_nodes = np.nan
            mean_bc_active_nodes = np.nan
            class_support_total = 0.0
            n_class_support = 0

        sample_rows.append(
            {
                "dataset": bundle.name,
                "config_id": _config_id(cfg),
                "seed": cfg.random_state,
                "sample_idx": idx,
                "y_true": y_true,
                "y_model_pred": y_hat,
                "y_local_pred": local_pred,
                "local_status": local_status,
                "local_error": local_error,
                "model_correct": y_hat == y_true,
                "local_matches_model": local_pred == y_hat if local_pred is not None else False,
                "local_correct": local_pred == y_true if local_pred is not None else False,
                "num_paths": num_paths,
                "num_active_nodes": num_active_nodes,
                "mean_lrc_active_nodes": mean_lrc_active_nodes,
                "mean_bc_active_nodes": mean_bc_active_nodes,
                "evidence_score_local_pred": local_pred_score,
                "evidence_score_true_class": true_score,
                "evidence_margin_pred_vs_competitor": margin,
                "top_competitor_class_pred": top_comp,
                "evidence_score_competitor_pred": top_comp_score,
                "class_support_total": class_support_total,
                "n_class_support": n_class_support,
            }
        )

    df = pd.DataFrame(sample_rows)

    summary: Dict[str, float | int | str | None] = {
        "dataset": bundle.name,
        "config_id": _config_id(cfg),
        "seed": cfg.random_state,
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "perc_var": cfg.perc_var,
        "decimal_threshold": cfg.decimal_threshold,
        "n_test_total": int(n_test),
        "n_test_evaluated": int(n_eval),
        "model_accuracy": model_acc,
        "local_matches_model_rate": float(df["local_matches_model"].mean()) if len(df) else np.nan,
        "local_accuracy": float(df["local_correct"].mean()) if len(df) else np.nan,
        "avg_num_paths": float(df["num_paths"].mean()) if len(df) else np.nan,
        "avg_num_active_nodes": float(df["num_active_nodes"].mean()) if len(df) else np.nan,
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
    else:
        summary["avg_margin_model_correct"] = np.nan
        summary["avg_margin_model_wrong"] = np.nan

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
        grouped = (
            summary_df.groupby("dataset", as_index=False)
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
