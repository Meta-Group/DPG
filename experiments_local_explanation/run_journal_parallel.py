#!/usr/bin/env python3
"""Journal experiment launcher with split protocol and hardware-aware defaults."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


CURATED_15 = (
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
)

SMOKE_DATASETS = ("iris", "vehicle", "spambase")

NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _load_recommendations(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(profile.get("recommendations", {}) or {})


def _default_parallel(kind: str, recommendations: Dict[str, Any]) -> int:
    key = "dpg_parallel_workers" if kind == "dpg" else "baseline_parallel_workers"
    value = recommendations.get(key)
    if value is not None:
        return max(int(value), 1)
    return 12 if kind == "dpg" else 6


def _split_args(phase: str) -> tuple[str, str]:
    if phase == "validation":
        return "train_core", "validation"
    if phase == "test":
        return "train_core_validation", "test"
    raise ValueError(f"Unsupported phase: {phase}")


def _dataset_arg(selection: str, explicit: str) -> str:
    if explicit:
        return explicit
    if selection == "smoke":
        return ",".join(SMOKE_DATASETS)
    if selection == "curated_15":
        return ",".join(CURATED_15)
    return ""


def _native_env(recommendations: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    native = recommendations.get("native_thread_env") or {}
    for key in NATIVE_THREAD_ENV_VARS:
        env[key] = str(native.get(key, "1"))
    env.setdefault("MPLCONFIGDIR", "/tmp/dpg_matplotlib")
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _build_cmd(args: argparse.Namespace, parallel: int, train_split: str, eval_split: str) -> List[str]:
    script_dir = Path(__file__).resolve().parent
    if args.kind == "dpg":
        runner = script_dir / "run_per_dataset.py"
        out_root = args.out_root or f"experiments_local_explanation/results_journal_v1/dpg_{args.phase}_by_dataset"
        cmd = [
            sys.executable,
            str(runner),
            "--data_dir",
            args.data_dir,
            "--out_root",
            out_root,
            "--datasets",
            _dataset_arg(args.dataset_set, args.datasets),
            "--parallel",
            str(parallel),
            "--model_families",
            args.model_families,
            "--n_estimators",
            args.n_estimators,
            "--rf_n_jobs",
            str(args.rf_n_jobs),
            "--max_depth",
            args.max_depth,
            "--perc_var",
            args.perc_var,
            "--decimal_threshold",
            args.decimal_threshold,
            "--seeds",
            args.seeds,
            "--graph_construction_modes",
            args.graph_construction_modes,
            "--max_test_samples",
            str(args.max_test_samples),
            "--progress_every",
            str(args.progress_every),
        ]
    else:
        runner = script_dir / "run_per_dataset_baselines.py"
        out_root = args.out_root or f"experiments_local_explanation/results_journal_v1/baselines_{args.phase}_by_dataset"
        cmd = [
            sys.executable,
            str(runner),
            "--data_dir",
            args.data_dir,
            "--out_root",
            out_root,
            "--datasets",
            _dataset_arg(args.dataset_set, args.datasets),
            "--parallel",
            str(parallel),
            "--methods",
            args.methods,
            "--n_estimators",
            args.n_estimators,
            "--rf_n_jobs",
            str(args.rf_n_jobs),
            "--max_depth",
            args.max_depth,
            "--perc_var",
            args.perc_var,
            "--decimal_threshold",
            args.decimal_threshold,
            "--seeds",
            args.seeds,
            "--max_test_samples",
            str(args.max_test_samples),
            "--progress_every",
            str(args.progress_every),
            "--shap_background_size",
            str(args.shap_background_size),
            "--lime_num_features",
            str(args.lime_num_features),
            "--lime_num_samples",
            str(args.lime_num_samples),
            "--ice_grid_points",
            str(args.ice_grid_points),
            "--anchor_precision_target",
            str(args.anchor_precision_target),
            "--anchor_max_rule_len",
            str(args.anchor_max_rule_len),
            "--anchor_max_candidates",
            str(args.anchor_max_candidates),
            "--lore_neighborhood_size",
            str(args.lore_neighborhood_size),
            "--lore_max_depth",
            str(args.lore_max_depth),
            "--lore_min_samples_leaf",
            str(args.lore_min_samples_leaf),
            "--lore_neighborhood_scale",
            str(args.lore_neighborhood_scale),
        ]

    cmd.extend(
        [
            "--split_registry",
            args.split_registry,
            "--train_split",
            train_split,
            "--eval_split",
            eval_split,
            "--native_threads",
            str(args.native_threads),
        ]
    )
    if args.no_resume:
        cmd.append("--no-resume")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run journal experiments with split and hardware defaults.")
    parser.add_argument("--kind", choices=["dpg", "baselines"], required=True)
    parser.add_argument("--phase", choices=["validation", "test"], default="validation")
    parser.add_argument("--dataset_set", choices=["smoke", "curated_15", "all"], default="smoke")
    parser.add_argument("--datasets", type=str, default="", help="Explicit comma-separated datasets.")
    parser.add_argument("--data_dir", type=str, default="experiments_local_explanation/data_numeric")
    parser.add_argument("--out_root", type=str, default="")
    parser.add_argument(
        "--split_registry",
        type=str,
        default="experiments_local_explanation/results_journal_v1/splits/split_registry.json",
    )
    parser.add_argument(
        "--hardware_profile",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/hardware/hardware_profile.json"),
    )
    parser.add_argument("--parallel", type=int, default=0, help="0 means use hardware recommendation.")
    parser.add_argument("--native_threads", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--n_estimators", type=str, default="20,50,100")
    parser.add_argument("--model_families", type=str, default="random_forest")
    parser.add_argument("--rf_n_jobs", type=int, default=1)
    parser.add_argument("--max_depth", type=str, default="4,8,12,None")
    parser.add_argument("--perc_var", type=str, default="0.0001,0.00001")
    parser.add_argument("--decimal_threshold", type=str, default="6")
    parser.add_argument("--seeds", type=str, default="27,42,100,101,102,103,104,105,106,107")
    parser.add_argument("--graph_construction_modes", type=str, default="aggregated_transitions,execution_trace")
    parser.add_argument("--methods", type=str, default="shap,lime,anchors,tree_path,lore")
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=25)
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

    recommendations = _load_recommendations(args.hardware_profile)
    parallel = int(args.parallel) if int(args.parallel) > 0 else _default_parallel(args.kind, recommendations)
    train_split, eval_split = _split_args(args.phase)
    cmd = _build_cmd(args, parallel=parallel, train_split=train_split, eval_split=eval_split)
    env = _native_env(recommendations)

    print("Journal launcher")
    print(f"  kind={args.kind}")
    print(f"  phase={args.phase}")
    print(f"  train_split={train_split}")
    print(f"  eval_split={eval_split}")
    print(f"  parallel={parallel}")
    print(f"  rf_n_jobs={args.rf_n_jobs}")
    print("  native thread caps:", " ".join(f"{key}={env[key]}" for key in NATIVE_THREAD_ENV_VARS))
    print("CMD:", " ".join(cmd))

    if args.dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
