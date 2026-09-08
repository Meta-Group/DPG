#!/usr/bin/env python3
"""Run final journal test evaluations from validation-selected configs."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

import pandas as pd


NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def _value(row: pd.Series, column: str, default: object = "") -> object:
    if column not in row or pd.isna(row[column]):
        return default
    return row[column]


def _int_text(value: object) -> str:
    return str(int(float(value)))


def _depth_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "None"
    text = str(value)
    if text.strip().lower() in {"", "none", "nan", "null"}:
        return "None"
    return str(int(float(value)))


def _build_command(row: pd.Series, out_dir: Path, args: argparse.Namespace) -> List[str]:
    script_dir = Path(__file__).resolve().parent
    dataset = str(row["dataset"])
    method = str(row["method"])
    selection_kind = str(row.get("selection_kind", ""))

    common = [
        "--data_dir",
        args.data_dir,
        "--out_dir",
        str(out_dir),
        "--datasets",
        dataset,
        "--n_estimators",
        _int_text(_value(row, "n_estimators")),
        "--rf_n_jobs",
        str(args.rf_n_jobs),
        "--max_depth",
        _depth_text(_value(row, "max_depth")),
        "--perc_var",
        str(float(_value(row, "perc_var"))),
        "--decimal_threshold",
        _int_text(_value(row, "decimal_threshold")),
        "--seeds",
        _int_text(_value(row, "seed")),
        "--max_test_samples",
        str(args.max_test_samples),
        "--progress_every",
        str(args.progress_every),
        "--split_registry",
        args.split_registry,
        "--train_split",
        "train_core_validation",
        "--eval_split",
        "test",
    ]

    if selection_kind == "dpg_validation" or method.startswith("dpg"):
        graph_mode = str(_value(row, "graph_construction_mode", "aggregated_transitions"))
        return [
            sys.executable,
            str(script_dir / "run_local_experiments.py"),
            *common,
            "--graph_construction_modes",
            graph_mode,
        ]

    return [
        sys.executable,
        str(script_dir / "run_baseline_experiments.py"),
        *common,
        "--methods",
        method,
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


def _env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    for key in NATIVE_THREAD_ENV_VARS:
        env[key] = str(args.native_threads)
    env.setdefault("MPLCONFIGDIR", "/tmp/dpg_matplotlib")
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _collect_outputs(out_root: Path) -> None:
    summary_frames = []
    detail_frames = []
    for run_dir in sorted((out_root / "runs").glob("*")):
        if (run_dir / "summary.csv").exists():
            summary_frames.append(pd.read_csv(run_dir / "summary.csv"))
        if (run_dir / "summary_baselines.csv").exists():
            summary_frames.append(pd.read_csv(run_dir / "summary_baselines.csv"))
        if (run_dir / "per_sample.csv").exists():
            detail_frames.append(pd.read_csv(run_dir / "per_sample.csv"))
        if (run_dir / "per_sample_baselines.csv").exists():
            detail_frames.append(pd.read_csv(run_dir / "per_sample_baselines.csv"))

    if summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(out_root / "summary_selected_test.csv", index=False)
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(out_root / "per_sample_selected_test.csv", index=False)


def _selected_rows(selection_path: Path, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(selection_path)
    if max_rows > 0:
        df = df.head(max_rows).copy()
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final test split from validation-selected configs.")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--data_dir", type=str, default="experiments_local_explanation/data_numeric")
    parser.add_argument(
        "--split_registry",
        type=str,
        default="experiments_local_explanation/results_journal_v1/splits/split_registry.json",
    )
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--rf_n_jobs", type=int, default=1)
    parser.add_argument("--native_threads", type=int, default=1)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
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

    rows = _selected_rows(args.selection, max_rows=int(args.max_rows))
    runs_root = args.out_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for idx, row in rows.iterrows():
        slug = "__".join(
            [
                f"{idx:04d}",
                _safe_slug(row["dataset"]),
                _safe_slug(row["method"]),
                _safe_slug(row["config_id"]),
            ]
        )
        out_dir = runs_root / slug
        cmd = _build_command(row, out_dir=out_dir, args=args)
        tasks.append((slug, cmd))
        print("CMD:", " ".join(cmd), flush=True)

    if args.dry_run:
        return

    env = _env(args)
    pending = list(tasks)
    active: List[tuple[str, subprocess.Popen[str]]] = []
    failed: List[tuple[str, int]] = []
    parallel = max(int(args.parallel), 1)

    while pending or active:
        while pending and len(active) < parallel:
            slug, cmd = pending.pop(0)
            print(f"START {slug}", flush=True)
            active.append((slug, subprocess.Popen(cmd, env=env)))

        time.sleep(0.5)
        next_active: List[tuple[str, subprocess.Popen[str]]] = []
        for slug, proc in active:
            rc = proc.poll()
            if rc is None:
                next_active.append((slug, proc))
            elif rc == 0:
                print(f"DONE {slug}", flush=True)
            else:
                print(f"FAIL {slug} (exit={rc})", flush=True)
                failed.append((slug, rc))
        active = next_active

    if failed:
        names = ", ".join(f"{slug}:{rc}" for slug, rc in failed)
        raise RuntimeError(f"Some selected test runs failed: {names}")

    _collect_outputs(args.out_root)
    print(f"Saved: {args.out_root / 'summary_selected_test.csv'}")
    print(f"Saved: {args.out_root / 'per_sample_selected_test.csv'}")


if __name__ == "__main__":
    main()
