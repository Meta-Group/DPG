#!/usr/bin/env python3
"""Launch one baseline-experiment run per dataset.

Each dataset is executed as an independent process of run_baseline_experiments.py,
writing results to:
    <out_root>/<dataset>/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

EXCLUDED_DATASETS = {"fashion_mnist_784", "mnist_784"}


def _list_datasets(data_dir: Path, only: Sequence[str] | None) -> List[str]:
    ds = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and p.name not in EXCLUDED_DATASETS])
    if not only:
        return ds
    selected = set(only)
    return [name for name in ds if name in selected and name not in EXCLUDED_DATASETS]


def _build_cmd(
    python_exec: str,
    runner: Path,
    dataset: str,
    data_dir: Path,
    out_root: Path,
    args: argparse.Namespace,
) -> List[str]:
    out_dir = out_root / dataset
    cmd = [
        python_exec,
        str(runner),
        "--data_dir",
        str(data_dir),
        "--out_dir",
        str(out_dir),
        "--datasets",
        dataset,
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
    ]
    if args.no_resume:
        cmd.append("--no-resume")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one baseline-experiment process per dataset.")
    parser.add_argument("--data_dir", type=str, default="experiments_local_explanation/data_numeric")
    parser.add_argument("--out_root", type=str, default="experiments_local_explanation/results_baselines_by_dataset")
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated dataset names. Empty = all.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of dataset processes to run concurrently.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")

    # Forwarded to run_baseline_experiments.py
    parser.add_argument("--methods", type=str, default="dpg,shap,lime,ice")
    parser.add_argument("--n_estimators", type=str, default="10,20")
    parser.add_argument("--rf_n_jobs", type=int, default=-1)
    parser.add_argument("--max_depth", type=str, default="2,4,None")
    parser.add_argument("--perc_var", type=str, default="0.0,0.001")
    parser.add_argument("--decimal_threshold", type=str, default="4,6")
    parser.add_argument("--seeds", type=str, default="27,42")
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shap_background_size", type=int, default=200)
    parser.add_argument("--lime_num_features", type=int, default=20)
    parser.add_argument("--lime_num_samples", type=int, default=1000)
    parser.add_argument("--ice_grid_points", type=int, default=21)
    parser.add_argument("--anchor_precision_target", type=float, default=0.95)
    parser.add_argument("--anchor_max_rule_len", type=int, default=4)
    parser.add_argument("--anchor_max_candidates", type=int, default=12)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    runner = script_dir / "run_baseline_experiments.py"
    data_dir = Path(args.data_dir).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    only = [x.strip() for x in args.datasets.split(",") if x.strip()]
    datasets = _list_datasets(data_dir, only=only)
    if not datasets:
        raise ValueError(f"No datasets selected in {data_dir}")

    cmds = [_build_cmd(sys.executable, runner, ds, data_dir, out_root, args) for ds in datasets]

    for cmd in cmds:
        print("CMD:", " ".join(cmd))
    if args.dry_run:
        return

    parallel = max(1, int(args.parallel))
    if parallel == 1:
        for cmd in cmds:
            subprocess.run(cmd, check=True)
        return

    pending = list(zip(datasets, cmds))
    active: List[tuple[str, subprocess.Popen[str]]] = []
    failed: List[tuple[str, int]] = []

    while pending or active:
        while pending and len(active) < parallel:
            ds, cmd = pending.pop(0)
            print(f"START {ds}")
            proc = subprocess.Popen(cmd)
            active.append((ds, proc))

        time.sleep(0.2)
        next_active: List[tuple[str, subprocess.Popen[str]]] = []
        for ds, proc in active:
            rc = proc.poll()
            if rc is None:
                next_active.append((ds, proc))
                continue
            if rc == 0:
                print(f"DONE {ds}")
            else:
                print(f"FAIL {ds} (exit={rc})")
                failed.append((ds, rc))
        active = next_active

    if failed:
        names = ", ".join(f"{ds}:{rc}" for ds, rc in failed)
        raise RuntimeError(f"Some dataset runs failed: {names}")


if __name__ == "__main__":
    main()
