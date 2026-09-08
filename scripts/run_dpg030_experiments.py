#!/usr/bin/env python3
"""Run the reproducible DPG 0.3.0 benchmark grid.

Each task owns one sklearn worker thread; the process pool fans tasks across
all detected CPUs without multiplying BLAS/OpenMP threads inside each task.
Every task returns a row, including failures, so interrupted or unsupported
cells remain auditable.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}
MODELS = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "BaggingClassifier": BaggingClassifier,
}
VARIANTS = {
    "aggregated": ("aggregated_transitions", 1, 6),
    "execution_trace_k1": ("execution_trace", 1, 6),
    "dpg_k_auto": ("execution_trace", "auto", "auto"),
}


@dataclass
class Result:
    dataset: str
    model: str
    n_estimators: int
    seed: int
    variant: str
    status: str
    error: str = ""
    seconds: float = 0.0
    nodes: int = 0
    edges: int = 0
    total_edge_weight: float = 0.0
    resolved_context_order: str = ""
    context_violations: str = ""
    decimal_threshold: str = ""
    warnings: int = 0
    git_commit: str = "unknown"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _make_model(name: str, n_estimators: int, seed: int):
    cls = MODELS[name]
    kwargs = {"n_estimators": n_estimators, "random_state": seed}
    if name in {"RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier"}:
        kwargs["n_jobs"] = 1
    return cls(**kwargs)


def run_task(task: tuple[str, str, int, int, str, str]) -> dict:
    dataset_name, model_name, n_estimators, seed, variant, commit = task
    result = Result(dataset_name, model_name, n_estimators, seed, variant, "error", git_commit=commit)
    started = time.perf_counter()
    try:
        dataset = DATASETS[dataset_name](as_frame=False)
        model = _make_model(model_name, n_estimators, seed)
        model.fit(dataset.data, dataset.target)

        from dpg.core import DecisionPredicateGraph

        mode, context_order, decimal_threshold = VARIANTS[variant]
        config = {
            "dpg": {
                "default": {
                    "perc_var": 1e-9,
                    "decimal_threshold": decimal_threshold,
                    "n_jobs": 1,
                },
                "graph_construction": {
                    "mode": mode,
                    "context_order": context_order,
                },
            }
        }
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            builder = DecisionPredicateGraph(
                model=model,
                feature_names=dataset.feature_names,
                target_names=[str(label) for label in np.unique(dataset.target)],
                dpg_config=config,
            )
            dot = builder.fit(dataset.data)
        graph, _ = builder.to_networkx(dot)
        result.status = "ok"
        result.seconds = time.perf_counter() - started
        result.nodes = graph.number_of_nodes()
        result.edges = graph.number_of_edges()
        result.total_edge_weight = float(sum(data.get("weight", 0.0) for _, _, data in graph.edges(data=True)))
        result.resolved_context_order = str(builder.get_context_order())
        result.context_violations = repr(builder.get_context_order_history())
        result.decimal_threshold = str(builder.get_decimal_threshold())
        result.warnings = len(captured)
    except Exception as exc:  # noqa: BLE001 - failures are benchmark rows
        result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - started
    return asdict(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/dpg_0_3_0/results/benchmark.csv"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--learners", default="5,10,25,50,100")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DPG_WORKERS", os.cpu_count() or 1)))
    parser.add_argument("--notify-every", type=int, default=25)
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    learners = [int(x) for x in args.learners.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = (set(datasets) - set(DATASETS)) | (set(models) - set(MODELS)) | (set(variants) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown benchmark values: {sorted(unknown)}")

    tasks = [
        (dataset, model, n, seed, variant, _git_commit())
        for dataset in datasets
        for model in models
        for n in learners
        for seed in seeds
        for variant in variants
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(Result("", "", 0, 0, "", "")).keys())
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"DPG 0.3.0 benchmark start: {len(tasks)} tasks, workers={args.workers}", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.notify_telegram import send_message
    send_message(f"DPG 0.3.0 benchmark started on {os.cpu_count()} CPUs: {len(tasks)} tasks, {args.workers} workers.")

    completed = 0
    ok = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_task, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                writer.writerow(row)
                handle.flush()
                completed += 1
                ok += row["status"] == "ok"
                if completed % args.notify_every == 0 or completed == len(tasks):
                    message = f"DPG 0.3.0 progress: {completed}/{len(tasks)} complete; {ok} ok; output={args.output}"
                    print(message, flush=True)
                    send_message(message)

    message = f"DPG 0.3.0 benchmark finished: {ok}/{len(tasks)} ok; started {started}; output={args.output}"
    print(message, flush=True)
    send_message(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
