#!/usr/bin/env python3
"""Run E4 scale experiments on the synthetic S5/S6 datasets."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

SCENARIOS = {
    "S5_100k_x50": (100_000, 50, 10, 0.05),
    "S6_10k_x500": (10_000, 500, 20, 0.05),
}
MODELS = {"RandomForestClassifier": RandomForestClassifier, "ExtraTreesClassifier": ExtraTreesClassifier}


def run_task(task):
    scenario, model_name, learners, seed, commit = task
    n_samples, n_features, informative, flip_y = SCENARIOS[scenario]
    row = {"scenario": scenario, "model": model_name, "n_samples": n_samples, "n_features": n_features,
           "n_estimators": learners, "seed": seed, "status": "error", "error": "",
           "k1_seconds": 0.0, "kauto_seconds": 0.0, "k1_nodes": 0, "kauto_nodes": 0,
           "node_ratio": "", "kauto": "", "kauto_violations": "", "git_commit": commit}
    try:
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=informative, n_redundant=0,
                                   n_classes=3, n_clusters_per_class=1,
                                   flip_y=flip_y, random_state=seed)
        model = MODELS[model_name](n_estimators=learners, random_state=seed, n_jobs=1).fit(X, y)
        from dpg.core import DecisionPredicateGraph
        names = [f"feature_{i}" for i in range(n_features)]
        targets = ["0", "1", "2"]
        config = lambda order: {"dpg": {"default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
                                         "graph_construction": {"mode": "execution_trace", "context_order": order}}}
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            dpg1 = DecisionPredicateGraph(model, names, target_names=targets, dpg_config=config(1))
            graph1, _ = dpg1.to_networkx(dpg1.fit(X))
        k1_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            dpgk = DecisionPredicateGraph(model, names, target_names=targets, dpg_config=config("auto"))
            graphk, _ = dpgk.to_networkx(dpgk.fit(X))
        row.update({"status": "ok", "k1_seconds": k1_seconds, "kauto_seconds": time.perf_counter() - start,
                    "k1_nodes": graph1.number_of_nodes(), "kauto_nodes": graphk.number_of_nodes(),
                    "node_ratio": graphk.number_of_nodes() / max(1, graph1.number_of_nodes()),
                    "kauto": str(dpgk.get_context_order()), "kauto_violations": repr(dpgk.get_context_order_history())})
    except Exception as exc:  # noqa: BLE001 - retain every scale cell
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/dpg_0_3_0/results/e4_scale.csv"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DPG_WORKERS", min(os.cpu_count() or 1, 18))))
    parser.add_argument("--notify-every", type=int, default=5)
    args = parser.parse_args()
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    tasks = [(scenario, model, learners, seed, commit)
             for scenario in SCENARIOS for model in MODELS
             for learners in (10, 25) for seed in (0, 1)]
    fields = ["scenario", "model", "n_samples", "n_features", "n_estimators", "seed", "status", "error",
              "k1_seconds", "kauto_seconds", "k1_nodes", "kauto_nodes", "node_ratio", "kauto",
              "kauto_violations", "git_commit"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.notify_telegram import send_message
    send_message(f"DPG E4 started: {len(tasks)} scale tasks, {args.workers} workers.")
    completed = ok = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
                    message = f"DPG E4 progress: {completed}/{len(tasks)} complete; {ok} ok; output={args.output}"
                    print(message, flush=True)
                    send_message(message)
    message = f"DPG E4 finished: {ok}/{len(tasks)} ok; output={args.output}"
    print(message, flush=True)
    send_message(message)


if __name__ == "__main__":
    main()
