#!/usr/bin/env python3
"""Run E3: controlled irrelevant-feature experiments S1-S4."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

SCENARIOS = {
    "S1_clean_20": (20, 0.0),
    "S2_noise_20": (20, 0.10),
    "S3_clean_50": (50, 0.0),
    "S4_clean_200": (200, 0.0),
}
MODELS = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "BaggingClassifier": BaggingClassifier,
}


def _make_model(name, learners, seed):
    kwargs = {"n_estimators": learners, "random_state": seed}
    if name in {"RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier"}:
        kwargs["n_jobs"] = 1
    return MODELS[name](**kwargs)


def _config(context_order):
    return {"dpg": {"default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
                     "graph_construction": {"mode": "execution_trace", "context_order": context_order}}}


def _fit(builder, X):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dot = builder.fit(X)
    graph, _ = builder.to_networkx(dot)
    return graph


def run_task(task):
    scenario, model_name, learners, seed, commit = task
    n_features, flip_y = SCENARIOS[scenario]
    row = {"scenario": scenario, "n_features": n_features, "irrelevant_features": n_features - 5,
           "flip_y": flip_y, "model": model_name, "n_estimators": learners, "seed": seed,
           "status": "error", "error": "", "k": "", "nodes_k1": 0, "nodes_kauto": 0,
           "node_ratio": 0.0, "seconds_k1": 0.0, "seconds_kauto": 0.0,
           "violations": "", "git_commit": commit}
    try:
        X, y = make_classification(n_samples=1000, n_features=n_features, n_informative=5,
                                   n_redundant=0, n_repeated=0, n_classes=3,
                                   n_clusters_per_class=1, flip_y=flip_y,
                                   class_sep=1.0, random_state=seed)
        from dpg.core import DecisionPredicateGraph
        names = [f"feature_{i}" for i in range(n_features)]
        target_names = ["0", "1", "2"]
        model = _make_model(model_name, learners, seed).fit(X, y)
        start = time.perf_counter()
        k1 = DecisionPredicateGraph(model, names, target_names=target_names, dpg_config=_config(1))
        graph1 = _fit(k1, X)
        seconds_k1 = time.perf_counter() - start
        start = time.perf_counter()
        auto = DecisionPredicateGraph(model, names, target_names=target_names, dpg_config=_config("auto"))
        graphk = _fit(auto, X)
        seconds_kauto = time.perf_counter() - start
        row.update({"status": "ok", "k": str(auto.get_context_order()),
                    "nodes_k1": graph1.number_of_nodes(), "nodes_kauto": graphk.number_of_nodes(),
                    "node_ratio": graphk.number_of_nodes() / max(1, graph1.number_of_nodes()),
                    "seconds_k1": seconds_k1, "seconds_kauto": seconds_kauto,
                    "violations": repr(auto.get_context_order_history())})
    except Exception as exc:  # noqa: BLE001 - preserve failed cells
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/dpg_0_3_0/results/e3_irrelevant_features.csv"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DPG_WORKERS", os.cpu_count() or 1)))
    parser.add_argument("--notify-every", type=int, default=25)
    args = parser.parse_args()
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    tasks = [(scenario, model, learners, seed, commit)
             for scenario in SCENARIOS for model in MODELS
             for learners in (10, 25, 50) for seed in (0, 1, 2, 3, 4)]
    fields = ["scenario", "n_features", "irrelevant_features", "flip_y", "model", "n_estimators", "seed",
              "status", "error", "k", "nodes_k1", "nodes_kauto", "node_ratio", "seconds_k1",
              "seconds_kauto", "violations", "git_commit"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.notify_telegram import send_message
    send_message(f"DPG E3 started: {len(tasks)} tasks, {args.workers} workers.")
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
                    message = f"DPG E3 progress: {completed}/{len(tasks)} complete; {ok} ok; output={args.output}"
                    print(message, flush=True)
                    send_message(message)
    message = f"DPG E3 finished: {ok}/{len(tasks)} ok; output={args.output}"
    print(message, flush=True)
    send_message(message)


if __name__ == "__main__":
    main()
