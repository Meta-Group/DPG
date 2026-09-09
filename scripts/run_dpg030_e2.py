#!/usr/bin/env python3
"""Run E2 context-order, phantom-path, and construction-cost experiments."""

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

import networkx as nx
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


DATASETS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}
MODELS = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "BaggingClassifier": BaggingClassifier,
}


def _make_model(name, n_estimators, seed):
    kwargs = {"n_estimators": n_estimators, "random_state": seed}
    if name in {"RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier"}:
        kwargs["n_jobs"] = 1
    return MODELS[name](**kwargs)


def _config(context_order):
    return {"dpg": {"default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
                     "graph_construction": {"mode": "execution_trace", "context_order": context_order}}}


def _phantom_metrics(builder, graph, log, budget=20000):
    traces = builder._trace_sequences(log)
    observed = {tuple(builder.get_node_ids_for_trace(trace)) for trace in traces}
    sinks = [node for node, data in graph.nodes(data=True) if data.get("predicate", "").startswith("Class ")]
    roots = [node for node in graph if graph.in_degree(node) == 0]
    longest = max((len(trace) for trace in traces), default=1)
    total = phantom = 0
    total_mass = phantom_mass = 0.0
    exact = True
    for root in roots:
        for sink in sinks:
            for path in nx.all_simple_paths(graph, root, sink, cutoff=longest):
                total += 1
                edge_weights = [graph.edges[source, target].get("weight", 1.0) for source, target in zip(path, path[1:])]
                weight = min(edge_weights, default=1.0)
                total_mass += weight
                if tuple(path) not in observed:
                    phantom += 1
                    phantom_mass += weight
                if total >= budget:
                    exact = False
                    break
            if not exact:
                break
        if not exact:
            break
    return {
        "phantom_rate": (phantom / total) if total else 0.0,
        "phantom_mass": (phantom_mass / total_mass) if total_mass else 0.0,
        "enumerated_paths": total,
        "enumeration_exact": exact,
    }


def _fit(builder, X):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dot = builder.fit(X)
    graph, _ = builder.to_networkx(dot)
    log = builder._extract_trace_log(X)
    return graph, log


def run_task(task):
    dataset_name, model_name, n_estimators, seed, commit, path_budget = task
    row = {"dataset": dataset_name, "model": model_name, "n_estimators": n_estimators, "seed": seed,
           "status": "error", "error": "", "k1_seconds": 0.0, "kauto_seconds": 0.0,
           "k1_nodes": 0, "kauto_nodes": 0, "node_ratio": 0.0, "kauto": "",
           "k1_phantom_rate": "", "k1_phantom_mass": "", "kauto_phantom_rate": "",
           "kauto_phantom_mass": "", "k1_enumeration_exact": False,
           "kauto_enumeration_exact": False, "k1_paths": 0, "kauto_paths": 0,
           "git_commit": commit}
    try:
        dataset = DATASETS[dataset_name](as_frame=False)
        model = _make_model(model_name, n_estimators, seed).fit(dataset.data, dataset.target)
        from dpg.core import DecisionPredicateGraph

        started = time.perf_counter()
        legacy = DecisionPredicateGraph(model, dataset.feature_names,
                                        target_names=[str(x) for x in np.unique(dataset.target)],
                                        dpg_config=_config(1))
        graph1, log1 = _fit(legacy, dataset.data)
        row["k1_seconds"] = time.perf_counter() - started
        metrics1 = _phantom_metrics(legacy, graph1, log1, budget=path_budget)

        started = time.perf_counter()
        contextual = DecisionPredicateGraph(model, dataset.feature_names,
                                             target_names=[str(x) for x in np.unique(dataset.target)],
                                             dpg_config=_config("auto"))
        graphk, logk = _fit(contextual, dataset.data)
        row["kauto_seconds"] = time.perf_counter() - started
        metricsk = _phantom_metrics(contextual, graphk, logk, budget=path_budget)
        row.update({"status": "ok", "k1_nodes": graph1.number_of_nodes(), "kauto_nodes": graphk.number_of_nodes(),
                    "node_ratio": graphk.number_of_nodes() / max(1, graph1.number_of_nodes()),
                    "kauto": str(contextual.get_context_order()),
                    "k1_phantom_rate": metrics1["phantom_rate"], "k1_phantom_mass": metrics1["phantom_mass"],
                    "kauto_phantom_rate": metricsk["phantom_rate"], "kauto_phantom_mass": metricsk["phantom_mass"],
                    "k1_enumeration_exact": metrics1["enumeration_exact"],
                    "kauto_enumeration_exact": metricsk["enumeration_exact"],
                    "k1_paths": metrics1["enumerated_paths"], "kauto_paths": metricsk["enumerated_paths"]})
    except Exception as exc:  # noqa: BLE001 - preserve failed cells
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/dpg_0_3_0/results/e2_context_phantoms.csv"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DPG_WORKERS", os.cpu_count() or 1)))
    parser.add_argument("--notify-every", type=int, default=25)
    parser.add_argument("--path-budget", type=int, default=20000)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--learners", default="5,10,25,50,100")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    learners_list = [int(x) for x in args.learners.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    unknown = (set(datasets) - set(DATASETS)) | (set(models) - set(MODELS))
    if unknown or args.path_budget < 1:
        raise SystemExit(f"Invalid E2 selection or path budget: {sorted(unknown)} / {args.path_budget}")
    tasks = [(dataset, model, learners, seed, commit, args.path_budget)
             for dataset in datasets for model in models
             for learners in learners_list for seed in seeds]
    fields = ["dataset", "model", "n_estimators", "seed", "status", "error", "k1_seconds", "kauto_seconds",
              "k1_nodes", "kauto_nodes", "node_ratio", "kauto", "k1_phantom_rate", "k1_phantom_mass",
              "kauto_phantom_rate", "kauto_phantom_mass", "k1_enumeration_exact", "kauto_enumeration_exact",
              "k1_paths", "kauto_paths", "git_commit"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.notify_telegram import send_message
    send_message(f"DPG E2 started: {len(tasks)} tasks, {args.workers} workers.")
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
                    message = f"DPG E2 progress: {completed}/{len(tasks)} complete; {ok} ok; output={args.output}"
                    print(message, flush=True)
                    send_message(message)
    message = f"DPG E2 finished: {ok}/{len(tasks)} ok; output={args.output}"
    print(message, flush=True)
    send_message(message)


if __name__ == "__main__":
    main()
