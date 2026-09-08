#!/usr/bin/env python3
"""Run E0 invariant and E1 routing-fidelity validation for DPG 0.3.0."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
PRECISIONS = (1, 2, 4, 6, "auto")


def _make_model(name, n_estimators, seed):
    kwargs = {"n_estimators": n_estimators, "random_state": seed}
    if name in {"RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier"}:
        kwargs["n_jobs"] = 1
    return MODELS[name](**kwargs)


def _config(decimal_threshold):
    return {
        "dpg": {
            "default": {"perc_var": 1e-9, "decimal_threshold": decimal_threshold, "n_jobs": 1},
            "graph_construction": {"mode": "execution_trace", "context_order": 1},
        }
    }


def _misroute_rate(builder, X):
    total = 0
    mismatches = 0
    for tree_index, tree in enumerate(builder.model.estimators_):
        tree_ = tree.tree_
        for sample in np.asarray(X):
            indicator = tree.decision_path(np.asarray(sample).reshape(1, -1))
            path = indicator.indices[indicator.indptr[0] : indicator.indptr[1]]
            labels = builder._trace_tree_labels(tree_index, tree, sample)
            for position, node in enumerate(path[:-1]):
                feature = int(tree_.feature[node])
                expected = "<=" if int(path[position + 1]) == int(tree_.children_left[node]) else ">"
                prefix = f"{builder.feature_names[feature]} {expected} "
                total += 1
                mismatches += not labels[position].startswith(prefix)
    return float(mismatches / total) if total else 0.0


def run_task(task):
    dataset_name, model_name, n_estimators, seed, decimal_threshold, commit = task
    row = {
        "phase": "E0-E1", "dataset": dataset_name, "model": model_name,
        "n_estimators": n_estimators, "seed": seed,
        "decimal_threshold": str(decimal_threshold), "status": "error", "error": "",
        "identity_k1": False, "sink_count_ok": False, "edge_mass_ok": False,
        "misroute_rate": "", "resolved_k": "", "violations_at_k": "", "warnings": 0,
        "seconds": 0.0, "git_commit": commit,
    }
    started = time.perf_counter()
    try:
        dataset = DATASETS[dataset_name](as_frame=False)
        model = _make_model(model_name, n_estimators, seed).fit(dataset.data, dataset.target)
        from dpg.core import DecisionPredicateGraph

        with warnings.catch_warnings(record=True) as captured, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            builder = DecisionPredicateGraph(
                model, dataset.feature_names,
                target_names=[str(label) for label in np.unique(dataset.target)],
                dpg_config=_config(decimal_threshold),
            )
            log = builder._extract_trace_log(dataset.data)
            dot = builder.fit(dataset.data)
        graph, nodes = builder.to_networkx(dot)
        labels = [label for _, label in nodes]
        sink_labels = {label for label in labels if label.startswith("Class ")}
        expected_mass = sum(max(0, len(sequence) - 1) for sequence in builder._trace_sequences(log))
        actual_mass = sum(data.get("weight", 0.0) for _, _, data in graph.edges(data=True))
        row.update({
            "status": "ok",
            "identity_k1": builder.discover_dfg(log) == builder.discover_dfg_execution_trace(log),
            "sink_count_ok": len(sink_labels) == len(np.unique(dataset.target)),
            "edge_mass_ok": actual_mass == expected_mass,
            "misroute_rate": _misroute_rate(builder, dataset.data),
            "resolved_k": str(builder.get_context_order()),
            "violations_at_k": builder.get_context_order_history().get(builder.get_context_order(), ""),
            "warnings": len(captured), "seconds": time.perf_counter() - started,
        })
    except Exception as exc:  # noqa: BLE001 - failures must be retained as rows
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = time.perf_counter() - started
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/dpg_0_3_0/results/e0_e1_validation.csv"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DPG_WORKERS", os.cpu_count() or 1)))
    parser.add_argument("--notify-every", type=int, default=25)
    args = parser.parse_args()
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    tasks = [
        (dataset, model, learners, seed, precision, commit)
        for dataset in DATASETS for model in MODELS
        for learners in (5, 10, 25) for seed in (0, 1, 2)
        for precision in PRECISIONS
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "phase", "dataset", "model", "n_estimators", "seed", "decimal_threshold", "status", "error",
        "identity_k1", "sink_count_ok", "edge_mass_ok", "misroute_rate", "resolved_k",
        "violations_at_k", "warnings", "seconds", "git_commit",
    ]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.notify_telegram import send_message
    send_message(f"DPG 0.3.0 E0/E1 validation started: {len(tasks)} tasks, {args.workers} workers.")
    print(f"DPG 0.3.0 E0/E1 validation: {len(tasks)} tasks, workers={args.workers}", flush=True)
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
                    message = f"DPG E0/E1 progress: {completed}/{len(tasks)} complete; {ok} ok; output={args.output}"
                    print(message, flush=True)
                    send_message(message)
    message = f"DPG E0/E1 validation finished: {ok}/{len(tasks)} ok; output={args.output}"
    print(message, flush=True)
    send_message(message)


if __name__ == "__main__":
    main()
