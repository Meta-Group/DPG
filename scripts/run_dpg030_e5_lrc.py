#!/usr/bin/env python3
"""Run E5: LRC alignment against ensemble feature_importances_.

Compares execution-trace k=1 against execution-trace auto-k, and compares
the three LRC-to-predicate aggregations (sum, max, weighted_sum) evaluated
against the SAME node-level LRC values from one fitted graph.

Aggregation is a fixed comparison for this analysis, not a knob to pick a
production default from: `sum` (unweighted) is what
`DecisionPredicateGraph.get_predicate_lrc()` already ships (CHANGELOG
0.3.0). `max` and `weighted_sum` live only in `scripts/lrc_aggregation.py`
for this comparison -- see that module's docstring for why picking one by
correlation would be circular. This script reports all three; it does not
recommend switching the shipped default.

Datasets, model families, learner counts, and seeds are fixed for
reproducibility (matches the grid used by E2/E3): iris/wine/breast_cancer x
5 ensemble families x {10, 25, 50} learners x 5 seeds = 225 cells.
BaggingClassifier has no `feature_importances_`; its graph-construction
metrics are still recorded, with status="skipped_no_feature_importances"
for the alignment columns.
"""

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
from scipy.stats import spearmanr
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
AGGREGATIONS = ("sum", "max", "weighted_sum")


def _make_model(name, n_estimators, seed):
    kwargs = {"n_estimators": n_estimators, "random_state": seed}
    if name in {"RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier"}:
        kwargs["n_jobs"] = 1
    return MODELS[name](**kwargs)


def _config(context_order):
    return {"dpg": {"default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
                     "graph_construction": {"mode": "execution_trace", "context_order": context_order}}}


def _fit(builder, X):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dot = builder.fit(X)
    return builder.to_networkx(dot)


def _ranking_overlap(vec_a, vec_b, top_k):
    top_a = set(np.argsort(vec_a)[::-1][:top_k].tolist())
    top_b = set(np.argsort(vec_b)[::-1][:top_k].tolist())
    return len(top_a & top_b) / top_k


def _alignment_metrics(graph, feature_names, importances, top_k):
    """Return {"spearman_<agg>": ..., "overlap_<agg>": ...} for every aggregation."""
    from scripts.lrc_aggregation import feature_scores, predicate_lrc

    out = {}
    for how in AGGREGATIONS:
        pred_scores = predicate_lrc(graph, how)
        feat_scores = feature_scores(pred_scores, feature_names)
        lrc_vec = np.array([feat_scores[name] for name in feature_names])
        if np.allclose(lrc_vec, lrc_vec[0]) or np.allclose(importances, importances[0]):
            rho = float("nan")
        else:
            rho = float(spearmanr(importances, lrc_vec).correlation)
        out[f"spearman_{how}"] = rho
        out[f"overlap_{how}"] = _ranking_overlap(importances, lrc_vec, top_k)
    return out


def run_task(task):
    dataset_name, model_name, n_estimators, seed, commit = task
    row = {
        "dataset": dataset_name, "model": model_name, "n_estimators": n_estimators, "seed": seed,
        "status": "error", "error": "", "has_feature_importances": False, "top_k": 0,
        "k1_seconds": 0.0, "kauto_seconds": 0.0, "kauto": "", "kauto_violations": "",
        "k1_nodes": 0, "k1_edges": 0, "kauto_nodes": 0, "kauto_edges": 0, "node_ratio": 0.0,
        "git_commit": commit,
    }
    for how in AGGREGATIONS:
        row[f"spearman_k1_{how}"] = ""
        row[f"spearman_kauto_{how}"] = ""
        row[f"overlap_k1_{how}"] = ""
        row[f"overlap_kauto_{how}"] = ""
    try:
        dataset = DATASETS[dataset_name](as_frame=False)
        feature_names = list(dataset.feature_names)
        model = _make_model(model_name, n_estimators, seed).fit(dataset.data, dataset.target)
        target_names = [str(x) for x in np.unique(dataset.target)]
        from dpg.core import DecisionPredicateGraph

        started = time.perf_counter()
        k1 = DecisionPredicateGraph(model, feature_names, target_names=target_names, dpg_config=_config(1))
        graph1, _ = _fit(k1, dataset.data)
        row["k1_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        kauto = DecisionPredicateGraph(model, feature_names, target_names=target_names, dpg_config=_config("auto"))
        graphk, _ = _fit(kauto, dataset.data)
        row["kauto_seconds"] = time.perf_counter() - started

        row.update({
            "status": "ok",
            "kauto": str(kauto.get_context_order()),
            "kauto_violations": repr(kauto.get_context_order_history()),
            "k1_nodes": graph1.number_of_nodes(), "k1_edges": graph1.number_of_edges(),
            "kauto_nodes": graphk.number_of_nodes(), "kauto_edges": graphk.number_of_edges(),
            "node_ratio": graphk.number_of_nodes() / max(1, graph1.number_of_nodes()),
        })

        if not hasattr(model, "feature_importances_"):
            row["status"] = "skipped_no_feature_importances"
            return row

        importances = np.asarray(model.feature_importances_, dtype=float)
        top_k = max(1, round(len(feature_names) * 0.25))
        row["top_k"] = top_k

        for graph, prefix in ((graph1, "k1"), (graphk, "kauto")):
            metrics = _alignment_metrics(graph, feature_names, importances, top_k)
            for how in AGGREGATIONS:
                row[f"spearman_{prefix}_{how}"] = metrics[f"spearman_{how}"]
                row[f"overlap_{prefix}_{how}"] = metrics[f"overlap_{how}"]
        row["has_feature_importances"] = True
    except Exception as exc:  # noqa: BLE001 - preserve failed cells
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/dpg_0_3_0/results/e5_lrc_alignment.csv"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DPG_WORKERS", 2)))
    parser.add_argument("--notify-every", type=int, default=25)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--learners", default="10,25,50")
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
    if unknown:
        raise SystemExit(f"Unknown E5 values: {sorted(unknown)}")

    tasks = [(dataset, model, learners, seed, commit)
             for dataset in datasets for model in models
             for learners in learners_list for seed in seeds]
    fields = (
        ["dataset", "model", "n_estimators", "seed", "status", "error", "has_feature_importances", "top_k",
         "k1_seconds", "kauto_seconds", "kauto", "kauto_violations",
         "k1_nodes", "k1_edges", "kauto_nodes", "kauto_edges", "node_ratio"]
        + [f"spearman_k1_{how}" for how in AGGREGATIONS]
        + [f"spearman_kauto_{how}" for how in AGGREGATIONS]
        + [f"overlap_k1_{how}" for how in AGGREGATIONS]
        + [f"overlap_kauto_{how}" for how in AGGREGATIONS]
        + ["git_commit"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.notify_telegram import send_message
    send_message(f"DPG E5 started: {len(tasks)} tasks, {args.workers} workers.")
    completed = ok = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def _drain(row):
            nonlocal completed, ok
            writer.writerow(row)
            handle.flush()
            completed += 1
            ok += row["status"] in ("ok", "skipped_no_feature_importances")
            if completed % args.notify_every == 0 or completed == len(tasks):
                message = f"DPG E5 progress: {completed}/{len(tasks)} complete; {ok} ok; output={args.output}"
                print(message, flush=True)
                send_message(message)

        if args.workers == 1:
            for task in tasks:
                _drain(run_task(task))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_task, task) for task in tasks]
                for future in as_completed(futures):
                    _drain(future.result())
    message = f"DPG E5 finished: {ok}/{len(tasks)} ok; output={args.output}"
    print(message, flush=True)
    send_message(message)


if __name__ == "__main__":
    main()
