"""RQ1 scaling study: route faithfulness, graph size and cost vs ensemble size/depth.

Fixed (not validation-selected) grids on representative datasets, DPG only.
Writes results/scaling/<dataset>__<family>__n<trees>__d<depth>__s<seed>.csv.
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import RESULTS, build_model, limit_native_threads, load_split, parse_list

DEFAULT_DATASETS = ["iris", "banknote-authentication", "vehicle", "spambase", "madelon", "isolet"]
GRID = {
    "rf": {"n_estimators": [25, 50, 100, 200, 400], "max_depth": [4, 8, 12, None]},
    "gbm": {"n_estimators": [25, 50, 100, 200], "max_depth": [2, 3, 5, 8]},
}


def run(job) -> str:
    dataset, family, n_estimators, depth, seed, max_samples = job
    tag = f"{dataset}__{family}__n{n_estimators}__d{depth}__s{seed}"
    out = RESULTS / "scaling" / f"{tag}.csv"
    if out.exists():
        return f"{tag} cached"
    from dpg.local_dpg import build_local_dpg

    split = load_split(dataset)
    params = {"n_estimators": n_estimators, "max_depth": depth}
    if family == "gbm":
        params["learning_rate"] = 0.1
    t0 = time.time()
    model = build_model(family, params, seed).fit(split.X_train, split.y_train)
    fit_s = time.time() - t0
    classes = [str(c) for c in model.classes_]
    n_trees = int(np.size(model.estimators_))
    rows = []
    for i in range(min(max_samples, len(split.X_test))):
        x = split.X_test[i]
        t0 = time.time()
        local = build_local_dpg(model, x, split.feature_names, classes, compute_pivots=False)
        build_ms = 1000 * (time.time() - t0)
        t0 = time.time()
        build_local_dpg(model, x, split.feature_names, classes, compute_pivots=True)
        pivots_ms = 1000 * (time.time() - t0) - build_ms
        s = local.summary()
        rows.append({**s, "dataset": dataset, "family": family, "n_estimators": n_estimators,
                     "max_depth": depth, "n_trees": n_trees, "seed": seed, "i": i,
                     "k1_predicate_nodes": len({st.label for t in local.traces for st in t.steps}),
                     "mean_trace_length": float(np.mean([len(t.steps) for t in local.traces])),
                     "build_ms": build_ms, "pivots_ms": max(pivots_ms, 0.0), "fit_s": fit_s})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return f"{tag} ok build_ms={np.mean([r['build_ms'] for r in rows]):.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets")
    ap.add_argument("--families", default="rf,gbm")
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    limit_native_threads()
    depth = lambda d: None if d in (None, "None") else int(d)
    jobs = [(d, f, n, md, a.seed, a.max_samples)
            for f in parse_list(a.families, GRID) for d in parse_list(a.datasets, DEFAULT_DATASETS)
            for n in GRID[f]["n_estimators"] for md in map(depth, GRID[f]["max_depth"])]
    jobs.sort(key=lambda j: -(j[2] * (j[3] or 16)))  # most expensive first
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for message in pool.imap_unordered(run, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
