"""Select one black box per (dataset, family, seed) on validation accuracy only.

Writes results/selection/<dataset>__<family>__s<seed>.json. Ties keep the first
(shallowest) configuration in the grid.
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

from sklearn.metrics import accuracy_score

from common import DATASETS, FAMILIES, GRIDS, RESULTS, SEEDS, build_model, config_id, limit_native_threads, load_split, parse_list, write_json


def select(job):
    dataset, family, seed = job
    out = RESULTS / "selection" / f"{dataset}__{family}__s{seed}.json"
    if out.exists():
        return job, "cached"
    split = load_split(dataset)
    rows = []
    for params in GRIDS[family]:
        t0 = time.time()
        model = build_model(family, params, seed).fit(split.X_fit, split.y_fit)
        rows.append({"params": params, "config_id": config_id(params),
                     "val_accuracy": float(accuracy_score(split.y_val, model.predict(split.X_val))),
                     "fit_seconds": time.time() - t0})
    best = max(rows, key=lambda r: r["val_accuracy"])  # max() keeps the first of tied maxima
    write_json(out, {"dataset": dataset, "family": family, "seed": seed, "criterion": "validation_accuracy",
                     "selected": best, "candidates": rows})
    return job, f"ok {best['config_id']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets")
    ap.add_argument("--families")
    ap.add_argument("--seeds")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    limit_native_threads()
    jobs = [(d, f, s) for s in parse_list(a.seeds, SEEDS, int)
            for f in parse_list(a.families, FAMILIES) for d in parse_list(a.datasets, DATASETS)]
    with Pool(a.workers) as pool:
        for job, result in pool.imap_unordered(select, jobs):
            print(*job, result, flush=True)


if __name__ == "__main__":
    main()
