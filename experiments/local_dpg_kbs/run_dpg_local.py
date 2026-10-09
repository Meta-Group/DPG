"""Main KBS run: trace-indexed local DPGs on the validation-selected black box.

For every test sample this records the local-DPG summary (route faithfulness,
contrastive support, critical predicate), model-native uncertainty baselines,
and two validations of the critical predicate against controls:

  * whole-model intervention: cross the predicate's threshold and re-evaluate;
  * neighbour separation: share of the k nearest training points with a
    different (or the same) model prediction that lie across the threshold.

Controls are executed predicates of the same sample chosen at random, by
smallest slack (nearest threshold), by most trees (most frequent), and by the
top |TreeSHAP| feature (when shap is installed and the family is supported).

Writes results/dpg_local/<dataset>__<family>__s<seed>.csv and a .json sidecar.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from multiprocessing import Pool
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from common import (
    DATASETS, FAMILIES, RESULTS, SEEDS, NeighbourIndex, build_model, limit_native_threads,
    load_split, parse_list, uncertainty_features, violation_rate, write_json,
)

CONTROLS = ("critical", "random", "nearest", "frequent", "shap_top")


def _shap_top_features(model: Any, family: str, X: np.ndarray, pred_idx: np.ndarray) -> Optional[np.ndarray]:
    if family not in {"rf", "et", "gbm"}:
        return None
    try:
        import shap
    except ImportError:
        return None
    try:
        values = shap.TreeExplainer(model).shap_values(X)
    except Exception:  # e.g. TreeSHAP does not support multiclass GradientBoostingClassifier
        return None
    values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    if values.ndim == 2:  # binary GBM: one log-odds column; ranking by |phi| is class-independent
        return np.abs(values).argmax(axis=1)
    return np.abs(values[np.arange(len(X)), :, pred_idx]).argmax(axis=1)


def _choose(local, rng: np.random.Generator, shap_feature: Optional[int]) -> Dict[str, Any]:
    pivots = local.pivots
    if not pivots:
        return {}
    chosen = {
        "critical": local.critical_predicate,
        "random": pivots[int(rng.integers(len(pivots)))],
        "nearest": min(pivots, key=lambda p: (p.slack, p.label)),
        "frequent": min(pivots, key=lambda p: (-len(p.trees), p.slack, p.label)),
    }
    if shap_feature is not None:
        on_feature = [p for p in pivots if p.feature_index == shap_feature]
        chosen["shap_top"] = min(on_feature, key=lambda p: (p.slack, p.label)) if on_feature else None
    return chosen


def run(job) -> str:
    dataset, family, seed, max_samples, k_neighbours = job
    tag = f"{dataset}__{family}__s{seed}"
    out_csv = RESULTS / "dpg_local" / f"{tag}.csv"
    if out_csv.exists():
        return f"{tag} cached"
    try:
        from dpg.local_dpg import build_local_dpg, intervene

        selection = json.loads((RESULTS / "selection" / f"{tag}.json").read_text())
        params = selection["selected"]["params"]
        split = load_split(dataset)
        t0 = time.time()
        model = build_model(family, params, seed).fit(split.X_train, split.y_train)
        fit_seconds = time.time() - t0
        class_names = [str(c) for c in model.classes_]
        pred_test = model.predict(split.X_test)
        neighbours = NeighbourIndex(split.X_train, model.predict(split.X_train), k=k_neighbours)

        n_eval = len(split.X_test) if max_samples <= 0 else min(max_samples, len(split.X_test))
        pred_idx = np.searchsorted(model.classes_, pred_test[:n_eval])
        t0 = time.time()
        shap_top = _shap_top_features(model, family, split.X_test[:n_eval], pred_idx)
        shap_seconds = time.time() - t0 if shap_top is not None else None

        rows = []
        for i in range(n_eval):
            x = split.X_test[i]
            t0 = time.time()
            local = build_local_dpg(model, x, split.feature_names, class_names, context_order="auto", sample_id=i)
            runtime_ms = 1000 * (time.time() - t0)
            s = local.summary()
            support = np.array([local.class_support[c] for c in class_names])
            row: Dict[str, Any] = {"dataset": dataset, "family": family, "seed": seed, "i": i,
                                   "y": str(split.y_test[i]), "runtime_ms": runtime_ms, **s,
                                   **uncertainty_features(support)}
            row["model_error"] = int(row["model_prediction"] != row["y"])
            outcomes = pd.Series([t.outcome for t in local.traces]).value_counts()
            if family in {"rf", "et", "bagging"}:
                shares = outcomes.values / len(local.traces)
                row["u_vote_margin"] = float(shares[0] - (shares[1] if len(shares) > 1 else 0.0))
            pred_labels = {st.label for t in local.traces_for_class(local.predicted_class) for st in t.steps}
            comp_labels = {st.label for t in local.traces_for_class(local.top_competitor) for st in t.steps} if local.top_competitor else set()
            row.update(
                k1_predicate_nodes=len({st.label for t in local.traces for st in t.steps}),
                mean_trace_length=float(np.mean([len(t.steps) for t in local.traces])),
                n_features_used=len({st.feature_index for t in local.traces for st in t.steps}),
                n_competitor_traces=int(outcomes.get(local.top_competitor, 0)) if local.top_competitor else 0,
                n_contested_predicates=len(pred_labels & comp_labels),
            )

            comp_idx = class_names.index(local.top_competitor) if local.top_competitor else None
            diff_nb = neighbours.neighbours(x, pred_test[i], same=False)
            same_nb = neighbours.neighbours(x, pred_test[i], same=True)
            rng = np.random.default_rng([seed, i])
            chosen = _choose(local, rng, int(shap_top[i]) if shap_top is not None else None)
            for name in CONTROLS:
                p = chosen.get(name)
                if p is None or comp_idx is None:
                    continue
                effect = intervene(model, x, p, comp_idx)
                row.update({
                    f"{name}_feature": p.feature, f"{name}_slack": p.slack, f"{name}_n_trees": len(p.trees),
                    f"{name}_first_order_gain": p.competitor_gain,
                    f"{name}_delta_comp": effect["target_delta"], f"{name}_flip": int(effect["flipped"]),
                    f"{name}_sep_diff": violation_rate(diff_nb, p.feature_index, p.threshold, p.operator),
                    f"{name}_sep_same": violation_rate(same_nb, p.feature_index, p.threshold, p.operator),
                })
            if shap_top is not None:
                row["shap_top_on_trace"] = int(chosen.get("shap_top") is not None)
            rows.append(row)

        df = pd.DataFrame(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv.with_suffix(".csv.tmp"), index=False)
        out_csv.with_suffix(".csv.tmp").replace(out_csv)
        write_json(out_csv.with_suffix(".json"), {
            "dataset": dataset, "family": family, "seed": seed, "params": params,
            "test_accuracy": float(accuracy_score(split.y_test, pred_test)),
            "n_test": len(split.X_test), "n_evaluated": n_eval, "fit_seconds": fit_seconds,
            "shap_seconds": shap_seconds, "k_neighbours": k_neighbours,
            "mean_runtime_ms": float(df.runtime_ms.mean()),
        })
        return f"{tag} ok n={n_eval} acc={accuracy_score(split.y_test, pred_test):.3f} ms={df.runtime_ms.mean():.1f}"
    except Exception:  # keep the pool alive; the failure is logged for rerun
        err = RESULTS / "dpg_local" / f"{tag}.error.txt"
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc())
        return f"{tag} FAILED (see {err.name})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets")
    ap.add_argument("--families")
    ap.add_argument("--seeds")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all test samples")
    ap.add_argument("--k_neighbours", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    limit_native_threads()
    jobs = [(d, f, s, a.max_samples, a.k_neighbours) for s in parse_list(a.seeds, SEEDS, int)
            for f in parse_list(a.families, FAMILIES) for d in parse_list(a.datasets, DATASETS)]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for message in pool.imap_unordered(run, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
