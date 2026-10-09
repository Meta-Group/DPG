"""Pilot: do DPG-local (AAAI27 implementation) diagnostics add information beyond
forest-native uncertainty? Addresses AAAI27 reviewers y8XZ (W2) and kpr7.

Runs the AAAI-branch DPGExplainer (execution_trace, top_competitor evidence) on a
fixed RandomForest per dataset and records per-sample DPG scores alongside forest
uncertainty baselines (max prob, prob margin, entropy, hard-vote share/margin).
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO = Path("/home/barbon/Python/DPG")
DATA = REPO / "experiments_local_explanation" / "data_numeric"
sys.path.insert(0, str(REPO))

DATASETS = [
    "banknote-authentication", "breast_cancer", "diabetes", "digits", "ionosphere",
    "iris", "isolet", "madelon", "phoneme", "qsar-biodeg", "segment", "spambase",
    "vehicle", "wine",
]


def run(args):
    dataset, n_estimators, max_depth, seed, out_dir = args
    from dpg.explainer import DPGExplainer  # AAAI branch implementation

    d = DATA / dataset
    X_tr = np.load(d / "X_train.npy", allow_pickle=True).astype(float)
    y_tr = np.load(d / "y_train.npy", allow_pickle=True)
    X_te = np.load(d / "X_test.npy", allow_pickle=True).astype(float)
    y_te = np.load(d / "y_test.npy", allow_pickle=True)
    feats = json.loads((d / "feature_names.json").read_text())

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=1
    ).fit(X_tr, y_tr)
    classes = list(model.classes_)
    proba = model.predict_proba(X_te)
    soft_pred = model.classes_[proba.argmax(1)]
    tree_votes = np.stack([model.classes_[t.predict(X_te).astype(int)] for t in model.estimators_], 1)

    t0 = time.time()
    exp = DPGExplainer(
        model=model,
        feature_names=feats,
        target_names=[str(c) for c in classes],
        dpg_config={"dpg": {
            "default": {"perc_var": 0.0001, "decimal_threshold": 6, "n_jobs": 1},
            "graph_construction": {"mode": "execution_trace"},
            "local_evidence": {"variant": "top_competitor", "base_lambda": 0.8},
        }},
    ).fit(X_tr)
    fit_s = time.time() - t0

    rows = []
    for i in range(len(X_te)):
        p = np.sort(proba[i])[::-1]
        vc = pd.Series(tree_votes[i]).value_counts()
        vshare = vc.values / n_estimators
        hard_pred = vc.index[0]
        t1 = time.time()
        loc = exp.explain_local(X_te[i], sample_id=i)
        rt = time.time() - t1
        sc = loc.sample_confidence
        rows.append({
            "dataset": dataset, "i": i, "y": str(y_te[i]),
            "soft_pred": str(soft_pred[i]), "hard_pred": str(hard_pred),
            "dpg_class": str(loc.majority_vote),
            "f_maxprob": p[0], "f_margin": p[0] - (p[1] if len(p) > 1 else 0.0),
            "f_entropy": float(-(proba[i][proba[i] > 0] * np.log(proba[i][proba[i] > 0])).sum()),
            "v_share": vshare[0], "v_margin": vshare[0] - (vshare[1] if len(vshare) > 1 else 0.0),
            "support_margin": sc.get("support_margin"),
            "concentration": sc.get("predicted_class_concentration_top3"),
            "vote_agreement": sc.get("model_vote_agreement"),
            "coverage": sc.get("trace_coverage_score"),
            "confidence": sc.get("explanation_confidence"),
            "competitor_exposure": sc.get("competitor_exposure"),
            "edge_recall": sc.get("edge_recall"),
            "edge_precision": sc.get("edge_precision"),
            "critical": sc.get("critical_node_label") is not None,
            "n_active_nodes": sc.get("num_active_nodes"),
            "runtime_s": rt,
        })
    df = pd.DataFrame(rows)
    df["fit_s"] = fit_s
    df["n_estimators"], df["max_depth"], df["seed"] = n_estimators, max_depth, seed
    tag = f"{dataset}_n{n_estimators}_d{max_depth}_s{seed}"
    df.to_csv(Path(out_dir) / f"{tag}.csv", index=False)
    return tag, len(df), round(fit_s, 1), round(df.runtime_s.mean() * 1000, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_estimators", type=int, default=20)
    ap.add_argument("--max_depth", type=lambda s: None if s == "None" else int(s), default=4)
    ap.add_argument("--seeds", default="27")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(ds, a.n_estimators, a.max_depth, int(s), a.out)
            for s in a.seeds.split(",") for ds in a.datasets.split(",")]
    # largest first so the pool drains evenly
    jobs.sort(key=lambda j: -(DATA / j[0] / "X_test.npy").stat().st_size)
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(run, jobs):
            print(res, flush=True)
