"""Pilot B: structural fragility signals read off executed paths.

For every tree t and every predicate on the executed path of x, build the minimal
counterfactual x' that crosses that one threshold (x_j := theta -/+ eps) and re-run
the tree. A predicate is *pivotal* if the tree's class distribution moves toward the
forest's top competitor. Per-sample features (data are standardized, so slack is in
SD units):
  - min_slack_pivot: smallest |x_j - theta| among pivotal predicates
  - frag_d: fraction of trees with a pivotal predicate within slack d
  - comp_gain_d: forest-level competitor probability gain reachable by single flips within d
Critical predicate (redefined): predicate (feature, theta) with the largest aggregated
competitor gain across trees. Validated by intervention on the whole forest vs a random
executed predicate (control).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA = Path("/home/barbon/Python/DPG/experiments_local_explanation/data_numeric")
EPS = 1e-6
DELTAS = (0.1, 0.25, 0.5)


def run(args):
    dataset, n_est, depth, seed, out = args
    d = DATA / dataset
    Xtr = np.load(d / "X_train.npy", allow_pickle=True).astype(float)
    ytr = np.load(d / "y_train.npy", allow_pickle=True)
    Xte = np.load(d / "X_test.npy", allow_pickle=True).astype(float)
    yte = np.load(d / "y_test.npy", allow_pickle=True)
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=seed, n_jobs=1).fit(Xtr, ytr)
    B, C = len(rf.estimators_), len(rf.classes_)
    P = rf.predict_proba(Xte)
    rng = np.random.default_rng(seed)
    rows = []
    for i, x in enumerate(Xte):
        order = np.argsort(P[i])[::-1]
        pred, comp = order[0], (order[1] if C > 1 else order[0])
        pivots = []  # (tree, feature, theta, slack, comp_gain, side_value)
        executed = []
        for t, tree in enumerate(rf.estimators_):
            tr = tree.tree_
            path = tree.decision_path(x[None])[0].indices
            base = tree.predict_proba(x[None])[0]
            cands = []
            for node in path:
                if tr.children_left[node] == tr.children_right[node]:
                    continue
                j, th = tr.feature[node], tr.threshold[node]
                xv = x.copy()
                xv[j] = th + EPS if x[j] <= th else th - EPS
                cands.append((j, th, abs(x[j] - th), xv[j]))
            executed += [(t, *c) for c in cands]
            if not cands:
                continue
            Xc = np.array([np.where(np.arange(len(x)) == c[0], c[3], x) for c in cands])
            Pc = tree.predict_proba(Xc)
            for c, pc in zip(cands, Pc):
                gain = (pc[comp] - base[comp]) / B
                if gain > 0:
                    pivots.append((t, c[0], c[1], c[2], gain, c[3]))
        r = {"dataset": dataset, "i": i, "y": yte[i], "pred": rf.classes_[pred],
             "f_maxprob": P[i, pred], "f_margin": P[i, pred] - P[i, comp]}
        slacks = np.array([p[3] for p in pivots]) if pivots else np.array([])
        r["min_slack_pivot"] = slacks.min() if len(slacks) else 10.0
        for dl in DELTAS:
            sel = [p for p in pivots if p[3] < dl]
            r[f"frag_{dl}"] = len({p[0] for p in sel}) / B
            best = {}
            for p in sel:
                best[p[0]] = max(best.get(p[0], 0), p[4])
            r[f"comp_gain_{dl}"] = sum(best.values())
        # executed-path slack summary (no pivot condition)
        ex_sl = np.array([e[3] for e in executed])
        r["mean_exec_slack"] = ex_sl.mean()
        r["min_exec_slack"] = ex_sl.min()
        # critical predicate = (feature, rounded theta) with max aggregated competitor gain
        agg = {}
        for p in pivots:
            key = (p[1], round(p[2], 6))
            g, s, v = agg.get(key, (0.0, 1e9, None))
            agg[key] = (g + p[4], min(s, p[3]), p[5])
        r["has_critical"] = bool(agg)
        r["crit_gain_pred"] = 0.0
        r["crit_dp_comp"] = np.nan
        r["crit_flip"] = np.nan
        r["ctrl_dp_comp"] = np.nan
        r["ctrl_flip"] = np.nan
        if agg:
            (j, th), (g, s, v) = max(agg.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
            xc = x.copy(); xc[j] = v
            pc = rf.predict_proba(xc[None])[0]
            r.update(crit_gain_pred=g, crit_slack=s, crit_feature=int(j),
                     crit_dp_comp=pc[comp] - P[i, comp], crit_flip=float(pc.argmax() != pred))
            # control: random executed predicate (any tree), same single-threshold crossing
            e = executed[rng.integers(len(executed))]
            xr = x.copy(); xr[e[1]] = e[4]
            pr = rf.predict_proba(xr[None])[0]
            r.update(ctrl_dp_comp=pr[comp] - P[i, comp], ctrl_flip=float(pr.argmax() != pred),
                     ctrl_slack=e[3])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / f"{dataset}_n{n_est}_d{depth}_s{seed}.csv", index=False)
    return dataset, n_est, depth, seed, len(df)


if __name__ == "__main__":
    n_est, depth, seeds, out = int(sys.argv[1]), (None if sys.argv[2] == "None" else int(sys.argv[2])), sys.argv[3], sys.argv[4]
    Path(out).mkdir(parents=True, exist_ok=True)
    ds = ["banknote-authentication", "breast_cancer", "diabetes", "digits", "ionosphere", "iris",
          "isolet", "madelon", "phoneme", "qsar-biodeg", "segment", "spambase", "vehicle", "wine"]
    jobs = [(x, n_est, depth, int(s), out) for s in seeds.split(",") for x in ds]
    jobs.sort(key=lambda j: -(DATA / j[0] / "X_test.npy").stat().st_size)
    with Pool(6) as pool:
        for res in pool.imap_unordered(run, jobs):
            print(res, flush=True)
