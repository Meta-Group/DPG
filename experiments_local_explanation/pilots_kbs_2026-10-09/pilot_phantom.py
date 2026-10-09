"""Pilot C: phantom routes in merged local DPGs (k=1) and local context order k*."""
import sys
from multiprocessing import Pool
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# dpg.context_order ships with DPG >= 0.3.0 (origin/main). The pilot ran against a
# verbatim copy of v0.3.4's module because this branch predates it.
sys.path.insert(0, str(Path(__file__).parent))
try:
    from dpg.context_order import path_violations, resolve_context_order, _node_windows
except ImportError:
    from context_order_v034 import path_violations, resolve_context_order, _node_windows

DATA = Path("/home/barbon/Python/DPG/experiments_local_explanation/data_numeric")

def traces_for(rf, x, dec=6):
    out = []
    for tree in rf.estimators_:
        tr = tree.tree_
        path = tree.decision_path(x[None])[0].indices
        labs = []
        for a, b in zip(path, path[1:]):
            j, th = tr.feature[a], round(float(tr.threshold[a]), dec)
            labs.append(f"f{j} {'<=' if b == tr.children_left[a] else '>'} {th}")
        labs.append(f"Class {int(tr.value[path[-1]].argmax())}")
        out.append(tuple(labs))
    return out

def merged_graph(traces, k):
    g = nx.DiGraph()
    for t in traces:
        nodes = _node_windows(t, k)
        g.add_node(("ROOT",))
        prev = ("ROOT",)
        for n in nodes:
            g.add_edge(prev, n); prev = n
    return g

def n_routes(g):
    order = list(nx.topological_sort(g))
    cnt = {n: 0 for n in g}; cnt[("ROOT",)] = 1
    for n in order:
        for s in g.successors(n):
            cnt[s] += cnt[n]
    return sum(c for n, c in cnt.items() if g.out_degree(n) == 0)

def run(a):
    ds, n_est, depth, seed = a
    d = DATA / ds
    Xtr = np.load(d / "X_train.npy", allow_pickle=True).astype(float); ytr = np.load(d / "y_train.npy", allow_pickle=True)
    Xte = np.load(d / "X_test.npy", allow_pickle=True).astype(float)
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=seed, n_jobs=1).fit(Xtr, ytr)
    rows = []
    for i, x in enumerate(Xte[:300]):
        tr = traces_for(rf, x)
        distinct = len(set(tr))
        g1 = merged_graph(tr, 1)
        cyc = not nx.is_directed_acyclic_graph(g1)
        prec = distinct / n_routes(g1) if not cyc else np.nan
        kstar, hist = resolve_context_order(tr)
        gk = merged_graph(tr, kstar)
        rows.append(dict(dataset=ds, i=i, viol_k1=hist[1], phantom_k1=hist[1] > 0, cyclic_k1=cyc,
                         route_precision_k1=prec, kstar=kstar, nodes_k1=g1.number_of_nodes() - 1,
                         nodes_kstar=gk.number_of_nodes() - 1, distinct_traces=distinct))
    return pd.DataFrame(rows).assign(n_est=n_est, depth=depth)

if __name__ == "__main__":
    n_est, depth = int(sys.argv[1]), (None if sys.argv[2] == "None" else int(sys.argv[2]))
    ds = ["banknote-authentication", "breast_cancer", "diabetes", "digits", "ionosphere", "iris", "isolet", "madelon", "phoneme", "qsar-biodeg", "segment", "spambase", "vehicle", "wine"]
    with Pool(2) as p:
        df = pd.concat(p.map(run, [(x, n_est, depth, 27) for x in ds]))
    df.to_csv(f"phantom_n{n_est}_d{depth}.csv", index=False)
    s = df.groupby("dataset").agg(phantom_k1=("phantom_k1", "mean"), cyclic_k1=("cyclic_k1", "mean"),
        route_prec_k1=("route_precision_k1", "mean"), kstar_mean=("kstar", "mean"), kstar_max=("kstar", "max"),
        nodes_k1=("nodes_k1", "mean"), nodes_kstar=("nodes_kstar", "mean"))
    pd.set_option("display.width", 200)
    print(f"== n_estimators={n_est} max_depth={depth} (<=300 test samples/dataset) ==")
    print(s.round(3)); print("MEAN", s.mean().round(3).to_dict())
