import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 220)
df = pd.concat([pd.read_csv(p) for p in Path(sys.argv[1]).glob("*.csv")], ignore_index=True)
df["seed"] = [p for p in df.get("seed", [0] * len(df))] if "seed" in df else 0
df["err"] = (df.y.astype(str) != df.pred.astype(str)).astype(int)
files = list(Path(sys.argv[1]).glob("*.csv"))
df = pd.concat([pd.read_csv(p).assign(run=p.stem) for p in files], ignore_index=True)
df["err"] = (df.y.astype(str) != df.pred.astype(str)).astype(int)

RISK = {
    "forest 1-maxprob": lambda d: 1 - d.f_maxprob,
    "forest -margin": lambda d: -d.f_margin,
    "-min_slack_pivot": lambda d: -d.min_slack_pivot,
    "frag_0.1": lambda d: d["frag_0.1"],
    "frag_0.25": lambda d: d["frag_0.25"],
    "comp_gain_0.25": lambda d: d["comp_gain_0.25"],
    "comp_gain_0.5": lambda d: d["comp_gain_0.5"],
    "-min_exec_slack": lambda d: -d.min_exec_slack,
}
recs = []
for (run, ds), d in df.groupby(["run", "dataset"]):
    y = d.err.values
    if y.sum() < 5 or y.sum() == len(y):
        continue
    recs.append({"dataset": ds, **{k: roc_auc_score(y, f(d)) for k, f in RISK.items()}})
t = pd.DataFrame(recs).groupby("dataset").mean()
print("== model-error AUROC ==")
print(t.round(3))
print(t.mean().round(4).to_string())

BASE = ["f_margin", "f_maxprob"]
FR = ["min_slack_pivot", "frag_0.1", "frag_0.25", "frag_0.5", "comp_gain_0.1", "comp_gain_0.25", "comp_gain_0.5", "min_exec_slack", "mean_exec_slack"]
recs = []
for (run, ds), d in df.groupby(["run", "dataset"]):
    y = d.err.values
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        continue
    out = {}
    for name, cols in [("base", BASE), ("base+frag", BASE + FR)]:
        X = d[cols].fillna(10).values
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        out[name] = roc_auc_score(y, oof)
    recs.append({"dataset": ds, **out})
inc = pd.DataFrame(recs).groupby("dataset").mean()
inc["delta"] = inc["base+frag"] - inc["base"]
print("\n== incremental model-error AUROC (OOF logistic) ==")
print(inc.round(4))
print(inc.mean().round(4).to_dict())
print(wilcoxon(inc["base+frag"], inc["base"]))

print("\n== redefined critical predicate: coverage and intervention vs random-predicate control ==")
c = df.groupby("dataset").agg(
    coverage=("has_critical", "mean"),
    crit_dp=("crit_dp_comp", "mean"), ctrl_dp=("ctrl_dp_comp", "mean"),
    crit_flip=("crit_flip", "mean"), ctrl_flip=("ctrl_flip", "mean"),
    crit_slack=("crit_slack", "median"), ctrl_slack=("ctrl_slack", "median"),
)
print(c.round(4))
print(c.mean().round(4).to_dict())
print("dp: ", wilcoxon(c.crit_dp, c.ctrl_dp))
print("flip:", wilcoxon(c.crit_flip, c.ctrl_flip))
