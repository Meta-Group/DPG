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
pd.set_option("display.max_columns", 40)

df = pd.concat([pd.read_csv(p) for p in Path(sys.argv[1]).glob("*.csv")], ignore_index=True)
for c in ["y", "soft_pred", "hard_pred", "dpg_class"]:
    df[c] = df[c].astype(str)
df["model_error"] = (df.soft_pred != df.y).astype(int)
df["disagree"] = (df.dpg_class != df.soft_pred).astype(int)
df["conf_wo_vote"] = df.coverage * (df.support_margin + df.concentration) / 2.0

# Higher score = higher risk.
RISK = {
    "forest: 1-max prob": lambda d: 1 - d.f_maxprob,
    "forest: -prob margin": lambda d: -d.f_margin,
    "forest: entropy": lambda d: d.f_entropy,
    "forest: -hard-vote margin": lambda d: -d.v_margin,
    "DPG: 1-confidence (AAAI)": lambda d: 1 - d.confidence,
    "DPG: 1-conf w/o vote agr.": lambda d: 1 - d.conf_wo_vote,
    "DPG: 1-vote agreement": lambda d: 1 - d.vote_agreement,
    "DPG: -support margin": lambda d: -d.support_margin,
    "DPG: competitor exposure": lambda d: d.competitor_exposure,
    "DPG: -concentration": lambda d: -d.concentration,
}


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) and y.sum() >= 5 else np.nan


print("== Sanity / rates (mean over runs) ==")
g = df.groupby(["dataset", "seed"])
rates = g.agg(
    n=("i", "size"), model_err=("model_error", "mean"), disagree=("disagree", "mean"),
    hard_ne_soft=("hard_pred", lambda s: (s != df.loc[s.index, "soft_pred"]).mean()),
    dpg_ne_hard=("dpg_class", lambda s: (s != df.loc[s.index, "hard_pred"]).mean()),
    edge_recall=("edge_recall", "mean"), edge_rec_lt1=("edge_recall", lambda s: (s < 0.999999).mean()),
    critical_n=("critical", "sum"), active_nodes=("n_active_nodes", "mean"),
    ms=("runtime_s", lambda s: 1000 * s.mean()), fit_s=("fit_s", "first"),
).reset_index()
print(rates.groupby("dataset").mean(numeric_only=True).round(3).drop(columns="seed"))
print("pooled:", rates.mean(numeric_only=True).round(4).to_dict())

for target in ["model_error", "disagree"]:
    recs = []
    for (ds, seed), d in g:
        r = {"dataset": ds, "seed": seed, "pos": int(d[target].sum())}
        for k, f in RISK.items():
            r[k] = auc(d[target].values, f(d).values)
        recs.append(r)
    t = pd.DataFrame(recs)
    print(f"\n== AUROC for target={target} (dataset means over seeds) ==")
    per_ds = t.groupby("dataset").mean(numeric_only=True).drop(columns="seed")
    print(per_ds.round(3).T)
    print("MEAN over datasets:")
    print(per_ds.drop(columns="pos").mean().round(4).to_string())

# Incremental value: does adding DPG features to forest-uncertainty features improve
# out-of-fold model-error AUROC? Vote agreement is excluded (it is a forest vote statistic).
BASE = ["f_margin", "f_entropy", "v_margin"]
DPGF = ["support_margin", "concentration", "competitor_exposure", "coverage"]
recs = []
for (ds, seed), d in g:
    y = d.model_error.values
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        continue
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    out = {}
    for name, cols in [("base", BASE), ("base+dpg", BASE + DPGF), ("dpg", DPGF)]:
        oof = np.zeros(len(y))
        X = d[cols].fillna(0).values
        for tr, te in skf.split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        out[name] = roc_auc_score(y, oof)
    recs.append({"dataset": ds, "seed": seed, **out})
inc = pd.DataFrame(recs).groupby("dataset").mean(numeric_only=True).drop(columns="seed")
inc["delta"] = inc["base+dpg"] - inc["base"]
print("\n== Incremental model-error AUROC (5-fold OOF logistic) ==")
print(inc.round(4))
print("mean:", inc.mean().round(4).to_dict())
if len(inc) >= 5:
    print("Wilcoxon base+dpg vs base:", wilcoxon(inc["base+dpg"], inc["base"]))
