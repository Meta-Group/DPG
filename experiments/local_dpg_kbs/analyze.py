"""Aggregate KBS runs into paper tables (results/report/*.csv and summary.md).

Units: per-sample rows are averaged within (dataset, family, seed), then over
seeds, giving one value per (dataset, family). Tests across datasets use these
dataset-level values (Wilcoxon signed-rank, Holm-corrected within each table).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RESULTS

CONTROLS = ["critical", "random", "nearest", "frequent", "shap_top", "lime_top", "anchors_top", "lore_top"]
BASE_RISK = {
    "1 - max support": lambda d: 1 - d.u_maxprob,
    "-support margin": lambda d: -d.u_margin,
    "entropy": lambda d: d.u_entropy,
}
STRUCT_RISK = {
    "k1 phantom violations": lambda d: d.route_violations_k1,
    "context order k*": lambda d: d.context_order,
    "contested predicates": lambda d: d.n_contested_predicates,
    "competitor traces": lambda d: d.n_competitor_traces,
    "-critical slack": lambda d: -d.critical_slack,
    "critical first-order gain": lambda d: d.critical_first_order_gain,
}
STRUCT_FEATURES = ["route_violations_k1", "context_order", "n_contested_predicates", "n_competitor_traces",
                   "critical_slack", "critical_first_order_gain", "n_features_used", "mean_trace_length"]


def md_table(df: pd.DataFrame, index: bool = False, digits: int = 3) -> str:
    """Minimal markdown table (avoids the optional tabulate dependency)."""
    if index:
        df = df.reset_index()
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def boot_ci(values, n=10000, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def dataset_level(df: pd.DataFrame, cols) -> pd.DataFrame:
    run = df.groupby(["family", "dataset", "seed"])[cols].mean()
    return run.groupby(["family", "dataset"]).mean().reset_index()


def rq1_route_faithfulness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(phantom_k1=(df.route_violations_k1 > 0).astype(float), cyclic_k1=(~df.k1_acyclic.astype(bool)).astype(float),
                   node_overhead=df.n_nodes / df.k1_predicate_nodes, k_ge_2=(df.context_order >= 2).astype(float))
    cols = ["phantom_k1", "cyclic_k1", "k1_route_precision", "context_order", "k_ge_2", "k1_predicate_nodes", "n_nodes",
            "node_overhead", "is_output_faithful", "runtime_ms"]
    ds = dataset_level(df, cols)
    rows = []
    for fam, g in ds.groupby("family"):
        r = {"family": fam, "n_datasets": len(g)}
        for c in cols:
            lo, hi = boot_ci(g[c])
            r[c] = g[c].mean()
            r[f"{c}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(r)
    return pd.DataFrame(rows)


def rq2_localisation(df: pd.DataFrame):
    metrics = ["delta_comp", "flip", "sep_diff", "sep_same"]
    present = [c for c in CONTROLS if f"{c}_delta_comp" in df]
    for c in present:
        df[f"{c}_sep_contrast"] = df[f"{c}_sep_diff"] - df[f"{c}_sep_same"]
    metrics.append("sep_contrast")
    cols = [f"{c}_{m}" for c in present for m in metrics]
    ds = dataset_level(df, cols)
    summary, tests = [], []
    for fam, g in ds.groupby("family"):
        for c in present:
            summary.append({"family": fam, "control": c, **{m: g[f"{c}_{m}"].mean() for m in metrics},
                            "n_datasets": int(g[f"{c}_delta_comp"].notna().sum())})
        for m in ("delta_comp", "flip", "sep_contrast"):
            block = []
            for c in present[1:]:
                pair = g[[f"critical_{m}", f"{c}_{m}"]].dropna()
                if len(pair) < 5 or np.allclose(pair.iloc[:, 0], pair.iloc[:, 1]):
                    continue
                stat = wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1])
                block.append({"family": fam, "metric": m, "control": c, "n": len(pair),
                              "critical_mean": pair.iloc[:, 0].mean(), "control_mean": pair.iloc[:, 1].mean(),
                              "wins": int((pair.iloc[:, 0] > pair.iloc[:, 1]).sum()), "p": stat.pvalue})
            if block:
                for row, p in zip(block, holm([b["p"] for b in block])):
                    row["p_holm"] = p
                tests.extend(block)
    return pd.DataFrame(summary), pd.DataFrame(tests)


def _auc(y, s):
    if y.sum() < 5 or y.sum() == len(y):
        return np.nan, np.nan
    s = np.nan_to_num(np.asarray(s, dtype=float), nan=np.nanmin(s) if np.isfinite(np.nanmin(s)) else 0.0)
    return roc_auc_score(y, s), average_precision_score(y, s)


def rq3_error_detection(df: pd.DataFrame):
    rows, inc = [], []
    for (fam, ds, seed), d in df.groupby(["family", "dataset", "seed"]):
        y = d.model_error.values
        r = {"family": fam, "dataset": ds, "seed": seed, "prevalence": y.mean(), "n": len(y)}
        risks = dict(BASE_RISK)
        if "u_vote_margin" in d and d.u_vote_margin.notna().all():
            risks["-hard-vote margin"] = lambda d: -d.u_vote_margin
        for name, f in {**risks, **STRUCT_RISK}.items():
            r[f"auroc|{name}"], r[f"auprc|{name}"] = _auc(y, f(d))
        rows.append(r)
        if y.sum() >= 10 and (len(y) - y.sum()) >= 10:
            base = ["u_maxprob", "u_margin", "u_entropy"]
            out = {}
            for name, cols in (("base", base), ("base+structure", base + STRUCT_FEATURES)):
                X = d[cols].fillna(d[cols].median()).fillna(0).values
                oof = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
                    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], y[tr])
                    oof[te] = m.predict_proba(X[te])[:, 1]
                out[name] = roc_auc_score(y, oof)
            inc.append({"family": fam, "dataset": ds, "seed": seed, **out})
    per_run = pd.DataFrame(rows)
    per_ds = per_run.groupby(["family", "dataset"]).mean(numeric_only=True).reset_index()
    table = per_ds.groupby("family").mean(numeric_only=True).drop(columns=["seed"]).T
    incr = pd.DataFrame(inc)
    if len(incr):
        incr = incr.groupby(["family", "dataset"]).mean(numeric_only=True).reset_index()
        incr["delta"] = incr["base+structure"] - incr["base"]
    return table, incr


def rq4_cost(df: pd.DataFrame, extra) -> pd.DataFrame:
    """Runtime (ms) and native explanation size per method, dataset-level means.

    DPG-local runtime includes pivot evaluation. Baselines are evaluated on the
    first samples of fewer seeds (see run_baselines.py), so DPG rows are restricted
    to the same (dataset, family, seed, i) keys when baselines are present.
    """
    keys = ["dataset", "family", "seed", "i"]
    dpg = df[keys + ["runtime_ms", "n_nodes", "k1_predicate_nodes"]]
    if extra is not None:
        dpg = dpg.merge(extra[keys], on=keys)
    rows = [("DPG-local (k*)", dpg.assign(size=dpg.n_nodes)), ("DPG-local (k=1)", dpg.assign(size=dpg.k1_predicate_nodes))]
    if extra is not None:
        for m in ("shap", "lime", "anchors", "lore"):
            if f"{m}_runtime_ms" in extra:
                rows.append((m, extra.assign(runtime_ms=extra[f"{m}_runtime_ms"], size=extra[f"{m}_size"])))
    out = []
    for name, d in rows:
        ds = d.groupby(["family", "dataset", "seed"])[["runtime_ms", "size"]].mean().groupby(["family", "dataset"]).mean()
        for fam, g in ds.groupby(level="family"):
            out.append({"method": name, "family": fam, "n_datasets": len(g),
                        "runtime_ms": g.runtime_ms.mean(), "size": g["size"].mean()})
    return pd.DataFrame(out)


def scaling_table() -> pd.DataFrame:
    files = glob.glob(str(RESULTS / "scaling" / "*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["phantom_k1"] = (df.route_violations_k1 > 0).astype(float)
    df["cyclic_k1"] = (~df.k1_acyclic.astype(bool)).astype(float)
    df["max_depth"] = df.max_depth.fillna(-1).astype(int).replace(-1, "None")
    cols = ["phantom_k1", "cyclic_k1", "k1_route_precision", "context_order", "k1_predicate_nodes", "n_nodes", "build_ms", "pivots_ms"]
    return df.groupby(["family", "n_estimators", "max_depth"])[cols].mean().reset_index()


def blackbox_table() -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(RESULTS / "dpg_local" / "*.json")):
        m = json.loads(Path(p).read_text())
        rows.append({"dataset": m["dataset"], "family": m["family"], "seed": m["seed"], "test_accuracy": m["test_accuracy"],
                     "config": "_".join(f"{k}={v}" for k, v in sorted(m["params"].items()))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RESULTS / "report"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(RESULTS / "dpg_local" / "*.csv"))
    if not files:
        raise SystemExit("No results/dpg_local/*.csv found.")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    base_files = glob.glob(str(RESULTS / "baselines" / "*.csv"))
    if base_files:
        extra = pd.concat([pd.read_csv(p) for p in base_files], ignore_index=True)
        keep = ["dataset", "family", "seed", "i"] + [c for c in extra if c.startswith(("lime_top_", "anchors_top_", "lore_top_"))]
        df = df.merge(extra[keep], on=["dataset", "family", "seed", "i"], how="left")
    else:
        extra = None

    md = ["# KBS local-DPG report", "", f"Rows: {len(df)} samples, {df.dataset.nunique()} datasets, "
          f"{df.family.nunique()} families, {df.seed.nunique()} seeds.", ""]
    bb = blackbox_table()
    bb.to_csv(out / "blackbox_selected.csv", index=False)
    md += ["## Selected black boxes (validation accuracy only)", "",
           bb.groupby(["family", "config"]).size().rename("runs").reset_index().pipe(md_table), ""]

    r1 = rq1_route_faithfulness(df)
    r1.to_csv(out / "rq1_route_faithfulness.csv", index=False)
    md += ["## RQ1 Route faithfulness (dataset-level means, bootstrap 95% CI over datasets)", "", md_table(r1), ""]

    summary, tests = rq2_localisation(df)
    summary.to_csv(out / "rq2_localisation_summary.csv", index=False)
    tests.to_csv(out / "rq2_localisation_tests.csv", index=False)
    md += ["## RQ2 Critical predicate vs controls", "", md_table(summary), "",
           md_table(tests, digits=4) if len(tests) else "(too few datasets for tests)", ""]

    table, incr = rq3_error_detection(df)
    table.to_csv(out / "rq3_error_detection_auc.csv")
    incr.to_csv(out / "rq3_incremental_value.csv", index=False)
    md += ["## RQ3 Error detection: model-native vs structural scores (mean over datasets)", "", md_table(table, index=True), ""]
    if len(incr):
        agg = incr.groupby("family")[["base", "base+structure", "delta"]].mean()
        tests = []
        for fam, g in incr.groupby("family"):
            p = wilcoxon(g["base+structure"], g["base"]).pvalue if len(g) >= 5 else np.nan
            tests.append(p)
        agg["wilcoxon_p"] = tests
        md += ["Incremental out-of-fold AUROC (logistic, 5-fold):", "", md_table(agg, index=True, digits=4), ""]

    r4 = rq4_cost(df, extra)
    r4.to_csv(out / "rq4_cost_size.csv", index=False)
    md += ["## RQ4 Cost and native explanation size", "", md_table(r4), ""]

    sc = scaling_table()
    if len(sc):
        sc.to_csv(out / "scaling.csv", index=False)
        md += ["## Scaling (RQ1/RQ4, fixed grids, all datasets pooled)", "", md_table(sc), ""]

    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
