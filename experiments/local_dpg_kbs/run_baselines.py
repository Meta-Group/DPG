"""Baseline explainers on the same validation-selected black box (RQ4 + RQ2 controls).

Official implementations: shap (TreeSHAP; tree-supported families only), lime,
anchor-exp (Ribeiro et al.), lore-sa (Guidotti et al.). Each method is timed and
sized per sample. The top feature named by LIME, Anchors and LORE becomes an
attribution-guided control: the executed predicate on that feature with the
smallest slack is crossed exactly as in run_dpg_local.py.

Writes results/baselines/<dataset>__<family>__s<seed>.csv (first --max_samples
test samples, aligned with run_dpg_local.py by the sample index ``i``).
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

from common import (
    DATASETS, FAMILIES, RESULTS, SEEDS, NeighbourIndex, build_model, limit_native_threads,
    load_split, parse_list, violation_rate,
)

METHODS = ("shap", "lime", "anchors", "lore")


class _Baselines:
    def __init__(self, model, family, split, seed, methods, lore_generator):
        self.model, self.family, self.split, self.seed = model, family, split, seed
        self.class_names = [str(c) for c in model.classes_]
        self.methods = {}
        if "shap" in methods and family in {"rf", "et", "gbm"}:
            import shap
            try:
                self.methods["shap"] = shap.TreeExplainer(model)
            except Exception:  # e.g. TreeSHAP does not support multiclass GradientBoostingClassifier
                pass
        if "lime" in methods:
            from lime.lime_tabular import LimeTabularExplainer
            self.methods["lime"] = LimeTabularExplainer(
                split.X_train, feature_names=split.feature_names, class_names=self.class_names,
                discretize_continuous=True, random_state=seed)
        if "anchors" in methods:
            from anchor.anchor_tabular import AnchorTabularExplainer
            self.methods["anchors"] = AnchorTabularExplainer(self.class_names, split.feature_names, split.X_train, {})
        if "lore" in methods:
            from lore_sa.bbox.bbox import AbstractBBox
            from lore_sa.dataset import TabularDataset
            from lore_sa import lore as lore_module

            class _StringLabelBBox(AbstractBBox):
                """lore-sa needs a categorical target, so labels are exchanged as strings."""

                def __init__(self, classifier):
                    self.classifier = classifier

                def predict(self, X):
                    return np.asarray([str(v) for v in self.classifier.predict(np.asarray(X, dtype=float))])

                def predict_proba(self, X):
                    return self.classifier.predict_proba(np.asarray(X, dtype=float))

            frame = pd.DataFrame(split.X_train, columns=split.feature_names)
            frame["__class__"] = [str(v) for v in split.y_train]
            generator = {"random": lore_module.TabularRandomGeneratorLore,
                         "genetic": lore_module.TabularGeneticGeneratorLore}[lore_generator]
            self.methods["lore"] = generator(_StringLabelBBox(model), TabularDataset(frame, class_name="__class__"))
        self.feature_pos = {name: j for j, name in enumerate(split.feature_names)}

    def explain(self, name: str, x: np.ndarray, pred_idx: int) -> Dict[str, Any]:
        explainer = self.methods[name]
        if name == "shap":
            phi = explainer.shap_values(x.reshape(1, -1))
            phi = np.stack(phi, axis=-1) if isinstance(phi, list) else np.asarray(phi)
            phi = phi[0] if phi.ndim == 2 else phi[0, :, pred_idx]
            return {"size": int(np.sum(np.abs(phi) > 1e-12)), "top": int(np.abs(phi).argmax())}
        if name == "lime":
            exp = explainer.explain_instance(x, self.model.predict_proba, labels=(pred_idx,),
                                             num_features=min(10, len(x)), num_samples=1000)
            weights = exp.as_map()[pred_idx]
            return {"size": len(weights), "top": int(max(weights, key=lambda fw: abs(fw[1]))[0])}
        if name == "anchors":
            exp = explainer.explain_instance(x, self.model.predict, threshold=0.95)
            feats = list(exp.features())
            return {"size": len(exp.names()), "n_features": len(set(feats)), "top": int(feats[0]) if feats else None,
                    "precision": float(exp.precision()), "coverage": float(exp.coverage())}
        if name == "lore":
            exp = explainer.explain(x, num_instances=1000)
            premise = exp["rule"].get("premises", [])
            top = None
            importances = exp.get("feature_importances") or []
            if importances:  # (feature, importance) pairs in feature order
                top = self.feature_pos.get(str(max(importances, key=lambda fi: abs(fi[1]))[0]))
            if top is None and premise:
                top = self.feature_pos.get(str(premise[0].get("attr")))
            return {"size": len(premise), "top": top, "fidelity": float(exp.get("fidelity", np.nan)),
                    "n_counterfactuals": len(exp.get("counterfactuals", []))}
        raise ValueError(name)


def run(job) -> str:
    dataset, family, seed, max_samples, methods, lore_generator, k_neighbours = job
    tag = f"{dataset}__{family}__s{seed}"
    out = RESULTS / "baselines" / f"{tag}.csv"
    if out.exists():
        return f"{tag} cached"
    try:
        from dpg.local_dpg import build_local_dpg, intervene

        params = json.loads((RESULTS / "selection" / f"{tag}.json").read_text())["selected"]["params"]
        split = load_split(dataset)
        model = build_model(family, params, seed).fit(split.X_train, split.y_train)
        class_names = [str(c) for c in model.classes_]
        pred_test = model.predict(split.X_test)
        neighbours = NeighbourIndex(split.X_train, model.predict(split.X_train), k=k_neighbours)
        baselines = _Baselines(model, family, split, seed, methods, lore_generator)
        rows = []
        for i in range(min(max_samples, len(split.X_test))):
            x = split.X_test[i]
            pred_idx = int(np.searchsorted(model.classes_, pred_test[i]))
            row: Dict[str, Any] = {"dataset": dataset, "family": family, "seed": seed, "i": i}
            local = build_local_dpg(model, x, split.feature_names, class_names)
            comp_idx = class_names.index(local.top_competitor) if local.top_competitor else None
            diff_nb = neighbours.neighbours(x, pred_test[i], same=False)
            same_nb = neighbours.neighbours(x, pred_test[i], same=True)
            for name in baselines.methods:
                t0 = time.time()
                try:
                    result = baselines.explain(name, x, pred_idx)
                except Exception as exc:  # record and continue: one failing instance must not void the run
                    row[f"{name}_error"] = repr(exc)[:200]
                    continue
                row[f"{name}_runtime_ms"] = 1000 * (time.time() - t0)
                row.update({f"{name}_{k}": v for k, v in result.items() if k != "top"})
                top: Optional[int] = result.get("top")
                row[f"{name}_top_feature"] = split.feature_names[top] if top is not None else None
                if name == "shap" or top is None or comp_idx is None:
                    continue  # SHAP-top control is already evaluated in run_dpg_local.py
                on_feature = [p for p in local.pivots if p.feature_index == top]
                key = f"{name}_top"
                row[f"{key}_on_trace"] = int(bool(on_feature))
                if not on_feature:
                    continue
                p = min(on_feature, key=lambda q: (q.slack, q.label))
                effect = intervene(model, x, p, comp_idx)
                row.update({f"{key}_slack": p.slack, f"{key}_delta_comp": effect["target_delta"],
                            f"{key}_flip": int(effect["flipped"]),
                            f"{key}_sep_diff": violation_rate(diff_nb, p.feature_index, p.threshold, p.operator),
                            f"{key}_sep_same": violation_rate(same_nb, p.feature_index, p.threshold, p.operator)})
            rows.append(row)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out.with_suffix(".csv.tmp"), index=False)
        out.with_suffix(".csv.tmp").replace(out)
        return f"{tag} ok n={len(rows)} methods={sorted(baselines.methods)}"
    except Exception:
        err = RESULTS / "baselines" / f"{tag}.error.txt"
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc())
        return f"{tag} FAILED (see {err.name})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets")
    ap.add_argument("--families")
    ap.add_argument("--seeds")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--lore_generator", choices=["random", "genetic"], default="genetic")
    ap.add_argument("--k_neighbours", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    limit_native_threads()
    methods = tuple(parse_list(a.methods, METHODS))
    jobs = [(d, f, s, a.max_samples, methods, a.lore_generator, a.k_neighbours)
            for s in parse_list(a.seeds, SEEDS, int) for f in parse_list(a.families, FAMILIES)
            for d in parse_list(a.datasets, DATASETS)]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for message in pool.imap_unordered(run, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
