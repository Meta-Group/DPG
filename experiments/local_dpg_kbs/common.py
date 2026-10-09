"""Shared protocol for the KBS local-DPG experiments.

Protocol (fixes the AAAI-27 per-method model selection):
  * one black box per (dataset, family, seed), selected on validation accuracy only;
  * every explainer and control explains that same refitted model;
  * DPG-local has no tuned hyper-parameters.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
# Prefer this checkout's dpg over any installed copy (the experiments need dpg.local_dpg).
sys.path.insert(0, str(HERE.parents[1]))
DATA_DIR = HERE / "data_numeric"
REGISTRY = HERE / "splits" / "split_registry.json"
# Override with KBS_RESULTS (e.g. for smoke runs) so partial outputs never mask full-run jobs.
RESULTS = Path(os.environ.get("KBS_RESULTS", HERE / "results"))

# wdbc (= sklearn breast_cancer) and optdigits (superset of sklearn digits) are excluded as duplicates.
DATASETS = [
    "banknote-authentication", "breast_cancer", "diabetes", "digits", "ionosphere", "iris",
    "isolet", "madelon", "phoneme", "qsar-biodeg", "satimage", "segment", "spambase",
    "vehicle", "wine", "wine_quality",
]
FAMILIES = ["rf", "et", "gbm", "adaboost", "bagging"]
SEEDS = [27, 42, 100, 101, 102, 103, 104, 105, 106, 107]

# Validation grids: selection uses accuracy only, never an explanation metric.
GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "rf": [{"n_estimators": 100, "max_depth": d} for d in (4, 8, None)],
    "et": [{"n_estimators": 100, "max_depth": d} for d in (4, 8, None)],
    "bagging": [{"n_estimators": 100, "max_depth": d} for d in (4, 8, None)],
    "adaboost": [{"n_estimators": 100, "max_depth": d} for d in (1, 2, 3)],
    "gbm": [{"n_estimators": 100, "max_depth": d, "learning_rate": 0.1} for d in (2, 3, 5)],
}


def limit_native_threads() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


@dataclass
class Split:
    name: str
    feature_names: List[str]
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_split(name: str) -> Split:
    d = DATA_DIR / name
    X_train = np.load(d / "X_train.npy", allow_pickle=True).astype(float)
    y_train = np.load(d / "y_train.npy", allow_pickle=True)
    X_test = np.load(d / "X_test.npy", allow_pickle=True).astype(float)
    y_test = np.load(d / "y_test.npy", allow_pickle=True)
    feature_names = [str(f) for f in json.loads((d / "feature_names.json").read_text())]
    reg = json.loads(REGISTRY.read_text())["datasets"][name]
    core = np.asarray(reg["train_core_indices"], dtype=int)
    val = np.asarray(reg["validation_indices"], dtype=int)
    if len(reg["test_indices"]) != len(X_test):
        raise ValueError(f"{name}: registry test size does not match X_test.")
    return Split(name, feature_names, X_train[core], y_train[core], X_train[val], y_train[val],
                 X_train, y_train, X_test, y_test)


def build_model(family: str, params: Dict[str, Any], seed: int) -> Any:
    p = dict(params)
    if family == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    if family == "et":
        return ExtraTreesClassifier(random_state=seed, n_jobs=1, **p)
    if family == "gbm":
        return GradientBoostingClassifier(random_state=seed, **p)
    if family == "adaboost":
        depth = p.pop("max_depth")
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth), random_state=seed, **p)
    if family == "bagging":
        depth = p.pop("max_depth")
        return BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=depth), random_state=seed, n_jobs=1, **p)
    raise ValueError(f"Unknown family {family!r}")


def config_id(params: Dict[str, Any]) -> str:
    return "_".join(f"{k}={'None' if v is None else v}" for k, v in sorted(params.items()))


def uncertainty_features(support: np.ndarray) -> Dict[str, float]:
    """Model-native uncertainty baselines computed from the class support vector."""
    s = np.sort(np.asarray(support, dtype=float))[::-1]
    p = np.clip(support, 1e-300, 1.0)
    return {
        "u_maxprob": float(s[0]),
        "u_margin": float(s[0] - (s[1] if len(s) > 1 else 0.0)),
        "u_entropy": float(-(np.asarray(support) * np.log(p)).sum()),
    }


class NeighbourIndex:
    """Nearest reference points with a different / same model prediction."""

    def __init__(self, X_ref: np.ndarray, pred_ref: np.ndarray, k: int = 10) -> None:
        self.X_ref, self.pred_ref, self.k = X_ref, np.asarray(pred_ref), k
        self._other: Dict[Any, tuple] = {}
        self._same: Dict[Any, tuple] = {}
        for c in np.unique(self.pred_ref):
            for store, mask in ((self._other, self.pred_ref != c), (self._same, self.pred_ref == c)):
                idx = np.flatnonzero(mask)
                if len(idx):
                    nn = NearestNeighbors(n_neighbors=min(k, len(idx))).fit(X_ref[idx])
                    store[c] = (nn, idx)

    def neighbours(self, x: np.ndarray, pred: Any, same: bool) -> np.ndarray:
        store = self._same if same else self._other
        if pred not in store:
            return np.empty((0, self.X_ref.shape[1]))
        nn, idx = store[pred]
        _, pos = nn.kneighbors(x.reshape(1, -1))
        return self.X_ref[idx[pos[0]]]


def violation_rate(neighbours: np.ndarray, feature_index: int, threshold: float, operator: str) -> float:
    """Fraction of neighbours on the other side of the predicate's threshold."""
    if len(neighbours) == 0:
        return float("nan")
    values = neighbours[:, feature_index].astype(np.float32).astype(float)
    holds = values <= threshold if operator == "<=" else values > threshold
    return float(1.0 - holds.mean())


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def parse_list(text: Optional[str], default: Iterable[Any], cast=str) -> List[Any]:
    if not text:
        return list(default)
    return [cast(t.strip()) for t in text.split(",") if t.strip()]
