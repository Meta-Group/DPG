#!/usr/bin/env python3
"""
prepare_datasets.py

Download + standardize NUMERIC-ONLY tabular classification datasets for experiments
(DPG local evidence vs LIME vs SHAP).

Key design:
- Prefers datasets that are already numeric (no categorical).
- If a dataset contains non-numeric columns, it is SKIPPED (not coerced), per your request.
- Produces a consistent artifact per dataset:
    <out_dir>/<dataset_name>/
        X_train.npy
        y_train.npy
        X_test.npy
        y_test.npy
        feature_names.json
        meta.json

Requirements:
    pip install scikit-learn pandas numpy

Optional:
    pip install openml  (not required if you use sklearn.fetch_openml)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    fetch_openml,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class DatasetSpec:
    name: str
    source: str  # "sklearn" | "openml"
    openml_id: Optional[int] = None
    openml_name: Optional[str] = None  # optional; id is preferred
    openml_version: Optional[int] = None
    target_col: Optional[str] = None  # optional override


# ---- Candidate datasets (numeric-only expected) ----
# Note: we still *verify* numerics at runtime. If any dataset is non-numeric, we skip it.

SKLEARN_SPECS: List[DatasetSpec] = [
    DatasetSpec(name="iris", source="sklearn"),
    DatasetSpec(name="wine", source="sklearn"),
    DatasetSpec(name="breast_cancer", source="sklearn"),
    DatasetSpec(name="digits", source="sklearn"),  # 8x8 pixels => numeric tabular
]

OPENML_SPECS: List[DatasetSpec] = [
    # Classic numeric tabular classification datasets on OpenML
    DatasetSpec(name="diabetes", source="openml", openml_id=37),        # Pima Indians Diabetes
    DatasetSpec(name="ionosphere", source="openml", openml_id=59),
    DatasetSpec(name="spambase", source="openml", openml_id=44),
    DatasetSpec(name="banknote-authentication", source="openml", openml_id=1462),
    DatasetSpec(name="phoneme", source="openml", openml_id=1489),
    DatasetSpec(name="wdbc", source="openml", openml_id=1510),          # Breast Cancer WDBC
    DatasetSpec(name="wine_quality", source="openml", openml_id=287),   # wine_quality (red/white combined on OpenML)
    DatasetSpec(name="isolet", source="openml", openml_id=300),
    # Larger numeric datasets (some versions may include sparse or huge; still numeric)
    DatasetSpec(name="covertype", source="openml", openml_id=1596),     # classic 7-class Covertype
    # A few extra numeric candidates (if you want up to 20+ after filtering)
    DatasetSpec(name="qsar-biodeg", source="openml", openml_id=1494),
    DatasetSpec(name="madelon", source="openml", openml_id=1485),
    DatasetSpec(name="satimage", source="openml", openml_id=182),
    DatasetSpec(name="segment", source="openml", openml_id=36),
    DatasetSpec(name="vehicle", source="openml", openml_id=54),
    DatasetSpec(name="optdigits", source="openml", openml_id=28),
    DatasetSpec(name="pendigits", source="openml", openml_id=32),
    DatasetSpec(name="texture", source="openml", openml_id=40499),
    DatasetSpec(name="jungle_chess_2pcs_raw_endgame_complete", source="openml", openml_id=40927),
]


@dataclass
class SavedMeta:
    dataset_name: str
    source: str
    openml_id: Optional[int]
    n_train: int
    n_test: int
    n_features: int
    n_classes: int
    class_counts_train: Dict[str, int]
    class_counts_test: Dict[str, int]
    standardized: bool
    notes: str


def _is_numeric_df(X: pd.DataFrame) -> bool:
    """Strict numeric check: all columns must be numeric dtype."""
    return all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)


def _load_sklearn_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if name == "iris":
        d = load_iris(as_frame=True)
        X = d.data
        y = d.target
        feature_names = list(X.columns)
        return X, y, feature_names
    if name == "wine":
        d = load_wine(as_frame=True)
        X = d.data
        y = d.target
        feature_names = list(X.columns)
        return X, y, feature_names
    if name == "breast_cancer":
        d = load_breast_cancer(as_frame=True)
        X = d.data
        y = d.target
        feature_names = list(X.columns)
        return X, y, feature_names
    if name == "digits":
        d = load_digits(as_frame=True)
        X = d.data
        y = d.target
        feature_names = list(X.columns)
        return X, y, feature_names
    raise ValueError(f"Unknown sklearn dataset: {name}")


def _load_openml_dataset(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.Series, List[str], int]:
    # Prefer data_id for stability.
    if spec.openml_id is None and spec.openml_name is None:
        raise ValueError(f"OpenML spec must define openml_id or openml_name: {spec}")

    # Use as_frame=True to preserve dtypes reliably.
    # Newer scikit-learn versions reject version=None, so include the
    # parameter only when explicitly set; otherwise use active version.
    fetch_kwargs: Dict[str, Any] = {
        "as_frame": True,
        "parser": "auto",
    }
    if spec.openml_id is not None:
        fetch_kwargs["data_id"] = spec.openml_id
    if spec.openml_name is not None:
        fetch_kwargs["name"] = spec.openml_name
    fetch_kwargs["version"] = spec.openml_version if spec.openml_version is not None else "active"

    data = fetch_openml(**fetch_kwargs)

    X = data.data
    y = data.target
    # y can be DataFrame/Series; convert to Series.
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"Multi-target dataset not supported: {spec.name} -> {y.shape[1]} targets")
        y = y.iloc[:, 0]
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="target")

    feature_names = list(X.columns)
    openml_id = spec.openml_id if spec.openml_id is not None else int(data.details["id"])
    return X, y, feature_names, openml_id


def _encode_target(y: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Make y integer-coded 0..C-1.
    Returns y_encoded and class counts (stringified class -> count) for reporting.
    """
    # If y numeric but not contiguous, still encode to 0..C-1.
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str).values)
    counts = {cls: int((y_enc == i).sum()) for i, cls in enumerate(le.classes_)}
    return y_enc.astype(np.int64), counts


def _standardize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s.astype(np.float32), X_test_s.astype(np.float32)


def _save_dataset(
    out_dir: Path,
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    meta: SavedMeta,
) -> None:
    ds_dir = out_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    np.save(ds_dir / "X_train.npy", X_train)
    np.save(ds_dir / "y_train.npy", y_train)
    np.save(ds_dir / "X_test.npy", X_test)
    np.save(ds_dir / "y_test.npy", y_test)

    with open(ds_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    with open(ds_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)


def _prepare_one(
    spec: DatasetSpec,
    out_dir: Path,
    test_size: float,
    random_state: int,
    standardize: bool,
    max_rows: Optional[int],
) -> Tuple[bool, str]:
    """
    Returns (saved?, message).
    """
    try:
        if spec.source == "sklearn":
            X_df, y_ser, feature_names = _load_sklearn_dataset(spec.name)
            openml_id = None
        elif spec.source == "openml":
            X_df, y_ser, feature_names, openml_id = _load_openml_dataset(spec)
        else:
            return False, f"[SKIP] Unknown source '{spec.source}' for {spec.name}"

        # Optional downsample rows for huge datasets (MNIST, Covertype, etc.)
        if max_rows is not None and len(X_df) > max_rows:
            X_df = X_df.iloc[:max_rows].copy()
            y_ser = y_ser.iloc[:max_rows].copy()

        # Enforce numeric-only datasets
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df)

        if not _is_numeric_df(X_df):
            return False, f"[SKIP] {spec.name}: contains non-numeric columns (categorical present)."

        # Convert to numpy
        X = X_df.to_numpy(dtype=np.float32, copy=True)

        # Encode target as int classes
        y, class_counts_all = _encode_target(y_ser)

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        if standardize:
            X_train, X_test = _standardize(X_train, X_test)

        # Per-split class counts
        def _counts(y_arr: np.ndarray) -> Dict[str, int]:
            vals, cnts = np.unique(y_arr, return_counts=True)
            return {str(int(v)): int(c) for v, c in zip(vals, cnts)}

        meta = SavedMeta(
            dataset_name=spec.name,
            source=spec.source,
            openml_id=openml_id,
            n_train=int(X_train.shape[0]),
            n_test=int(X_test.shape[0]),
            n_features=int(X_train.shape[1]),
            n_classes=int(len(np.unique(y))),
            class_counts_train=_counts(y_train),
            class_counts_test=_counts(y_test),
            standardized=bool(standardize),
            notes="Numeric-only enforced; non-numeric datasets skipped. y label-encoded to 0..C-1.",
        )

        _save_dataset(
            out_dir=out_dir,
            dataset_name=spec.name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            meta=meta,
        )

        return True, f"[OK] {spec.name}: saved ({meta.n_train}+{meta.n_test} rows, {meta.n_features} feats, {meta.n_classes} classes)"
    except Exception as e:
        return False, f"[FAIL] {spec.name}: {type(e).__name__}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + prepare numeric-only tabular datasets for experiments.")
    parser.add_argument("--out_dir", type=str, default="data_tabular_numeric", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--standardize", action="store_true", help="Apply StandardScaler (fit on train, transform test)")
    parser.add_argument("--max_datasets", type=int, default=20, help="Stop after saving this many datasets")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional row cap (useful for huge OpenML datasets)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = SKLEARN_SPECS + OPENML_SPECS

    saved = 0
    report: List[str] = []
    for spec in specs:
        if saved >= args.max_datasets:
            break
        ok, msg = _prepare_one(
            spec=spec,
            out_dir=out_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            standardize=args.standardize,
            max_rows=args.max_rows,
        )
        report.append(msg)
        if ok:
            saved += 1

    # Save run report
    with open(out_dir / "RUN_REPORT.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
        f.write(f"\nSaved datasets: {saved}\n")

    print("\n".join(report))
    print(f"\nSaved datasets: {saved}")
    print(f"Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
