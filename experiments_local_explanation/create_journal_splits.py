#!/usr/bin/env python3
"""Create fixed train/validation/test split registries for journal experiments.

The prepared datasets in ``data_numeric`` already contain train/test arrays.
For the journal protocol we keep those test arrays untouched and split the
existing training rows into ``train_core`` and ``validation`` indices. This
prevents configuration selection from peeking at the final test set while
preserving compatibility with the current prepared data artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EXCLUDED_DATASETS = {"fashion_mnist_784", "mnist_784"}


@dataclass
class DatasetSplit:
    dataset: str
    n_existing_train: int
    n_train_core: int
    n_validation: int
    n_test: int
    n_features: int
    n_classes_train: int
    validation_size_from_existing_train: float
    split_seed: int
    stratified_validation: bool
    train_core_indices: List[int]
    validation_indices: List[int]
    test_indices: List[int]
    class_counts_train_core: Dict[str, int]
    class_counts_validation: Dict[str, int]
    class_counts_test: Dict[str, int]


def _parse_dataset_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _dataset_dirs(data_dir: Path, only: Sequence[str] | None = None) -> List[Path]:
    dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name not in EXCLUDED_DATASETS)
    if not only:
        return dirs
    selected = set(only)
    return [p for p in dirs if p.name in selected]


def _counts(y: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _can_stratify(y: np.ndarray, validation_size: float) -> bool:
    values, counts = np.unique(y, return_counts=True)
    if len(values) <= 1:
        return False
    if int(np.min(counts)) < 2:
        return False
    n_validation = int(np.ceil(float(validation_size) * y.shape[0]))
    return n_validation >= len(values)


def _make_split(dataset_dir: Path, validation_size: float, seed: int) -> DatasetSplit:
    X_train = np.load(dataset_dir / "X_train.npy")
    y_train = np.load(dataset_dir / "y_train.npy")
    X_test = np.load(dataset_dir / "X_test.npy")
    y_test = np.load(dataset_dir / "y_test.npy")

    train_indices = np.arange(y_train.shape[0], dtype=int)
    stratified = _can_stratify(y_train, validation_size)
    train_core_idx, validation_idx = train_test_split(
        train_indices,
        test_size=float(validation_size),
        random_state=int(seed),
        stratify=y_train if stratified else None,
    )

    train_core_idx = np.sort(train_core_idx.astype(int))
    validation_idx = np.sort(validation_idx.astype(int))
    test_idx = np.arange(y_test.shape[0], dtype=int)

    return DatasetSplit(
        dataset=dataset_dir.name,
        n_existing_train=int(y_train.shape[0]),
        n_train_core=int(train_core_idx.shape[0]),
        n_validation=int(validation_idx.shape[0]),
        n_test=int(y_test.shape[0]),
        n_features=int(X_train.shape[1]),
        n_classes_train=int(len(np.unique(y_train))),
        validation_size_from_existing_train=float(validation_size),
        split_seed=int(seed),
        stratified_validation=bool(stratified),
        train_core_indices=train_core_idx.tolist(),
        validation_indices=validation_idx.tolist(),
        test_indices=test_idx.tolist(),
        class_counts_train_core=_counts(y_train[train_core_idx]),
        class_counts_validation=_counts(y_train[validation_idx]),
        class_counts_test=_counts(y_test),
    )


def _summary_rows(splits: Sequence[DatasetSplit]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split in splits:
        rows.append(
            {
                "dataset": split.dataset,
                "n_existing_train": split.n_existing_train,
                "n_train_core": split.n_train_core,
                "n_validation": split.n_validation,
                "n_test": split.n_test,
                "n_features": split.n_features,
                "n_classes_train": split.n_classes_train,
                "validation_size_from_existing_train": split.validation_size_from_existing_train,
                "split_seed": split.split_seed,
                "stratified_validation": split.stratified_validation,
                "class_counts_train_core": json.dumps(split.class_counts_train_core, sort_keys=True),
                "class_counts_validation": json.dumps(split.class_counts_validation, sort_keys=True),
                "class_counts_test": json.dumps(split.class_counts_test, sort_keys=True),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create journal train/validation/test split registry.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("experiments_local_explanation/data_numeric"),
        help="Prepared dataset directory containing X_train/X_test arrays.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/splits"),
        help="Directory where split registry files will be written.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names. Empty means all prepared datasets.",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.20,
        help="Fraction of existing training rows reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=20260617, help="Split seed.")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    only = _parse_dataset_list(args.datasets)
    dataset_dirs = _dataset_dirs(data_dir, only=only)
    if not dataset_dirs:
        raise ValueError(f"No dataset directories found in {data_dir}")

    splits = [_make_split(ds_dir, validation_size=args.validation_size, seed=args.seed) for ds_dir in dataset_dirs]

    registry = {
        "schema_version": 1,
        "description": (
            "Indices are relative to the existing prepared arrays. "
            "train_core_indices and validation_indices index X_train/y_train; "
            "test_indices index X_test/y_test."
        ),
        "data_dir": str(data_dir),
        "validation_size_from_existing_train": float(args.validation_size),
        "split_seed": int(args.seed),
        "datasets": {split.dataset: asdict(split) for split in splits},
    }

    registry_path = out_dir / "split_registry.json"
    summary_path = out_dir / "split_summary.csv"

    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(_summary_rows(splits)).to_csv(summary_path, index=False)

    print(f"Saved split registry: {registry_path}")
    print(f"Saved split summary: {summary_path}")
    print(f"Datasets: {len(splits)}")


if __name__ == "__main__":
    main()
