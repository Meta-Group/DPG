"""Utilities for applying fixed journal train/validation/test splits."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np


class SplitBundle(Protocol):
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_split: str
    eval_split: str
    split_registry_path: str | None


TBundle = TypeVar("TBundle", bound=SplitBundle)


def load_split_registry(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not str(p):
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def apply_split_registry(
    bundle: TBundle,
    registry: dict[str, Any] | None,
    registry_path: Path | str | None,
    train_split: str,
    eval_split: str,
) -> TBundle:
    """Return a bundle with X_train/y_train and X_test/y_test remapped.

    Indices in the registry are relative to the existing prepared arrays:
    train_core/validation index the existing X_train/y_train, and test indexes
    the existing X_test/y_test.
    """

    if registry is None:
        return replace(
            bundle,
            train_split="existing_train",
            eval_split="existing_test",
            split_registry_path=None,
        )

    dataset_splits = registry.get("datasets", {})
    if bundle.name not in dataset_splits:
        raise KeyError(f"Dataset '{bundle.name}' is missing from split registry.")
    split = dataset_splits[bundle.name]

    train_core_idx = np.asarray(split["train_core_indices"], dtype=int)
    validation_idx = np.asarray(split["validation_indices"], dtype=int)
    test_idx = np.asarray(split["test_indices"], dtype=int)

    train_split_key = train_split.strip().lower()
    eval_split_key = eval_split.strip().lower()

    if train_split_key == "train_core":
        fit_idx = train_core_idx
    elif train_split_key in {"train_core_validation", "train_plus_validation"}:
        fit_idx = np.sort(np.concatenate([train_core_idx, validation_idx]))
    elif train_split_key in {"existing_train", "prepared_train"}:
        fit_idx = np.arange(bundle.y_train.shape[0], dtype=int)
    else:
        raise ValueError(
            f"Unsupported train_split '{train_split}'. "
            "Use train_core, train_core_validation, or existing_train."
        )

    if eval_split_key == "validation":
        X_eval = bundle.X_train[validation_idx]
        y_eval = bundle.y_train[validation_idx]
    elif eval_split_key in {"test", "existing_test", "prepared_test"}:
        X_eval = bundle.X_test[test_idx]
        y_eval = bundle.y_test[test_idx]
        eval_split_key = "test"
    else:
        raise ValueError(f"Unsupported eval_split '{eval_split}'. Use validation or test.")

    return replace(
        bundle,
        X_train=bundle.X_train[fit_idx],
        y_train=bundle.y_train[fit_idx],
        X_test=X_eval,
        y_test=y_eval,
        train_split=train_split_key,
        eval_split=eval_split_key,
        split_registry_path=str(Path(registry_path).resolve()) if registry_path else None,
    )
