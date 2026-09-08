"""Model factory for journal local-explanation experiments."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier


SUPPORTED_CLASSIFICATION_MODEL_FAMILIES = {
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "adaboost",
    "bagging",
}


def normalize_model_family(name: str) -> str:
    """Return the canonical model-family name used in experiment outputs."""
    raw = name.strip().lower().replace("-", "_")
    aliases = {
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest_classifier": "random_forest",
        "extratrees": "extra_trees",
        "extra_trees_classifier": "extra_trees",
        "gb": "gradient_boosting",
        "gbc": "gradient_boosting",
        "gradientboosting": "gradient_boosting",
        "gradient_boosting_classifier": "gradient_boosting",
        "ada": "adaboost",
        "adaboost_classifier": "adaboost",
        "bag": "bagging",
        "bagging_classifier": "bagging",
    }
    canonical = aliases.get(raw, raw)
    if canonical not in SUPPORTED_CLASSIFICATION_MODEL_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_CLASSIFICATION_MODEL_FAMILIES))
        raise ValueError(f"Unsupported model family '{name}'. Supported: {supported}")
    return canonical


def parse_model_families(text: str) -> list[str]:
    """Parse a comma-separated model-family list."""
    families = [normalize_model_family(item) for item in text.split(",") if item.strip()]
    return families or ["random_forest"]


def build_classifier(
    model_family: str,
    *,
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
    n_jobs: int | None = None,
) -> Any:
    """Build a DPG-supported sklearn classification ensemble."""
    family = normalize_model_family(model_family)
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if family == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    if family == "adaboost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
            n_estimators=n_estimators,
            random_state=random_state,
            algorithm="SAMME",
        )
    if family == "bagging":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    raise AssertionError(f"Unhandled model family: {family}")
