"""
Normalizer for scikit-learn ensemble models.

Handles differences in tree storage between RandomForest, GradientBoosting,
AdaBoost, and other ensemble methods to provide a consistent interface for DPG.
"""

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


class SklearnEnsembleNormalizer:
    """
    Normalizes sklearn ensemble models to have consistent tree access interface.

    Problem: Different ensemble methods store trees differently:
    - RandomForest: estimators_ is 1D list of DecisionTree objects
    - GradientBoosting: estimators_ is 2D (n_classes, n_estimators) array
    - AdaBoost: estimators_ is 1D list of DecisionTree objects

    Solution: Normalize to 1D list for consistent access.
    """

    GB_MODELS = (GradientBoostingClassifier, GradientBoostingRegressor)

    @staticmethod
    def needs_normalization(model):
        """
        Check if a model needs tree structure normalization.

        Args:
            model: sklearn ensemble model

        Returns:
            bool: True if normalization needed
        """
        return isinstance(model, SklearnEnsembleNormalizer.GB_MODELS)

    @staticmethod
    def normalize(model):
        """
        Normalize a sklearn ensemble model's tree structure.

        For GradientBoosting models:
        - Converts 2D estimators_ to 1D list

        Args:
            model: sklearn ensemble model

        Returns:
            model: Modified model with normalized estimators_
        """
        if not SklearnEnsembleNormalizer.needs_normalization(model):
            return model

        # Check if already normalized
        if isinstance(model.estimators_, list):
            return model

        # Store original shape for potential restoration
        if not hasattr(model, '_original_estimators_shape'):
            model._original_estimators_shape = model.estimators_.shape
            model._normalized_for_dpg = True

        # Flatten 2D (n_classes, n_estimators) to 1D list
        flat_estimators = []
        for row in model.estimators_:
            for tree in row:
                flat_estimators.append(tree)

        model.estimators_ = flat_estimators

        return model
