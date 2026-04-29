"""
Normalizer for scikit-learn ensemble models.

Handles differences in tree storage between RandomForest, GradientBoosting,
AdaBoost, and other ensemble methods to provide a consistent interface for DPG.
"""

import copy
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
    - GradientBoosting: estimators_ is 2D (n_estimators, n_trees_per_iteration)
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
        - Preserves the class-slot index for multiclass classifiers

        Args:
            model: sklearn ensemble model

        Returns:
            model: Normalized shallow copy with DPG-specific metadata
        """
        if not SklearnEnsembleNormalizer.needs_normalization(model):
            return model

        # Check if already normalized
        if isinstance(model.estimators_, list):
            return model

        normalized_model = copy.copy(model)
        normalized_model._original_estimators_shape = model.estimators_.shape
        normalized_model._normalized_for_dpg = True

        # Flatten 2D (n_estimators, n_trees_per_iteration) to 1D list.
        # For multiclass GradientBoostingClassifier the column index is the class slot.
        flat_estimators = []
        tree_class_indices = []
        for row in model.estimators_:
            for class_index, tree in enumerate(row):
                flat_estimators.append(tree)
                if isinstance(model, GradientBoostingClassifier) and len(row) > 1:
                    tree_class_indices.append(class_index)
                else:
                    tree_class_indices.append(None)

        normalized_model.estimators_ = flat_estimators
        normalized_model._dpg_tree_class_indices = tree_class_indices

        return normalized_model

    @staticmethod
    def get_tree_class_index(model, tree_index):
        """Return the preserved class-slot index for a normalized GB tree."""
        indices = getattr(model, "_dpg_tree_class_indices", None)
        if indices is None:
            return None
        if tree_index < 0 or tree_index >= len(indices):
            return None
        return indices[tree_index]
