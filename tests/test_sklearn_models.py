"""Test DPGExplainer with various sklearn ensemble models, especially GradientBoosting."""

import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_diabetes
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)

from dpg import DPGExplainer


class TestGradientBoostingSupport:
    """Test Gradient Boosting support for DPGExplainer."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset."""
        iris = load_iris()
        return iris.data, iris.target, iris.feature_names, iris.target_names

    @pytest.fixture
    def wine_data(self):
        """Load wine dataset."""
        wine = load_wine()
        return wine.data, wine.target, wine.feature_names, wine.target_names

    @pytest.fixture
    def diabetes_data(self):
        """Load diabetes dataset (regression)."""
        diabetes = load_diabetes()
        return diabetes.data, diabetes.target, diabetes.feature_names, None

    @pytest.fixture
    def breast_cancer_data(self):
        """Load breast cancer dataset."""
        cancer = load_breast_cancer()
        return cancer.data, cancer.target, cancer.feature_names, cancer.target_names

    def test_gradient_boosting_classifier_binary(self, iris_data):
        """Test GradientBoostingClassifier on binary classification."""
        X, y, feature_names, target_names = iris_data
        X_binary = X[:100]
        y_binary = y[:100]

        gb = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        gb.fit(X_binary, y_binary)

        explainer = DPGExplainer(gb, feature_names, list(target_names[:2]))
        explanation = explainer.explain_global(X_binary)

        assert explanation is not None
        assert explanation.nodes is not None
        assert len(explanation.nodes) > 0

    def test_gradient_boosting_classifier_multiclass(self, iris_data):
        """Test GradientBoostingClassifier on multiclass classification."""
        X, y, feature_names, target_names = iris_data

        gb = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        gb.fit(X, y)

        explainer = DPGExplainer(gb, feature_names, list(target_names))
        explanation = explainer.explain_global(X)

        assert explanation is not None
        assert explanation.nodes is not None
        assert len(explanation.nodes) > 0
        class_labels = sorted(
            explanation.node_metrics.loc[
                explanation.node_metrics["Label"].astype(str).str.startswith("Class "),
                "Label",
            ].unique().tolist()
        )
        assert class_labels == sorted([f"Class {name}" for name in target_names])

    def test_gradient_boosting_classifier_wine(self, wine_data):
        """Test GradientBoostingClassifier on wine dataset (3 classes)."""
        X, y, feature_names, target_names = wine_data

        gb = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        gb.fit(X, y)

        explainer = DPGExplainer(gb, feature_names, list(target_names))
        explanation = explainer.explain_global(X)

        assert explanation is not None
        assert explanation.nodes is not None
        assert len(explanation.nodes) > 0
        class_labels = sorted(
            explanation.node_metrics.loc[
                explanation.node_metrics["Label"].astype(str).str.startswith("Class "),
                "Label",
            ].unique().tolist()
        )
        assert class_labels == sorted([f"Class {name}" for name in target_names])

    def test_gradient_boosting_classifier_binary_breast_cancer(self, breast_cancer_data):
        """Binary GradientBoostingClassifier should expose both class nodes."""
        X, y, feature_names, target_names = breast_cancer_data

        gb = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        gb.fit(X, y)

        explainer = DPGExplainer(gb, feature_names, list(target_names))
        explanation = explainer.explain_global(X)

        class_labels = sorted(
            explanation.node_metrics.loc[
                explanation.node_metrics["Label"].astype(str).str.startswith("Class "),
                "Label",
            ].unique().tolist()
        )
        assert class_labels == sorted([f"Class {name}" for name in target_names])

    def test_gradient_boosting_regressor(self, diabetes_data):
        """Test GradientBoostingRegressor on regression task."""
        X, y, feature_names, _ = diabetes_data

        gb = GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=42)
        gb.fit(X, y)

        target_names = ["prediction"]
        explainer = DPGExplainer(gb, feature_names, target_names)
        explanation = explainer.explain_global(X)

        assert explanation is not None
        assert explanation.nodes is not None
        assert len(explanation.nodes) > 0

    def test_gb_normalizer_flattens_estimators(self, iris_data):
        """Test that normalizer properly flattens GB estimators_ array."""
        X, y, feature_names, target_names = iris_data

        gb = GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=42)
        gb.fit(X, y)

        # Before DPGExplainer: estimators_ is 2D
        original_shape = gb.estimators_.shape
        assert len(original_shape) == 2

        # After DPGExplainer: should be normalized to list
        explainer = DPGExplainer(gb, feature_names, list(target_names))
        assert isinstance(explainer._builder.model.estimators_, list)
        assert gb.estimators_.shape == original_shape

    def test_gb_original_model_predict_still_works_after_explainer(self, iris_data):
        """DPG normalization should not mutate the caller's sklearn model."""
        X, y, feature_names, target_names = iris_data

        gb = GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=42)
        gb.fit(X, y)
        baseline = gb.predict(X[:5])

        explainer = DPGExplainer(gb, feature_names, list(target_names))
        assert isinstance(explainer._builder.model.estimators_, list)

        after = gb.predict(X[:5])
        np.testing.assert_array_equal(after, baseline)

    def test_gb_vs_rf_same_structure(self, iris_data):
        """Test that GB and RF produce same output structure."""
        X, y, feature_names, target_names = iris_data

        # Random Forest
        rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        rf.fit(X, y)
        rf_explainer = DPGExplainer(rf, feature_names, list(target_names))
        rf_explanation = rf_explainer.explain_global(X)

        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        gb.fit(X, y)
        gb_explainer = DPGExplainer(gb, feature_names, list(target_names))
        gb_explanation = gb_explainer.explain_global(X)

        # Both should produce valid output
        assert rf_explanation is not None
        assert gb_explanation is not None
        assert len(rf_explanation.nodes) > 0
        assert len(gb_explanation.nodes) > 0


class TestBackwardCompatibility:
    """Ensure existing functionality still works after GB support added."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset."""
        iris = load_iris()
        return iris.data, iris.target, iris.feature_names, iris.target_names

    @pytest.fixture
    def diabetes_data(self):
        """Load diabetes dataset (regression)."""
        diabetes = load_diabetes()
        return diabetes.data, diabetes.target, diabetes.feature_names, None

    def test_random_forest_classifier_still_works(self, iris_data):
        """Test that RandomForestClassifier still works."""
        X, y, feature_names, target_names = iris_data

        rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        rf.fit(X, y)

        explainer = DPGExplainer(rf, feature_names, list(target_names))
        explanation = explainer.explain_global(X)

        assert explanation is not None
        assert explanation.nodes is not None
        assert len(explanation.nodes) > 0

    def test_random_forest_regressor_still_works(self, diabetes_data):
        """Test that RandomForestRegressor still works."""
        X, y, feature_names, _ = diabetes_data

        rf = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        rf.fit(X, y)

        target_names = ["prediction"]
        explainer = DPGExplainer(rf, feature_names, target_names)
        explanation = explainer.explain_global(X)

        assert explanation is not None
        assert explanation.nodes is not None
        assert len(explanation.nodes) > 0

    def test_adaboost_classifier_still_works(self, iris_data):
        """Test that AdaBoostClassifier still works."""
        X, y, feature_names, target_names = iris_data

        ada = AdaBoostClassifier(n_estimators=5, random_state=42)
        ada.fit(X, y)

        explainer = DPGExplainer(ada, feature_names, list(target_names))
        explanation = explainer.explain_global(X)

        assert explanation is not None
        assert explanation.nodes is not None
        assert len(explanation.nodes) > 0

    def test_adaboost_regressor_still_works(self, diabetes_data):
        """Test that AdaBoostRegressor still works."""
        X, y, feature_names, _ = diabetes_data

        ada = AdaBoostRegressor(n_estimators=5, random_state=42)
        ada.fit(X, y)

        target_names = ["prediction"]
        explainer = DPGExplainer(ada, feature_names, target_names)
        explanation = explainer.explain_global(X)

        assert explanation is not None
        assert explanation.nodes is not None
        assert len(explanation.nodes) > 0
