"""
Tests for the high-level DPGExplainer API.

Validates the full explain_global workflow, DPGExplanation dataclass,
and the fit/explain lifecycle.
"""

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from dpg import DPGExplainer
from dpg.explainer import DPGExplanation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def iris_model():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(iris.data, iris.target)
    target_names = np.unique(iris.target).astype(str).tolist()
    return model, iris.data, iris.feature_names, target_names


@pytest.fixture(scope="module")
def explainer(iris_model):
    model, X, feature_names, target_names = iris_model
    return DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=target_names,
    )


@pytest.fixture(scope="module")
def explanation(explainer, iris_model):
    _, X, _, _ = iris_model
    return explainer.explain_global(X)


@pytest.fixture(scope="module")
def explanation_with_communities(iris_model):
    model, X, feature_names, target_names = iris_model
    exp = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=target_names,
    )
    return exp.explain_global(X, communities=True)


# ---------------------------------------------------------------------------
# DPGExplanation structure
# ---------------------------------------------------------------------------


class TestDPGExplanationStructure:
    def test_is_dpg_explanation_instance(self, explanation):
        assert isinstance(explanation, DPGExplanation)

    def test_graph_is_not_none(self, explanation):
        assert explanation.graph is not None

    def test_nodes_is_list(self, explanation):
        assert isinstance(explanation.nodes, list)
        assert len(explanation.nodes) > 0

    def test_dot_is_graphviz(self, explanation):
        import graphviz

        assert isinstance(explanation.dot, graphviz.Digraph)

    def test_node_metrics_is_dataframe(self, explanation):
        assert isinstance(explanation.node_metrics, pd.DataFrame)
        assert not explanation.node_metrics.empty

    def test_edge_metrics_is_dataframe(self, explanation):
        assert isinstance(explanation.edge_metrics, pd.DataFrame)
        assert not explanation.edge_metrics.empty

    def test_class_boundaries_is_dict(self, explanation):
        assert isinstance(explanation.class_boundaries, dict)

    def test_communities_none_by_default(self, explanation):
        assert explanation.communities is None
        assert explanation.community_threshold is None


# ---------------------------------------------------------------------------
# Explanation values (Iris, seed=42, full dataset)
# ---------------------------------------------------------------------------


class TestExplanationValues:
    def test_graph_node_count(self, explanation):
        assert explanation.graph.number_of_nodes() == 47

    def test_graph_edge_count(self, explanation):
        assert explanation.graph.number_of_edges() == 90

    def test_node_metrics_row_count(self, explanation):
        assert len(explanation.node_metrics) == 47

    def test_edge_metrics_row_count(self, explanation):
        assert len(explanation.edge_metrics) == 90

    def test_class_boundaries_have_all_classes(self, explanation):
        bounds = explanation.class_boundaries.get("Class Bounds", {})
        assert sorted(bounds.keys()) == ["Class 0", "Class 1", "Class 2"]

    def test_class_boundary_predicates_non_empty(self, explanation):
        bounds = explanation.class_boundaries["Class Bounds"]
        for cls, preds in bounds.items():
            assert len(preds) > 0, f"{cls} has no boundary predicates"

    def test_class_boundary_predicate_format(self, explanation):
        valid_re = re.compile(r"(<=|>|<)")
        bounds = explanation.class_boundaries["Class Bounds"]
        for cls, preds in bounds.items():
            for p in preds:
                assert valid_re.search(p), f"Invalid predicate: {p!r}"

    def test_specific_class0_boundaries(self, explanation):
        """Class 0 (setosa) boundaries should involve petal features."""
        bounds = explanation.class_boundaries["Class Bounds"]["Class 0"]
        combined = " ".join(bounds).lower()
        assert "petal" in combined


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------


class TestExplanationCommunities:
    def test_communities_returned(self, explanation_with_communities):
        assert explanation_with_communities.communities is not None

    def test_community_threshold_stored(self, explanation_with_communities):
        assert explanation_with_communities.community_threshold == 0.2

    def test_communities_has_clusters_key(self, explanation_with_communities):
        assert "Clusters" in explanation_with_communities.communities

    def test_communities_has_probability_key(self, explanation_with_communities):
        assert "Probability" in explanation_with_communities.communities

    def test_communities_has_confidence_key(self, explanation_with_communities):
        assert "Confidence Interval" in explanation_with_communities.communities

    def test_cluster_labels_contain_class_names(self, explanation_with_communities):
        clusters = explanation_with_communities.communities["Clusters"]
        for expected in ["Class 0", "Class 1", "Class 2"]:
            assert expected in clusters


# ---------------------------------------------------------------------------
# as_dict
# ---------------------------------------------------------------------------


class TestAsDict:
    def test_returns_dict(self, explanation):
        d = explanation.as_dict()
        assert isinstance(d, dict)

    def test_all_keys_present(self, explanation):
        d = explanation.as_dict()
        expected_keys = {
            "graph",
            "nodes",
            "dot",
            "node_metrics",
            "edge_metrics",
            "class_boundaries",
            "communities",
            "community_threshold",
        }
        assert set(d.keys()) == expected_keys

    def test_values_match_attributes(self, explanation):
        d = explanation.as_dict()
        assert d["graph"] is explanation.graph
        assert d["node_metrics"] is explanation.node_metrics
        assert d["edge_metrics"] is explanation.edge_metrics


# ---------------------------------------------------------------------------
# Fit / lifecycle
# ---------------------------------------------------------------------------


class TestExplainerLifecycle:
    def test_explain_global_without_fit_needs_X(self, iris_model):
        model, _, feature_names, target_names = iris_model
        exp = DPGExplainer(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
        )
        with pytest.raises(ValueError, match="not fitted"):
            exp.explain_global(X=None)

    def test_fit_then_explain_without_X(self, iris_model):
        model, X, feature_names, target_names = iris_model
        exp = DPGExplainer(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
        )
        exp.fit(X)
        explanation = exp.explain_global()
        assert explanation.graph is not None
        assert len(explanation.node_metrics) > 0

    def test_builder_property(self, explainer):
        from dpg.core import DecisionPredicateGraph

        assert isinstance(explainer.builder, DecisionPredicateGraph)


class TestAdditionalExplainerApis:
    def test_sample_bc_weights(self, explainer, explanation, iris_model):
        _, X, _, _ = iris_model
        X_df = pd.DataFrame(X, columns=explainer.builder.feature_names)
        weights = explainer.sample_bc_weights(
            X_df=X_df,
            explanation=explanation,
            top_k=5,
        )
        assert len(weights) == len(X_df)
        assert np.all(np.asarray(weights) >= 0)

    def test_plot_sample_using_bc_weights(self, explainer, explanation, iris_model, tmp_path):
        _, X, _, target_names = iris_model
        X_df = pd.DataFrame(X, columns=explainer.builder.feature_names)
        y = explainer.builder.model.predict(X_df)
        fig = explainer.plot_sample_using_bc_weights(
            X_df=X_df,
            y=y,
            explanation=explanation,
            top_k=5,
            dataset_name="Iris",
            class_names=target_names,
            save_path=str(tmp_path / "iris_bc_weights.png"),
            show=False,
        )
        assert fig is not None
        assert (tmp_path / "iris_bc_weights.png").exists()
