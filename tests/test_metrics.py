"""
Tests for node-level, edge-level, and graph-level metrics computed on a DPG.

All tests use Iris with a fixed seed (160898) so that the expected metric
values are deterministic and reproducible.
"""

import math
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from dpg.core import DecisionPredicateGraph
from metrics.edges import EdgeMetrics
from metrics.graph import GraphMetrics
from metrics.nodes import NodeMetrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 160898


@pytest.fixture(scope="module")
def iris_dpg():
    """Build an Iris DPG and return (graph, nodes_list, target_names)."""
    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=SEED
    )
    model = RandomForestClassifier(n_estimators=5, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    target_names = np.unique(iris.target).astype(str).tolist()
    dpg = DecisionPredicateGraph(
        model=model,
        feature_names=iris.feature_names,
        target_names=target_names,
    )
    dot = dpg.fit(X_train)
    dpg_graph, nodes_list = dpg.to_networkx(dot)
    return dpg_graph, nodes_list, target_names


@pytest.fixture(scope="module")
def node_metrics(iris_dpg):
    dpg_graph, nodes_list, _ = iris_dpg
    return NodeMetrics.extract_node_metrics(dpg_graph, nodes_list)


@pytest.fixture(scope="module")
def edge_metrics(iris_dpg):
    dpg_graph, nodes_list, _ = iris_dpg
    return EdgeMetrics.extract_edge_metrics(dpg_graph, nodes_list)


# ---------------------------------------------------------------------------
# Node metrics
# ---------------------------------------------------------------------------


class TestNodeMetrics:
    def test_returns_dataframe(self, node_metrics):
        assert isinstance(node_metrics, pd.DataFrame)

    def test_expected_shape(self, node_metrics):
        assert node_metrics.shape == (31, 7)

    def test_expected_columns(self, node_metrics):
        expected = {
            "Node",
            "Degree",
            "In degree nodes",
            "Out degree nodes",
            "Betweenness centrality",
            "Local reaching centrality",
            "Label",
        }
        assert set(node_metrics.columns) == expected

    def test_degree_range(self, node_metrics):
        assert node_metrics["Degree"].min() == 1
        assert node_metrics["Degree"].max() == 7

    def test_mean_degree(self, node_metrics):
        assert round(node_metrics["Degree"].mean(), 4) == 3.2903

    def test_in_out_degree_sum_equals_degree(self, node_metrics):
        computed = node_metrics["In degree nodes"] + node_metrics["Out degree nodes"]
        pd.testing.assert_series_equal(
            computed, node_metrics["Degree"], check_names=False
        )

    def test_betweenness_centrality_range(self, node_metrics):
        bc = node_metrics["Betweenness centrality"]
        assert bc.min() == pytest.approx(0.0)
        assert bc.max() == pytest.approx(0.072414, abs=1e-4)
        assert (bc >= 0).all()
        assert (bc <= 1).all()

    def test_local_reaching_centrality_range(self, node_metrics):
        lrc = node_metrics["Local reaching centrality"]
        assert lrc.min() == pytest.approx(0.0)
        assert lrc.max() == pytest.approx(0.927617, abs=1e-4)
        assert (lrc >= 0).all()
        assert (lrc <= 1).all()

    def test_class_nodes_have_zero_out_degree(self, node_metrics):
        class_rows = node_metrics[node_metrics["Label"].str.startswith("Class ")]
        assert (class_rows["Out degree nodes"] == 0).all()

    def test_class_nodes_have_positive_in_degree(self, node_metrics):
        class_rows = node_metrics[node_metrics["Label"].str.startswith("Class ")]
        assert (class_rows["In degree nodes"] > 0).all()

    def test_all_nodes_have_labels(self, node_metrics):
        assert node_metrics["Label"].notna().all()
        assert (node_metrics["Label"].str.len() > 0).all()


# ---------------------------------------------------------------------------
# Edge metrics
# ---------------------------------------------------------------------------


class TestEdgeMetrics:
    def test_returns_dataframe(self, edge_metrics):
        assert isinstance(edge_metrics, pd.DataFrame)

    def test_expected_row_count(self, edge_metrics):
        assert len(edge_metrics) == 51

    def test_expected_columns(self, edge_metrics):
        expected = {
            "Edge",
            "Weight",
            "Node_u_label",
            "Node_v_label",
            "Source_id",
            "Target_id",
        }
        assert set(edge_metrics.columns) == expected

    def test_weight_range(self, edge_metrics):
        assert edge_metrics["Weight"].min() == 1.0
        assert edge_metrics["Weight"].max() == 70.0

    def test_total_weight(self, edge_metrics):
        assert edge_metrics["Weight"].sum() == 1159.0

    def test_all_weights_positive(self, edge_metrics):
        assert (edge_metrics["Weight"] > 0).all()

    def test_edge_ids_are_unique(self, edge_metrics):
        assert edge_metrics["Edge"].nunique() == len(edge_metrics)

    def test_source_target_labels_present(self, edge_metrics):
        assert edge_metrics["Node_u_label"].notna().all()
        assert edge_metrics["Node_v_label"].notna().all()

    def test_edges_to_class_nodes_exist(self, edge_metrics):
        """At least some edges should target class nodes."""
        class_targets = edge_metrics[
            edge_metrics["Node_v_label"].str.startswith("Class ")
        ]
        assert len(class_targets) >= 3  # at least one edge per class


# ---------------------------------------------------------------------------
# Graph-level metrics (LPA communities + class boundaries)
# ---------------------------------------------------------------------------


class TestGraphMetrics:
    @pytest.fixture(scope="class")
    def graph_metrics(self, iris_dpg):
        dpg_graph, nodes_list, target_names = iris_dpg
        return GraphMetrics.extract_graph_metrics_lpa(
            dpg_graph, nodes_list, target_names=target_names
        )

    def test_returns_dict(self, graph_metrics):
        assert isinstance(graph_metrics, dict)

    def test_contains_communities_key(self, graph_metrics):
        assert "Communities" in graph_metrics

    def test_contains_class_bounds_key(self, graph_metrics):
        assert "Class Bounds" in graph_metrics

    def test_community_count(self, graph_metrics):
        assert len(graph_metrics["Communities"]) == 3

    def test_class_bounds_has_all_classes(self, graph_metrics):
        bounds = graph_metrics["Class Bounds"]
        assert sorted(bounds.keys()) == ["Class 0", "Class 1", "Class 2"]

    def test_class_bounds_contain_predicates(self, graph_metrics):
        """Each class boundary list should have at least one predicate string."""
        predicate_re = re.compile(r"(<=|>|<)")
        for cls, preds in graph_metrics["Class Bounds"].items():
            assert len(preds) > 0, f"{cls} has no boundary predicates"
            for p in preds:
                assert predicate_re.search(p), f"Invalid predicate format: {p!r}"


# ---------------------------------------------------------------------------
# Class boundaries (community-based extract_class_boundaries)
# ---------------------------------------------------------------------------


class TestClassBoundaries:
    @pytest.fixture(scope="class")
    def class_boundaries(self, iris_dpg):
        dpg_graph, nodes_list, target_names = iris_dpg
        return GraphMetrics.extract_class_boundaries(
            dpg_graph, nodes_list, target_names=target_names
        )

    def test_returns_dict_with_class_bounds_key(self, class_boundaries):
        assert "Class Bounds" in class_boundaries

    def test_all_three_classes_present(self, class_boundaries):
        keys = sorted(class_boundaries["Class Bounds"].keys())
        assert keys == ["Class 0", "Class 1", "Class 2"]

    def test_each_class_has_boundaries(self, class_boundaries):
        for cls, preds in class_boundaries["Class Bounds"].items():
            assert len(preds) >= 1, f"{cls} has no boundaries"

    def test_boundary_predicate_format(self, class_boundaries):
        """Boundaries should match patterns like 'feat <= val', 'feat > val', or 'val < feat <= val'."""
        valid_re = re.compile(
            r"^([\w\s()]+\s*(<=|>)\s*[\d.eE+-]+|[\d.eE+-]+\s*<\s*[\w\s()]+\s*<=\s*[\d.eE+-]+)$"
        )
        for cls, preds in class_boundaries["Class Bounds"].items():
            for p in preds:
                assert valid_re.match(p.strip()), (
                    f"Predicate {p!r} for {cls} doesn't match expected pattern"
                )


# ---------------------------------------------------------------------------
# Clustering (absorbing Markov chain)
# ---------------------------------------------------------------------------


class TestClustering:
    @pytest.fixture(scope="class")
    def clustering_results(self, iris_dpg):
        dpg_graph, nodes_list, _ = iris_dpg
        class_nodes = {n[0]: n[1] for n in nodes_list if "Class" in n[1]}
        clusters, node_prob, confidence = GraphMetrics.clustering(
            dpg_graph, class_nodes, threshold=0.2
        )
        return clusters, node_prob, confidence

    def test_cluster_keys_include_all_classes(self, clustering_results):
        clusters, _, _ = clustering_results
        assert "Class 0" in clusters
        assert "Class 1" in clusters
        assert "Class 2" in clusters

    def test_ambiguous_cluster_exists_with_threshold(self, clustering_results):
        clusters, _, _ = clustering_results
        assert "Ambiguous" in clusters

    def test_ambiguous_cluster_empty(self, clustering_results):
        """With a 0.2 threshold on Iris, no ambiguous nodes are expected."""
        clusters, _, _ = clustering_results
        assert len(clusters["Ambiguous"]) == 0

    def test_cluster_sizes(self, clustering_results):
        clusters, _, _ = clustering_results
        assert len(clusters["Class 0"]) == 7
        assert len(clusters["Class 1"]) == 12
        assert len(clusters["Class 2"]) == 12

    def test_total_clustered_nodes_match_graph(self, clustering_results, iris_dpg):
        clusters, _, _ = clustering_results
        dpg_graph, _, _ = iris_dpg
        total = sum(len(v) for v in clusters.values())
        assert total == dpg_graph.number_of_nodes()

    def test_node_probabilities_sum_to_one(self, clustering_results):
        _, node_prob, _ = clustering_results
        for node_id, probs in node_prob.items():
            total = sum(probs.values())
            # Clustering rounds each class probability to 2 decimals,
            # so the sum can drift by up to n_classes * 0.005.
            assert total == pytest.approx(1.0, abs=0.05), (
                f"Node {node_id} probs sum to {total}"
            )

    def test_class_node_probability_is_one_for_own_class(self, clustering_results, iris_dpg):
        _, node_prob, _ = clustering_results
        _, nodes_list, _ = iris_dpg
        class_nodes = {n[0]: n[1] for n in nodes_list if "Class" in n[1]}
        for node_id, label in class_nodes.items():
            probs = node_prob[node_id]
            assert probs[label] == pytest.approx(1.0)

    def test_confidence_values_count(self, clustering_results, iris_dpg):
        _, _, confidence = clustering_results
        dpg_graph, _, _ = iris_dpg
        assert len(confidence) == dpg_graph.number_of_nodes()

    def test_confidence_values_in_valid_range(self, clustering_results):
        _, _, confidence = clustering_results
        for node_id, conf in confidence.items():
            assert 0.0 <= conf <= 1.0, f"Node {node_id} confidence {conf} out of range"

    def test_clustering_without_threshold(self, iris_dpg):
        """Without threshold=None, there should be no Ambiguous cluster."""
        dpg_graph, nodes_list, _ = iris_dpg
        class_nodes = {n[0]: n[1] for n in nodes_list if "Class" in n[1]}
        clusters, _, _ = GraphMetrics.clustering(dpg_graph, class_nodes, threshold=None)
        assert "Ambiguous" not in clusters
        total = sum(len(v) for v in clusters.values())
        assert total == dpg_graph.number_of_nodes()


# ---------------------------------------------------------------------------
# Wine dataset (different dimensionality, same API)
# ---------------------------------------------------------------------------


class TestMetricsOnWine:
    @pytest.fixture(scope="class")
    def wine_dpg(self):
        wine = load_wine()
        X_train, _, y_train, _ = train_test_split(
            wine.data, wine.target, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        target_names = np.unique(wine.target).astype(str).tolist()
        dpg = DecisionPredicateGraph(
            model=model,
            feature_names=wine.feature_names,
            target_names=target_names,
        )
        dot = dpg.fit(X_train)
        dpg_graph, nodes_list = dpg.to_networkx(dot)
        return dpg_graph, nodes_list, target_names

    def test_node_metrics_shape(self, wine_dpg):
        dpg_graph, nodes_list, _ = wine_dpg
        df = NodeMetrics.extract_node_metrics(dpg_graph, nodes_list)
        assert len(df) == 87

    def test_edge_metrics_shape(self, wine_dpg):
        dpg_graph, nodes_list, _ = wine_dpg
        df = EdgeMetrics.extract_edge_metrics(dpg_graph, nodes_list)
        assert len(df) == 126

    def test_edge_weight_range(self, wine_dpg):
        dpg_graph, nodes_list, _ = wine_dpg
        df = EdgeMetrics.extract_edge_metrics(dpg_graph, nodes_list)
        assert df["Weight"].min() == 1.0
        assert df["Weight"].max() == 57.0

    def test_class_boundaries_per_class(self, wine_dpg):
        dpg_graph, nodes_list, target_names = wine_dpg
        cb = GraphMetrics.extract_class_boundaries(
            dpg_graph, nodes_list, target_names=target_names
        )
        bounds = cb["Class Bounds"]
        assert sorted(bounds.keys()) == ["Class 0", "Class 1", "Class 2"]
        assert len(bounds["Class 0"]) == 8
        assert len(bounds["Class 1"]) == 11
        assert len(bounds["Class 2"]) == 9


# ---------------------------------------------------------------------------
# PR #32 addition: extract_communities defensive guard against regression DPGs
# ---------------------------------------------------------------------------


class TestExtractCommunitiesClassifierGuard:
    """``GraphMetrics.extract_communities`` must still work on classifiers
    and raise a clear error on regression DPGs (which have only ``Pred ``
    sinks, no ``Class `` sink)."""

    def test_classifier_dpg_still_returns_communities(self):
        from sklearn.datasets import load_iris

        from dpg import DPGExplainer

        iris = load_iris()
        model = RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=1).fit(
            iris.data, iris.target
        )
        explainer = DPGExplainer(
            model,
            iris.feature_names,
            target_names=["0", "1", "2"],
            dpg_config=_config_for_test(),
        )
        explanation = explainer.explain_global(iris.data, communities=True)

        assert explanation.communities is not None
        assert "Clusters" in explanation.communities
        assert "Probability" in explanation.communities
        assert "Confidence Interval" in explanation.communities

    def test_direct_call_on_regression_dpg_raises_clear_error(self):
        """``GraphMetrics.extract_communities`` must raise a clear
        ``ValueError`` instead of an opaque numpy LinAlgError when no
        ``Class ...`` sink node is present."""
        from sklearn.datasets import load_diabetes

        from dpg.core import DecisionPredicateGraph

        X, y = load_diabetes(return_X_y=True)
        feature_names = [f"f_{i}" for i in range(X.shape[1])]
        model = RandomForestRegressor(
            n_estimators=3, max_depth=3, random_state=0, n_jobs=1
        ).fit(X, y)

        dpg = DecisionPredicateGraph(
            model, feature_names, dpg_config=_config_for_test()
        )
        graph, nodes = dpg.to_networkx(dpg.fit(X))

        df_node_metrics = pd.DataFrame(
            {"Node": [nid for nid, _ in nodes], "Label": [lbl for _, lbl in nodes]}
        )

        with pytest.raises(ValueError, match="requires a classifier DPG"):
            GraphMetrics.extract_communities(graph, df_node_metrics, nodes)


def _config_for_test():
    """Minimal dpg_config for CLI-style tests."""
    return {
        "dpg": {
            "default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
            "graph_construction": {"mode": "execution_trace", "context_order": 1},
        }
    }
