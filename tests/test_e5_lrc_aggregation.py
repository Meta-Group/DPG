"""Focused tests for scripts/lrc_aggregation.py (E5 LRC-alignment analysis).

These pin down the three aggregation variants compared in E5 before the
broader experiment runner is exercised:

- "sum" must be byte-for-byte identical to the production
  ``DecisionPredicateGraph.get_predicate_lrc()`` (CHANGELOG 0.3.0's shipped
  default) -- this is the contract that keeps the analysis script honest
  about which aggregation is actually shipped.
- "max" and "weighted_sum" are analysis-only variants; verify their formulas
  directly against manually computed values rather than only smoke-testing.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpg.core import DecisionPredicateGraph
from scripts.lrc_aggregation import AGGREGATIONS, _node_traffic, feature_scores, predicate_lrc


def _config(context_order):
    return {
        "dpg": {
            "default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
            "graph_construction": {"mode": "execution_trace", "context_order": context_order},
        }
    }


@pytest.fixture(scope="module")
def contextual_graph():
    """A k>1 graph so predicates genuinely occupy multiple contextual nodes."""
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=10, random_state=7, n_jobs=1).fit(iris.data, iris.target)
    dpg = DecisionPredicateGraph(
        model, iris.feature_names, target_names=["0", "1", "2"], dpg_config=_config(2)
    )
    graph, _ = dpg.to_networkx(dpg.fit(iris.data))
    return dpg, graph, list(iris.feature_names)


def test_unknown_aggregation_rejected(contextual_graph):
    _, graph, _ = contextual_graph
    with pytest.raises(ValueError, match="Unknown aggregation"):
        predicate_lrc(graph, "average")


def test_sum_matches_production_get_predicate_lrc(contextual_graph):
    dpg, graph, _ = contextual_graph
    assert predicate_lrc(graph, "sum") == dpg.get_predicate_lrc(graph)


def test_a_predicate_occupies_multiple_contextual_nodes(contextual_graph):
    """Sanity check that k=2 actually produces the multi-node case this
    module exists to aggregate over; otherwise the other tests are vacuous."""
    _, graph, _ = contextual_graph
    labels = [
        data["predicate"]
        for _, data in graph.nodes(data=True)
        if DecisionPredicateGraph._is_predicate_label(data.get("predicate", ""))
    ]
    counts = {label: labels.count(label) for label in set(labels)}
    assert max(counts.values()) > 1


def test_max_equals_manual_max_per_predicate(contextual_graph):
    _, graph, _ = contextual_graph
    manual = {}
    for node, data in graph.nodes(data=True):
        label = data.get("predicate")
        if label is None or not DecisionPredicateGraph._is_predicate_label(label):
            continue
        value = float(nx.local_reaching_centrality(graph, node, weight=None))
        manual[label] = max(manual.get(label, float("-inf")), value)

    result = predicate_lrc(graph, "max")
    assert result.keys() == manual.keys()
    for label in manual:
        assert result[label] == pytest.approx(manual[label])


def test_weighted_sum_equals_manual_traffic_weighted_sum(contextual_graph):
    _, graph, _ = contextual_graph
    manual = {}
    for node, data in graph.nodes(data=True):
        label = data.get("predicate")
        if label is None or not DecisionPredicateGraph._is_predicate_label(label):
            continue
        value = float(nx.local_reaching_centrality(graph, node, weight=None))
        manual[label] = manual.get(label, 0.0) + value * _node_traffic(graph, node)

    result = predicate_lrc(graph, "weighted_sum")
    assert result.keys() == manual.keys()
    for label in manual:
        assert result[label] == pytest.approx(manual[label])


def test_node_traffic_falls_back_to_out_edges_for_root():
    graph = nx.DiGraph()
    graph.add_node("root", predicate="f <= 1.0")
    graph.add_node("leaf", predicate="Class 0")
    graph.add_edge("root", "leaf", weight=7.0)
    assert _node_traffic(graph, "root") == 7.0
    assert _node_traffic(graph, "leaf") == 7.0


def test_feature_scores_maps_predicates_to_features_and_ignores_sinks():
    predicate_scores = {
        "petal length (cm) <= 2.0": 1.5,
        "petal length (cm) > 2.0": 0.5,
        "sepal width (cm) <= 3.0": 2.0,
        "Class 0": 100.0,
        "unrelated feature <= 5.0": 9.0,
    }
    result = feature_scores(predicate_scores, ["petal length (cm)", "sepal width (cm)"])
    assert result == {"petal length (cm)": pytest.approx(2.0), "sepal width (cm)": pytest.approx(2.0)}


def test_all_three_aggregations_are_exposed():
    assert set(AGGREGATIONS) == {"sum", "max", "weighted_sum"}
