import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from dpg.core import DPGError, DecisionPredicateGraph


def _forest():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=5, random_state=7, n_jobs=1)
    model.fit(iris.data, iris.target)
    return iris, model


def _config(mode="execution_trace", context_order=1, decimal_threshold=6):
    return {
        "dpg": {
            "default": {
                "perc_var": 1e-9,
                "decimal_threshold": decimal_threshold,
                "n_jobs": 1,
            },
            "graph_construction": {
                "mode": mode,
                "context_order": context_order,
            },
        }
    }


def test_context_order_requires_execution_trace():
    iris, model = _forest()
    with pytest.raises(DPGError, match="requires mode='execution_trace'"):
        DecisionPredicateGraph(
            model, iris.feature_names, dpg_config=_config("aggregated_transitions", 2)
        )


def test_auto_context_has_one_sink_per_class_and_no_local_violations():
    iris, model = _forest()
    dpg = DecisionPredicateGraph(
        model,
        iris.feature_names,
        target_names=["0", "1", "2"],
        dpg_config=_config(context_order="auto"),
    )
    graph, nodes = dpg.to_networkx(dpg.fit(iris.data))

    assert dpg.get_context_order() >= 1
    assert dpg.get_context_order_history()[dpg.get_context_order()] == 0
    sinks = [label for _, label in nodes if label.startswith("Class ")]
    assert sorted(sinks) == ["Class 0", "Class 1", "Class 2"]
    assert all("context" in data and "predicate" in data for _, data in graph.nodes(data=True))


def test_k1_execution_trace_matches_legacy_edge_weights():
    iris, model = _forest()
    dpg = DecisionPredicateGraph(
        model,
        iris.feature_names,
        target_names=["0", "1", "2"],
        dpg_config=_config(context_order=1),
    )
    log = dpg._extract_trace_log(iris.data)
    assert dpg.discover_dfg(log) == dpg.discover_dfg_execution_trace(log)


def test_integer_data_auto_precision_is_lossless():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 20, size=(200, 3)).astype(float)
    y = (X[:, 0] > 9).astype(int)
    model = RandomForestClassifier(n_estimators=3, random_state=3, n_jobs=1).fit(X, y)
    dpg = DecisionPredicateGraph(
        model,
        ["a", "b", "c"],
        target_names=["0", "1"],
        dpg_config=_config(context_order=1, decimal_threshold="auto"),
    )
    dpg.fit(X)
    assert dpg.get_decimal_threshold() == 1
