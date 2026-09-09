import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from dpg.context_order import resolve_context_order
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


@pytest.mark.parametrize("context_order", [2, "auto"])
def test_context_order_requires_execution_trace(context_order):
    iris, model = _forest()
    with pytest.raises(DPGError, match="requires mode='execution_trace'"):
        DecisionPredicateGraph(
            model,
            iris.feature_names,
            dpg_config=_config("aggregated_transitions", context_order),
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


def test_auto_context_detects_longer_history_recombination():
    """Pairwise transitions can be consistent while a longer path is not."""
    traces = [
        ("A", "B", "C", "D", "Class 0"),
        ("X", "B", "C", "E", "Class 1"),
    ]

    resolved, history = resolve_context_order(traces)

    assert history[1] > 0
    assert resolved >= 3
    assert history[resolved] == 0


def test_execution_trace_graph_preserves_long_case_order():
    """A case-id sort must not scramble a long execution trace."""
    iris, model = _forest()
    dpg = DecisionPredicateGraph(
        model,
        iris.feature_names,
        dpg_config=_config(context_order=1),
    )
    labels = tuple(f"step-{index}" for index in range(20))
    log = pd.DataFrame(
        {
            "case:concept:name": ["case-0"] * len(labels),
            "concept:name": labels,
        }
    )

    assert dpg.discover_dfg(log) == {
        (source, target): 1 for source, target in zip(labels, labels[1:])
    }


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
