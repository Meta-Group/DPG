"""E6 — regression scope: sink behavior and downstream guards.

Regression leaves are labeled ``"Pred <value>"`` instead of ``"Class <name>"``.
``_context_node`` (dpg/core.py) already treats any ``"Pred "``-prefixed label
as a terminal sink, so DPG-k mechanically builds a graph for regressors at
``context_order`` 2 and "auto" without raising. But unlike classification,
where sink count equals the known, finite number of classes, a regression
sink is only as unique as the 2-decimal-rounded leaf value: two leaves collide
into one sink purely by coincidence of rounding, not because of any modeled
notion of "output". There is therefore no "one sink per output" invariant for
regression in 0.3.0 -- these tests document the current, explicitly
out-of-scope behavior rather than assert a guarantee that does not exist.

``GraphMetrics.extract_communities`` separately assumed a classifier (it
partitions the graph into absorbing "Class " states); called on a regression
DPG it used to crash with an opaque ``numpy.linalg.LinAlgError: Singular
matrix`` because no node was absorbing. It now raises a clear ``ValueError``
instead. This is a defensive fix, not a semantic change: no classifier
behavior is touched, and no regression numeric output changes.
"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from dpg import DPGExplainer
from dpg.core import DecisionPredicateGraph


def _diabetes():
    X, y = load_diabetes(return_X_y=True)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


def _config(context_order):
    return {
        "dpg": {
            "default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
            "graph_construction": {"mode": "execution_trace", "context_order": context_order},
        }
    }


@pytest.mark.parametrize("context_order", [1, 2, "auto"])
def test_regressor_execution_trace_builds_without_crashing(context_order):
    """DPG-k mechanically supports regressors: no invariant is claimed here."""
    X, y, feature_names = _diabetes()
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0, n_jobs=1).fit(X, y)
    dpg = DecisionPredicateGraph(model, feature_names, dpg_config=_config(context_order))
    graph, nodes = dpg.to_networkx(dpg.fit(X))

    assert graph.number_of_nodes() > 0
    leaf_labels = [label for _, label in nodes if str(label).startswith("Pred ")]
    assert leaf_labels, "regression leaves must be labeled 'Pred ...', not 'Class ...'"
    assert not any(str(label).startswith("Class ") for _, label in nodes)


def test_regressor_sink_count_is_not_a_fixed_output_count():
    """Sink count for regression tracks rounding collisions, not a fixed
    'number of outputs' the way classification tracks class count."""
    X, y, feature_names = _diabetes()
    model = RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=1).fit(X, y)
    dpg = DecisionPredicateGraph(model, feature_names, dpg_config=_config(1))
    graph, nodes = dpg.to_networkx(dpg.fit(X))

    distinct_sinks = {label for _, label in nodes if str(label).startswith("Pred ")}
    total_leaves = sum(int(tree.tree_.n_leaves) for tree in model.estimators_)
    # There is no guarantee these match; regression sinks are an artifact of
    # rounding collisions, not a modeled invariant. Document the relationship
    # instead of asserting equality with any "number of outputs".
    assert 0 < len(distinct_sinks) <= total_leaves


def test_regressor_class_boundaries_are_empty_by_design():
    """extract_class_boundaries only recognizes 'Class ' sinks; for a
    regression DPG it returns an empty mapping rather than raising. This is
    existing, documented behavior -- guard against a silent regression."""
    X, y, feature_names = _diabetes()
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0, n_jobs=1).fit(X, y)
    explainer = DPGExplainer(model, feature_names, target_names=["prediction"])
    explanation = explainer.explain_global(X)

    assert explanation.class_boundaries == {"Class Bounds": {}}


def test_regressor_communities_raises_clear_error_not_linalg_crash():
    """explain_global(communities=True) on a regressor must fail loudly and
    clearly, not with an internal numpy.linalg.LinAlgError."""
    X, y, feature_names = _diabetes()
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0, n_jobs=1).fit(X, y)
    explainer = DPGExplainer(model, feature_names, target_names=["prediction"])

    with pytest.raises(ValueError, match="requires a classifier DPG"):
        explainer.explain_global(X, communities=True)


def test_classifier_communities_unaffected_by_regression_guard():
    """The new guard must not change classifier behavior at all."""
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()
    model = RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=1).fit(iris.data, iris.target)
    target_names = np.unique(iris.target).astype(str).tolist()
    explainer = DPGExplainer(model, iris.feature_names, target_names=target_names)
    explanation = explainer.explain_global(iris.data, communities=True)

    assert explanation.communities is not None
    assert "Clusters" in explanation.communities
