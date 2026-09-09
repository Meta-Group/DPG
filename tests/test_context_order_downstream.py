"""E7 -- downstream compatibility of context_order > 1 / "auto".

Exercises the consumers named in the DPG 0.3.0 execution plan (plot_dpg,
class-boundary extraction, communities, DPGExplainer global/local
explanations, faithfulness evaluation) at context_order 2 and "auto", to
confirm DPG-k does not silently break them. No DPG-IF/DPG-CF consumers exist
in this codebase, so there is nothing to test for those.

Two pre-existing issues surfaced during this investigation are documented
here rather than fixed, because they are independent of context_order (both
reproduce identically at the k=1 default) and are therefore out of E7's
scope ("fix compatibility issues only within the requested scope"):

- `evaluate_faithfulness()` raises inside sklearn's own GradientBoosting
  internals because `SklearnEnsembleNormalizer.normalize()` flattens
  `estimators_` from a 2D array to a list for DPG's own tree iteration, and
  `evaluate_faithfulness` calls `.predict()` on that same normalized copy.
- `plot_dpg` does not visually disambiguate two nodes that share a predicate
  label but differ in context (the DOT `tooltip` carries context, but static
  PNG/PDF renders do not show tooltips). `get_node_context(node)` remains the
  documented way to recover per-node context.
"""

import os
import shutil

os.environ.setdefault("MPLBACKEND", "Agg")

import networkx as nx
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from dpg import DPGExplainer


def _require_graphviz_dot():
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' executable is unavailable")


def _iris():
    iris = load_iris()
    target_names = np.unique(iris.target).astype(str).tolist()
    return iris, target_names


def _config(context_order):
    return {
        "dpg": {
            "default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
            "graph_construction": {"mode": "execution_trace", "context_order": context_order},
        }
    }


@pytest.mark.parametrize("context_order", [1, 2, "auto"])
def test_to_networkx_two_tuple_shape_preserved(context_order):
    """Public API contract: to_networkx always returns (graph, nodes)."""
    iris, target_names = _iris()
    model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(iris.data, iris.target)
    explainer = DPGExplainer(model, iris.feature_names, target_names=target_names, dpg_config=_config(context_order))
    explainer.fit(iris.data)

    result = explainer.builder.to_networkx(explainer._dot)
    assert isinstance(result, tuple)
    assert len(result) == 2
    graph, nodes = result
    assert isinstance(graph, nx.DiGraph)
    assert isinstance(nodes, list)


@pytest.mark.parametrize("context_order", [2, "auto"])
def test_node_attributes_preserved_at_higher_context_order(context_order):
    iris, target_names = _iris()
    model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(iris.data, iris.target)
    explainer = DPGExplainer(model, iris.feature_names, target_names=target_names, dpg_config=_config(context_order))
    explanation = explainer.explain_global(iris.data)

    assert all(
        {"predicate", "context", "context_order"} <= data.keys()
        for _, data in explanation.graph.nodes(data=True)
    )
    resolved_k = explainer.builder.get_context_order()
    assert resolved_k >= 2


@pytest.mark.parametrize("context_order", [1, 2, "auto"])
def test_class_boundaries_and_communities_at_context_order(context_order):
    iris, target_names = _iris()
    model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(iris.data, iris.target)
    explainer = DPGExplainer(model, iris.feature_names, target_names=target_names, dpg_config=_config(context_order))
    explanation = explainer.explain_global(iris.data, communities=True)

    assert sorted(explanation.class_boundaries["Class Bounds"].keys()) == ["Class 0", "Class 1", "Class 2"]
    assert explanation.communities is not None
    assert set(explanation.communities.keys()) == {"Clusters", "Probability", "Confidence Interval"}


def test_local_explanation_class_votes_are_stable_across_context_order():
    """context_order changes predicate *identity*, not routing: the same
    sample must still land on the same per-tree leaf and the same class
    votes at k=1, k=2, and k='auto'."""
    iris, target_names = _iris()
    model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(iris.data, iris.target)
    sample = iris.data[0]

    votes_by_k = {}
    for context_order in (1, 2, "auto"):
        explainer = DPGExplainer(
            model, iris.feature_names, target_names=target_names, dpg_config=_config(context_order)
        )
        explainer.fit(iris.data)
        local = explainer.explain_local(sample=sample, sample_id=0)
        votes_by_k[context_order] = (local.majority_vote, dict(local.class_votes))

    assert votes_by_k[1] == votes_by_k[2] == votes_by_k["auto"]


@pytest.mark.parametrize("context_order", [1, 2, "auto"])
def test_faithfulness_evaluation_at_context_order_random_forest(context_order):
    iris, target_names = _iris()
    model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(iris.data, iris.target)
    explainer = DPGExplainer(model, iris.feature_names, target_names=target_names, dpg_config=_config(context_order))
    explainer.fit(iris.data)

    details = explainer.evaluate_faithfulness(iris.data[:20], y_true=iris.target[:20], return_details=True)

    assert 0.0 <= details["faithfulness_score"] <= 1.0
    assert 0.0 <= details["output_fidelity"] <= 1.0
    assert "mean_trace_coverage_score" in details
    assert "mean_recombination_rate" in details


@pytest.mark.parametrize("context_order", [2, "auto"])
def test_plot_dpg_and_communities_render_at_context_order(tmp_path, context_order):
    _require_graphviz_dot()
    iris, target_names = _iris()
    model = RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=1).fit(iris.data, iris.target)
    explainer = DPGExplainer(model, iris.feature_names, target_names=target_names, dpg_config=_config(context_order))
    explanation = explainer.explain_global(iris.data, communities=True)

    explainer.plot(f"dpg_k_{context_order}", explanation, save_dir=str(tmp_path), show=False)
    assert (tmp_path / f"dpg_k_{context_order}.png").exists()

    explainer.plot_communities(f"dpg_k_{context_order}_communities", explanation, save_dir=str(tmp_path), show=False)
    assert (tmp_path / f"dpg_k_{context_order}_communities_communities.png").exists()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing bug, independent of context_order: SklearnEnsembleNormalizer "
        "flattens GradientBoostingClassifier.estimators_ from a (n_stages, n_classes) "
        "ndarray to a flat list for DPG's own traversal, and evaluate_faithfulness() "
        "calls .predict() on that same normalized copy, so sklearn's internal "
        "estimators_[0, 0] 2D indexing fails. Reproduces identically at k=1 with the "
        "default aggregated_transitions mode; out of scope for E7 (context_order "
        "compatibility), tracked here so it is not silently lost."
    ),
)
def test_gradient_boosting_faithfulness_known_issue_reproduces_at_k1():
    iris, target_names = _iris()
    model = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(iris.data, iris.target)
    explainer = DPGExplainer(model, iris.feature_names, target_names=target_names)
    explainer.fit(iris.data)
    explainer.evaluate_faithfulness(iris.data[:20], y_true=iris.target[:20], return_details=True)
