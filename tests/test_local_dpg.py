import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from dpg.context_order import path_violations
from dpg.local_dpg import LocalTrace, PredicateStep, build_local_dpg, crossing_value, intervene, merge_traces


def _models(seed=0):
    return {
        "rf": RandomForestClassifier(n_estimators=15, max_depth=4, random_state=seed),
        "et": ExtraTreesClassifier(n_estimators=15, max_depth=4, random_state=seed),
        "bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3), n_estimators=10, max_features=0.6, random_state=seed
        ),
        "adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2), n_estimators=10, random_state=seed
        ),
        "gbm": GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=seed),
    }


DATASETS = {"iris": load_iris, "breast_cancer": load_breast_cancer, "wine": load_wine}


@pytest.fixture(scope="module")
def fitted():
    out = {}
    for ds_name, loader in DATASETS.items():
        data = loader()
        X, y = data.data, data.target
        for name, model in _models().items():
            out[(ds_name, name)] = (model.fit(X, y), X, list(data.feature_names), [str(c) for c in data.target_names])
    return out


@pytest.mark.parametrize("dataset", sorted(DATASETS))
@pytest.mark.parametrize("family", ["rf", "et", "bagging", "adaboost", "gbm"])
def test_explained_class_is_model_prediction(fitted, dataset, family):
    model, X, features, targets = fitted[(dataset, family)]
    for i in range(0, len(X), max(1, len(X) // 25)):
        local = build_local_dpg(model, X[i], features, targets, compute_pivots=False)
        assert local.is_output_faithful
        assert local.model_prediction == targets[int(model.predict(X[i : i + 1])[0])]
        assert sum(local.class_support.values()) == pytest.approx(1.0)
        assert local.decomposition_residual < 1e-9
        assert local.top_competitor != local.predicted_class


@pytest.mark.parametrize("family", ["rf", "bagging", "gbm"])
def test_edges_store_the_trees_that_executed_them(fitted, family):
    model, X, features, targets = fitted[("wine", family)]
    local = build_local_dpg(model, X[3], features, targets, context_order=1, compute_pivots=False)
    by_tree = {trace.tree_index: trace.labels for trace in local.traces}
    for u, v, data in local.graph.edges(data=True):
        for tree in data["trees"]:
            labels = by_tree[tree]
            if u == ("source",):
                assert local.graph.nodes[v]["label"] == labels[0]
            else:
                pair = (local.graph.nodes[u]["label"], local.graph.nodes[v]["label"])
                assert pair in set(zip(labels, labels[1:]))


@pytest.mark.parametrize("family", ["rf", "et", "gbm"])
def test_routes_at_resolved_order_are_executed_traces(fitted, family):
    model, X, features, targets = fitted[("breast_cancer", family)]
    for i in (0, 50, 200, 400):
        local = build_local_dpg(model, X[i], features, targets, compute_pivots=False)
        assert local.context_history[local.context_order] == 0
        assert set(local.graph_routes()) == set(local.routes())


def _trace(tree_index, labels, outcome):
    steps = [PredicateStep(0, 0, "f", 0.0, "<=", label, 0.0) for label in labels]
    return LocalTrace(tree_index, steps, 0, outcome, np.zeros(2))


def test_reviewer_counterexample_has_phantom_route_at_k1_only():
    traces = [_trace(0, ["A", "B", "C"], "0"), _trace(1, ["D", "B", "E"], "1")]
    labels = [t.labels for t in traces]
    assert path_violations(labels, 1) > 0
    assert path_violations(labels, 2) == 0
    k1 = merge_traces(traces, 1, 2)
    nodes = {d["label"]: n for n, d in k1.nodes(data=True)}
    phantom = [("source",), nodes["A"], nodes["B"], nodes["E"]]
    assert all(k1.has_edge(u, v) for u, v in zip(phantom, phantom[1:]))
    k2 = merge_traces(traces, 2, 2)
    assert k2.number_of_nodes() == k1.number_of_nodes() + 1


@pytest.mark.parametrize("threshold", [0.5, -1.25, 3.1415926535, 1e-7, 2.4999999])
def test_crossing_value_switches_branch_under_float32_routing(threshold):
    for value in (threshold - 1.0, threshold + 1.0, threshold):
        crossed = crossing_value(value, threshold)
        before_left = float(np.float32(value)) <= threshold
        after_left = float(np.float32(crossed)) <= threshold
        assert before_left != after_left
        assert abs(crossed - threshold) < 1e-5 * max(1.0, abs(threshold))


@pytest.mark.parametrize("family", ["rf", "bagging", "adaboost", "gbm"])
def test_pivots_reroute_trees_and_critical_is_deterministic(fitted, family):
    model, X, features, targets = fitted[("iris", family)]
    local = build_local_dpg(model, X[70], features, targets)
    again = build_local_dpg(model, X[70], features, targets)
    assert [p.label for p in local.pivots] == [p.label for p in again.pivots]
    gains = [p.competitor_gain for p in local.pivots]
    assert gains == sorted(gains, reverse=True)
    critical = local.critical_predicate
    if critical is None:
        return
    x_new = X[70].copy()
    x_new[critical.feature_index] = critical.crossing_value
    trees = model.estimators_.ravel() if family == "gbm" else model.estimators_
    feature_maps = getattr(model, "estimators_features_", None)
    for t in critical.trees:
        cols = feature_maps[t] if feature_maps is not None else slice(None)
        before = trees[t].apply(X[70][cols].reshape(1, -1))[0]
        after = trees[t].apply(x_new[cols].reshape(1, -1))[0]
        assert before != after
    result = intervene(model, X[70], critical, targets.index(local.top_competitor))
    assert result["perturbation"] == pytest.approx(critical.slack, abs=1e-5)
    assert np.isfinite(result["target_delta"])


def test_rejects_regressors_and_bad_inputs(fitted):
    from sklearn.ensemble import RandomForestRegressor

    X, y = load_iris(return_X_y=True)
    reg = RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y)
    with pytest.raises(NotImplementedError):
        build_local_dpg(reg, X[0], ["a", "b", "c", "d"])
    model, X, features, targets = fitted[("iris", "rf")]
    with pytest.raises(ValueError):
        build_local_dpg(model, X[0][:3], features, targets)
    with pytest.raises(ValueError):
        build_local_dpg(model, X[0], features, targets, context_order=0)


def test_explainer_entry_point_matches_function(fitted):
    from dpg import DPGExplainer

    model, X, features, targets = fitted[("wine", "gbm")]
    explainer = DPGExplainer(model, features, targets, dpg_config={"dpg": {"default": {"decimal_threshold": 4}}})
    local = explainer.explain_local_dpg(X[10], sample_id=10)
    direct = build_local_dpg(model, X[10], features, targets, decimal_threshold=4, sample_id=10)
    assert local.summary() == direct.summary()
    assert all(step.label.split()[-1] == str(round(step.threshold, 4)) for t in local.traces for step in t.steps)

    auto = DPGExplainer(model, features, targets, dpg_config={"dpg": {"default": {"decimal_threshold": "auto"}}})
    with pytest.raises(ValueError):
        auto.explain_local_dpg(X[10])
