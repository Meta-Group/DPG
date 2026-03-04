import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from dpg import DPGExplainer, DPGTreePathExplanation


def _load_custom_dataset():
    base_dir = os.getcwd()
    dataset_path = os.path.join(base_dir, "datasets", "custom.csv")
    dataset_raw = pd.read_csv(dataset_path, index_col=0)

    features = dataset_raw.iloc[:, :-1]
    labels = dataset_raw.iloc[:, -1]
    features = features.replace([np.inf, -np.inf], np.nan).fillna(features.mean())
    features = np.round(features, 2)
    return features, labels


def test_explain_local_uses_dpg_graph_paths():
    X, y = _load_custom_dataset()
    model = RandomForestClassifier(n_estimators=7, random_state=27)
    model.fit(X, y)

    explainer = DPGExplainer(
        model=model,
        feature_names=X.columns.tolist(),
        target_names=model.classes_.astype(str).tolist(),
        dpg_config={"dpg": {"default": {"perc_var": 0.0, "decimal_threshold": 6, "n_jobs": 1}}},
    )
    explainer.fit(X.values)

    local = explainer.explain_local(X.values[0], sample_id=0, validate_graph=True)

    assert local.path_mode == "dpg_graph"
    assert local.graph_validated is True
    assert len(local.tree_paths) > 0
    assert local.all_trees_valid is True
    assert all(path.starts_from_root for path in local.tree_paths)
    assert all(path.ends_in_leaf for path in local.tree_paths)
    assert all(all(path.predicate_truths) for path in local.tree_paths)
    assert all(all(path.edge_exists) for path in local.tree_paths)
    assert all(path.mean_lrc is not None for path in local.tree_paths)
    assert all(path.mean_bc is not None for path in local.tree_paths)
    assert all(path.path_confidence is not None for path in local.tree_paths)


def test_explain_local_returns_class_votes_and_confidence():
    X, y = _load_custom_dataset()
    model = RandomForestClassifier(n_estimators=5, random_state=27)
    model.fit(X, y)

    explainer = DPGExplainer(
        model=model,
        feature_names=X.columns.tolist(),
        target_names=model.classes_.astype(str).tolist(),
        dpg_config={"dpg": {"default": {"perc_var": 0.0, "decimal_threshold": 6, "n_jobs": 1}}},
    )
    explainer.fit(X.values)

    sample_idx = 3
    local = explainer.explain_local(X.values[sample_idx], sample_id=sample_idx, validate_graph=True)

    assert isinstance(local.class_votes, dict)
    assert len(local.class_votes) > 0
    assert local.majority_vote in local.class_votes
    assert local.sample_confidence["num_paths"] >= 1
    assert local.sample_confidence["num_active_nodes"] >= 1
    assert local.sample_confidence["mean_lrc_active_nodes"] is not None
    assert local.sample_confidence["mean_bc_active_nodes"] is not None
    assert local.sample_confidence["evidence_score_pred"] is not None
    assert "top_competitor_class_pred" in local.sample_confidence
    assert "evidence_margin_pred_vs_competitor" in local.sample_confidence
    assert local.sample_confidence["evidence_variant"] == "base"
    assert local.sample_confidence["evidence_lambda_rule"] == "constant"
    assert local.sample_confidence["evidence_score_rule"] == "ratio_total"
    assert local.sample_confidence["evidence_lambda"] == 0.8

    sample_df = X.iloc[[sample_idx]]
    model_pred = str(model.predict(sample_df)[0])
    assert model_pred in local.class_votes

    pred = local.majority_vote
    scores = local.sample_confidence.get("evidence_scores", {})
    competitor = local.sample_confidence.get("top_competitor_class_pred")
    margin = local.sample_confidence.get("evidence_margin_pred_vs_competitor")
    if pred in scores:
        if competitor is None:
            assert len(scores) <= 1
            assert margin == scores[pred]
        else:
            assert competitor in scores
            assert np.isclose(margin, scores[pred] - scores[competitor])


def test_explain_local_stress_randomized_samples():
    X, y = make_classification(
        n_samples=220,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=13,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y.astype(str))

    model = RandomForestClassifier(n_estimators=17, random_state=17)
    model.fit(X_df, y_s)

    explainer = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=model.classes_.astype(str).tolist(),
        dpg_config={"dpg": {"default": {"perc_var": 0.0, "decimal_threshold": 6, "n_jobs": 1}}},
    )
    explainer.fit(X_df.values)

    rng = np.random.default_rng(17)
    sample_idxs = rng.choice(X_df.shape[0], size=30, replace=False)
    for idx in sample_idxs:
        local = explainer.explain_local(X_df.values[idx], sample_id=int(idx), validate_graph=True)
        assert len(local.tree_paths) > 0
        assert local.all_trees_valid is True
        assert all(path.starts_from_root for path in local.tree_paths)
        assert all(path.ends_in_leaf for path in local.tree_paths)
        assert all(all(path.predicate_truths) for path in local.tree_paths)
        assert all(all(path.edge_exists) for path in local.tree_paths)


def test_explain_local_supports_execution_trace_graph_construction():
    X, y = make_classification(
        n_samples=180,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        n_classes=3,
        random_state=41,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y.astype(str))

    model = RandomForestClassifier(n_estimators=9, random_state=9)
    model.fit(X_df, y_s)

    explainer = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=model.classes_.astype(str).tolist(),
        dpg_config={
            "dpg": {
                "default": {"perc_var": 0.0, "decimal_threshold": 6, "n_jobs": 1},
                "graph_construction": {"mode": "execution_trace"},
                "local_evidence": {"variant": "top_competitor", "base_lambda": 0.8},
            }
        },
    )
    explainer.fit(X_df.values)
    local = explainer.explain_local(X_df.values[0], sample_id=0, validate_graph=True)

    assert local.path_mode == "execution_trace"
    assert local.sample_confidence["graph_construction_mode"] == "execution_trace"
    assert local.sample_confidence["num_executed_paths"] == len(model.estimators_)
    assert local.sample_confidence["trace_node_coverage"] is not None
    assert local.sample_confidence["trace_edge_coverage"] is not None
    assert local.sample_confidence["trace_node_coverage"] <= 1.0
    assert local.sample_confidence["trace_edge_coverage"] <= 1.0


def test_explain_local_reports_advanced_local_metrics():
    X, y = make_classification(
        n_samples=180,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        n_classes=3,
        random_state=23,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y.astype(str))

    model = RandomForestClassifier(n_estimators=11, max_depth=4, random_state=31)
    model.fit(X_df, y_s)

    explainer = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=model.classes_.astype(str).tolist(),
        dpg_config={
            "dpg": {
                "default": {"perc_var": 0.0, "decimal_threshold": 6, "n_jobs": 1},
                "local_evidence": {"variant": "top_competitor", "base_lambda": 0.8},
            }
        },
    )
    explainer.fit(X_df.values)
    local = explainer.explain_local(X_df.values[0], sample_id=0, validate_graph=True)
    conf = local.sample_confidence

    assert conf["support_pred_class"] is not None
    assert conf["support_pred_score"] is not None
    assert 0.0 <= conf["support_pred_score"] <= 1.0
    assert 0.0 <= conf["path_purity"] <= 1.0
    assert 0.0 <= conf["competitor_exposure"] <= 1.0
    assert np.isclose(conf["path_purity"] + conf["competitor_exposure"], 1.0, atol=1e-6)
    assert 0.0 <= conf["predicted_class_concentration_top3"] <= 1.0
    assert 0.0 <= conf["model_vote_agreement"] <= 1.0
    assert 0.0 <= conf["node_recall"] <= 1.0
    assert 0.0 <= conf["node_precision"] <= 1.0
    assert 0.0 <= conf["edge_recall"] <= 1.0
    assert 0.0 <= conf["edge_precision"] <= 1.0
    assert 0.0 <= conf["recombination_rate"] <= 1.0
    assert 0.0 <= conf["trace_coverage_score"] <= 1.0
    assert 0.0 <= conf["explanation_confidence"] <= 1.0
    assert conf["critical_split_depth"] is None or conf["critical_split_depth"] >= 0


def test_execution_trace_advanced_metrics_have_zero_recombination():
    X, y = make_classification(
        n_samples=200,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=51,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y.astype(str))

    model = RandomForestClassifier(n_estimators=9, max_depth=4, random_state=9)
    model.fit(X_df, y_s)

    explainer = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=model.classes_.astype(str).tolist(),
        dpg_config={
            "dpg": {
                "default": {"perc_var": 0.0, "decimal_threshold": 6, "n_jobs": 1},
                "graph_construction": {"mode": "execution_trace"},
                "local_evidence": {"variant": "top_competitor", "base_lambda": 0.8},
            }
        },
    )
    explainer.fit(X_df.values)
    local = explainer.explain_local(X_df.values[3], sample_id=3, validate_graph=True)
    conf = local.sample_confidence

    assert np.isclose(conf["recombination_rate"], 0.0, atol=1e-12)
    assert 0.0 <= conf["node_recall"] <= 1.0
    assert 0.0 <= conf["edge_recall"] <= 1.0
    assert 0.0 <= conf["explanation_confidence"] <= 1.0
    assert conf["trace_node_count_unique"] >= conf["explanation_node_count_unique"]


def test_root_only_shared_prefix_is_not_a_critical_node():
    explainer = object.__new__(DPGExplainer)
    explainer._trace_reference_sets = lambda sample_arr: {"trace_node_labels": set(), "trace_edge_labels": set()}
    explainer._path_leaf_class = (
        lambda path: path.labels[-1].replace("Class ", "", 1) if str(path.labels[-1]).startswith("Class ") else None
    )
    explainer._path_weight = lambda path: float(path.path_confidence or 0.0)
    explainer._is_leaf_label = lambda label: str(label).startswith("Class ")

    pred_path = DPGTreePathExplanation(
        tree_index=0,
        tree_prefix="source:root",
        labels=["root <= 0.5", "pred > 0.1", "Class A"],
        node_ids=["n0", "n1", "n2"],
        predicate_truths=[True, True, True],
        edge_exists=[True, True],
        starts_from_root=True,
        ends_in_leaf=True,
        graph_path_valid=True,
        path_confidence=0.6,
    )
    comp_path = DPGTreePathExplanation(
        tree_index=1,
        tree_prefix="source:root",
        labels=["root <= 0.5", "comp <= -0.2", "Class B"],
        node_ids=["n0", "n3", "n4"],
        predicate_truths=[True, True, True],
        edge_exists=[True, True],
        starts_from_root=True,
        ends_in_leaf=True,
        graph_path_valid=True,
        path_confidence=0.4,
    )

    conf = explainer._compute_advanced_local_metrics(
        sample_arr=np.array([0.0]),
        tree_paths=[pred_path, comp_path],
        class_support={"A": 0.6, "B": 0.4},
        class_votes={"A": 6, "B": 4},
    )

    assert conf["critical_node_label"] is None
    assert conf["critical_split_depth"] is None
    assert conf["critical_successor_pred"] is None
    assert conf["critical_successor_comp"] is None
    assert conf["critical_node_contrast"] is None
