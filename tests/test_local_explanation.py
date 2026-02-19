import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from dpg import DPGExplainer


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
    assert local.sample_confidence["vote_confidence"] is not None

    sample_df = X.iloc[[sample_idx]]
    model_pred = str(model.predict(sample_df)[0])
    assert model_pred in local.class_votes


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
