import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dpg import DPGExplainer
from dpg.visualizer import (
    class_feature_predicate_counts,
    class_lookup_from_target_names,
    plot_dpg_class_bounds_vs_dataset_feature_ranges,
    plot_lrc_vs_rf_importance,
    plot_top_lrc_predicate_splits,
)


def _build_explanation():
    base_dir = os.getcwd()
    dataset_path = os.path.join(base_dir, "datasets", "custom.csv")
    dataset_raw = pd.read_csv(dataset_path, index_col=0)

    X = dataset_raw.iloc[:, :-1]
    y = dataset_raw.iloc[:, -1]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    X = np.round(X, 2)

    model = RandomForestClassifier(n_estimators=5, random_state=27)
    model.fit(X, y)

    target_names = np.unique(y).astype(str).tolist()
    explainer = DPGExplainer(
        model=model,
        feature_names=X.columns,
        target_names=target_names,
    )
    explanation = explainer.explain_global(X.values, communities=True)
    return explainer, explanation, X, y


def test_additional_visualization_apis(tmp_path):
    explainer, explanation, X, y = _build_explanation()

    heat = class_feature_predicate_counts(explanation)
    assert not heat.empty

    fig_lrc = plot_lrc_vs_rf_importance(
        explanation=explanation,
        model=explainer.builder.model,
        X_df=X,
        top_k=8,
        dataset_name="Custom",
        save_path=str(tmp_path / "lrc_vs_rf.png"),
        show=False,
    )
    assert fig_lrc is not None
    assert (tmp_path / "lrc_vs_rf.png").exists()

    fig_split = plot_top_lrc_predicate_splits(
        explanation=explanation,
        X_df=X,
        y=y,
        top_predicates=6,
        top_features=2,
        dataset_name="Custom",
        save_path=str(tmp_path / "top_lrc_splits.png"),
        show=False,
    )
    assert fig_split is not None
    assert (tmp_path / "top_lrc_splits.png").exists()

    lookup = class_lookup_from_target_names(explainer.builder.target_names)
    fig_bounds = plot_dpg_class_bounds_vs_dataset_feature_ranges(
        explanation=explanation,
        X_df=X,
        y=y,
        dataset_name="Custom",
        top_features=3,
        class_lookup=lookup,
        save_path=str(tmp_path / "bounds_vs_dataset.png"),
        show=False,
    )
    assert fig_bounds is not None
    assert (tmp_path / "bounds_vs_dataset.png").exists()


def test_local_path_on_dpg_plot_api(tmp_path):
    explainer, _, X, _ = _build_explanation()
    sample = X.iloc[0].to_numpy()
    local = explainer.explain_local(sample=sample, sample_id="s0", validate_graph=True)
    num_paths = len(local.tree_paths)
    assert num_paths > 0

    selected = [0] if num_paths == 1 else [0, 1]
    fig = explainer.plot_local_on_dpg(
        plot_name="local_on_dpg",
        local_explanation=local,
        save_dir=str(tmp_path),
        path_indices=selected,
        show=False,
        export_pdf=False,
    )
    assert fig is not None
    sample_token = str(local.sample_id).replace(" ", "_")
    assert (tmp_path / f"local_on_dpg_sid_{sample_token}.png").exists()
