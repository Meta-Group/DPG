import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dpg import DPGExplainer
from dpg.visualizer import (
    class_feature_predicate_counts,
    class_lookup_from_target_names,
    plot_class_feature_complexity,
    plot_dpg_class_bounds_vs_dataset_feature_ranges,
    plot_lrc_vs_rf_importance,
    plot_sample_using_bc_weights,
    plot_top_lrc_predicate_splits,
    sample_bc_weights,
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

    bc_weights = sample_bc_weights(
        explanation=explanation,
        X_df=X,
        top_k=8,
    )
    assert len(bc_weights) == len(X)
    assert np.all(np.asarray(bc_weights) >= 0)

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

    fig_bc = plot_sample_using_bc_weights(
        explanation=explanation,
        X_df=X,
        y=y,
        top_k=8,
        dataset_name="Custom",
        class_names=explainer.builder.target_names,
        save_path=str(tmp_path / "sample_bc_weights.png"),
        show=False,
    )
    assert fig_bc is not None
    assert (tmp_path / "sample_bc_weights.png").exists()

    fig_heat, fig_bars = plot_class_feature_complexity(
        heat_df=heat,
        dataset_name="Custom",
        class_names=explainer.builder.target_names,
        top_n_features=3,
        save_prefix=str(tmp_path / "community_complexity"),
        show=False,
    )
    assert fig_heat is not None
    assert fig_bars is not None
    assert (tmp_path / "community_complexity_heatmap.png").exists()
    assert (tmp_path / "community_complexity_bars.png").exists()

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
