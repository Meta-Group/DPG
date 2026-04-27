import os
import shutil

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from dpg import DPGExplainer
from dpg.visualizer import (
    class_feature_predicate_counts,
    class_lookup_from_target_names,
    plot_class_feature_complexity,
    plot_dpg_class_bounds_vs_dataset_feature_ranges,
    plot_dpg_local_paths_aggregate,
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


def _require_graphviz_dot():
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' executable is unavailable")


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


def test_plot_local_on_dpg_writes_png_and_returns_figure(tmp_path):
    _require_graphviz_dot()
    explainer, explanation, X, y = _build_explanation()
    local_explanation = explainer.explain_local(sample=X.iloc[0].values, sample_id=5)

    fig = explainer.plot_local_on_dpg(
        plot_name="local_paths",
        local_explanation=local_explanation,
        true_class_label=str(y.iloc[0]),
        save_dir=str(tmp_path),
        show=False,
    )

    assert fig is not None
    assert (tmp_path / "local_paths.png").exists()


def test_plot_local_on_dpg_with_path_indices(tmp_path):
    _require_graphviz_dot()
    explainer, explanation, X, _ = _build_explanation()
    local_explanation = explainer.explain_local(sample=X.iloc[0].values, sample_id=6)

    fig = explainer.plot_local_on_dpg(
        plot_name="local_paths_subset",
        local_explanation=local_explanation,
        path_indices=[0, 1],
        save_dir=str(tmp_path),
        show=False,
    )

    assert fig is not None
    assert (tmp_path / "local_paths_subset.png").exists()


def test_plot_local_on_dpg_invalid_path_indices_raise(tmp_path):
    _require_graphviz_dot()
    explainer, explanation, X, _ = _build_explanation()
    local_explanation = explainer.explain_local(sample=X.iloc[0].values)

    with pytest.raises(ValueError, match="path_indices"):
        explainer.plot_local_on_dpg(
            plot_name="invalid_local_paths",
            local_explanation=local_explanation,
            path_indices=[999],
            save_dir=str(tmp_path),
            show=False,
        )


def test_plot_local_on_dpg_with_local_explanation_avoids_recompute(tmp_path, monkeypatch):
    _require_graphviz_dot()
    explainer, explanation, X, _ = _build_explanation()
    local_explanation = explainer.explain_local(sample=X.iloc[0].values, sample_id=7)

    def fail_explain_local(*args, **kwargs):
        raise AssertionError("explain_local should not be called when local_explanation is provided")

    monkeypatch.setattr(explainer, "explain_local", fail_explain_local)
    fig = explainer.plot_local_on_dpg(
        plot_name="local_paths_no_recompute",
        local_explanation=local_explanation,
        save_dir=str(tmp_path),
        show=False,
    )

    assert fig is not None


def test_plot_local_on_dpg_does_not_mutate_base_dot(tmp_path):
    _require_graphviz_dot()
    explainer, explanation, X, _ = _build_explanation()
    local_explanation = explainer.explain_local(sample=X.iloc[0].values, sample_id=8)
    original_source = explainer._dot.source

    explainer.plot_local_on_dpg(
        plot_name="local_paths_immutable",
        local_explanation=local_explanation,
        save_dir=str(tmp_path),
        show=False,
    )

    assert explainer._dot.source == original_source


def test_plot_dpg_local_paths_aggregate_returns_figure(tmp_path):
    _require_graphviz_dot()
    explainer, explanation, X, _ = _build_explanation()
    local_explanation = explainer.explain_local(sample=X.iloc[0].values, sample_id=9)

    fig = plot_dpg_local_paths_aggregate(
        plot_name="local_paths_direct",
        dot=explainer._dot,
        df=explanation.node_metrics,
        df_edges=explanation.edge_metrics,
        paths_node_ids=[path.node_ids for path in local_explanation.tree_paths],
        path_confidences=[path.path_confidence for path in local_explanation.tree_paths],
        sample_id=local_explanation.sample_id,
        true_class_label=None,
        obtained_class_label=local_explanation.majority_vote,
        sample_metrics=local_explanation.sample_confidence,
        save_dir=str(tmp_path),
        show=False,
    )

    assert fig is not None
    assert (tmp_path / "local_paths_direct.png").exists()
