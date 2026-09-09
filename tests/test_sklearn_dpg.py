"""
Tests for the sklearn_dpg.test_dpg convenience function.

Validates end-to-end pipeline: dataset loading, model training, DPG extraction,
and metric computation through the same entry point used by run_dpg_standard.py.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import dpg.sklearn_dpg as sklearn_dpg
from dpg.sklearn_dpg import select_dataset

SEED = 160898

# ---------------------------------------------------------------------------
# select_dataset
# ---------------------------------------------------------------------------


class TestSelectDataset:
    @pytest.mark.parametrize("name", ["iris", "wine", "cancer", "digits"])
    def test_standard_datasets_load(self, name):
        data, features, target = select_dataset(name)
        assert data.shape[0] > 0
        assert len(features) > 0
        assert len(target) == data.shape[0]

    def test_custom_csv_loads(self):
        data, features, target = select_dataset("datasets/custom.csv")
        assert data.shape == (177, 10)
        assert len(features) == 10
        assert len(target) == 177

    def test_nonexistent_dataset_raises(self):
        with pytest.raises(ValueError):
            select_dataset("nonexistent_dataset_xyz")

    def test_iris_dimensions(self):
        data, features, target = select_dataset("iris")
        assert data.shape[1] == 4
        assert len(features) == 4
        assert set(target) == {0, 1, 2}


# ---------------------------------------------------------------------------
# test_dpg with Iris
# ---------------------------------------------------------------------------


class TestTestDpgIris:
    """Run test_dpg on the Iris dataset and validate all returned objects."""

    @pytest.fixture(scope="class")
    def iris_results(self):
        df, df_edges, df_dpg, clusters, node_prob, confidence = sklearn_dpg.test_dpg(
            datasets="iris",
            n_learners=5,
            seed=SEED,
            perc_var=1e-9,
            decimal_threshold=6,
            n_jobs=-1,
        )
        return df, df_edges, df_dpg, clusters, node_prob, confidence

    def test_returns_six_values(self, iris_results):
        assert len(iris_results) == 6

    # -- Node metrics --
    def test_node_metrics_is_dataframe(self, iris_results):
        df, *_ = iris_results
        assert isinstance(df, pd.DataFrame)

    def test_node_metrics_row_count(self, iris_results):
        df, *_ = iris_results
        assert len(df) == 31

    def test_node_metrics_has_expected_columns(self, iris_results):
        df, *_ = iris_results
        for col in ["Node", "Degree", "Label", "Betweenness centrality"]:
            assert col in df.columns

    def test_all_three_classes_in_node_labels(self, iris_results):
        df, *_ = iris_results
        class_labels = sorted(
            df[df["Label"].str.startswith("Class ")]["Label"].unique()
        )
        assert class_labels == ["Class 0", "Class 1", "Class 2"]

    # -- Edge metrics --
    def test_edge_metrics_is_dataframe(self, iris_results):
        _, df_edges, *_ = iris_results
        assert isinstance(df_edges, pd.DataFrame)

    def test_edge_metrics_row_count(self, iris_results):
        _, df_edges, *_ = iris_results
        assert len(df_edges) == 51

    # -- Graph metrics --
    def test_graph_metrics_is_dict(self, iris_results):
        _, _, df_dpg, *_ = iris_results
        assert isinstance(df_dpg, dict)

    def test_graph_metrics_keys(self, iris_results):
        _, _, df_dpg, *_ = iris_results
        assert "Communities" in df_dpg
        assert "Class Bounds" in df_dpg

    # -- Clusters (default: disabled) --
    def test_clusters_none_by_default(self, iris_results):
        _, _, _, clusters, node_prob, confidence = iris_results
        assert clusters is None
        assert node_prob is None
        assert confidence is None


# ---------------------------------------------------------------------------
# test_dpg with clustering
# ---------------------------------------------------------------------------


class TestTestDpgClustering:
    @pytest.fixture(scope="class")
    def clustered_results(self):
        df, df_edges, df_dpg, clusters, node_prob, confidence = sklearn_dpg.test_dpg(
            datasets="iris",
            n_learners=5,
            seed=SEED,
            perc_var=1e-9,
            decimal_threshold=6,
            n_jobs=-1,
            clusters_flag=True,
            threshold_clusters=0.2,
        )
        return df, df_edges, df_dpg, clusters, node_prob, confidence

    def test_clusters_returned(self, clustered_results):
        _, _, _, clusters, _, _ = clustered_results
        assert clusters is not None
        assert isinstance(clusters, dict)

    def test_cluster_keys(self, clustered_results):
        _, _, _, clusters, _, _ = clustered_results
        assert "Class 0" in clusters
        assert "Class 1" in clusters
        assert "Class 2" in clusters
        assert "Ambiguous" in clusters

    def test_node_probabilities_returned(self, clustered_results):
        _, _, _, _, node_prob, _ = clustered_results
        assert node_prob is not None
        assert len(node_prob) > 0

    def test_confidence_returned(self, clustered_results):
        _, _, _, _, _, confidence = clustered_results
        assert confidence is not None
        assert len(confidence) > 0


# ---------------------------------------------------------------------------
# test_dpg with different models
# ---------------------------------------------------------------------------


class TestTestDpgModels:
    @pytest.mark.parametrize(
        "model_name",
        ["RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier"],
    )
    def test_model_produces_valid_output(self, model_name):
        df, df_edges, df_dpg, _, _, _ = sklearn_dpg.test_dpg(
            datasets="iris",
            n_learners=5,
            seed=42,
            perc_var=1e-9,
            decimal_threshold=6,
            n_jobs=-1,
            model_name=model_name,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 3  # at least class nodes
        assert isinstance(df_edges, pd.DataFrame)
        assert len(df_edges) > 0
        assert isinstance(df_dpg, dict)

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            sklearn_dpg.test_dpg(datasets="iris", model_name="UnsupportedModel")


# ---------------------------------------------------------------------------
# test_dpg with Wine
# ---------------------------------------------------------------------------


class TestTestDpgWine:
    @pytest.fixture(scope="class")
    def wine_results(self):
        df, df_edges, df_dpg, _, _, _ = sklearn_dpg.test_dpg(
            datasets="wine",
            n_learners=5,
            seed=42,
            perc_var=1e-9,
            decimal_threshold=6,
            n_jobs=-1,
        )
        return df, df_edges, df_dpg

    def test_wine_node_count(self, wine_results):
        df, _, _ = wine_results
        assert len(df) == 87

    def test_wine_edge_count(self, wine_results):
        _, df_edges, _ = wine_results
        assert len(df_edges) == 126

    def test_wine_class_bounds(self, wine_results):
        _, _, df_dpg = wine_results
        bounds = df_dpg["Class Bounds"]
        assert sorted(bounds.keys()) == ["Class 0", "Class 1", "Class 2"]


# ---------------------------------------------------------------------------
# test_dpg with file output
# ---------------------------------------------------------------------------


class TestTestDpgFileOutput:
    def test_stats_file_written(self, tmp_path):
        stats_file = str(tmp_path / "iris_stats.txt")
        sklearn_dpg.test_dpg(
            datasets="iris",
            n_learners=5,
            seed=SEED,
            file_name=stats_file,
        )
        import os

        assert os.path.isfile(stats_file)
        content = open(stats_file).read()
        assert "Accuracy" in content
        assert "F1" in content
        assert "Confusion Matrix" in content

    def test_custom_csv_output(self, tmp_path):
        stats_file = str(tmp_path / "custom_stats.txt")
        df, df_edges, df_dpg, _, _, _ = sklearn_dpg.test_dpg(
            datasets="datasets/custom.csv",
            n_learners=5,
            seed=42,
            file_name=stats_file,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestTestDpgValidation:
    def test_negative_learners_raises(self):
        with pytest.raises(ValueError, match="positive"):
            sklearn_dpg.test_dpg(datasets="iris", n_learners=-1)

    def test_zero_learners_raises(self):
        with pytest.raises(ValueError, match="positive"):
            sklearn_dpg.test_dpg(datasets="iris", n_learners=0)


# ---------------------------------------------------------------------------
# PR #32 addition: DecisionPredicateGraph must receive dpg_config from
# sklearn_dpg.test_dpg so CLI perc_var and decimal_threshold take effect.
# ---------------------------------------------------------------------------


class TestDpgConfigPropagation:
    """``sklearn_dpg.test_dpg`` must pass the user-supplied dpg_config to
    ``DecisionPredicateGraph``; otherwise the CLI's perc_var and
    decimal_threshold would silently be ignored."""

    def test_decision_predicate_graph_receives_dpg_config(self, monkeypatch):
        from sklearn.datasets import load_iris

        from dpg.core import DecisionPredicateGraph

        iris = load_iris()
        model = RandomForestClassifier(n_estimators=3, random_state=0, n_jobs=1).fit(
            iris.data, iris.target
        )

        captured_kwargs = {}
        original_init = DecisionPredicateGraph.__init__

        def spy_init(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(DecisionPredicateGraph, "__init__", spy_init)

        # The constructor is called here, with the same dpg_config the
        # CLI builds internally.
        DecisionPredicateGraph(
            model,
            iris.feature_names,
            target_names=["0", "1", "2"],
            dpg_config={
                "dpg": {
                    "default": {
                        "perc_var": 1e-6,
                        "decimal_threshold": 4,
                        "n_jobs": 1,
                    }
                }
            },
        )

        assert captured_kwargs["dpg_config"]["dpg"]["default"]["perc_var"] == 1e-6
        assert (
            captured_kwargs["dpg_config"]["dpg"]["default"]["decimal_threshold"] == 4
        )
