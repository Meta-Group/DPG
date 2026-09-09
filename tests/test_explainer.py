"""
Tests for the high-level DPGExplainer API.

Validates the full explain_global workflow, DPGExplanation dataclass,
and the fit/explain lifecycle.
"""

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from dpg import DPGExplainer
from dpg.explainer import DPGExplanation, DPGLocalExplanation, DPGTreePathExplanation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def iris_model():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(iris.data, iris.target)
    target_names = np.unique(iris.target).astype(str).tolist()
    return model, iris.data, iris.feature_names, target_names


@pytest.fixture(scope="module")
def explainer(iris_model):
    model, X, feature_names, target_names = iris_model
    return DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=target_names,
    )


@pytest.fixture(scope="module")
def explanation(explainer, iris_model):
    _, X, _, _ = iris_model
    return explainer.explain_global(X)


@pytest.fixture(scope="module")
def explanation_with_communities(iris_model):
    model, X, feature_names, target_names = iris_model
    exp = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=target_names,
    )
    return exp.explain_global(X, communities=True)


# ---------------------------------------------------------------------------
# DPGExplanation structure
# ---------------------------------------------------------------------------


class TestDPGExplanationStructure:
    def test_is_dpg_explanation_instance(self, explanation):
        assert isinstance(explanation, DPGExplanation)

    def test_graph_is_not_none(self, explanation):
        assert explanation.graph is not None

    def test_nodes_is_list(self, explanation):
        assert isinstance(explanation.nodes, list)
        assert len(explanation.nodes) > 0

    def test_dot_is_graphviz(self, explanation):
        import graphviz

        assert isinstance(explanation.dot, graphviz.Digraph)

    def test_node_metrics_is_dataframe(self, explanation):
        assert isinstance(explanation.node_metrics, pd.DataFrame)
        assert not explanation.node_metrics.empty

    def test_edge_metrics_is_dataframe(self, explanation):
        assert isinstance(explanation.edge_metrics, pd.DataFrame)
        assert not explanation.edge_metrics.empty

    def test_class_boundaries_is_dict(self, explanation):
        assert isinstance(explanation.class_boundaries, dict)

    def test_communities_none_by_default(self, explanation):
        assert explanation.communities is None
        assert explanation.community_threshold is None


# ---------------------------------------------------------------------------
# Explanation values (Iris, seed=42, full dataset)
# ---------------------------------------------------------------------------


class TestExplanationValues:
    def test_graph_node_count(self, explanation):
        assert explanation.graph.number_of_nodes() == 47

    def test_graph_edge_count(self, explanation):
        assert explanation.graph.number_of_edges() == 90

    def test_node_metrics_row_count(self, explanation):
        assert len(explanation.node_metrics) == 47

    def test_edge_metrics_row_count(self, explanation):
        assert len(explanation.edge_metrics) == 90

    def test_class_boundaries_have_all_classes(self, explanation):
        bounds = explanation.class_boundaries.get("Class Bounds", {})
        assert sorted(bounds.keys()) == ["Class 0", "Class 1", "Class 2"]

    def test_class_boundary_predicates_non_empty(self, explanation):
        bounds = explanation.class_boundaries["Class Bounds"]
        for cls, preds in bounds.items():
            assert len(preds) > 0, f"{cls} has no boundary predicates"

    def test_class_boundary_predicate_format(self, explanation):
        valid_re = re.compile(r"(<=|>|<)")
        bounds = explanation.class_boundaries["Class Bounds"]
        for cls, preds in bounds.items():
            for p in preds:
                assert valid_re.search(p), f"Invalid predicate: {p!r}"

    def test_specific_class0_boundaries(self, explanation):
        """Class 0 (setosa) boundaries should involve petal features."""
        bounds = explanation.class_boundaries["Class Bounds"]["Class 0"]
        combined = " ".join(bounds).lower()
        assert "petal" in combined


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------


class TestExplanationCommunities:
    def test_communities_returned(self, explanation_with_communities):
        assert explanation_with_communities.communities is not None

    def test_community_threshold_stored(self, explanation_with_communities):
        assert explanation_with_communities.community_threshold == 0.2

    def test_communities_has_clusters_key(self, explanation_with_communities):
        assert "Clusters" in explanation_with_communities.communities

    def test_communities_has_probability_key(self, explanation_with_communities):
        assert "Probability" in explanation_with_communities.communities

    def test_communities_has_confidence_key(self, explanation_with_communities):
        assert "Confidence Interval" in explanation_with_communities.communities

    def test_cluster_labels_contain_class_names(self, explanation_with_communities):
        clusters = explanation_with_communities.communities["Clusters"]
        for expected in ["Class 0", "Class 1", "Class 2"]:
            assert expected in clusters


# ---------------------------------------------------------------------------
# as_dict
# ---------------------------------------------------------------------------


class TestAsDict:
    def test_returns_dict(self, explanation):
        d = explanation.as_dict()
        assert isinstance(d, dict)

    def test_all_keys_present(self, explanation):
        d = explanation.as_dict()
        expected_keys = {
            "graph",
            "nodes",
            "dot",
            "node_metrics",
            "edge_metrics",
            "class_boundaries",
            "communities",
            "community_threshold",
        }
        assert set(d.keys()) == expected_keys

    def test_values_match_attributes(self, explanation):
        d = explanation.as_dict()
        assert d["graph"] is explanation.graph
        assert d["node_metrics"] is explanation.node_metrics
        assert d["edge_metrics"] is explanation.edge_metrics


# ---------------------------------------------------------------------------
# Fit / lifecycle
# ---------------------------------------------------------------------------


class TestExplainerLifecycle:
    def test_explain_global_without_fit_needs_X(self, iris_model):
        model, _, feature_names, target_names = iris_model
        exp = DPGExplainer(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
        )
        with pytest.raises(ValueError, match="not fitted"):
            exp.explain_global(X=None)

    def test_fit_then_explain_without_X(self, iris_model):
        model, X, feature_names, target_names = iris_model
        exp = DPGExplainer(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
        )
        exp.fit(X)
        explanation = exp.explain_global()
        assert explanation.graph is not None
        assert len(explanation.node_metrics) > 0

    def test_builder_property(self, explainer):
        from dpg.core import DecisionPredicateGraph

        assert isinstance(explainer.builder, DecisionPredicateGraph)


class TestAdditionalExplainerApis:
    def test_sample_bc_weights(self, explainer, explanation, iris_model):
        _, X, _, _ = iris_model
        X_df = pd.DataFrame(X, columns=explainer.builder.feature_names)
        weights = explainer.sample_bc_weights(
            X_df=X_df,
            explanation=explanation,
            top_k=5,
        )
        assert len(weights) == len(X_df)
        assert np.all(np.asarray(weights) >= 0)

    def test_plot_sample_using_bc_weights(self, explainer, explanation, iris_model, tmp_path):
        _, X, _, target_names = iris_model
        X_df = pd.DataFrame(X, columns=explainer.builder.feature_names)
        y = explainer.builder.model.predict(X_df)
        fig = explainer.plot_sample_using_bc_weights(
            X_df=X_df,
            y=y,
            explanation=explanation,
            top_k=5,
            dataset_name="Iris",
            class_names=target_names,
            save_path=str(tmp_path / "iris_bc_weights.png"),
            show=False,
        )
        assert fig is not None
        assert (tmp_path / "iris_bc_weights.png").exists()


class TestLocalExplanation:
    def test_explain_local_requires_fit_or_X(self, iris_model):
        model, X, feature_names, target_names = iris_model
        exp = DPGExplainer(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
        )
        with pytest.raises(ValueError, match="not fitted"):
            exp.explain_local(sample=X[0])

    def test_explain_local_rejects_wrong_sample_shape(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        with pytest.raises(ValueError, match="expected 4"):
            explainer.explain_local(sample=X[0][:3])

    def test_explain_local_returns_one_path_per_estimator(self, explainer, iris_model):
        model, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0], sample_id=7)

        assert isinstance(explanation, DPGLocalExplanation)
        assert len(explanation.tree_paths) == len(model.estimators_)
        assert all(isinstance(path, DPGTreePathExplanation) for path in explanation.tree_paths)
        assert explanation.sample_id == 7

    def test_every_path_ends_in_class_leaf_for_classifier_models(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])

        for path in explanation.tree_paths:
            assert path.ends_in_leaf
            assert path.labels[-1].startswith("Class ")

    def test_class_votes_and_majority_vote_are_populated(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])

        assert explanation.class_votes
        assert explanation.majority_vote is not None
        assert explanation.majority_vote in explanation.class_votes
        assert sum(explanation.class_votes.values()) == len(explanation.tree_paths)
        assert all(not label.startswith("Class ") for label in explanation.class_votes)
        assert not explanation.majority_vote.startswith("Class ")

    def test_path_labels_keep_class_prefix(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])

        assert all(path.labels[-1].startswith("Class ") for path in explanation.tree_paths)

    def test_local_paths_include_metrics(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])

        for path in explanation.tree_paths:
            assert hasattr(path, "mean_lrc")
            assert hasattr(path, "mean_bc")
            assert hasattr(path, "path_confidence")
            assert path.path_confidence is not None
            assert 0.0 <= path.path_confidence <= 1.0

    def test_sample_confidence_contains_required_keys(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])

        expected_keys = {
            "num_paths",
            "num_valid_paths",
            "num_active_nodes",
            "mean_lrc_active_nodes",
            "mean_bc_active_nodes",
            "graph_path_valid_rate",
            "vote_confidence",
            "class_scores",
            "score_margin",
            "class_support",
            "evidence_scores",
            "evidence_score_pred",
            "evidence_score_margin",
            "top_competitor_class_pred",
            "evidence_score_competitor_pred",
            "evidence_margin_pred_vs_competitor",
            "trace_node_count_unique",
            "trace_edge_count_unique",
            "explanation_node_count_unique",
            "explanation_edge_count_unique",
            "node_recall",
            "node_precision",
            "edge_recall",
            "edge_precision",
            "trace_coverage_score",
            "recombination_rate",
        }
        assert explanation.sample_confidence is not None
        assert expected_keys.issubset(explanation.sample_confidence.keys())

    def test_vote_confidence_and_class_scores_match_class_votes(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])

        total_votes = sum(explanation.class_votes.values())
        expected_scores = {
            label: votes / total_votes
            for label, votes in explanation.class_votes.items()
        }
        assert explanation.sample_confidence["class_scores"] == expected_scores
        assert explanation.sample_confidence["vote_confidence"] == max(expected_scores.values())

        sorted_scores = sorted(expected_scores.values(), reverse=True)
        expected_margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        assert explanation.sample_confidence["score_margin"] == expected_margin

    def test_class_support_equals_summed_path_confidence_by_class(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])

        expected_support = {}
        for path in explanation.tree_paths:
            leaf = path.labels[-1]
            if leaf.startswith("Class "):
                class_name = leaf[len("Class ") :]
                expected_support[class_name] = expected_support.get(class_name, 0.0) + path.path_confidence

        assert explanation.sample_confidence["class_support"] == expected_support

    def test_evidence_scores_sum_to_one_when_support_exists(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])
        class_support = explanation.sample_confidence["class_support"]
        evidence_scores = explanation.sample_confidence["evidence_scores"]

        if sum(class_support.values()) > 0:
            assert np.isclose(sum(evidence_scores.values()), 1.0)

    def test_evidence_score_pred_matches_majority_vote(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])

        if explanation.majority_vote is not None:
            assert (
                explanation.sample_confidence["evidence_score_pred"]
                == explanation.sample_confidence["evidence_scores"][explanation.majority_vote]
            )

    def test_top_competitor_and_evidence_margins_are_correct(self, explainer):
        paths = [
            DPGTreePathExplanation(
                tree_index=0,
                tree_prefix="sample0_dt0",
                labels=["f0 <= 0.5", "Class 0"],
                node_ids=["1", "2"],
                predicate_truths=[True],
                edge_exists=[True],
                starts_from_root=True,
                ends_in_leaf=True,
                graph_path_valid=True,
                mean_lrc=0.5,
                mean_bc=0.25,
                path_confidence=0.7,
            ),
            DPGTreePathExplanation(
                tree_index=1,
                tree_prefix="sample0_dt1",
                labels=["f0 > 0.5", "Class 0"],
                node_ids=["3", "2"],
                predicate_truths=[True],
                edge_exists=[True],
                starts_from_root=True,
                ends_in_leaf=True,
                graph_path_valid=True,
                mean_lrc=0.4,
                mean_bc=0.15,
                path_confidence=0.5,
            ),
            DPGTreePathExplanation(
                tree_index=2,
                tree_prefix="sample0_dt2",
                labels=["f1 <= 1.5", "Class 1"],
                node_ids=["4", "5"],
                predicate_truths=[True],
                edge_exists=[True],
                starts_from_root=True,
                ends_in_leaf=True,
                graph_path_valid=True,
                mean_lrc=0.3,
                mean_bc=0.1,
                path_confidence=0.4,
            ),
        ]

        sample_confidence = explainer._compute_sample_confidence(
            paths,
            {"0": 2, "1": 1},
            np.asarray([5.1, 3.5, 1.4, 0.2]),
        )
        assert sample_confidence["evidence_scores"] == pytest.approx({"0": 0.75, "1": 0.25})
        assert sample_confidence["evidence_score_pred"] == pytest.approx(0.75)
        assert sample_confidence["top_competitor_class_pred"] == "1"
        assert sample_confidence["evidence_score_competitor_pred"] == pytest.approx(0.25)
        assert sample_confidence["evidence_margin_pred_vs_competitor"] == pytest.approx(0.5)
        assert sample_confidence["evidence_score_margin"] == pytest.approx(0.5)

    def test_single_class_evidence_has_no_competitor(self, explainer):
        single_class_paths = [
            DPGTreePathExplanation(
                tree_index=0,
                tree_prefix="sample0_dt0",
                labels=["f0 <= 0.5", "Class 0"],
                node_ids=["1", "2"],
                predicate_truths=[True],
                edge_exists=[True],
                starts_from_root=True,
                ends_in_leaf=True,
                graph_path_valid=True,
                mean_lrc=0.5,
                mean_bc=0.25,
                path_confidence=0.8,
            ),
            DPGTreePathExplanation(
                tree_index=1,
                tree_prefix="sample0_dt1",
                labels=["f0 > 0.5", "Class 0"],
                node_ids=["3", "2"],
                predicate_truths=[True],
                edge_exists=[True],
                starts_from_root=True,
                ends_in_leaf=True,
                graph_path_valid=True,
                mean_lrc=0.4,
                mean_bc=0.15,
                path_confidence=0.6,
            ),
        ]

        sample_confidence = explainer._compute_sample_confidence(
            single_class_paths,
            {"0": 2},
            np.asarray([5.1, 3.5, 1.4, 0.2]),
        )
        assert sample_confidence["top_competitor_class_pred"] is None
        assert sample_confidence["evidence_score_competitor_pred"] is None
        assert sample_confidence["evidence_margin_pred_vs_competitor"] == sample_confidence["evidence_score_pred"]
        assert sample_confidence["evidence_score_margin"] == sample_confidence["evidence_score_pred"]

    def test_zero_support_fallback_uses_vote_based_class_scores(self, explainer):
        zero_support_paths = [
            DPGTreePathExplanation(
                tree_index=0,
                tree_prefix="sample0_dt0",
                labels=["f0 <= 0.5", "Class 0"],
                node_ids=[None, None],
                predicate_truths=[True],
                edge_exists=[False],
                starts_from_root=True,
                ends_in_leaf=True,
                graph_path_valid=False,
                mean_lrc=None,
                mean_bc=None,
                path_confidence=0.0,
            ),
            DPGTreePathExplanation(
                tree_index=1,
                tree_prefix="sample0_dt1",
                labels=["f0 > 0.5", "Class 1"],
                node_ids=[None, None],
                predicate_truths=[True],
                edge_exists=[False],
                starts_from_root=True,
                ends_in_leaf=True,
                graph_path_valid=False,
                mean_lrc=None,
                mean_bc=None,
                path_confidence=0.0,
            ),
        ]

        sample_confidence = explainer._compute_sample_confidence(
            zero_support_paths,
            {"0": 3, "1": 1},
            np.asarray([5.1, 3.5, 1.4, 0.2]),
        )
        assert sample_confidence["class_support"] == {"0": 0.0, "1": 0.0}
        assert sample_confidence["evidence_scores"] == sample_confidence["class_scores"]

    def test_exact_execution_trace_helper_returns_one_trace_per_estimator(self, explainer, iris_model):
        model, X, _, _ = iris_model
        explainer.fit(X)
        traces = explainer._extract_execution_trace_labels(np.asarray(X[0]))

        assert len(traces) == len(model.estimators_)
        assert all(len(trace) > 0 for trace in traces)

    def test_trace_precision_and_recall_are_in_unit_interval(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])
        confidence = explanation.sample_confidence

        for key in ["node_recall", "node_precision", "edge_recall", "edge_precision"]:
            assert 0.0 <= confidence[key] <= 1.0

    def test_trace_coverage_score_is_in_unit_interval(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])

        assert 0.0 <= explanation.sample_confidence["trace_coverage_score"] <= 1.0

    def test_recombination_rate_is_in_unit_interval_and_near_zero(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])

        assert 0.0 <= explanation.sample_confidence["recombination_rate"] <= 1.0
        assert explanation.sample_confidence["recombination_rate"] == pytest.approx(0.0, abs=1e-12)

    def test_missing_pruned_nodes_or_edges_lower_coverage_without_crashing(self, iris_model):
        model, X, feature_names, target_names = iris_model
        exp = DPGExplainer(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config={
                "dpg": {
                    "default": {
                        "perc_var": 0.1,
                        "decimal_threshold": 6,
                        "n_jobs": 1,
                    },
                    "graph_construction": {
                        "mode": "execution_trace",
                    },
                }
            },
        )
        exp.fit(X)
        explanation = exp.explain_local(sample=X[0], validate_graph=True)
        confidence = explanation.sample_confidence

        assert 0.0 <= confidence["node_recall"] <= 1.0
        assert 0.0 <= confidence["edge_recall"] <= 1.0
        assert confidence["explanation_node_count_unique"] <= confidence["trace_node_count_unique"]
        assert confidence["explanation_edge_count_unique"] <= confidence["trace_edge_count_unique"]
        assert confidence["trace_coverage_score"] <= 1.0

    def test_local_as_dict_works(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])
        payload = explanation.as_dict()

        assert isinstance(payload, dict)
        assert payload["sample_id"] == explanation.sample_id
        assert payload["tree_paths"]
        assert isinstance(payload["tree_paths"][0], dict)
        assert "sample_confidence" in payload
        assert "mean_lrc" in payload["tree_paths"][0]
        assert "mean_bc" in payload["tree_paths"][0]
        assert "path_confidence" in payload["tree_paths"][0]
        assert "class_support" in payload["sample_confidence"]
        assert "evidence_scores" in payload["sample_confidence"]
        assert "trace_coverage_score" in payload["sample_confidence"]
        assert "recombination_rate" in payload["sample_confidence"]

    def test_node_ids_and_edge_exists_handle_pruned_graph(self, iris_model):
        model, X, feature_names, target_names = iris_model
        exp = DPGExplainer(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config={
                "dpg": {
                    "default": {
                        "perc_var": 0.1,
                        "decimal_threshold": 6,
                        "n_jobs": 1,
                    },
                    "graph_construction": {
                        "mode": "execution_trace",
                    },
                }
            },
        )
        exp.fit(X)
        explanation = exp.explain_local(sample=X[0], validate_graph=True)

        assert isinstance(explanation, DPGLocalExplanation)
        assert explanation.graph_validated is True
        assert any(
            any(node_id is None for node_id in path.node_ids) or not all(path.edge_exists)
            for path in explanation.tree_paths
        )
        for path in explanation.tree_paths:
            assert len(path.node_ids) == len(path.labels)
            assert len(path.edge_exists) == max(0, len(path.labels) - 1)
            assert path.path_confidence is not None

    def test_local_path_dataframe_returns_one_row_per_path_label(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0], sample_id=3)
        df = explainer.local_path_dataframe(explanation)

        assert len(df) == sum(len(path.labels) for path in explanation.tree_paths)

    def test_local_path_dataframe_has_required_columns(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])
        df = explainer.local_path_dataframe(explanation)

        expected_columns = [
            "sample_id",
            "tree_index",
            "step_index",
            "label",
            "node_id",
            "is_leaf",
            "predicate_true",
            "edge_exists_from_prev",
            "starts_from_root",
            "ends_in_leaf",
            "graph_path_valid",
            "mean_lrc",
            "mean_bc",
            "path_confidence",
        ]
        assert list(df.columns) == expected_columns

    def test_local_path_dataframe_rows_are_sorted(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])
        df = explainer.local_path_dataframe(explanation)

        expected_pairs = [
            (path.tree_index, step_index)
            for path in sorted(explanation.tree_paths, key=lambda path: path.tree_index)
            for step_index in range(len(path.labels))
        ]
        assert list(zip(df["tree_index"], df["step_index"])) == expected_pairs

    def test_local_path_dataframe_edge_alignment(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0])
        df = explainer.local_path_dataframe(explanation)

        for path in explanation.tree_paths:
            path_df = df[df["tree_index"] == path.tree_index].sort_values("step_index")
            assert path_df.iloc[0]["edge_exists_from_prev"]
            for step_index in range(1, len(path.labels)):
                assert (
                    path_df.iloc[step_index]["edge_exists_from_prev"]
                    == path.edge_exists[step_index - 1]
                )

    def test_local_path_dataframe_empty_explanation(self):
        exp = DPGExplainer(
            model=RandomForestClassifier(n_estimators=1, random_state=42).fit(
                np.array([[0, 0], [1, 1]]),
                np.array([0, 1]),
            ),
            feature_names=["f0", "f1"],
            target_names=["0", "1"],
        )
        empty_explanation = DPGLocalExplanation(
            sample_id=0,
            sample=[],
            tree_paths=[],
            graph_validated=True,
            all_trees_valid=True,
            majority_vote=None,
            class_votes={},
            path_mode="execution_trace",
            sample_confidence={},
        )

        df = exp.local_path_dataframe(empty_explanation)

        assert df.empty
        assert list(df.columns) == [
            "sample_id",
            "tree_index",
            "step_index",
            "label",
            "node_id",
            "is_leaf",
            "predicate_true",
            "edge_exists_from_prev",
            "starts_from_root",
            "ends_in_leaf",
            "graph_path_valid",
            "mean_lrc",
            "mean_bc",
            "path_confidence",
        ]

    def test_local_path_dataframe_values_match_explanation(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        explanation = explainer.explain_local(sample=X[0], sample_id=11)
        df = explainer.local_path_dataframe(explanation)

        first_path = min(explanation.tree_paths, key=lambda path: path.tree_index)
        first_row = df[df["tree_index"] == first_path.tree_index].sort_values("step_index").iloc[0]

        assert first_row["sample_id"] == explanation.sample_id
        assert first_row["label"] == first_path.labels[0]
        assert first_row["node_id"] == first_path.node_ids[0]
        assert first_row["mean_lrc"] == first_path.mean_lrc
        assert first_row["mean_bc"] == first_path.mean_bc
        assert first_row["path_confidence"] == first_path.path_confidence


class TestFaithfulnessEvaluation:
    def test_returns_scalar_in_unit_interval(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        score = explainer.evaluate_faithfulness(X[:5], max_samples=5)
        assert 0.0 <= score <= 1.0

    def test_return_details_contains_required_keys(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        details = explainer.evaluate_faithfulness(X[:5], max_samples=5, return_details=True)

        expected_keys = {
            "faithfulness_score",
            "weights",
            "n_samples",
            "n_successful",
            "n_local_failures",
            "output_fidelity",
            "mean_node_recall",
            "mean_node_precision",
            "mean_edge_recall",
            "mean_edge_precision",
            "mean_trace_coverage_score",
            "mean_recombination_rate",
            "mean_vote_confidence",
            "mean_evidence_score_pred",
            "mean_evidence_score_margin",
            "mean_evidence_margin_pred_vs_competitor",
            "mean_path_purity",
            "mean_competitor_exposure",
            "mean_explanation_confidence",
            "per_sample",
        }
        assert expected_keys.issubset(details.keys())
        assert isinstance(details["per_sample"], pd.DataFrame)

    def test_works_with_dataframe_and_numpy_array(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        score_df = explainer.evaluate_faithfulness(pd.DataFrame(X), max_samples=5)
        score_np = explainer.evaluate_faithfulness(np.asarray(X), max_samples=5)
        assert 0.0 <= score_df <= 1.0
        assert 0.0 <= score_np <= 1.0

    def test_uses_y_true_to_compute_local_accuracy(self, explainer, iris_model):
        _, X, _, target_names = iris_model
        y_true = [str(i) for i in load_iris().target[:5]]
        explainer.fit(X)
        details = explainer.evaluate_faithfulness(X[:5], y_true=y_true, max_samples=5, return_details=True)
        assert "local_accuracy" in details
        assert 0.0 <= details["local_accuracy"] <= 1.0

    def test_max_samples_limits_evaluation(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        details = explainer.evaluate_faithfulness(X, max_samples=3, return_details=True)
        assert details["n_samples"] == 3
        assert len(details["per_sample"]) == 3

    def test_invalid_weights_raise_value_error(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        with pytest.raises(ValueError, match="Supported keys are"):
            explainer.evaluate_faithfulness(
                X[:3],
                weights={
                    "output_fidelity": 0.5,
                    "trace_coverage": 0.3,
                    "anti_recombination": 0.2,
                    "extra": 0.0,
                },
            )

    def test_empty_x_raises_value_error(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        with pytest.raises(ValueError, match="at least one sample"):
            explainer.evaluate_faithfulness(np.asarray([]).reshape(0, X.shape[1]))

    def test_non_positive_max_samples_raises_value_error(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        with pytest.raises(ValueError, match="max_samples must be a positive integer"):
            explainer.evaluate_faithfulness(X, max_samples=0)

    def test_weights_not_summing_to_one_raise_value_error(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        with pytest.raises(ValueError, match="sum to 1.0"):
            explainer.evaluate_faithfulness(
                X[:3],
                weights={
                    "output_fidelity": 0.4,
                    "trace_coverage": 0.3,
                    "anti_recombination": 0.2,
                    "evidence_margin": 0.2,
                },
            )

    def test_sample_ids_length_mismatch_raises_value_error(self, explainer, iris_model):
        _, X, _, _ = iris_model
        explainer.fit(X)
        with pytest.raises(ValueError, match="sample_ids length"):
            explainer.evaluate_faithfulness(X[:3], sample_ids=[10, 11])

    def test_y_true_labels_are_normalized_consistently(self, explainer, iris_model):
        _, X, _, _ = iris_model
        y_true = [0, 0, 0, 0, 0]
        explainer.fit(X)
        details = explainer.evaluate_faithfulness(X[:5], y_true=y_true, max_samples=5, return_details=True)
        assert "local_accuracy" in details
        assert 0.0 <= details["local_accuracy"] <= 1.0

    def test_all_local_failures_raise_clear_value_error(self, explainer, iris_model, monkeypatch):
        _, X, _, _ = iris_model
        explainer.fit(X)

        def always_fail(*args, **kwargs):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(explainer, "explain_local", always_fail)
        with pytest.raises(ValueError, match="All local explanations failed"):
            explainer.evaluate_faithfulness(X[:3], return_details=True)

    def test_local_failures_are_counted_without_crashing(self, explainer, iris_model, monkeypatch):
        _, X, _, _ = iris_model
        explainer.fit(X)
        original_explain_local = explainer.explain_local

        def maybe_fail(sample, sample_id=0, X=None, validate_graph=True):
            if sample_id == 1:
                raise RuntimeError("synthetic local failure")
            return original_explain_local(sample, sample_id=sample_id, X=X, validate_graph=validate_graph)

        monkeypatch.setattr(explainer, "explain_local", maybe_fail)
        details = explainer.evaluate_faithfulness(X[:3], sample_ids=[0, 1, 2], return_details=True)

        assert details["n_local_failures"] == 1
        assert details["n_successful"] == 2
        assert len(details["per_sample"]) == 3
        assert details["per_sample"]["error"].notna().sum() == 1


# ---------------------------------------------------------------------------
# PR #32 additions: explainer stays consistent across context orders, and
# uses ``builder.get_node_ids_for_trace`` for local node-id mapping.
# ---------------------------------------------------------------------------


def _config(mode="execution_trace", context_order=1, decimal_threshold=6, perc_var=1e-9, n_jobs=1):
    return {
        "dpg": {
            "default": {
                "perc_var": perc_var,
                "decimal_threshold": decimal_threshold,
                "n_jobs": n_jobs,
            },
            "graph_construction": {
                "mode": mode,
                "context_order": context_order,
            },
        }
    }


class TestContextOrderIntegration:
    """``DPGExplainer.explain_local`` must produce the same routing at any k."""

    def test_local_node_ids_match_builder_helper(self):
        """Each tree path's node ids must come from
        ``builder.get_node_ids_for_trace`` (same length as labels)."""
        from sklearn.datasets import load_iris

        from dpg import DPGExplainer

        iris = load_iris()
        model = RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=1).fit(
            iris.data, iris.target
        )
        explainer = DPGExplainer(
            model, iris.feature_names, dpg_config=_config(context_order=2)
        )
        explainer.fit(iris.data)

        local = explainer.explain_local(iris.data[0], sample_id=0)
        for tree_path in local.tree_paths:
            node_ids = explainer.builder.get_node_ids_for_trace(tree_path.labels)
            assert len(node_ids) == len(tree_path.labels)

    def test_explain_local_returns_same_votes_across_context_order(self):
        """Routing is independent of context order -- so local votes
        (which depend on routing) must not change between k=1 and k=2."""
        from sklearn.datasets import load_iris

        from dpg import DPGExplainer

        iris = load_iris()
        model = RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=1).fit(
            iris.data, iris.target
        )
        sample = iris.data[0]
        truth = {}

        for k in (1, 2):
            explainer = DPGExplainer(
                model, iris.feature_names, dpg_config=_config(context_order=k)
            )
            explainer.fit(iris.data)
            local = explainer.explain_local(sample, sample_id=0)
            truth[k] = (local.majority_vote, dict(local.class_votes))

        assert truth[1] == truth[2]
