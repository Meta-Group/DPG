"""
Tests for DPG core pipeline: DecisionPredicateGraph construction,
NetworkX conversion, and graph structure validation.

Uses the Iris dataset with a fixed random seed so that expected node/edge
counts and metric ranges are deterministic and reproducible.
"""

import re

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split

from dpg.core import DecisionPredicateGraph, DPGError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 160898


@pytest.fixture(scope="module")
def iris_split():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=SEED
    )
    target_names = np.unique(iris.target).astype(str).tolist()
    return X_train, X_test, y_train, y_test, iris.feature_names, target_names


@pytest.fixture(scope="module")
def iris_rf(iris_split):
    X_train, _, y_train, _, _, _ = iris_split
    model = RandomForestClassifier(n_estimators=5, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def iris_dpg(iris_rf, iris_split):
    """Build a DPG for Iris and return (dpg_graph, nodes_list, dot)."""
    X_train, _, _, _, feature_names, target_names = iris_split
    dpg = DecisionPredicateGraph(
        model=iris_rf,
        feature_names=feature_names,
        target_names=target_names,
    )
    dot = dpg.fit(X_train)
    dpg_graph, nodes_list = dpg.to_networkx(dot)
    return dpg_graph, nodes_list, dot


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------


class TestGraphStructure:
    """Validate the DPG graph has the expected topology for Iris."""

    def test_graph_is_directed(self, iris_dpg):
        dpg_graph, _, _ = iris_dpg
        assert isinstance(dpg_graph, nx.DiGraph)

    def test_exact_node_count(self, iris_dpg):
        dpg_graph, _, _ = iris_dpg
        assert dpg_graph.number_of_nodes() == 31

    def test_exact_edge_count(self, iris_dpg):
        dpg_graph, _, _ = iris_dpg
        assert dpg_graph.number_of_edges() == 51

    def test_nodes_list_matches_graph(self, iris_dpg):
        """nodes_list (non-edge entries) should match the graph node set."""
        dpg_graph, nodes_list, _ = iris_dpg
        node_ids = {n[0] for n in nodes_list if "->" not in n[0]}
        assert node_ids == set(dpg_graph.nodes())

    def test_all_edges_have_weight(self, iris_dpg):
        dpg_graph, _, _ = iris_dpg
        for u, v, data in dpg_graph.edges(data=True):
            assert "weight" in data
            assert data["weight"] >= 1

    def test_class_nodes_are_sinks(self, iris_dpg):
        """Class nodes should have out-degree 0 (terminal/absorbing)."""
        dpg_graph, nodes_list, _ = iris_dpg
        class_node_ids = [n[0] for n in nodes_list if "Class" in n[1]]
        for node_id in class_node_ids:
            assert dpg_graph.out_degree(node_id) == 0

    def test_three_class_nodes_present(self, iris_dpg):
        _, nodes_list, _ = iris_dpg
        class_labels = sorted(
            {n[1] for n in nodes_list if n[1].startswith("Class ")}
        )
        assert class_labels == ["Class 0", "Class 1", "Class 2"]

    def test_predicate_node_label_format(self, iris_dpg):
        """Non-class node labels should be feature predicates like 'feat <= val' or 'feat > val'."""
        _, nodes_list, _ = iris_dpg
        predicate_re = re.compile(r"^.+\s*(<=|>)\s*[\d.eE+-]+$")
        for node_id, label in nodes_list:
            if "->" in node_id or label.startswith("Class "):
                continue
            assert predicate_re.match(label), f"Unexpected predicate format: {label!r}"

    def test_graph_is_weakly_connected(self, iris_dpg):
        dpg_graph, _, _ = iris_dpg
        assert nx.is_weakly_connected(dpg_graph)


# ---------------------------------------------------------------------------
# Dot / Graphviz output tests
# ---------------------------------------------------------------------------


class TestDotOutput:
    def test_dot_is_graphviz_digraph(self, iris_dpg):
        import graphviz

        _, _, dot = iris_dpg
        assert isinstance(dot, graphviz.Digraph)

    def test_dot_source_contains_class_labels(self, iris_dpg):
        _, _, dot = iris_dpg
        source = dot.source
        for cls in ["Class 0", "Class 1", "Class 2"]:
            assert cls in source


# ---------------------------------------------------------------------------
# Initialization validation
# ---------------------------------------------------------------------------


class TestDPGInitValidation:
    def test_rejects_non_ensemble_model(self, iris_split):
        _, _, _, _, feature_names, _ = iris_split
        with pytest.raises(DPGError, match="tree-based ensemble"):
            DecisionPredicateGraph(model="not_a_model", feature_names=feature_names)

    def test_rejects_empty_feature_names(self, iris_rf):
        with pytest.raises(DPGError, match="Feature names cannot be empty"):
            DecisionPredicateGraph(model=iris_rf, feature_names=[])

    def test_accepts_custom_dpg_config(self, iris_rf, iris_split):
        _, _, _, _, feature_names, target_names = iris_split
        custom_config = {
            "dpg": {
                "default": {
                    "perc_var": 0.01,
                    "decimal_threshold": 3,
                    "n_jobs": 1,
                }
            }
        }
        dpg = DecisionPredicateGraph(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config=custom_config,
        )
        assert dpg.perc_var == 0.01
        assert dpg.decimal_threshold == 3
        assert dpg.n_jobs == 1

    def test_default_graph_construction_mode_is_aggregated_transitions(
        self, iris_rf, iris_split
    ):
        X_train, _, _, _, feature_names, target_names = iris_split
        dpg = DecisionPredicateGraph(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
        )
        assert dpg.graph_construction_mode == "aggregated_transitions"

        default_log = dpg._extract_trace_log(X_train)
        default_edges = set(
            dpg.discover_dfg(dpg.filter_log(default_log)).keys()
        )

        explicit_dpg = DecisionPredicateGraph(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config={
                "dpg": {
                    "default": {
                        "perc_var": dpg.perc_var,
                        "decimal_threshold": dpg.decimal_threshold,
                        "n_jobs": 1,
                    },
                    "graph_construction": {
                        "mode": "aggregated_transitions",
                    },
                }
            },
        )
        explicit_log = explicit_dpg._extract_trace_log(X_train)
        explicit_edges = set(
            explicit_dpg.discover_dfg(explicit_dpg.filter_log(explicit_log)).keys()
        )

        assert default_edges == explicit_edges

    def test_explicit_aggregated_transitions_works(self, iris_rf, iris_split):
        X_train, _, _, _, feature_names, target_names = iris_split
        dpg = DecisionPredicateGraph(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config={
                "dpg": {
                    "default": {
                        "perc_var": 1e-9,
                        "decimal_threshold": 6,
                        "n_jobs": 1,
                    },
                    "graph_construction": {
                        "mode": "aggregated_transitions",
                    },
                }
            },
        )

        dot = dpg.fit(X_train)
        graph, _ = dpg.to_networkx(dot)

        assert dpg.graph_construction_mode == "aggregated_transitions"
        assert graph.number_of_edges() > 0

    def test_explicit_execution_trace_works(self, iris_rf, iris_split):
        X_train, _, _, _, feature_names, target_names = iris_split
        dpg = DecisionPredicateGraph(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config={
                "dpg": {
                    "default": {
                        "perc_var": 1e-9,
                        "decimal_threshold": 6,
                        "n_jobs": 1,
                    },
                    "graph_construction": {
                        "mode": "execution_trace",
                    },
                }
            },
        )

        dot = dpg.fit(X_train)
        graph, _ = dpg.to_networkx(dot)

        assert dpg.graph_construction_mode == "execution_trace"
        assert graph.number_of_edges() > 0

    def test_invalid_graph_construction_mode_raises(self, iris_rf, iris_split):
        _, _, _, _, feature_names, target_names = iris_split
        with pytest.raises(DPGError, match="Unsupported graph construction mode"):
            DecisionPredicateGraph(
                model=iris_rf,
                feature_names=feature_names,
                target_names=target_names,
                dpg_config={
                    "dpg": {
                        "default": {
                            "perc_var": 1e-9,
                            "decimal_threshold": 6,
                            "n_jobs": 1,
                        },
                        "graph_construction": {
                            "mode": "not_a_real_mode",
                        },
                    }
                },
            )

    def test_graph_construction_modes_can_produce_different_edges(self):
        iris = load_iris()
        X_train, _, y_train, _ = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(
            n_estimators=3, max_depth=2, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        feature_names = iris.feature_names
        target_names = np.unique(iris.target).astype(str).tolist()

        aggregated_dpg = DecisionPredicateGraph(
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
                        "mode": "aggregated_transitions",
                    },
                }
            },
        )
        aggregated_log = aggregated_dpg._extract_trace_log(X_train)
        aggregated_edges = set(
            aggregated_dpg.discover_dfg(
                aggregated_dpg.filter_log(aggregated_log)
            ).keys()
        )

        execution_trace_dpg = DecisionPredicateGraph(
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
        execution_trace_log = execution_trace_dpg._extract_trace_log(X_train)
        execution_trace_edges = set(
            execution_trace_dpg.discover_dfg_execution_trace(
                execution_trace_log
            ).keys()
        )

        assert aggregated_edges != execution_trace_edges


# ---------------------------------------------------------------------------
# Different ensemble models
# ---------------------------------------------------------------------------


class TestMultipleModels:
    def test_extra_trees_produces_valid_dpg(self, iris_split):
        X_train, _, y_train, _, feature_names, target_names = iris_split
        model = ExtraTreesClassifier(n_estimators=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        dpg = DecisionPredicateGraph(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
        )
        dot = dpg.fit(X_train)
        graph, nodes = dpg.to_networkx(dot)
        assert graph.number_of_nodes() > 3
        assert graph.number_of_edges() > 3
        class_labels = {n[1] for n in nodes if n[1].startswith("Class ")}
        assert len(class_labels) == 3


class TestDifferentDataset:
    """Validate DPG works on Wine (more features, same 3-class setup)."""

    def test_wine_graph_structure(self):
        wine = load_wine()
        X_train, _, y_train, _ = train_test_split(
            wine.data, wine.target, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        target_names = np.unique(wine.target).astype(str).tolist()
        dpg = DecisionPredicateGraph(
            model=model,
            feature_names=wine.feature_names,
            target_names=target_names,
        )
        dot = dpg.fit(X_train)
        graph, nodes = dpg.to_networkx(dot)

        assert graph.number_of_nodes() == 87
        assert graph.number_of_edges() == 126
        class_labels = sorted({n[1] for n in nodes if n[1].startswith("Class ")})
        assert class_labels == ["Class 0", "Class 1", "Class 2"]


class TestTraceArtifacts:
    """
    Trace-consistent artefacts (LRC, downstream sets, signatures) built in
    ``execution_trace`` mode. These must only report relations witnessed
    within a single observed sample-tree execution, never relations that
    only exist after pooling edges from different traces.
    """

    def _log(self, rows):
        return pd.DataFrame(rows, columns=["case:concept:name", "concept:name"])

    def _dpg(self, iris_rf, iris_split, mode="execution_trace"):
        _, _, _, _, feature_names, target_names = iris_split
        return DecisionPredicateGraph(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config={
                "dpg": {
                    "default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
                    "graph_construction": {"mode": mode},
                }
            },
        )

    def test_getters_empty_before_fit(self, iris_rf, iris_split):
        dpg = self._dpg(iris_rf, iris_split)
        assert dpg.get_trace_consistent_lrc() == {}
        assert dpg.get_trace_consistent_trc() == {}
        assert dpg.get_trace_signatures() == []

    def test_cross_trace_phantom_path_guard(self, iris_rf, iris_split):
        """Pooled A->B and B->C must not be reported as a witnessed A->B->C."""
        dpg = self._dpg(iris_rf, iris_split)
        log = self._log([
            ("sample0_dt0", "A <= 0.5"),
            ("sample0_dt0", "B <= 1.0"),
            ("sample1_dt0", "B <= 1.0"),
            ("sample1_dt0", "C <= 2.0"),
        ])
        dpg._build_trace_artifacts(log)

        trc = dpg.get_trace_consistent_trc()
        assert trc["A <= 0.5"] == ("B <= 1.0",)
        assert "C <= 2.0" not in trc["A <= 0.5"]

        signatures = dpg.get_trace_signatures()
        for sig in signatures:
            assert sig.predicate_sequence != ("A <= 0.5", "B <= 1.0", "C <= 2.0")

    def test_single_trace_preservation(self, iris_rf, iris_split):
        dpg = self._dpg(iris_rf, iris_split)
        log = self._log([
            ("sample0_dt0", "A <= 0.5"),
            ("sample0_dt0", "B <= 1.0"),
            ("sample0_dt0", "Class 0"),
        ])
        dpg._build_trace_artifacts(log)

        signatures = dpg.get_trace_signatures()
        assert len(signatures) == 1
        assert signatures[0].predicate_sequence == ("A <= 0.5", "B <= 1.0", "Class 0")
        assert signatures[0].path_count == 1

    def test_repeated_predicate_feature_not_deduplicated(self, iris_rf, iris_split):
        dpg = self._dpg(iris_rf, iris_split)
        log = self._log([
            ("sample0_dt0", "A <= 0.5"),
            ("sample0_dt0", "A <= 2.0"),
            ("sample0_dt0", "Class 0"),
        ])
        dpg._build_trace_artifacts(log)

        signatures = dpg.get_trace_signatures()
        assert signatures[0].signature == ("A", "A", "Class 0")
        assert signatures[0].predicate_sequence == ("A <= 0.5", "A <= 2.0", "Class 0")

    def test_trace_count_aggregation(self, iris_rf, iris_split):
        dpg = self._dpg(iris_rf, iris_split)
        log = self._log([
            ("sample0_dt0", "A <= 0.5"),
            ("sample0_dt0", "Class 0"),
            ("sample1_dt0", "A <= 0.5"),
            ("sample1_dt0", "Class 0"),
            ("sample2_dt0", "A <= 0.5"),
            ("sample2_dt0", "Class 1"),
        ])
        dpg._build_trace_artifacts(log)

        signatures = {sig.predicate_sequence: sig.path_count for sig in dpg.get_trace_signatures()}
        assert signatures[("A <= 0.5", "Class 0")] == 2
        assert signatures[("A <= 0.5", "Class 1")] == 1

    def test_downstream_set_provenance(self, iris_rf, iris_split):
        dpg = self._dpg(iris_rf, iris_split)
        log = self._log([
            ("sample0_dt0", "A <= 0.5"),
            ("sample0_dt0", "B <= 1.0"),
            ("sample0_dt0", "C <= 2.0"),
            ("sample0_dt0", "Class 0"),
        ])
        dpg._build_trace_artifacts(log)

        sequences = [sig.predicate_sequence for sig in dpg.get_trace_signatures()]
        for label, downs in dpg.get_trace_consistent_trc().items():
            for down in downs:
                assert any(
                    label in seq and down in seq[seq.index(label) + 1:]
                    for seq in sequences
                )

    def test_non_predicate_labels_excluded_from_trc_keys(self, iris_rf, iris_split):
        dpg = self._dpg(iris_rf, iris_split)
        log = self._log([
            ("sample0_dt0", "A <= 0.5"),
            ("sample0_dt0", "Class 0"),
            ("sample1_dt0", "Pred 1.23"),
            ("sample1_dt0", "A <= 0.5"),
        ])
        dpg._build_trace_artifacts(log)

        trc = dpg.get_trace_consistent_trc()
        assert "Class 0" not in trc
        assert "Pred 1.23" not in trc

    def test_refit_isolation(self, iris_rf, iris_split):
        dpg = self._dpg(iris_rf, iris_split)
        dpg._build_trace_artifacts(self._log([
            ("sample0_dt0", "A <= 0.5"),
            ("sample0_dt0", "Class 0"),
        ]))
        assert "A <= 0.5" in dpg.get_trace_consistent_trc()

        dpg._build_trace_artifacts(self._log([
            ("sample0_dt0", "D <= 0.5"),
            ("sample0_dt0", "Class 0"),
        ]))
        assert "A <= 0.5" not in dpg.get_trace_consistent_trc()
        assert "D <= 0.5" in dpg.get_trace_consistent_trc()

    def test_aggregated_mode_leaves_trace_artifacts_empty(self, iris_rf, iris_split):
        dpg = self._dpg(iris_rf, iris_split, mode="aggregated_transitions")
        X_train, _, _, _, _, _ = iris_split
        dpg.fit(X_train)
        assert dpg.get_trace_consistent_lrc() == {}
        assert dpg.get_trace_consistent_trc() == {}
        assert dpg.get_trace_signatures() == []

    def test_execution_trace_mode_populates_artifacts_on_fit(self, iris_rf, iris_split):
        dpg = self._dpg(iris_rf, iris_split, mode="execution_trace")
        X_train, _, _, _, _, _ = iris_split
        dpg.fit(X_train)
        assert dpg.get_trace_consistent_lrc() != {}
        assert dpg.get_trace_signatures() != []

    def test_get_trace_consistent_trc_returns_sorted_tuples(self, iris_rf, iris_split):
        dpg = self._dpg(iris_rf, iris_split)
        log = self._log([
            ("sample0_dt0", "A <= 0.5"),
            ("sample0_dt0", "C <= 2.0"),
            ("sample0_dt0", "B <= 1.0"),
        ])
        dpg._build_trace_artifacts(log)

        trc = dpg.get_trace_consistent_trc()
        assert trc["A <= 0.5"] == ("B <= 1.0", "C <= 2.0")
        assert isinstance(trc["A <= 0.5"], tuple)

    def test_trace_artifacts_ignore_perc_var_filtering(self, iris_rf, iris_split):
        """
        Trace artefacts are built from the raw, unfiltered execution log:
        a rare predicate whose pooled edges get filtered out by perc_var
        must still appear in the trace artefacts.
        """
        _, _, _, _, feature_names, target_names = iris_split
        X_train, _, _, _, _, _ = iris_split

        # A high perc_var filters almost every pooled edge out of the graph...
        dpg_filtered = DecisionPredicateGraph(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config={
                "dpg": {
                    "default": {"perc_var": 0.5, "decimal_threshold": 6, "n_jobs": 1},
                    "graph_construction": {"mode": "execution_trace"},
                }
            },
        )
        dot = dpg_filtered.fit(X_train)
        filtered_graph, _ = dpg_filtered.to_networkx(dot)

        # ...but the same fit's trace artefacts are unaffected by perc_var.
        dpg_unfiltered = DecisionPredicateGraph(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config={
                "dpg": {
                    "default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
                    "graph_construction": {"mode": "execution_trace"},
                }
            },
        )
        dpg_unfiltered.fit(X_train)

        assert filtered_graph.number_of_edges() < len(
            dpg_unfiltered.discover_dfg(dpg_unfiltered._extract_trace_log(X_train))
        )
        assert dpg_filtered.get_trace_consistent_lrc() == dpg_unfiltered.get_trace_consistent_lrc()
        assert dpg_filtered.get_trace_signatures() == dpg_unfiltered.get_trace_signatures()


class TestTraceLRCNodeMetricsIntegration:
    """DPGExplainer forwards trace LRC only for execution_trace mode."""

    def test_explainer_uses_trace_lrc_in_execution_trace_mode(self, iris_rf, iris_split):
        from dpg.explainer import DPGExplainer

        X_train, _, _, _, feature_names, target_names = iris_split
        explainer = DPGExplainer(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
            dpg_config={
                "dpg": {
                    "default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
                    "graph_construction": {"mode": "execution_trace"},
                }
            },
        )
        explainer.fit(X_train)
        node_metrics = explainer._get_node_metrics()
        trace_lrc = explainer.builder.get_trace_consistent_lrc()

        by_label = node_metrics.set_index("Label")["Local reaching centrality"].to_dict()
        for label, score in trace_lrc.items():
            assert by_label[label] == pytest.approx(score)

    def test_explainer_uses_legacy_lrc_in_aggregated_mode(self, iris_rf, iris_split):
        from dpg.explainer import DPGExplainer

        X_train, _, _, _, feature_names, target_names = iris_split
        explainer = DPGExplainer(
            model=iris_rf,
            feature_names=feature_names,
            target_names=target_names,
        )
        explainer.fit(X_train)
        node_metrics = explainer._get_node_metrics()
        assert explainer.builder.get_trace_consistent_lrc() == {}
        assert not node_metrics.empty


class TestIrisLRCRankingComparison:
    """
    Fit Iris with both graph-construction implementations and compare the
    resulting predicate LRC rankings: pooled-graph NetworkX local reaching
    centrality (aggregated_transitions) vs. trace-consistent LRC
    (execution_trace).
    """

    def test_top_20_lrc_ranking_both_implementations(self, iris_rf, iris_split, capsys):
        from dpg.explainer import DPGExplainer

        X_train, _, _, _, feature_names, target_names = iris_split

        def fit_and_rank(mode):
            explainer = DPGExplainer(
                model=iris_rf,
                feature_names=feature_names,
                target_names=target_names,
                dpg_config={
                    "dpg": {
                        "default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
                        "graph_construction": {"mode": mode},
                    }
                },
            )
            explainer.fit(X_train)
            node_metrics = explainer._get_node_metrics()
            predicates = node_metrics[
                node_metrics["Label"].apply(DecisionPredicateGraph._is_predicate_label)
            ]
            ranked = predicates.sort_values(
                "Local reaching centrality", ascending=False
            ).head(20)
            return ranked[["Label", "Local reaching centrality"]].reset_index(drop=True)

        aggregated_ranking = fit_and_rank("aggregated_transitions")
        trace_ranking = fit_and_rank("execution_trace")

        assert not aggregated_ranking.empty
        assert not trace_ranking.empty
        assert aggregated_ranking["Local reaching centrality"].is_monotonic_decreasing
        assert trace_ranking["Local reaching centrality"].is_monotonic_decreasing

        with capsys.disabled():
            print("\n\nTop 20 predicates by LRC -- aggregated_transitions (pooled NetworkX LRC)")
            print("-" * 70)
            for rank, row in aggregated_ranking.iterrows():
                print(f"{rank + 1:>2}. {row['Label']:<35} {row['Local reaching centrality']:.4f}")

            print("\nTop 20 predicates by LRC -- execution_trace (trace-consistent LRC)")
            print("-" * 70)
            for rank, row in trace_ranking.iterrows():
                print(f"{rank + 1:>2}. {row['Label']:<35} {row['Local reaching centrality']:.4f}")
            print()


# ---------------------------------------------------------------------------
# PR #32 additions: context_order validation, decimal_threshold="auto",
# sink invariants, DOT/NetworkX round-trip for context-aware graphs.
# ---------------------------------------------------------------------------

import hashlib
import warnings

from dpg import DPGExplainer


def _config(mode="execution_trace", context_order=1, decimal_threshold=6, perc_var=1e-9, n_jobs=1):
    """Tiny helper to build a dpg_config for the new graph_construction key."""
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


class TestDecimalThresholdValidation:
    """``decimal_threshold`` accepts a non-negative int or ``"auto"``."""

    @pytest.mark.parametrize("bad_value", [-1, -10, True, False, 1.5, "six"])
    def test_invalid_decimal_threshold_is_rejected(self, iris_rf, iris_split, bad_value):
        _, _, _, _, feature_names, _ = iris_split
        with pytest.raises(
            DPGError, match="decimal_threshold must be a non-negative integer or 'auto'"
        ):
            DecisionPredicateGraph(
                iris_rf,
                feature_names,
                dpg_config=_config(decimal_threshold=bad_value),
            )

    def test_zero_decimal_threshold_is_accepted(self, iris_rf, iris_split):
        _, _, _, _, feature_names, _ = iris_split
        dpg = DecisionPredicateGraph(
            iris_rf, feature_names, dpg_config=_config(decimal_threshold=0)
        )
        assert dpg.decimal_threshold == 0


class TestContextOrderValidation:
    """``context_order`` accepts a positive int or ``"auto"``."""

    @pytest.mark.parametrize("bad_value", [0, -1, -2, True, False, 1.5])
    def test_invalid_context_order_is_rejected(self, iris_rf, iris_split, bad_value):
        _, _, _, _, feature_names, _ = iris_split
        with pytest.raises(
            DPGError, match="context_order must be a positive integer or 'auto'"
        ):
            DecisionPredicateGraph(
                iris_rf,
                feature_names,
                dpg_config=_config(context_order=bad_value),
            )

    def test_string_context_order_is_rejected(self, iris_rf, iris_split):
        _, _, _, _, feature_names, _ = iris_split
        with pytest.raises(
            DPGError, match="context_order must be a positive integer or 'auto'"
        ):
            DecisionPredicateGraph(
                iris_rf,
                feature_names,
                dpg_config=_config(context_order="two"),
            )


class TestDecimalPlacesHelper:
    """``_decimal_places`` is the building block of ``decimal_threshold='auto'``."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, 0),
            (1, 0),
            (-3, 0),
            (0.5, 1),
            (0.125, 3),
            (1.25, 2),
            (1.0001, 4),
        ],
    )
    def test_decimal_places_for_finite_values(self, value, expected):
        assert DecisionPredicateGraph._decimal_places(value) == expected

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_decimal_places_for_non_finite_values_returns_zero(self, bad_value):
        assert DecisionPredicateGraph._decimal_places(bad_value) == 0


class TestResolveDecimalThreshold:
    """``decimal_threshold='auto'`` derives precision from data and audits tree thresholds."""

    def test_non_auto_passes_through(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(decimal_threshold=4)
        )
        dpg._resolve_decimal_threshold(iris.data)
        assert dpg.get_decimal_threshold() == 4

    def test_auto_with_integer_data_returns_one(self):
        # 0 decimal places in data => precision 0, +1 = 1.
        rng = np.random.default_rng(0)
        X = rng.integers(0, 50, size=(40, 2)).astype(float)
        y = (X[:, 0] > 25).astype(int)
        model = RandomForestClassifier(n_estimators=3, random_state=0, n_jobs=1).fit(X, y)
        dpg = DecisionPredicateGraph(
            model, ["a", "b"], dpg_config=_config(decimal_threshold="auto")
        )
        dpg._resolve_decimal_threshold(X)
        assert dpg.get_decimal_threshold() == 1

    def test_auto_with_fractional_data_uses_max_precision(self):
        rng = np.random.default_rng(1)
        X = np.round(rng.random((40, 2)), 3)
        y = (X[:, 0] > 0.5).astype(int)
        model = RandomForestClassifier(n_estimators=3, random_state=0, n_jobs=1).fit(X, y)
        dpg = DecisionPredicateGraph(
            model, ["a", "b"], dpg_config=_config(decimal_threshold="auto")
        )
        dpg._resolve_decimal_threshold(X)
        # 3 decimal places in data => precision 3, +1 = 4.
        assert dpg.get_decimal_threshold() == 4

    def test_get_decimal_threshold_before_fit_raises(self, iris_rf):
        """Before ``fit()`` the public getter must fail loudly rather than
        silently return ``None``."""
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(decimal_threshold="auto")
        )
        dpg._resolved_decimal_threshold = None
        with pytest.raises(DPGError, match="decimal_threshold='auto' is resolved when fit"):
            dpg.get_decimal_threshold()


class TestAutoDecimalThresholdWarning:
    """Audit step in ``_resolve_decimal_threshold`` must warn on off-grid tree thresholds."""

    def test_off_grid_tree_threshold_emits_warning(self):
        """Tree thresholds off the data-derived grid warn -- but exact
        routing is preserved (only labels are rounded)."""
        rng = np.random.default_rng(42)
        X = rng.integers(0, 100, size=(60, 2)).astype(float)
        y = (X[:, 0] > 50).astype(int)

        model = RandomForestClassifier(n_estimators=2, random_state=0, n_jobs=1)
        model.fit(X, y)
        # Force one tree's threshold off the 1-decimal grid by a clear margin.
        first_tree = model.estimators_[0].tree_
        first_tree.threshold[0] = 50.123

        dpg = DecisionPredicateGraph(
            model, ["a", "b"], dpg_config=_config(decimal_threshold="auto")
        )
        with pytest.warns(RuntimeWarning, match="off the data-derived"):
            dpg._resolve_decimal_threshold(X)

    def test_on_grid_thresholds_emit_no_warning(self):
        """A tree whose thresholds already fit the 1-decimal grid must not warn."""
        rng = np.random.default_rng(123)
        X = rng.integers(0, 50, size=(40, 2)).astype(float)
        y = (X[:, 0] > 25).astype(int)
        model = RandomForestClassifier(n_estimators=2, random_state=0, n_jobs=1).fit(X, y)
        # Snap every threshold to the 1-decimal grid so the audit step finds nothing.
        for tree in model.estimators_:
            tree_ = tree.tree_
            for idx, feature_index in enumerate(tree_.feature):
                if feature_index < 0:
                    continue
                tree_.threshold[idx] = round(float(tree_.threshold[idx]), 1)
        dpg = DecisionPredicateGraph(
            model, ["a", "b"], dpg_config=_config(decimal_threshold="auto")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            dpg._resolve_decimal_threshold(X)
        assert dpg.get_decimal_threshold() == 1


class TestContextNodeHelpers:
    """The static ``_context_node`` helpers are pure functions."""

    def test_context_node_returns_sink_for_class_label(self):
        node = DecisionPredicateGraph._context_node(("a", "Class 0"), index=1, k=2)
        assert node == ("sink", "Class 0")

    def test_context_node_returns_sink_for_pred_label(self):
        node = DecisionPredicateGraph._context_node(("a", "Pred 1.5"), index=1, k=2)
        assert node == ("sink", "Pred 1.5")

    def test_context_node_window_at_start_truncates_to_seen_prefix(self):
        node = DecisionPredicateGraph._context_node(("A", "B"), index=0, k=3)
        assert node == ("ctx", ("A",))

    def test_context_node_full_window_after_k_steps(self):
        node = DecisionPredicateGraph._context_node(("A", "B", "C"), index=2, k=3)
        assert node == ("ctx", ("A", "B", "C"))

    def test_context_node_window_respects_k(self):
        node = DecisionPredicateGraph._context_node(("A", "B", "C"), index=2, k=2)
        assert node == ("ctx", ("B", "C"))

    @pytest.mark.parametrize(
        "node, expected",
        [
            (("sink", "Class 0"), ("Class 0", ())),
            (("ctx", ("A", "B")), ("B", ("A", "B"))),
        ],
    )
    def test_context_node_info_round_trip(self, node, expected):
        assert DecisionPredicateGraph._context_node_info(node) == expected


class TestDiscoverDFGContext:
    """Context-aware DFG construction must fall back at k=1 and split at k>1."""

    def test_k_one_falls_through_to_execution_trace(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=1)
        )
        dpg.fit(iris.data)
        log = dpg._extract_trace_log(iris.data)

        ctx = dpg.discover_dfg_context(log, 1)
        legacy = dpg.discover_dfg_execution_trace(log)
        assert ctx == legacy

    def test_k_two_splits_diverging_traces(self):
        """A hand-built trace log exposes the contextual split: the shared
        predicate ``p`` yields a (``ctx``, (``p``,)) node at k=2."""
        from sklearn.datasets import load_iris

        iris = load_iris()
        model = RandomForestClassifier(n_estimators=2, random_state=0, n_jobs=1).fit(
            iris.data, iris.target
        )
        dpg = DecisionPredicateGraph(
            model, iris.feature_names, dpg_config=_config(context_order=2)
        )

        log = pd.DataFrame(
            {
                "case:concept:name": ["c1", "c1", "c2", "c2"],
                "concept:name": ["p", "Class 0", "p", "Class 1"],
            }
        )

        dfg = dpg.discover_dfg_context(log, 2)
        ctx_nodes = [
            node for edge in dfg for node in edge
            if isinstance(node, tuple) and node[0] == "ctx"
        ]
        assert ("ctx", ("p",)) in ctx_nodes


class TestNodeLookupHelpers:
    """Public lookup helpers for graph nodes and traces."""

    def test_get_node_context_returns_empty_for_k1(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=1)
        )
        dpg.fit(iris.data)
        assert dpg.get_node_context("any-id") == ()

    def test_get_node_context_returns_empty_for_non_string_node(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=2)
        )
        dpg.fit(iris.data)

        class Weird:
            def __str__(self):
                return ""

        assert dpg.get_node_context(Weird()) == ()

    def test_get_node_ids_for_trace_k1_matches_string_keys(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=1)
        )
        dpg.fit(iris.data)

        labels = ("a", "b", "Class 0")
        ids = dpg.get_node_ids_for_trace(labels)
        assert len(ids) == 3
        # k=1 keys are just the label strings, hashed the same way the dot
        # generator hashes them.
        for label, node_id in zip(labels, ids):
            assert node_id == str(int(hashlib.sha1(label.encode()).hexdigest(), 16))

    def test_get_node_ids_for_trace_k_gt_1_uses_context_keys(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=2)
        )
        dpg.fit(iris.data)

        labels = ("a", "b", "Class 0")
        ids = dpg.get_node_ids_for_trace(labels)
        # The two non-sink labels live in distinct contexts, so they
        # must hash to different node ids.
        assert ids[0] != ids[1]


class TestGetPredicateLrc:
    """``get_predicate_lrc`` aggregates per-node LRC scores back to predicate labels."""

    def test_returns_scores_in_unit_interval(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=2)
        )
        graph, _ = dpg.to_networkx(dpg.fit(iris.data))
        scores = dpg.get_predicate_lrc(graph)
        assert isinstance(scores, dict)
        assert all(0.0 <= s <= 1.0 for s in scores.values())
        assert scores  # at least one predicate for the iris fit

    def test_aggregation_sums_node_lrcs(self, iris_rf):
        """A predicate appearing in two contexts must have score ==
        sum of its two node LRCs."""
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=2)
        )
        graph, _ = dpg.to_networkx(dpg.fit(iris.data))

        # Build the per-predicate sum ourselves and compare.
        expected = {}
        for node, data in graph.nodes(data=True):
            label = data.get("predicate")
            if label is None or not dpg._is_predicate_label(label):
                continue
            score = float(nx.local_reaching_centrality(graph, node, weight=None))
            expected[label] = expected.get(label, 0.0) + score

        actual = dpg.get_predicate_lrc(graph)
        assert set(actual) == set(expected)
        for label in actual:
            assert actual[label] == pytest.approx(expected[label], rel=1e-9)


class TestTraceConsistentLRCBackwardCompat:
    """``get_trace_consistent_lrc`` is kept for k=1 and emits a deprecation warning at k>1."""

    def test_k1_still_silently_returns_lrc(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=1)
        )
        dpg.fit(iris.data)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            dpg.get_trace_consistent_lrc()  # must not warn

    def test_k_gt_1_emits_deprecation_warning(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=2)
        )
        dpg.fit(iris.data)
        with pytest.warns(DeprecationWarning, match="get_predicate_lrc"):
            dpg.get_trace_consistent_lrc()


class TestDOTAndNetworkXRoundTrip:
    """The new DOT attribute ``dpg_context_order`` survives the NetworkX round trip."""

    def test_dot_attaches_context_order_to_every_node(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=2)
        )
        dot = dpg.fit(iris.data)
        body = "\n".join(dot.body)
        assert "dpg_context_order=" in body

    def test_to_networkx_parses_context_order_from_dot(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=2)
        )
        dot = dpg.fit(iris.data)
        graph, _ = dpg.to_networkx(dot)

        assert all(
            data["context_order"] == dpg.get_context_order() == 2
            for _, data in graph.nodes(data=True)
        )

    def test_node_records_attach_predicate_and_context(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=2)
        )
        dot = dpg.fit(iris.data)
        graph, _ = dpg.to_networkx(dot)
        for _, data in graph.nodes(data=True):
            assert "predicate" in data
            assert "context" in data
            assert isinstance(data["context"], tuple)


class TestTraceTreeLabels:
    """Both label extractors must produce the same leaf and the same length."""

    def test_legacy_and_native_have_same_length_and_same_leaf(self, iris_rf):
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=1)
        )
        dpg.fit(iris.data)
        sample = iris.data[0]
        tree = dpg.model.estimators_[0]

        legacy = dpg._trace_tree_labels_legacy(0, tree, sample)
        native = dpg._trace_tree_labels(0, tree, sample)

        assert legacy[-1] == native[-1]
        assert len(legacy) == len(native)

    def test_native_uses_decision_path_for_routing(self, iris_rf):
        """The native extractor must follow sklearn's ``decision_path``
        exactly -- both branches must come from sklearn, not from our own
        rounding comparison."""
        from sklearn.datasets import load_iris

        iris = load_iris()
        dpg = DecisionPredicateGraph(
            iris_rf, iris.feature_names, dpg_config=_config(context_order=1)
        )
        dpg.fit(iris.data)

        sample = iris.data[0].reshape(1, -1)
        tree = dpg.model.estimators_[0]
        indicator = tree.decision_path(sample)
        path = indicator.indices[indicator.indptr[0] : indicator.indptr[1]]
        leaf_id = int(tree.apply(sample)[0])
        # Number of predicates in the native trace == number of internal
        # nodes visited (path excludes the leaf itself).
        native = dpg._trace_tree_labels(0, tree, iris.data[0])
        predicates = [label for label in native if dpg._is_predicate_label(label)]
        assert len(predicates) == sum(1 for n in path if int(n) != leaf_id)
