"""
Smoke tests: verify all public packages and key symbols are importable.
"""


def test_import_dpg():
    import dpg


def test_import_metrics():
    import metrics


def test_import_core_classes():
    from dpg.core import DecisionPredicateGraph, DPGError


def test_import_explainer():
    from dpg.explainer import DPGExplainer, DPGExplanation


def test_import_node_metrics():
    from metrics.nodes import NodeMetrics


def test_import_edge_metrics():
    from metrics.edges import EdgeMetrics


def test_import_graph_metrics():
    from metrics.graph import GraphMetrics


def test_import_sklearn_dpg():
    from dpg.sklearn_dpg import test_dpg, select_dataset


def test_import_visualizer():
    from dpg.visualizer import plot_dpg, plot_dpg_communities
