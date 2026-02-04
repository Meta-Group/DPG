import os
import shutil

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dpg.core import DecisionPredicateGraph
from dpg.visualizer import plot_dpg, plot_dpg_communities
from metrics.graph import GraphMetrics
from metrics.nodes import NodeMetrics
from metrics.edges import EdgeMetrics


def _build_small_dpg():
    base_dir = os.getcwd()
    dataset_path = os.path.join(base_dir, "datasets", "custom.csv")
    dataset_raw = pd.read_csv(dataset_path, index_col=0)

    features = dataset_raw.iloc[:, :-1]
    labels = dataset_raw.iloc[:, -1]

    features = features.replace([np.inf, -np.inf], np.nan).fillna(features.mean())
    features = np.round(features, 2)

    model = RandomForestClassifier(n_estimators=5, random_state=27)
    model.fit(features, labels)

    target_names = np.unique(labels).astype(str).tolist()
    dpg_builder = DecisionPredicateGraph(
        model=model,
        feature_names=features.columns,
        target_names=target_names,
    )
    graph_dot = dpg_builder.fit(features.values)
    dpg_graph, dpg_nodes = dpg_builder.to_networkx(graph_dot)

    class_boundaries = GraphMetrics.extract_class_boundaries(
        dpg_graph, dpg_nodes, target_names=target_names
    )
    node_metrics = NodeMetrics.extract_node_metrics(dpg_graph, dpg_nodes)
    edge_metrics = EdgeMetrics.extract_edge_metrics(dpg_graph, dpg_nodes)

    return (
        dpg_graph,
        dpg_nodes,
        class_boundaries,
        node_metrics,
        edge_metrics,
        graph_dot,
    )


def test_dpg_end_to_end_small_dataset():
    dpg_graph, dpg_nodes, class_boundaries, node_metrics, edge_metrics, _ = _build_small_dpg()

    assert dpg_graph is not None
    assert len(dpg_nodes) > 0
    assert len(class_boundaries) > 0
    assert not node_metrics.empty
    assert not edge_metrics.empty


def test_dpg_plots_render(tmp_path):
    if shutil.which("dot") is None:
        import pytest

        pytest.skip("Graphviz 'dot' executable not available")

    dpg_graph, dpg_nodes, _, node_metrics, edge_metrics, graph_dot = _build_small_dpg()
    communities = GraphMetrics.extract_communities(dpg_graph, node_metrics, dpg_nodes)

    plot_dpg(
        "test_dpg_plot",
        graph_dot,
        node_metrics,
        edge_metrics,
        save_dir=str(tmp_path),
        show=False,
        export_pdf=False,
    )
    plot_dpg_communities(
        "test_dpg_plot",
        graph_dot,
        node_metrics,
        communities,
        save_dir=str(tmp_path),
        show=False,
        export_pdf=False,
    )

    assert (tmp_path / "test_dpg_plot.png").exists()
    assert (tmp_path / "test_dpg_plot_communities.png").exists()
