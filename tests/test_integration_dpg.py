import os
import shutil

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

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


def test_graph_construction_modes_can_change_global_graph():
    X, y = make_classification(
        n_samples=220,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        n_classes=3,
        random_state=17,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y.astype(str))

    model = RandomForestClassifier(n_estimators=11, max_depth=4, random_state=11)
    model.fit(X_df, y_s)

    common_cfg = {
        "dpg": {
            "default": {"perc_var": 0.02, "decimal_threshold": 6, "n_jobs": 1},
        }
    }
    agg_builder = DecisionPredicateGraph(
        model=model,
        feature_names=feature_names,
        target_names=model.classes_.astype(str).tolist(),
        dpg_config={
            "dpg": {
                **common_cfg["dpg"],
                "graph_construction": {"mode": "aggregated_transitions"},
            }
        },
    )
    trace_builder = DecisionPredicateGraph(
        model=model,
        feature_names=feature_names,
        target_names=model.classes_.astype(str).tolist(),
        dpg_config={
            "dpg": {
                **common_cfg["dpg"],
                "graph_construction": {"mode": "execution_trace"},
            }
        },
    )

    agg_graph, _ = agg_builder.to_networkx(agg_builder.fit(X_df.values))
    trace_graph, _ = trace_builder.to_networkx(trace_builder.fit(X_df.values))

    agg_edges = {(u, v, round(float(d.get("weight", 0.0)), 6)) for u, v, d in agg_graph.edges(data=True)}
    trace_edges = {(u, v, round(float(d.get("weight", 0.0)), 6)) for u, v, d in trace_graph.edges(data=True)}

    assert len(agg_edges) > 0
    assert len(trace_edges) > 0
    assert agg_edges != trace_edges
