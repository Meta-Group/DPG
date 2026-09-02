import pandas as pd
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple

class NodeMetrics:
    """Handles node-level metric calculations."""

    def extract_node_metrics(
        dpg_model,
        nodes_list: List[Tuple],
        trace_lrc_by_label: Optional[Dict[str, float]] = None,
    ) -> Any:
        """Compute per-node graph metrics for a DPG model.

        Args:
            dpg_model: NetworkX DiGraph representing the DPG.
            nodes_list: List of ``(node_id, label)`` tuples.
            trace_lrc_by_label: Optional mapping of node label to a
                trace-consistent local-reaching-centrality score (see
                ``DecisionPredicateGraph.get_trace_consistent_lrc``). When a
                label is present, its trace-consistent value is used in place
                of the pooled-graph NetworkX local reaching centrality.
                Labels absent from the mapping (including ``None``) keep the
                legacy NetworkX computation.

        Returns:
            DataFrame with columns ``['Node', 'Label', 'Degree', 'In degree nodes',
            'Out degree nodes', 'Betweenness centrality', 'Local reaching centrality']``.
        """
        in_nodes = {}
        out_nodes = {}
        degree = {}
        for node in dpg_model.nodes():
            in_nodes[node] = dpg_model.in_degree(node)
            out_nodes[node] = dpg_model.out_degree(node)
            degree[node] = in_nodes[node] + out_nodes[node]
        sample_size = int(1 * len(dpg_model.nodes()))
        betweenness_centrality = nx.betweenness_centrality(dpg_model, k=sample_size, normalized=True, weight='weight', endpoints=False)
        node_label_by_id = {node_id: label for node_id, label in nodes_list}
        local_reaching_centrality = {}
        for node in dpg_model.nodes():
            label = node_label_by_id.get(node)
            if trace_lrc_by_label is not None and label in trace_lrc_by_label:
                local_reaching_centrality[node] = trace_lrc_by_label[label]
            else:
                local_reaching_centrality[node] = nx.local_reaching_centrality(dpg_model, node, weight='weight')
        data_node = {
            "Node": list(dpg_model.nodes()),
            "Degree": list(degree.values()),
            "In degree nodes": list(in_nodes.values()),
            "Out degree nodes": list(out_nodes.values()),
            "Betweenness centrality": list(betweenness_centrality.values()),
            "Local reaching centrality": list(local_reaching_centrality.values()),
        }
        df_data_node = pd.DataFrame(data_node).set_index('Node')
        df_nodes_list = pd.DataFrame(nodes_list, columns=["Node", "Label"]).set_index('Node')
        return pd.concat([df_data_node, df_nodes_list], axis=1, join='inner').reset_index()