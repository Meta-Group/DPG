import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from typing import Dict, List, Set
import re
import math
from collections import defaultdict
import pandas as pd

class GraphMetrics:
    """Handles graph-level metric calculations"""
    
    def __init__(self, target_names=None):
        self.target_names = target_names

    @staticmethod
    def calculate_class_boundaries(key: str, nodes: List[str], class_names: List[str]) -> tuple:
        """Static method for boundary calculation"""
        feature_bounds = {}
        boundaries = []
        for node in nodes:
            parts = re.split(' <= | > ', node)
            if len(parts) != 2:
                continue
            feature, value_str = parts
            try:
                value = float(value_str)
            except ValueError:
                continue
                
            if feature not in feature_bounds:
                feature_bounds[feature] = [math.inf, -math.inf]
                
            if '>' in node:
                if value < feature_bounds[feature][0]:
                    feature_bounds[feature][0] = value
            else:
                if value > feature_bounds[feature][1]:
                    feature_bounds[feature][1] = value

        for feature, (min_greater, max_lessequal) in feature_bounds.items():
            if min_greater == math.inf:
                boundary = f"{feature} <= {max_lessequal}"
            elif max_lessequal == -math.inf:
                boundary = f"{feature} > {min_greater}"
            else:
                boundary = f"{min_greater} < {feature} <= {max_lessequal}"
            boundaries.append(boundary)
        return str(key), boundaries

    @classmethod
    def calculate_boundaries(cls, class_dict: Dict, class_names: List[str]) -> Dict:
        """Parallel boundary calculation"""
        results = Parallel(n_jobs=-1)(
            delayed(cls.calculate_class_boundaries)(key, nodes, class_names) 
            for key, nodes in class_dict.items()
        )
        return dict(results)

    @classmethod
    def extract_class_boundaries(cls, dpg_model: nx.DiGraph, nodes_list: List[tuple], target_names: List[str]) -> Dict:
        """Main interface for graph metrics"""
        # Create node mappings
        node_label_to_id = {node[1]: node[0] for node in nodes_list if "->" not in node[0]}
        node_id_to_label = {v: k for k, v in node_label_to_id.items()}
        
        # Class boundaries
        terminal_nodes = {
            k: v for k, v in node_label_to_id.items() 
            if any(x in k for x in ['Class', 'Pred'])
        }
        predecessors = {}
        
        for class_name, node_id in terminal_nodes.items():
            try:
                preds = nx.descendants(dpg_model.reverse(), node_id)
                predecessors[class_name] = [
                    node_id_to_label[p] for p in preds 
                    if p in node_id_to_label and not any(
                        x in node_id_to_label[p] for x in ['Class', 'Pred']
                    )
                ]
            except nx.NetworkXError:
                predecessors[class_name] = []
        
        # Calculate boundaries
        class_bounds = cls.calculate_boundaries(predecessors, target_names)
        
        return {
            "Class Bounds": class_bounds
        }

    @classmethod
    def extract_graph_metrics(cls, dpg_model: nx.DiGraph, nodes_list: List[tuple], target_names: List[str]) -> Dict:
        """Backwards-compatible graph metrics interface.

        This delegates to the current LPA-based implementation to keep
        older examples and notebooks working.
        """
        return cls.extract_graph_metrics_lpa(dpg_model, nodes_list, target_names)

    @classmethod
    def extract_graph_metrics_lpa(cls, dpg_model: nx.DiGraph, nodes_list: List[tuple], target_names: List[str]) -> Dict:
        """Main interface for graph metrics"""
        # Create node mappings
        node_label_to_id = {node[1]: node[0] for node in nodes_list if "->" not in node[0]}
        node_id_to_label = {v: k for k, v in node_label_to_id.items()}
        
        # Community detection
        communities = list(nx.community.asyn_lpa_communities(dpg_model, weight='weight'))
        communities_labels = [
            {node_id_to_label[str(node)] for node in community} 
            for community in communities
        ]
        
        # Class boundaries
        terminal_nodes = {
            k: v for k, v in node_label_to_id.items() 
            if any(x in k for x in ['Class', 'Pred'])
        }
        predecessors = {}
        
        for class_name, node_id in terminal_nodes.items():
            try:
                preds = nx.descendants(dpg_model.reverse(), node_id)
                predecessors[class_name] = [
                    node_id_to_label[p] for p in preds 
                    if p in node_id_to_label and not any(
                        x in node_id_to_label[p] for x in ['Class', 'Pred']
                    )
                ]
            except nx.NetworkXError:
                predecessors[class_name] = []
        
        # Calculate boundaries
        class_bounds = cls.calculate_boundaries(predecessors, target_names)
        
        return {
            "Communities": communities_labels,
            "Class Bounds": class_bounds
        }
    
    @classmethod
    def extract_communities(cls, dpg_model, df_node_metrics, nodes_list, threshold_clusters = 0.2) -> Dict:
        node_to_label = df_node_metrics.set_index('Node')['Label'].to_dict()

        class_nodes = {i[0] : i[1] for i in nodes_list if 'Class' in i[1]}
        clusters, node_prob, confidence = cls.clustering(dpg_model, class_nodes, threshold_clusters)

        clusters_labels = {k: [node_to_label.get(n, n) for n in v] for k, v in clusters.items()}
        node_probs_labels = {node_to_label.get(str(k), str(k)): v for k, v in node_prob.items()}
        confidence_labels = {node_to_label.get(str(k), str(k)): v for k, v in confidence.items()}
        #feature_count_df, feature_intervals_df = cls.create_dataframes(clusters_labels)

        return {"Clusters": clusters_labels, "Probability": node_probs_labels, "Confidence Interval": confidence_labels}

    @staticmethod
    def communities_to_csv(communities: Dict, file_path: str) -> None:
        """
        Save communities output to a CSV file.

        The CSV is written in a long format with columns:
        Section, Key, Value.
        """
        def _to_builtin(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_builtin(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                converted = [_to_builtin(v) for v in obj]
                return type(obj)(converted)
            return obj

        communities = _to_builtin(communities)
        rows = []
        for section, mapping in communities.items():
            if isinstance(mapping, dict):
                for key, value in mapping.items():
                    rows.append({"Section": section, "Key": key, "Value": value})
            else:
                rows.append({"Section": section, "Key": "", "Value": mapping})

        df = pd.DataFrame(rows, columns=["Section", "Key", "Value"])
        df.to_csv(file_path, index=False)
    
    @classmethod
    def clustering (cls, dpg_model, class_nodes, threshold = None):
    
        classes = sorted(set(class_nodes.values()))
        class_by_node = dict(class_nodes)
        class_set = set(class_by_node.keys())

        nodes = list(dpg_model.nodes())
        n = len(nodes)
        
        idx = {idx_node : node for node, idx_node in enumerate(nodes)}
        
        # P
        P = np.zeros((n, n), dtype = float)
        for node in nodes:
            i = idx[node]
            if node in class_set:
                P[i, i] = 1.0
                continue

            out_edges = list(dpg_model.out_edges(node, data=True))
            
            weight_sum = 0

            for out_node, in_node, weight in out_edges:
                weight_sum += weight.get('weight', 1)

            if weight_sum > 0:
                for out_node, in_node, weight in out_edges:
                    j = idx[in_node]
                    P[i, j] = weight.get('weight', 1) / weight_sum
            else:
                P[i, i] = 1.0
        
        # Order to obtain Q and R
        transient = []
        absorbing = []
        for node in nodes:
            if node not in class_set:
                transient.append(node)
            elif node in class_set:
                absorbing.append(node)

        t = len(transient)

        perm = transient + absorbing
        
        perm_idx = [idx[node] for node in perm]
        
        Pp = P[perm_idx][:, perm_idx]

        Q = Pp[:t, :t]
        R = Pp[:t, t:]

        # N
        I = np.eye(t)
        N = np.linalg.solve(I - Q, I)

        # Absorbing probability for each node
        B = N @ R

        # ----- #
        class_labels = [class_by_node[node] for node in absorbing]

        class_to_cols = {}
        for class_index in range(len(absorbing)):
            label = class_labels[class_index]
            if label not in class_to_cols:
                class_to_cols[label] = []
            class_to_cols[label].append(class_index)
        
        # Distribution for transient nodes
        node_probs = {}

        for index_row in range(len(transient)):
            node = transient[index_row]

            probs = {}
            for label in classes:
                probs[label] = 0.0
            
            # sum columns for class
            for label in classes:
                cols = class_to_cols.get(label, [])
                total = 0.0
                for index_col in cols:
                    total += B[index_row, index_col]
                probs[label] = np.round(total,2)

            node_probs[node] = probs
        
        # Distribution for absorbing nodes
        for node in absorbing:
            probs = {}
            for label in classes:
                probs[label] = 0.0
            probs[class_nodes[node]] = 1.0
            
            node_probs[node] = probs

        # Clusters
        clusters = {}
        for label in classes:
            clusters[label] = []
        
        if threshold is not None:
            clusters['Ambiguous'] = []

        confidence = {}

        for node in nodes:
            probs = node_probs[node]

            top_label = None
            top_prob = -1.0
            second_top_prob = -1.0

            # Top probability and cluster identification
            for label in classes:
                prob = probs[label]
                if prob > top_prob:
                    top_prob = prob
                    top_label = label

            # Second top probability
            for label in classes:
                prob = probs[label]
                if label != top_label and prob > second_top_prob:
                    second_top_prob = prob

            margin = top_prob - (second_top_prob if second_top_prob >= 0.0 else 0.0)

            confidence[node] = np.round(margin,2)

            
            if threshold is None:
                clusters[top_label].append(node)

            else:
                if top_prob > threshold:       
                    clusters[top_label].append(node)     
                else:
                    clusters['Ambiguous'].append(node)


        return clusters, node_probs, confidence
    
    @classmethod
    def extract_feature_intervals(cls, decisions):
            feature_count = defaultdict(int)
            feature_intervals = defaultdict(lambda: {"min": float('-inf'), "max": float('inf')})
            
            regex = r'([a-zA-Z0-9_]+)\s*([<=|>]+)\s*([-+]?[\d.]+)'
            
            for decision in decisions:
                match = re.search(regex, decision)
                if match:
                    feature, operator, value = match.groups()
                    value = float(value)
                    feature_count[feature] += 1
                    
                    if '>' in operator:
                        feature_intervals[feature]["min"] = max(feature_intervals[feature]["min"], value)
                    elif '<=' in operator:
                        feature_intervals[feature]["max"] = min(feature_intervals[feature]["max"], value)
                        
            return feature_count, feature_intervals

    @classmethod
    def create_dataframes(cls, data):
        all_found_features = set()
        temp_counts = {}
        temp_intervals = {}

        for class_name, decisions in data.items():
            counts, intervals = cls.extract_feature_intervals(decisions)
            temp_counts[class_name] = counts
            temp_intervals[class_name] = intervals
            all_found_features.update(counts.keys())

        sorted_features = sorted(list(all_found_features))
        
        feature_count_df = pd.DataFrame(index=sorted_features, columns=data.keys())
        
        interval_index = []
        for f in sorted_features:
            interval_index.extend([f"{f}_min", f"{f}_max"])
        feature_intervals_df = pd.DataFrame(index=interval_index, columns=data.keys())

        for class_name in data.keys():
            for feat in sorted_features:
                feature_count_df.loc[feat, class_name] = temp_counts[class_name].get(feat, 0)
                
                inter = temp_intervals[class_name].get(feat, {"min": float('-inf'), "max": float('inf')})
                feature_intervals_df.loc[f"{feat}_min", class_name] = inter["min"]
                feature_intervals_df.loc[f"{feat}_max", class_name] = inter["max"]

        return feature_count_df, feature_intervals_df
