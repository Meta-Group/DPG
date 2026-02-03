import pandas as pd
pd.set_option("display.max_colwidth", 255)
import re
import math
import os
import numpy as np

from tqdm import tqdm
import graphviz
import networkx as nx
import hashlib
import yaml
from joblib import Parallel, delayed

from typing import List, Dict, Union
from sklearn.base import is_classifier, is_regressor

# Handle OmegaConf DictConfig if available
try:
    from omegaconf import DictConfig, OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

from sklearn.ensemble import (AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor)

DEFAULT_DPG_CONFIG = {
    "dpg": {
        "default": {
            "perc_var": 0.000000001,
            "decimal_threshold": 6,
            "n_jobs": -1,
        },
        "visualization": {},
    }
}

class DPGError(Exception):
    """Base exception class for DPG-specific errors"""
    pass


class DecisionPredicateGraph:
    """
    Main class for converting tree-based ensemble models into interpretable graphs.
    
    Attributes:
        model: Trained tree ensemble model (RandomForest, AdaBoost, etc.)
        feature_names: List of feature names
        target_names: List of target class names
        perc_var: Minimum path frequency threshold (0-1)
        decimal_threshold: Rounding precision for feature values
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    def __init__(self, model, feature_names, target_names=None, config_file="config.yaml", dpg_config=None):
        """
        Initialize DPG converter with model and configuration.
        
        Args:
            model: Tree ensemble model with estimators_ attribute
            feature_names: List of feature names
            target_names: Optional list of target class names
            config_file: Path to YAML config file (fallback if dpg_config not provided)
            dpg_config: Optional dict with DPG config parameters (overrides config_file)
        """
        # Load configuration from provided config, file, or defaults
        if dpg_config is not None:
            config = dpg_config
        else:
            config = None
            if config_file:
                if os.path.exists(config_file):
                    with open(config_file) as f:
                        config = yaml.safe_load(f)
                else:
                    print(f"Config file not found at '{config_file}'. Using built-in defaults.")
            if config is None:
                config = DEFAULT_DPG_CONFIG
        
        # Convert OmegaConf DictConfig to regular dict if needed
        if HAS_OMEGACONF and isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        # Handle dict-like objects that have to_dict() method (like custom DictConfig)
        elif hasattr(config, 'to_dict'):
            config = config.to_dict()
        
        # Input validation
        if not hasattr(model, 'estimators_'):
            raise DPGError("Model must be a tree-based ensemble")
        if len(feature_names) == 0:
            raise DPGError("Feature names cannot be empty")
        
        # Initialize attributes
        self.model = model
        self.feature_names = feature_names
        self.target_names = target_names #TODO create "Class as class name"
        
        # Get config values with defaults
        dpg_config_section = config.get('dpg', {})
        default_config = dpg_config_section.get('default', {})

        self.perc_var = default_config.get('perc_var', DEFAULT_DPG_CONFIG["dpg"]["default"]["perc_var"])
        self.decimal_threshold = default_config.get('decimal_threshold', DEFAULT_DPG_CONFIG["dpg"]["default"]["decimal_threshold"])
        self.n_jobs = default_config.get('n_jobs', DEFAULT_DPG_CONFIG["dpg"]["default"]["n_jobs"])
        
        # Validate required config values
        if self.perc_var is None:
            raise DPGError("perc_var not found in DPG config")
        if self.decimal_threshold is None:
            raise DPGError("decimal_threshold not found in DPG config")
        if self.n_jobs is None:
            raise DPGError("n_jobs not found in DPG config")
        
        print(f"DPG initialized with perc_var={self.perc_var}, decimal_threshold={self.decimal_threshold}, n_jobs={self.n_jobs}")
        # Store visualization config for use by utils
        self.visualization_config = dpg_config_section.get('visualization', DEFAULT_DPG_CONFIG["dpg"]["visualization"])

    def fit(self, X_train):
        """
        Main pipeline: Extract decision paths → Build graph → Generate visualization.
        
        Args:
            X_train: Training data (n_samples, n_features)
            
        Returns:
            graphviz.Digraph: Visualizable graph object
        """
        print("\nStarting DPG extraction *****************************************")
        print("Model Class:", self.model.__class__.__name__)
        print("Model Class Module:", self.model.__class__.__module__)
        print("Model Estimators: ", len(self.model.estimators_))
        print("Model Params: ", self.model.get_params())
        print("*****************************************************************")

        # Extract decision paths (parallel or sequential)
        if self.n_jobs == 1:
            log = Parallel(n_jobs=self.n_jobs)(
                delayed(self.tracing_ensemble)(i, sample) for i, sample in tqdm(list(enumerate(X_train)), total=len(X_train))
            )
        else:
            log = Parallel(n_jobs=self.n_jobs)(
                delayed(self.tracing_ensemble_parallel)(i, sample) for i, sample in tqdm(list(enumerate(X_train)), total=len(X_train))
            )

        # Process extracted paths
        log = [item for sublist in log for item in sublist]
        log_df = pd.DataFrame(log, columns=["case:concept:name", "concept:name"])

        print(f'Total of paths: {len(log_df["case:concept:name"].unique())}')

        # Filter infrequent paths if threshold set
        if self.perc_var > 0:
            log_df = self.filter_log(log_df)

        print('Building DPG...')
        dfg = self.discover_dfg(log_df)

        print('Extracting graph...')
        return self.generate_dot(dfg)

    def tracing_ensemble(self, case_id, sample):
        """
        Extract decision path for a single sample (generator version).
        
        Args:
            case_id: Sample identifier
            sample: Feature values (1D array)
            
        Yields:
            List[str]: Path segments as [prefix, decision/prediction]
        """
        is_regressor = isinstance(self.model, (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor))
        sample = sample.reshape(-1)
        for i, tree in enumerate(self.model.estimators_):
            tree_ = tree.tree_
            node_index = 0
            prefix = f"sample{case_id}_dt{i}"
            while True:
                left = tree_.children_left[node_index]
                right = tree_.children_right[node_index]
                if left == right:
                    if is_regressor:
                        pred = round(tree_.value[node_index][0][0], 2)
                        yield [prefix, f"Pred {pred}"]
                    else:
                        pred_class = tree_.value[node_index].argmax()
                        #Using the original class name
                        if self.target_names is not None:
                            pred_class = self.target_names[pred_class]
                        yield [prefix, f"Class {pred_class}"]
                    break
                feature_index = tree_.feature[node_index]
                threshold = round(tree_.threshold[node_index], self.decimal_threshold)
                feature_name = self.feature_names[feature_index]
                sample_val = sample[feature_index]
                if sample_val <= threshold:
                    condition = f"{feature_name} <= {threshold}"
                    node_index = left
                else:
                    condition = f"{feature_name} > {threshold}"
                    node_index = right
                yield [prefix, condition]

    #for parallel processing, when using n_jobs>1
    def tracing_ensemble_parallel(self, case_id, sample):
        is_regressor = isinstance(self.model, (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor))
        sample = sample.reshape(-1)
        result = []
        for i, tree in enumerate(self.model.estimators_):
            tree_ = tree.tree_
            node_index = 0
            prefix = f"sample{case_id}_dt{i}"
            while True:
                left = tree_.children_left[node_index]
                right = tree_.children_right[node_index]
                if left == right:
                    if is_regressor:
                        pred = round(tree_.value[node_index][0][0], 2)
                        result.append([prefix, f"Pred {pred}"])
                    else:
                        pred_class = tree_.value[node_index].argmax()
                        if self.target_names is not None:
                            pred_class = self.target_names[pred_class]
                        result.append([prefix, f"Class {pred_class}"])
                    break
                feature_index = tree_.feature[node_index]
                threshold = round(tree_.threshold[node_index], self.decimal_threshold)
                feature_name = self.feature_names[feature_index]
                sample_val = sample[feature_index]
                if sample_val <= threshold:
                    condition = f"{feature_name} <= {threshold}"
                    node_index = left
                else:
                    condition = f"{feature_name} > {threshold}"
                    node_index = right
                result.append([prefix, condition])
        return result
            

    def filter_log(self, log):
        """
        Filter paths based on frequency threshold.
        
        Args:
            log: DataFrame of extracted paths
            
        Returns:
            pd.DataFrame: Filtered paths meeting perc_var threshold
        """
        from collections import defaultdict
        variant_map = defaultdict(list)
        for case_id, group in log.groupby("case:concept:name", sort=False):
            variant = "|".join(group["concept:name"].values)
            variant_map[variant].append(case_id)

        case_ids_to_keep = set()
        min_count = len(log["case:concept:name"].unique()) * self.perc_var
        for variant, case_ids in variant_map.items():
            if len(case_ids) >= min_count:
                case_ids_to_keep.update(case_ids)
        return log[log["case:concept:name"].isin(case_ids_to_keep)].copy()

    def discover_dfg(self, log):
        """
        Build directed frequency graph from path logs.
        
        Args:
            log: DataFrame of decision paths
            
        Returns:
            Dict[tuple, int]: Edge frequencies as {(source, target): count}
        """
        cases = log["case:concept:name"].unique()
        if len(cases) == 0:
            raise Exception("There is no paths with the current value of perc_var and decimal_threshold!")

        # Optimized: Group by case once, then process each group
        # This avoids repeated filtering of the dataframe for each case
        dfg = {}
        grouped = log.groupby("case:concept:name", sort=False)
        
        for case, trace_df in tqdm(grouped, desc="Processing cases", total=len(cases)):
            trace_df = trace_df.sort_values(by="case:concept:name")
            concepts = trace_df["concept:name"].values
            for i in range(len(concepts) - 1):
                key = (concepts[i], concepts[i + 1])
                dfg[key] = dfg.get(key, 0) + 1
        
        return dfg

    def generate_dot(self, dfg):
        """
        Convert frequency graph to Graphviz format.
        
        Args:
            dfg: Directed frequency graph
            
        Returns:
            graphviz.Digraph: Visualizable graph
        """
        # Get visualization config
        viz_config = self.visualization_config
        graph_attrs = viz_config.get('graph_attrs', {})
        node_attrs = viz_config.get('node_attrs', {})
        
        # Build graph_attr dict
        final_graph_attr = {
            "bgcolor": graph_attrs.get('bgcolor'),
            "rankdir": graph_attrs.get('rankdir'),
            "overlap": "false",
            "fontsize": "20"
        }
        
        # Build node_attr dict
        final_node_attr = {
            "shape": node_attrs.get('shape')
        }
        
        # Get fillcolor for regular nodes
        default_fillcolor = node_attrs.get('fillcolor')
        
        dot = graphviz.Digraph(
            "dpg",
            engine="dot",
            graph_attr=final_graph_attr,
            node_attr=final_node_attr,
        )
        def _escape_dot_label(label: str) -> str:
            # Escape characters that can break DOT parsing.
            return (
                label.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("[", "\\[")
                .replace("]", "\\]")
            )

        added_nodes = set()
        for k, v in sorted(dfg.items(), key=lambda item: item[1]):
            for activity in k:
                if activity not in added_nodes:
                    dot.node(
                        str(int(hashlib.sha1(activity.encode()).hexdigest(), 16)),
                        label=_escape_dot_label(activity),
                        style="filled",
                        fontsize="20",
                        fillcolor=default_fillcolor,
                    )
                    added_nodes.add(activity)
            dot.edge(
                str(int(hashlib.sha1(k[0].encode()).hexdigest(), 16)),
                str(int(hashlib.sha1(k[1].encode()).hexdigest(), 16)),
                label=str(v),
                penwidth="1",
                fontsize="18"
            )
        return dot

    def to_networkx(self, graphviz_graph):
        """
        Convert Graphviz graph to NetworkX format.
        
        Args:
            graphviz_graph: Input graph
            
        Returns:
            Tuple[nx.DiGraph, List]: NetworkX graph and node metadata
        """
        networkx_graph = nx.DiGraph()
        nodes_list = []
        edges = []
        weights = {}
        for edge in graphviz_graph.body:
            if "->" in edge:
                src, dest = edge.split("->")
                src = src.strip()
                dest = dest.split(" [label=")[0].strip()
                weight = None
                if "[label=" in edge:
                    attr = edge.split("[label=")[1].split("]")[0].split(" ")[0]
                    weight = float(attr) if attr.replace(".", "").isdigit() else None
                    weights[(src, dest)] = weight
                edges.append((src, dest))
            if "[label=" in edge:
                id, desc = edge.split("[label=")
                id = id.replace("\t", "").replace(" ", "")
                desc = desc.split(" fillcolor=")[0].replace('"', "")
                nodes_list.append([id, desc])
        for src, dest in edges:
            if (src, dest) in weights:
                networkx_graph.add_edge(src, dest, weight=weights[(src, dest)])
            else:
                networkx_graph.add_edge(src, dest)
        return networkx_graph, sorted(nodes_list, key=lambda x: x[0])
