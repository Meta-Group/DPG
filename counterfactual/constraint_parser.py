import ast
import re
import json
import os
import sys
import numpy as np

# Handle OmegaConf DictConfig if available
try:
    from omegaconf import DictConfig, OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

from dpg import DecisionPredicateGraph
from metrics.graph import GraphMetrics
from metrics.nodes import NodeMetrics

class ConstraintParser:
    """Parse, normalise, and extract feature constraints from DPG models.

    Reads constraint files or extracts constraints directly from a
    :class:`~dpg.DecisionPredicateGraph`, returning structured
    dictionaries suitable for counterfactual generation and scoring.
    """

    def __init__(self, filename=None):
        self.filename = filename
        self.constraints_dict = {}

    @staticmethod
    def parse_condition(condition):
        """Parse a single condition string into a list of dictionaries with feature, operator, and value."""
        parts = re.split(r" (<=|>=|<|>|==) ", condition.strip())
        if len(parts) == 3:
            feature, operator, value = parts
            return [{"feature": feature.strip(), "operator": operator, "value": float(value.strip())}]
        elif len(parts) == 5:
            value1, operator1, feature, operator2, value2 = parts
            return [
                {"feature": feature.strip(), "operator": operator1, "value": float(value1.strip())},
                {"feature": feature.strip(), "operator": operator2, "value": float(value2.strip())}
            ]
        else:
            return None

    @staticmethod
    def normalize_constraints(constraints):
        """Normalize DPG constraints into per-class, per-feature intervals.
        
        Converts constraint lists into a dictionary structure with min/max bounds
        per feature per class, keeping the most restrictive bounds when multiple
        constraints exist for the same feature.
        
        Args:
            constraints: Dict mapping class names to lists of constraint dicts.
                        Each constraint dict has 'feature', 'min', and 'max' keys.
                        
        Returns:
            Dict mapping class names to dicts of {feature: {'min': val, 'max': val}},
            with features ordered alphabetically for deterministic output.
            
        Example:
            >>> constraints = {
            ...     'Class 0': [
            ...         {'feature': 'age', 'min': 18, 'max': 65},
            ...         {'feature': 'age', 'min': 25, 'max': None}  # More restrictive min
            ...     ]
            ... }
            >>> ConstraintParser.normalize_constraints(constraints)
            {'Class 0': {'age': {'min': 25, 'max': 65}}}
        """
        normalized = {}
        for cname in sorted(constraints.keys()):
            feature_map = {}
            for entry in constraints[cname]:
                f = entry.get('feature')
                minv = entry.get('min')
                maxv = entry.get('max')
                if f not in feature_map:
                    feature_map[f] = {'min': minv, 'max': maxv}
                else:
                    cur = feature_map[f]
                    # For min (lower bound), keep the most restrictive (largest) value if present
                    if minv is not None:
                        if cur['min'] is None or minv > cur['min']:
                            cur['min'] = minv
                    # For max (upper bound), keep the most restrictive (smallest) value if present
                    if maxv is not None:
                        if cur['max'] is None or maxv < cur['max']:
                            cur['max'] = maxv
            # Order features alphabetically for deterministic display
            normalized[cname] = {k: feature_map[k] for k in sorted(feature_map.keys())}
        return normalized

    def read_constraints_from_file(self):
        def _convert_operator_list_to_minmax(operator_list):
            # operator_list: [{'feature':'Age','operator':'<=','value':3.4}, ...]
            features_map = {}
            for cond in operator_list:
                f = cond.get('feature')
                op = cond.get('operator')
                val = cond.get('value')
                if f not in features_map:
                    features_map[f] = {'min': None, 'max': None}
                if op in ('>', '>='):
                    if features_map[f]['min'] is None or val > features_map[f]['min']:
                        features_map[f]['min'] = val
                elif op in ('<', '<='):
                    if features_map[f]['max'] is None or val < features_map[f]['max']:
                        features_map[f]['max'] = val
            return [{'feature': f, 'min': mm['min'], 'max': mm['max']} for f, mm in features_map.items()]

        with open(self.filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                class_label, json_string = line.split(":", 1)
                class_label = class_label.strip()
                json_string = json_string.strip().replace("'", '"').replace("None", "null")
                try:
                    parsed = json.loads(json_string)
                    if class_label.lower() == 'class bounds' and isinstance(parsed, dict):
                        # Merge inner class bounds entries into top-level
                        for inner_class_label, conditions in parsed.items():
                            if isinstance(conditions, list) and len(conditions) and isinstance(conditions[0], str):
                                # Old style: list of constraint strings; parse and convert
                                nested = []
                                for cond_str in conditions:
                                    parsed_conditions = ConstraintParser.parse_condition(cond_str)
                                    if parsed_conditions:
                                        nested.extend(parsed_conditions)
                                converted = _convert_operator_list_to_minmax(nested)
                                self.constraints_dict[inner_class_label.strip()] = converted
                            elif isinstance(conditions, list) and len(conditions) and isinstance(conditions[0], dict) and ('operator' in conditions[0]):
                                # List of operator dicts; convert to min/max
                                converted = _convert_operator_list_to_minmax(conditions)
                                self.constraints_dict[inner_class_label.strip()] = converted
                            else:
                                # Already in expected min/max dict format
                                self.constraints_dict[inner_class_label.strip()] = conditions
                    else:
                        # Regular class label line: could contain list of min/max dicts or other
                        if isinstance(parsed, list) and len(parsed) and isinstance(parsed[0], str):
                            # If it's a list of strings, convert using parse_condition
                            nested = []
                            for cond_str in parsed:
                                parsed_conditions = ConstraintParser.parse_condition(cond_str)
                                if parsed_conditions:
                                    nested.extend(parsed_conditions)
                            converted = _convert_operator_list_to_minmax(nested)
                            self.constraints_dict[class_label] = converted
                        elif isinstance(parsed, list) and len(parsed) and isinstance(parsed[0], dict) and ('operator' in parsed[0]):
                            converted = _convert_operator_list_to_minmax(parsed)
                            self.constraints_dict[class_label] = converted
                        else:
                            self.constraints_dict[class_label] = parsed
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for {class_label}: {e}")
        return self.constraints_dict

    @staticmethod
    def extract_constraints_from_dataset(model, train_features, train_labels, feature_names, dpg_config=None):
        """
        Extract constraints and communities from the dataset using Decision Predicate Graph (DPG).
        Uses DPG's graph-based boundary extraction to determine decision boundaries
        and community detection from the ensemble model's decision paths.
        
        Args:
            model: Trained sklearn model
            train_features: Training features array
            train_labels: Training labels array
            feature_names: List of feature names
            dpg_config: Optional DPG config dict (from main config's counterfactual.dpg.config)
            
        Returns:
            Dictionary with keys:
                - 'constraints': Dict mapping class labels to feature constraints (min/max format)
                - 'communities': List of sets, where each set contains node labels belonging to a community
        """
        # Convert OmegaConf DictConfig to regular dict if needed
        if dpg_config is not None and HAS_OMEGACONF and isinstance(dpg_config, DictConfig):
            dpg_config = OmegaConf.to_container(dpg_config, resolve=True)
        
        # Build DPG from the trained model
        config_path = os.path.join(os.path.dirname(__file__), 'DPG', 'config.yaml')
        dpg = DecisionPredicateGraph(
            model=model,
            feature_names=feature_names,
            target_names=np.unique(train_labels).astype(str).tolist(),
            config_file=config_path,
            dpg_config=dpg_config
        )
        
        # Fit DPG to extract decision paths
        dot = dpg.fit(train_features)
        
        # Convert to NetworkX graph and extract metrics
        dpg_model, nodes_list = dpg.to_networkx(dot)
        
        if len(nodes_list) < 2:
            # Fallback: if DPG extraction fails, return empty constraints with expected structure
            return {
                'constraints': {},
                'communities': []
            }
        
        # Extract class boundaries using absorbing Markov chain method
        target_names = [str(c) for c in np.unique(train_labels)]
        df_dpg = GraphMetrics.extract_class_boundaries(
            dpg_model,
            nodes_list,
            target_names=target_names,
        )
        
        # Extract communities using absorbing Markov chain clustering
        node_metrics = NodeMetrics.extract_node_metrics(dpg_model, nodes_list)
        communities_result = GraphMetrics.extract_communities(
            dpg_model, node_metrics, nodes_list
        )
        communities = communities_result.get("Clusters", {})
        
        # Parse Class Bounds into expected format
        parsed_constraints = {}
        if "Class Bounds" in df_dpg:
            class_bounds = df_dpg["Class Bounds"]
            for class_name, bounds_list in class_bounds.items():
                parsed_list = []
                for bound_str in bounds_list:
                    # Parse bounds like "-0.25 < Family <= 4.09" or "Age <= 3.49" or "Age > -2.02"
                    bound_str = bound_str.strip()
                    if "<=" in bound_str and "<" in bound_str and bound_str.count("<") == 2:
                        # Format: min < feature <= max
                        parts = bound_str.split(" < ")
                        if len(parts) == 2:
                            min_val = float(parts[0].strip())
                            right_part = parts[1].strip()  # "feature <= max"
                            if " <= " in right_part:
                                feat, max_str = right_part.split(" <= ")
                                max_val = float(max_str.strip())
                                parsed_list.append({"feature": feat.strip(), "min": min_val, "max": max_val})
                    elif "<=" in bound_str:
                        # Format: feature <= max
                        parts = bound_str.split(" <= ")
                        if len(parts) == 2:
                            feat = parts[0].strip()
                            max_val = float(parts[1].strip())
                            parsed_list.append({"feature": feat, "min": None, "max": max_val})
                    elif ">" in bound_str:
                        # Format: feature > min
                        parts = bound_str.split(" > ")
                        if len(parts) == 2:
                            feat = parts[0].strip()
                            min_val = float(parts[1].strip())
                            parsed_list.append({"feature": feat, "min": min_val, "max": None})
                
                # Format class name to match expected format
                if class_name.startswith("Class "):
                    parsed_constraints[class_name] = parsed_list
                else:
                    parsed_constraints[f"Class {class_name}"] = parsed_list
        
        return {
            'constraints': parsed_constraints,
            'communities': communities
        }
