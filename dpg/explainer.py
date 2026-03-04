from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .core import DecisionPredicateGraph
from .visualizer import (
    class_feature_predicate_counts,
    class_lookup_from_target_names,
    plot_dpg,
    plot_dpg_class_bounds_vs_dataset_feature_ranges,
    plot_dpg_communities,
    plot_dpg_local_paths_aggregate,
    plot_lrc_vs_rf_importance,
    plot_top_lrc_predicate_splits,
)
from metrics.graph import GraphMetrics
from metrics.nodes import NodeMetrics
from metrics.edges import EdgeMetrics


@dataclass
class DPGExplanation:
    """Container for global DPG outputs."""

    graph: Any
    nodes: List[List[str]]
    dot: Any
    node_metrics: pd.DataFrame
    edge_metrics: pd.DataFrame
    class_boundaries: Dict[str, Any]
    graph_construction_mode: str = "aggregated_transitions"
    communities: Optional[Dict[str, Any]] = None
    community_threshold: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.graph,
            "nodes": self.nodes,
            "dot": self.dot,
            "node_metrics": self.node_metrics,
            "edge_metrics": self.edge_metrics,
            "class_boundaries": self.class_boundaries,
            "graph_construction_mode": self.graph_construction_mode,
            "communities": self.communities,
            "community_threshold": self.community_threshold,
        }


@dataclass
class DPGTreePathExplanation:
    """Per-path local details for one sample in the active DPG subgraph."""

    tree_index: int
    tree_prefix: str
    labels: List[str]
    node_ids: List[Optional[str]]
    predicate_truths: List[bool]
    edge_exists: List[bool]
    starts_from_root: Optional[bool]
    ends_in_leaf: bool
    graph_path_valid: Optional[bool]
    mean_lrc: Optional[float] = None
    mean_bc: Optional[float] = None
    path_confidence: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tree_index": self.tree_index,
            "tree_prefix": self.tree_prefix,
            "labels": self.labels,
            "node_ids": self.node_ids,
            "predicate_truths": self.predicate_truths,
            "edge_exists": self.edge_exists,
            "starts_from_root": self.starts_from_root,
            "ends_in_leaf": self.ends_in_leaf,
            "graph_path_valid": self.graph_path_valid,
            "mean_lrc": self.mean_lrc,
            "mean_bc": self.mean_bc,
            "path_confidence": self.path_confidence,
        }


@dataclass
class DPGLocalExplanation:
    """Container for graph-based local DPG path outputs."""

    sample_id: Union[int, str]
    sample: np.ndarray
    tree_paths: List[DPGTreePathExplanation]
    graph_validated: bool
    all_trees_valid: Optional[bool]
    majority_vote: Optional[str]
    class_votes: Dict[str, int]
    sample_confidence: Dict[str, Any]
    path_mode: str = "dpg_graph"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sample": self.sample,
            "tree_paths": [path.as_dict() for path in self.tree_paths],
            "graph_validated": self.graph_validated,
            "all_trees_valid": self.all_trees_valid,
            "majority_vote": self.majority_vote,
            "class_votes": self.class_votes,
            "sample_confidence": self.sample_confidence,
            "path_mode": self.path_mode,
        }


class DPGExplainer:
    """
    High-level, user-friendly API for building and plotting DPG explanations.

    This class wraps DecisionPredicateGraph and the metrics/visualization utilities
    into a cohesive workflow.
    """
    _PREDICATE_RE = re.compile(
        r"^(?P<feature>.+?)\s*(?P<op><=|>)\s*(?P<threshold>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)$"
    )
    _DEFAULT_LOCAL_EVIDENCE_CONFIG: Dict[str, Any] = {
        "variant": "base",
        "base_lambda": 0.8,
        "normalize_scores": True,
        "edge_threshold": 0.0,
        "path_prune_ratio": 0.0,
    }
    _DEFAULT_GRAPH_CONSTRUCTION_CONFIG: Dict[str, Any] = {
        "mode": "aggregated_transitions",
    }
    _VALID_GRAPH_CONSTRUCTION_MODES = {
        "aggregated_transitions",
        "execution_trace",
    }
    _LOCAL_EVIDENCE_VARIANTS: Dict[str, Dict[str, str]] = {
        "base": {
            "score_rule": "ratio_total",
            "lambda_rule": "constant",
        },
        "class_norm": {
            "score_rule": "ratio_mean_repulsion",
            "lambda_rule": "constant",
        },
        "log_class": {
            "score_rule": "ratio_total",
            "lambda_rule": "log_classes",
        },
        "sqrt_class": {
            "score_rule": "ratio_total",
            "lambda_rule": "sqrt_classes",
        },
        "class_feature": {
            "score_rule": "ratio_mean_repulsion",
            "lambda_rule": "class_feature",
        },
        "top_competitor": {
            "score_rule": "ratio_top_competitor",
            "lambda_rule": "log_classes",
        },
    }

    def __init__(
        self,
        model: Any,
        feature_names: Iterable[str],
        target_names: Optional[Iterable[str]] = None,
        config_file: str = "config.yaml",
        dpg_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw_graph_construction = (
            dpg_config.get("dpg", {}).get("graph_construction", {})
            if isinstance(dpg_config, dict)
            else {}
        )
        self._graph_construction_config = {
            **self._DEFAULT_GRAPH_CONSTRUCTION_CONFIG,
            **raw_graph_construction,
        }
        self._graph_construction_mode = str(
            self._graph_construction_config.get("mode", "aggregated_transitions")
        ).strip().lower()
        if self._graph_construction_mode not in self._VALID_GRAPH_CONSTRUCTION_MODES:
            valid = ", ".join(sorted(self._VALID_GRAPH_CONSTRUCTION_MODES))
            raise ValueError(
                f"Unknown graph_construction.mode '{self._graph_construction_mode}'. Valid: {valid}"
            )
        raw_local_evidence = (
            dpg_config.get("dpg", {}).get("local_evidence", {})
            if isinstance(dpg_config, dict)
            else {}
        )
        self._local_evidence_config = {
            **self._DEFAULT_LOCAL_EVIDENCE_CONFIG,
            **raw_local_evidence,
        }
        self._local_evidence_variant = str(self._local_evidence_config["variant"]).strip().lower()
        if self._local_evidence_variant not in self._LOCAL_EVIDENCE_VARIANTS:
            valid = ", ".join(sorted(self._LOCAL_EVIDENCE_VARIANTS))
            raise ValueError(
                f"Unknown local evidence variant '{self._local_evidence_variant}'. Valid: {valid}"
            )
        edge_threshold = float(self._local_evidence_config.get("edge_threshold", 0.0))
        if edge_threshold < 0:
            raise ValueError("local_evidence.edge_threshold must be >= 0.")
        self._local_evidence_config["edge_threshold"] = edge_threshold
        path_prune_ratio = float(self._local_evidence_config.get("path_prune_ratio", 0.0))
        if path_prune_ratio < 0 or path_prune_ratio > 1:
            raise ValueError("local_evidence.path_prune_ratio must be in [0, 1].")
        self._local_evidence_config["path_prune_ratio"] = path_prune_ratio
        self._builder = DecisionPredicateGraph(
            model=model,
            feature_names=list(feature_names),
            target_names=list(target_names) if target_names is not None else None,
            config_file=config_file,
            dpg_config=dpg_config,
        )
        self._is_fitted = False
        self._dot = None
        self._graph = None
        self._nodes = None
        self._node_metrics_cache: Optional[pd.DataFrame] = None

    @property
    def builder(self) -> DecisionPredicateGraph:
        return self._builder

    def fit(self, X: np.ndarray) -> "DPGExplainer":
        """Fit the DPG structure from training data."""
        self._dot = self._builder.fit(X)
        self._graph, self._nodes = self._builder.to_networkx(self._dot)
        self._node_metrics_cache = None
        self._is_fitted = True
        return self

    def explain_global(
        self,
        X: Optional[np.ndarray] = None,
        communities: bool = False,
        community_threshold: float = 0.2,
    ) -> DPGExplanation:
        """
        Build global DPG metrics and return a structured explanation object.

        Args:
            X: Optional training data. If provided, fit() is called before extracting metrics.
            communities: Whether to compute cluster-based communities.
            community_threshold: Threshold used by community extraction.
        """
        if X is not None:
            self.fit(X)
        if not self._is_fitted:
            raise ValueError("DPGExplainer is not fitted. Call fit(X) or explain_global(X=...).")

        node_metrics = NodeMetrics.extract_node_metrics(self._graph, self._nodes)
        edge_metrics = EdgeMetrics.extract_edge_metrics(self._graph, self._nodes)
        class_boundaries = GraphMetrics.extract_class_boundaries(
            self._graph,
            self._nodes,
            target_names=self._builder.target_names or [],
        )

        communities_out = None
        if communities:
            communities_out = GraphMetrics.extract_communities(
                self._graph,
                node_metrics,
                self._nodes,
                threshold_clusters=community_threshold,
            )

        return DPGExplanation(
            graph=self._graph,
            nodes=self._nodes,
            dot=self._dot,
            node_metrics=node_metrics,
            edge_metrics=edge_metrics,
            class_boundaries=class_boundaries,
            graph_construction_mode=self._graph_construction_mode,
            communities=communities_out,
            community_threshold=community_threshold if communities else None,
        )

    def _node_metrics_df(self) -> pd.DataFrame:
        if self._node_metrics_cache is None:
            if not self._is_fitted:
                raise ValueError("DPGExplainer is not fitted. Call fit(X) first.")
            self._node_metrics_cache = NodeMetrics.extract_node_metrics(self._graph, self._nodes)
        return self._node_metrics_cache

    def _predicate_is_true(
        self,
        label: str,
        sample: np.ndarray,
        feature_index: Dict[str, int],
    ) -> bool:
        if label.startswith("Class ") or label.startswith("Pred "):
            return True
        match = self._PREDICATE_RE.match(label.strip())
        if match is None:
            return False
        feat = match.group("feature").strip()
        op = match.group("op")
        threshold = float(match.group("threshold"))
        idx = feature_index.get(feat)
        if idx is None:
            return False
        value = float(sample[idx])
        if op == "<=":
            return value <= threshold
        return value > threshold

    @staticmethod
    def _is_leaf_label(label: str) -> bool:
        return label.startswith("Class ") or label.startswith("Pred ")

    @staticmethod
    def _path_leaf_class(path: DPGTreePathExplanation) -> Optional[str]:
        if not path.labels:
            return None
        label = str(path.labels[-1])
        if label.startswith("Class "):
            return label.replace("Class ", "", 1)
        return None

    @staticmethod
    def _path_weight(path: DPGTreePathExplanation) -> float:
        try:
            return max(float(path.path_confidence or 0.0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _trace_reference_sets(self, sample_arr: np.ndarray) -> Dict[str, set]:
        trace_paths = self._extract_execution_trace_labels(sample_arr)
        trace_node_labels: set[str] = set()
        trace_edge_labels: set[tuple[str, str]] = set()
        for labels in trace_paths:
            for label in labels[:-1]:
                if not self._is_leaf_label(label):
                    trace_node_labels.add(str(label))
            for src, dst in zip(labels, labels[1:]):
                trace_edge_labels.add((str(src), str(dst)))
        return {
            "trace_node_labels": trace_node_labels,
            "trace_edge_labels": trace_edge_labels,
        }

    def _compute_advanced_local_metrics(
        self,
        sample_arr: np.ndarray,
        tree_paths: List[DPGTreePathExplanation],
        class_support: Dict[str, float],
        class_votes: Dict[str, int],
    ) -> Dict[str, Any]:
        eps = 1e-12
        trace_ref = self._trace_reference_sets(sample_arr)
        trace_node_labels = trace_ref["trace_node_labels"]
        trace_edge_labels = trace_ref["trace_edge_labels"]

        explanation_node_labels: set[str] = set()
        explanation_edge_labels: set[tuple[str, str]] = set()
        weighted_paths: List[Tuple[DPGTreePathExplanation, str, float, float]] = []

        total_support = float(sum(max(float(v), 0.0) for v in class_support.values()))
        normalized_support = {
            str(cls): float(max(float(val), 0.0) / (total_support + eps))
            for cls, val in class_support.items()
        }

        for path in tree_paths:
            leaf_class = self._path_leaf_class(path)
            weight = self._path_weight(path)
            normalized_weight = float(weight / (total_support + eps)) if total_support > 0 else 0.0
            if leaf_class is not None:
                weighted_paths.append((path, leaf_class, weight, normalized_weight))

            for idx, label in enumerate(path.labels):
                label = str(label)
                node_id = path.node_ids[idx] if idx < len(path.node_ids) else None
                if self._is_leaf_label(label):
                    continue
                if node_id is not None:
                    explanation_node_labels.add(label)

            for idx, (src_label, dst_label) in enumerate(zip(path.labels, path.labels[1:])):
                src_id = path.node_ids[idx] if idx < len(path.node_ids) else None
                dst_id = path.node_ids[idx + 1] if idx + 1 < len(path.node_ids) else None
                edge_exists = bool(path.edge_exists[idx]) if idx < len(path.edge_exists) else False
                if src_id is not None and dst_id is not None and edge_exists:
                    explanation_edge_labels.add((str(src_label), str(dst_label)))

        node_intersection = explanation_node_labels & trace_node_labels
        edge_intersection = explanation_edge_labels & trace_edge_labels
        node_recall = (
            float(len(node_intersection) / (len(trace_node_labels) + eps))
            if trace_node_labels
            else 1.0
        )
        node_precision = (
            float(len(node_intersection) / (len(explanation_node_labels) + eps))
            if explanation_node_labels
            else 1.0
        )
        edge_recall = (
            float(len(edge_intersection) / (len(trace_edge_labels) + eps))
            if trace_edge_labels
            else 1.0
        )
        edge_precision = (
            float(len(edge_intersection) / (len(explanation_edge_labels) + eps))
            if explanation_edge_labels
            else 1.0
        )
        recombination_rate = (
            float(len(explanation_edge_labels - trace_edge_labels) / (len(explanation_edge_labels) + eps))
            if explanation_edge_labels
            else 0.0
        )

        sorted_support = sorted(normalized_support.items(), key=lambda kv: kv[1], reverse=True)
        support_pred_class = sorted_support[0][0] if sorted_support else None
        support_pred_score = float(sorted_support[0][1]) if sorted_support else None
        support_top_competitor_class = sorted_support[1][0] if len(sorted_support) > 1 else None
        support_top_competitor_score = float(sorted_support[1][1]) if len(sorted_support) > 1 else 0.0
        support_margin = (
            float(support_pred_score - support_top_competitor_score)
            if support_pred_score is not None
            else None
        )

        pred_paths = [rec for rec in weighted_paths if rec[1] == support_pred_class] if support_pred_class else []
        top_k = 3
        predicted_class_concentration = None
        if support_pred_class is not None and support_pred_score is not None:
            top_pred_weights = sorted((rec[3] for rec in pred_paths), reverse=True)[:top_k]
            predicted_class_concentration = float(
                sum(top_pred_weights) / (support_pred_score + eps)
            )

        total_votes = int(sum(int(v) for v in class_votes.values()))
        model_vote_agreement = (
            float(int(class_votes.get(support_pred_class, 0)) / total_votes)
            if support_pred_class is not None and total_votes > 0
            else None
        )
        path_purity = support_pred_score
        competitor_exposure = (
            float(1.0 - support_pred_score) if support_pred_score is not None else None
        )
        trace_coverage_score = float((node_recall + edge_recall) / 2.0)
        explanation_confidence = None
        if (
            support_margin is not None
            and predicted_class_concentration is not None
            and model_vote_agreement is not None
        ):
            explanation_confidence = float(
                trace_coverage_score
                * ((support_margin + predicted_class_concentration + model_vote_agreement) / 3.0)
            )

        critical_node_label = None
        critical_split_depth = None
        critical_successor_pred = None
        critical_successor_comp = None
        critical_node_contrast = None

        if support_pred_class is not None and support_top_competitor_class is not None:
            pred_candidates = [rec for rec in weighted_paths if rec[1] == support_pred_class]
            comp_candidates = [rec for rec in weighted_paths if rec[1] == support_top_competitor_class]
            if pred_candidates and comp_candidates:
                pred_path = max(pred_candidates, key=lambda rec: rec[2])[0]
                comp_path = max(comp_candidates, key=lambda rec: rec[2])[0]
                pred_seq = [str(label) for label in pred_path.labels[:-1]]
                comp_seq = [str(label) for label in comp_path.labels[:-1]]
                prefix_len = 0
                for pred_label, comp_label in zip(pred_seq, comp_seq):
                    if pred_label != comp_label:
                        break
                    prefix_len += 1
                # Do not treat the root predicate as a valid critical node.
                if prefix_len >= 2:
                    critical_split_depth = int(prefix_len)
                    critical_node_label = pred_seq[prefix_len - 1]
                    if prefix_len < len(pred_seq):
                        critical_successor_pred = pred_seq[prefix_len]
                    if prefix_len < len(comp_seq):
                        critical_successor_comp = comp_seq[prefix_len]

                if (
                    critical_split_depth is not None
                    and critical_successor_pred is not None
                    and critical_successor_comp is not None
                ):
                    pred_prefix = tuple(pred_seq[:prefix_len])

                    def _branch_support(successor: str) -> float:
                        total = 0.0
                        for path, _leaf_class, _weight, norm_weight in weighted_paths:
                            seq = tuple(str(label) for label in path.labels[:-1])
                            if len(seq) <= prefix_len:
                                continue
                            if seq[:prefix_len] != pred_prefix:
                                continue
                            if seq[prefix_len] == successor:
                                total += norm_weight
                        return float(total)

                    critical_node_contrast = float(
                        abs(_branch_support(critical_successor_pred) - _branch_support(critical_successor_comp))
                    )

        return {
            "support_pred_class": support_pred_class,
            "support_top_competitor_class": support_top_competitor_class,
            "support_pred_score": support_pred_score,
            "support_top_competitor_score": support_top_competitor_score
            if support_top_competitor_class is not None
            else None,
            "support_margin": support_margin,
            "predicted_class_concentration_top3": predicted_class_concentration,
            "model_vote_agreement": model_vote_agreement,
            "trace_coverage_score": trace_coverage_score,
            "explanation_confidence": explanation_confidence,
            "node_recall": node_recall,
            "node_precision": node_precision,
            "edge_recall": edge_recall,
            "edge_precision": edge_precision,
            "path_purity": path_purity,
            "competitor_exposure": competitor_exposure,
            "recombination_rate": recombination_rate,
            "critical_node_label": critical_node_label,
            "critical_split_depth": critical_split_depth,
            "critical_successor_pred": critical_successor_pred,
            "critical_successor_comp": critical_successor_comp,
            "critical_node_contrast": critical_node_contrast,
            "trace_node_count_unique": int(len(trace_node_labels)),
            "trace_edge_count_unique": int(len(trace_edge_labels)),
            "explanation_node_count_unique": int(len(explanation_node_labels)),
            "explanation_edge_count_unique": int(len(explanation_edge_labels)),
        }

    def explain_local(
        self,
        sample: np.ndarray,
        sample_id: Union[int, str] = 0,
        X: Optional[np.ndarray] = None,
        validate_graph: bool = True,
    ) -> DPGLocalExplanation:
        """
        Build a local explanation by traversing active paths in the fitted DPG graph.

        Args:
            sample: Single feature vector.
            sample_id: Label used in generated tree prefixes.
            X: Optional training data to fit before explaining.
            validate_graph: Kept for compatibility. Local traversal is graph-based.
        """
        if X is not None:
            self.fit(X)
        if not self._is_fitted:
            raise ValueError(
                "DPGExplainer is not fitted. Call fit(X), explain_global(X=...), "
                "or pass X=... to explain_local."
            )

        sample_arr = np.asarray(sample, dtype=float).reshape(-1)
        if sample_arr.shape[0] != len(self._builder.feature_names):
            raise ValueError(
                f"sample has {sample_arr.shape[0]} features, expected {len(self._builder.feature_names)}."
            )

        feature_index = {name: i for i, name in enumerate(self._builder.feature_names)}
        node_to_label: Dict[str, str] = {str(node_id): str(label) for node_id, label in (self._nodes or [])}
        if not node_to_label:
            raise ValueError("No DPG nodes available. Fit the explainer before explain_local.")

        if self._graph_construction_mode == "execution_trace":
            return self._explain_local_execution_trace(
                sample_arr=sample_arr,
                sample_id=sample_id,
                feature_index=feature_index,
                node_to_label=node_to_label,
            )
        return self._explain_local_aggregated_transitions(
            sample_arr=sample_arr,
            sample_id=sample_id,
            feature_index=feature_index,
            node_to_label=node_to_label,
        )

    def _explain_local_aggregated_transitions(
        self,
        sample_arr: np.ndarray,
        sample_id: Union[int, str],
        feature_index: Dict[str, int],
        node_to_label: Dict[str, str],
    ) -> DPGLocalExplanation:

        class_nodes = {
            node_id
            for node_id, label in node_to_label.items()
            if label.startswith("Class ") or label.startswith("Pred ")
        }
        true_pred_nodes = {
            node_id
            for node_id, label in node_to_label.items()
            if node_id not in class_nodes and self._predicate_is_true(label, sample_arr, feature_index)
        }
        active_nodes = sorted(true_pred_nodes | class_nodes)
        active_graph = self._graph.subgraph(active_nodes).copy()
        if active_graph.number_of_nodes() == 0:
            raise ValueError("No active DPG nodes for this sample.")

        raw_edge_weights = []
        for src, dst in active_graph.edges():
            w = active_graph.get_edge_data(src, dst, default={}).get("weight", 0.0)
            try:
                raw_edge_weights.append(float(w))
            except (TypeError, ValueError):
                raw_edge_weights.append(0.0)
        max_edge_weight = max(raw_edge_weights) if raw_edge_weights else 1.0
        if max_edge_weight <= 0:
            max_edge_weight = 1.0

        edge_threshold = float(self._local_evidence_config.get("edge_threshold", 0.0))
        num_active_edges_raw = int(active_graph.number_of_edges())
        if edge_threshold > 0:
            edges_to_remove = []
            for src, dst in active_graph.edges():
                w = active_graph.get_edge_data(src, dst, default={}).get("weight", 0.0)
                try:
                    w = float(w)
                except (TypeError, ValueError):
                    w = 0.0
                norm_w = max(0.0, w) / max_edge_weight
                if norm_w < edge_threshold:
                    edges_to_remove.append((src, dst))
            if edges_to_remove:
                active_graph.remove_edges_from(edges_to_remove)
                orphan_nodes = [
                    node
                    for node in list(active_graph.nodes())
                    if node not in class_nodes and active_graph.degree(node) == 0
                ]
                if orphan_nodes:
                    active_graph.remove_nodes_from(orphan_nodes)
        num_active_edges_filtered = int(active_graph.number_of_edges())

        # Starting nodes must be true predicates that are roots in the original DPG,
        # not just roots in the filtered active subgraph.
        source_nodes = sorted(
            [n for n in true_pred_nodes if self._graph.in_degree(n) == 0],
            key=lambda n: node_to_label[n],
        )
        if not source_nodes:
            # Fallback 1: roots in the active subgraph (after local filtering).
            source_nodes = sorted(
                [n for n in true_pred_nodes if active_graph.in_degree(n) == 0],
                key=lambda n: node_to_label[n],
            )
        if not source_nodes:
            # Fallback 2: dense/merged graphs may remove root status; start from
            # low in-degree true predicates instead of failing hard.
            source_nodes = sorted(
                list(true_pred_nodes),
                key=lambda n: (self._graph.in_degree(n), node_to_label[n]),
            )
        if not source_nodes:
            raise ValueError("No valid local path: no predicate node matched this sample.")

        raw_paths: List[List[str]] = []
        max_paths = 5000
        max_depth = max(1, active_graph.number_of_nodes())

        def _dfs(current: str, path_nodes: List[str]) -> None:
            if len(raw_paths) >= max_paths:
                return
            if current in class_nodes:
                raw_paths.append(path_nodes.copy())
                return
            if len(path_nodes) >= max_depth:
                return
            for nxt in sorted(active_graph.successors(current), key=lambda n: node_to_label[n]):
                if nxt in path_nodes:
                    continue
                _dfs(nxt, path_nodes + [nxt])

        for src in source_nodes:
            _dfs(src, [src])
            if len(raw_paths) >= max_paths:
                break

        node_metrics = self._node_metrics_df().copy()
        metric_by_node = node_metrics.set_index("Node")
        # Local edge weights are normalized inside the filtered active graph so path
        # supports are comparable across paths for this sample after thresholding.
        filtered_edge_weights = []
        for src, dst in active_graph.edges():
            w = active_graph.get_edge_data(src, dst, default={}).get("weight", 0.0)
            try:
                filtered_edge_weights.append(float(w))
            except (TypeError, ValueError):
                filtered_edge_weights.append(0.0)
        max_edge_weight = max(filtered_edge_weights) if filtered_edge_weights else 1.0
        if max_edge_weight <= 0:
            max_edge_weight = 1.0

        # Path-scoring hyperparameters.
        # Rationale:
        # - higher edge frequencies suggest stronger evidence in the DPG (edge term),
        # - higher-LRC nodes indicate more influential predicates (LRC term),
        # - shorter paths are preferred for local faithfulness and specificity (length penalty).
        # These produce an "attraction" support for the reached class.
        alpha_edge = 1.0
        beta_lrc = 1.0
        gamma_length = 0.35

        tree_paths_raw: List[DPGTreePathExplanation] = []
        path_records_raw: List[Dict[str, Any]] = []
        for idx, node_path in enumerate(raw_paths):
            labels = [node_to_label[n] for n in node_path]
            truths = [self._predicate_is_true(lbl, sample_arr, feature_index) for lbl in labels]
            edge_exists = [True for _ in zip(node_path, node_path[1:])]
            starts_from_root = bool(node_path[0] in source_nodes)
            ends_in_leaf = bool(labels and (labels[-1].startswith("Class ") or labels[-1].startswith("Pred ")))
            graph_path_valid = bool(starts_from_root and ends_in_leaf and all(truths) and all(edge_exists))

            lrc_vals = []
            bc_vals = []
            for n in node_path:
                if n in metric_by_node.index:
                    lrc_vals.append(float(metric_by_node.loc[n, "Local reaching centrality"]))
                    bc_vals.append(float(metric_by_node.loc[n, "Betweenness centrality"]))
            mean_lrc = float(np.mean(lrc_vals)) if lrc_vals else None
            mean_bc = float(np.mean(bc_vals)) if bc_vals else None

            # Build edge support as geometric mean of normalized edge weights.
            # Geometric mean avoids over-rewarding very long paths and reflects
            # consistent support along the whole route.
            edge_supports = []
            for src, dst in zip(node_path, node_path[1:]):
                ew = active_graph.get_edge_data(src, dst, default={}).get("weight", 0.0)
                try:
                    ew = float(ew)
                except (TypeError, ValueError):
                    ew = 0.0
                edge_supports.append(max(0.0, ew) / max_edge_weight)
            edge_support = (
                float(np.exp(np.mean(np.log(np.maximum(edge_supports, 1e-12)))))
                if edge_supports
                else 1.0
            )
            lrc_support = float(np.mean(lrc_vals)) if lrc_vals else 0.0
            length_penalty = float(np.exp(-gamma_length * max(0, len(node_path) - 1)))

            # Path confidence is now class-evidence oriented:
            # high with strong edges, influential predicates, and shorter path.
            path_confidence = float(
                (edge_support ** alpha_edge) * (max(lrc_support, 0.0) ** beta_lrc) * length_penalty
            )

            tree_paths_raw.append(
                DPGTreePathExplanation(
                    tree_index=idx,
                    tree_prefix=f"source:{labels[0]}",
                    labels=labels,
                    node_ids=node_path,
                    predicate_truths=truths,
                    edge_exists=edge_exists,
                    starts_from_root=starts_from_root,
                    ends_in_leaf=ends_in_leaf,
                    graph_path_valid=graph_path_valid,
                    mean_lrc=mean_lrc,
                    mean_bc=mean_bc,
                    path_confidence=path_confidence,
                )
            )
            leaf_class = None
            if labels and labels[-1].startswith("Class "):
                leaf_class = labels[-1].replace("Class ", "", 1)
            path_records_raw.append({"leaf_class": leaf_class, "path_confidence": path_confidence})

        path_prune_ratio = float(self._local_evidence_config.get("path_prune_ratio", 0.0))
        max_path_confidence = max(
            (float(rec["path_confidence"]) for rec in path_records_raw),
            default=0.0,
        )
        min_path_confidence = path_prune_ratio * max_path_confidence
        prune_eps = 1e-12
        if path_prune_ratio > 0 and max_path_confidence > 0:
            keep_mask = [
                float(rec["path_confidence"]) + prune_eps >= min_path_confidence
                for rec in path_records_raw
            ]
            tree_paths = [path for path, keep in zip(tree_paths_raw, keep_mask) if keep]
            path_records = [rec for rec, keep in zip(path_records_raw, keep_mask) if keep]
        else:
            tree_paths = tree_paths_raw
            path_records = path_records_raw

        leaf_classes = [
            p.labels[-1].replace("Class ", "", 1)
            for p in tree_paths
            if p.labels and p.labels[-1].startswith("Class ")
        ]
        class_votes: Dict[str, int] = (
            pd.Series(leaf_classes).value_counts().astype(int).to_dict() if leaf_classes else {}
        )

        # Class-level attraction/repulsion score.
        eps = 1e-12
        class_support: Dict[str, float] = {}
        for rec in path_records:
            cls = rec["leaf_class"]
            if cls is None:
                continue
            class_support[cls] = class_support.get(cls, 0.0) + float(rec["path_confidence"])
        evidence_scores, evidence_meta = self._compute_evidence_scores(
            class_support=class_support,
            n_model_classes=max(len(getattr(self._builder.model, "classes_", [])), len(class_support)),
            n_features=len(self._builder.feature_names),
            eps=eps,
        )
        majority_vote = (
            max(evidence_scores, key=evidence_scores.get)
            if evidence_scores
            else next(iter(class_votes.keys()), None)
        )
        top_competitor_class_pred = None
        evidence_score_competitor_pred = None
        evidence_margin_pred_vs_competitor = None
        if majority_vote is not None and evidence_scores:
            competitor_items = [(c, s) for c, s in evidence_scores.items() if c != majority_vote]
            if competitor_items:
                top_competitor_class_pred, top_competitor_score = max(
                    competitor_items, key=lambda kv: kv[1]
                )
                evidence_score_competitor_pred = float(top_competitor_score)
                evidence_margin_pred_vs_competitor = float(
                    float(evidence_scores.get(majority_vote, 0.0)) - top_competitor_score
                )
            else:
                evidence_margin_pred_vs_competitor = float(evidence_scores.get(majority_vote, 0.0))
        all_trees_valid = all(p.graph_path_valid for p in tree_paths) if tree_paths else False

        active_metric_nodes = metric_by_node.loc[
            metric_by_node.index.intersection(list({n for p in tree_paths for n in p.node_ids}))
        ]
        sample_confidence = {
            "graph_construction_mode": self._graph_construction_mode,
            "num_active_nodes": float(len(active_metric_nodes)),
            "num_active_edges_raw": float(num_active_edges_raw),
            "num_active_edges_filtered": float(num_active_edges_filtered),
            "num_paths_raw": float(len(tree_paths_raw)),
            "num_paths_pruned": float(len(tree_paths)),
            "num_paths": float(len(tree_paths)),
            "mean_lrc_active_nodes": (
                float(active_metric_nodes["Local reaching centrality"].mean())
                if len(active_metric_nodes)
                else None
            ),
            "mean_bc_active_nodes": (
                float(active_metric_nodes["Betweenness centrality"].mean())
                if len(active_metric_nodes)
                else None
            ),
            "evidence_score_pred": (
                float(evidence_scores.get(majority_vote)) if majority_vote in evidence_scores else None
            ),
            "top_competitor_class_pred": top_competitor_class_pred,
            "evidence_score_competitor_pred": evidence_score_competitor_pred,
            "evidence_margin_pred_vs_competitor": evidence_margin_pred_vs_competitor,
            "class_support": class_support,
            "evidence_scores": evidence_scores,
            "evidence_variant": evidence_meta["variant"],
            "evidence_score_rule": evidence_meta["score_rule"],
            "evidence_lambda_rule": evidence_meta["lambda_rule"],
            "evidence_lambda": evidence_meta["lambda_value"],
            "evidence_base_lambda": evidence_meta["base_lambda"],
            "edge_threshold": float(edge_threshold),
            "path_prune_ratio": float(path_prune_ratio),
            "path_confidence_max": float(max_path_confidence) if tree_paths_raw else None,
            "path_confidence_min_kept": float(min_path_confidence) if tree_paths else None,
            "n_model_classes": evidence_meta["n_model_classes"],
            "n_features": evidence_meta["n_features"],
            "evidence_score_margin": (
                float(sorted(evidence_scores.values(), reverse=True)[0] - sorted(evidence_scores.values(), reverse=True)[1])
                if len(evidence_scores) >= 2
                else (float(next(iter(evidence_scores.values()))) if len(evidence_scores) == 1 else None)
            ),
            # Backward-compatible aliases (deprecated).
            "vote_confidence": (
                float(evidence_scores.get(majority_vote)) if majority_vote in evidence_scores else None
            ),
            "class_scores": evidence_scores,
            "score_margin": (
                float(sorted(evidence_scores.values(), reverse=True)[0] - sorted(evidence_scores.values(), reverse=True)[1])
                if len(evidence_scores) >= 2
                else (float(next(iter(evidence_scores.values()))) if len(evidence_scores) == 1 else None)
            ),
        }
        sample_confidence.update(
            self._compute_advanced_local_metrics(
                sample_arr=sample_arr,
                tree_paths=tree_paths,
                class_support=class_support,
                class_votes=class_votes,
            )
        )

        return DPGLocalExplanation(
            sample_id=sample_id,
            sample=sample_arr,
            tree_paths=tree_paths,
            graph_validated=True,
            all_trees_valid=all_trees_valid,
            majority_vote=majority_vote,
            class_votes=class_votes,
            sample_confidence=sample_confidence,
            path_mode="dpg_graph",
        )

    def _explain_local_execution_trace(
        self,
        sample_arr: np.ndarray,
        sample_id: Union[int, str],
        feature_index: Dict[str, int],
        node_to_label: Dict[str, str],
    ) -> DPGLocalExplanation:
        label_to_node = {label: node_id for node_id, label in node_to_label.items()}
        node_metrics = self._node_metrics_df().copy()
        metric_by_node = node_metrics.set_index("Node")

        graph_edge_weights = []
        for src, dst in self._graph.edges():
            w = self._graph.get_edge_data(src, dst, default={}).get("weight", 0.0)
            try:
                graph_edge_weights.append(float(w))
            except (TypeError, ValueError):
                graph_edge_weights.append(0.0)
        max_edge_weight = max(graph_edge_weights) if graph_edge_weights else 1.0
        if max_edge_weight <= 0:
            max_edge_weight = 1.0

        alpha_edge = 1.0
        beta_lrc = 1.0
        gamma_length = 0.35

        tree_paths: List[DPGTreePathExplanation] = []
        path_records: List[Dict[str, Any]] = []
        trace_predicates_missing = 0
        trace_edges_missing = 0
        total_trace_predicates = 0
        total_trace_edges = 0
        mapped_node_ids: set[str] = set()
        mapped_edges: set[tuple[str, str]] = set()

        for idx, labels in enumerate(self._extract_execution_trace_labels(sample_arr)):
            node_ids = [label_to_node.get(label) for label in labels]
            truths = [self._predicate_is_true(label, sample_arr, feature_index) for label in labels]
            edge_exists = []
            for src, dst in zip(node_ids, node_ids[1:]):
                total_trace_edges += 1
                exists = bool(src is not None and dst is not None and self._graph.has_edge(src, dst))
                if exists:
                    mapped_edges.add((str(src), str(dst)))
                else:
                    trace_edges_missing += 1
                edge_exists.append(exists)
            total_trace_predicates += max(len(labels) - 1, 0)
            trace_predicates_missing += sum(
                1 for label, node_id in zip(labels[:-1], node_ids[:-1]) if not label.startswith(("Class ", "Pred ")) and node_id is None
            )
            for node_id in node_ids:
                if node_id is not None:
                    mapped_node_ids.add(str(node_id))

            lrc_vals = []
            bc_vals = []
            for node_id in node_ids:
                if node_id is not None and node_id in metric_by_node.index:
                    lrc_vals.append(float(metric_by_node.loc[node_id, "Local reaching centrality"]))
                    bc_vals.append(float(metric_by_node.loc[node_id, "Betweenness centrality"]))
            mean_lrc = float(np.mean(lrc_vals)) if lrc_vals else None
            mean_bc = float(np.mean(bc_vals)) if bc_vals else None

            edge_supports = []
            for src, dst in zip(node_ids, node_ids[1:]):
                if src is None or dst is None or not self._graph.has_edge(src, dst):
                    edge_supports.append(1e-12)
                    continue
                ew = self._graph.get_edge_data(src, dst, default={}).get("weight", 0.0)
                try:
                    ew = float(ew)
                except (TypeError, ValueError):
                    ew = 0.0
                edge_supports.append(max(0.0, ew) / max_edge_weight)
            edge_support = (
                float(np.exp(np.mean(np.log(np.maximum(edge_supports, 1e-12)))))
                if edge_supports
                else 1.0
            )
            lrc_support = float(np.mean(lrc_vals)) if lrc_vals else 0.0
            length_penalty = float(np.exp(-gamma_length * max(0, len(labels) - 1)))
            path_confidence = float(
                (edge_support ** alpha_edge) * (max(lrc_support, 0.0) ** beta_lrc) * length_penalty
            )

            graph_path_valid = bool(
                len(labels) > 0
                and all(truths)
                and all(node_id is not None for node_id in node_ids)
                and all(edge_exists)
            )
            tree_paths.append(
                DPGTreePathExplanation(
                    tree_index=idx,
                    tree_prefix=f"tree:{idx}",
                    labels=labels,
                    node_ids=node_ids,
                    predicate_truths=truths,
                    edge_exists=edge_exists,
                    starts_from_root=True,
                    ends_in_leaf=bool(labels and labels[-1].startswith(("Class ", "Pred "))),
                    graph_path_valid=graph_path_valid,
                    mean_lrc=mean_lrc,
                    mean_bc=mean_bc,
                    path_confidence=path_confidence,
                )
            )
            leaf_class = None
            if labels and labels[-1].startswith("Class "):
                leaf_class = labels[-1].replace("Class ", "", 1)
            path_records.append({"leaf_class": leaf_class, "path_confidence": path_confidence})

        leaf_classes = [
            p.labels[-1].replace("Class ", "", 1)
            for p in tree_paths
            if p.labels and p.labels[-1].startswith("Class ")
        ]
        class_votes: Dict[str, int] = (
            pd.Series(leaf_classes).value_counts().astype(int).to_dict() if leaf_classes else {}
        )

        eps = 1e-12
        class_support: Dict[str, float] = {}
        for rec in path_records:
            cls = rec["leaf_class"]
            if cls is None:
                continue
            class_support[cls] = class_support.get(cls, 0.0) + float(rec["path_confidence"])
        evidence_scores, evidence_meta = self._compute_evidence_scores(
            class_support=class_support,
            n_model_classes=max(len(getattr(self._builder.model, "classes_", [])), len(class_support)),
            n_features=len(self._builder.feature_names),
            eps=eps,
        )
        majority_vote = (
            max(evidence_scores, key=evidence_scores.get)
            if evidence_scores
            else next(iter(class_votes.keys()), None)
        )
        top_competitor_class_pred = None
        evidence_score_competitor_pred = None
        evidence_margin_pred_vs_competitor = None
        if majority_vote is not None and evidence_scores:
            competitor_items = [(c, s) for c, s in evidence_scores.items() if c != majority_vote]
            if competitor_items:
                top_competitor_class_pred, top_competitor_score = max(
                    competitor_items, key=lambda kv: kv[1]
                )
                evidence_score_competitor_pred = float(top_competitor_score)
                evidence_margin_pred_vs_competitor = float(
                    float(evidence_scores.get(majority_vote, 0.0)) - top_competitor_score
                )
            else:
                evidence_margin_pred_vs_competitor = float(evidence_scores.get(majority_vote, 0.0))

        active_metric_nodes = metric_by_node.loc[metric_by_node.index.intersection(list(mapped_node_ids))]
        all_trees_valid = all(p.graph_path_valid for p in tree_paths) if tree_paths else False
        trace_node_coverage = (
            float((total_trace_predicates - trace_predicates_missing) / total_trace_predicates)
            if total_trace_predicates > 0
            else 1.0
        )
        trace_edge_coverage = (
            float((total_trace_edges - trace_edges_missing) / total_trace_edges)
            if total_trace_edges > 0
            else 1.0
        )
        sample_confidence = {
            "graph_construction_mode": self._graph_construction_mode,
            "num_active_nodes": float(len(active_metric_nodes)),
            "num_active_edges_raw": float(total_trace_edges),
            "num_active_edges_filtered": float(len(mapped_edges)),
            "num_paths_raw": float(len(tree_paths)),
            "num_paths_pruned": float(len(tree_paths)),
            "num_paths": float(len(tree_paths)),
            "num_executed_paths": float(len(tree_paths)),
            "num_executed_predicates": float(total_trace_predicates),
            "num_executed_edges": float(total_trace_edges),
            "num_trace_predicates_missing_from_dpg": float(trace_predicates_missing),
            "num_trace_edges_missing_from_dpg": float(trace_edges_missing),
            "trace_node_coverage": trace_node_coverage,
            "trace_edge_coverage": trace_edge_coverage,
            "mean_lrc_active_nodes": (
                float(active_metric_nodes["Local reaching centrality"].mean())
                if len(active_metric_nodes)
                else None
            ),
            "mean_bc_active_nodes": (
                float(active_metric_nodes["Betweenness centrality"].mean())
                if len(active_metric_nodes)
                else None
            ),
            "evidence_score_pred": (
                float(evidence_scores.get(majority_vote)) if majority_vote in evidence_scores else None
            ),
            "top_competitor_class_pred": top_competitor_class_pred,
            "evidence_score_competitor_pred": evidence_score_competitor_pred,
            "evidence_margin_pred_vs_competitor": evidence_margin_pred_vs_competitor,
            "class_support": class_support,
            "evidence_scores": evidence_scores,
            "evidence_variant": evidence_meta["variant"],
            "evidence_score_rule": evidence_meta["score_rule"],
            "evidence_lambda_rule": evidence_meta["lambda_rule"],
            "evidence_lambda": evidence_meta["lambda_value"],
            "evidence_base_lambda": evidence_meta["base_lambda"],
            "edge_threshold": 0.0,
            "path_prune_ratio": 0.0,
            "path_confidence_max": (
                float(max((p.path_confidence or 0.0) for p in tree_paths))
                if tree_paths
                else None
            ),
            "path_confidence_min_kept": (
                float(min((p.path_confidence or 0.0) for p in tree_paths))
                if tree_paths
                else None
            ),
            "n_model_classes": evidence_meta["n_model_classes"],
            "n_features": evidence_meta["n_features"],
            "evidence_score_margin": (
                float(sorted(evidence_scores.values(), reverse=True)[0] - sorted(evidence_scores.values(), reverse=True)[1])
                if len(evidence_scores) >= 2
                else (float(next(iter(evidence_scores.values()))) if len(evidence_scores) == 1 else None)
            ),
            "vote_confidence": (
                float(evidence_scores.get(majority_vote)) if majority_vote in evidence_scores else None
            ),
            "class_scores": evidence_scores,
            "score_margin": (
                float(sorted(evidence_scores.values(), reverse=True)[0] - sorted(evidence_scores.values(), reverse=True)[1])
                if len(evidence_scores) >= 2
                else (float(next(iter(evidence_scores.values()))) if len(evidence_scores) == 1 else None)
            ),
        }
        sample_confidence.update(
            self._compute_advanced_local_metrics(
                sample_arr=sample_arr,
                tree_paths=tree_paths,
                class_support=class_support,
                class_votes=class_votes,
            )
        )
        return DPGLocalExplanation(
            sample_id=sample_id,
            sample=sample_arr,
            tree_paths=tree_paths,
            graph_validated=True,
            all_trees_valid=all_trees_valid,
            majority_vote=majority_vote,
            class_votes=class_votes,
            sample_confidence=sample_confidence,
            path_mode="execution_trace",
        )

    def _extract_execution_trace_labels(self, sample_arr: np.ndarray) -> List[List[str]]:
        traces: List[List[str]] = []
        sample = sample_arr.reshape(-1)
        for tree in getattr(self._builder.model, "estimators_", []):
            tree_ = tree.tree_
            node_index = 0
            labels: List[str] = []
            while True:
                left = tree_.children_left[node_index]
                right = tree_.children_right[node_index]
                if left == right:
                    if hasattr(self._builder.model, "classes_"):
                        pred_class = tree_.value[node_index].argmax()
                        if self._builder.target_names is not None:
                            pred_class = self._builder.target_names[pred_class]
                        labels.append(f"Class {pred_class}")
                    else:
                        pred = round(tree_.value[node_index][0][0], 2)
                        labels.append(f"Pred {pred}")
                    break
                feature_index = tree_.feature[node_index]
                threshold = round(tree_.threshold[node_index], self._builder.decimal_threshold)
                feature_name = self._builder.feature_names[feature_index]
                sample_val = sample[feature_index]
                if sample_val <= threshold:
                    labels.append(f"{feature_name} <= {threshold}")
                    node_index = left
                else:
                    labels.append(f"{feature_name} > {threshold}")
                    node_index = right
            traces.append(labels)
        return traces

    def _compute_evidence_scores(
        self,
        class_support: Dict[str, float],
        n_model_classes: int,
        n_features: int,
        eps: float,
    ) -> Tuple[Dict[str, float], Dict[str, float | int | str | bool]]:
        if not class_support:
            return {}, {
                "variant": self._local_evidence_variant,
                "score_rule": self._LOCAL_EVIDENCE_VARIANTS[self._local_evidence_variant]["score_rule"],
                "lambda_rule": self._LOCAL_EVIDENCE_VARIANTS[self._local_evidence_variant]["lambda_rule"],
                "lambda_value": float(self._local_evidence_config["base_lambda"]),
                "base_lambda": float(self._local_evidence_config["base_lambda"]),
                "n_model_classes": int(max(n_model_classes, 1)),
                "n_features": int(max(n_features, 1)),
                "normalize_scores": bool(self._local_evidence_config["normalize_scores"]),
            }

        variant_spec = self._LOCAL_EVIDENCE_VARIANTS[self._local_evidence_variant]
        base_lambda = float(self._local_evidence_config["base_lambda"])
        lambda_value = self._resolve_repulsion_lambda(
            rule=variant_spec["lambda_rule"],
            base_lambda=base_lambda,
            n_model_classes=n_model_classes,
            n_features=n_features,
        )
        total_support = float(sum(class_support.values()))

        evidence_scores: Dict[str, float] = {}
        for cls, attraction in class_support.items():
            total_repulsion = max(0.0, total_support - attraction)
            mean_repulsion = (
                total_repulsion / max(n_model_classes - 1, 1)
                if n_model_classes > 1
                else total_repulsion
            )
            top_competitor_repulsion = max(
                (support for other_cls, support in class_support.items() if other_cls != cls),
                default=0.0,
            )

            if variant_spec["score_rule"] == "ratio_total":
                repulsion = total_repulsion
            elif variant_spec["score_rule"] == "ratio_mean_repulsion":
                repulsion = mean_repulsion
            elif variant_spec["score_rule"] == "ratio_top_competitor":
                repulsion = top_competitor_repulsion
            else:
                raise ValueError(f"Unsupported score_rule '{variant_spec['score_rule']}'")

            evidence_scores[cls] = float(attraction / (attraction + lambda_value * repulsion + eps))

        if bool(self._local_evidence_config["normalize_scores"]):
            score_sum = float(sum(evidence_scores.values()))
            if score_sum > 0:
                evidence_scores = {k: float(v / score_sum) for k, v in evidence_scores.items()}

        return evidence_scores, {
            "variant": self._local_evidence_variant,
            "score_rule": variant_spec["score_rule"],
            "lambda_rule": variant_spec["lambda_rule"],
            "lambda_value": float(lambda_value),
            "base_lambda": float(base_lambda),
            "n_model_classes": int(max(n_model_classes, 1)),
            "n_features": int(max(n_features, 1)),
            "normalize_scores": bool(self._local_evidence_config["normalize_scores"]),
        }

    @staticmethod
    def _resolve_repulsion_lambda(
        rule: str,
        base_lambda: float,
        n_model_classes: int,
        n_features: int,
    ) -> float:
        n_model_classes = max(int(n_model_classes), 1)
        n_features = max(int(n_features), 1)

        if rule == "constant":
            return float(base_lambda)
        if rule == "log_classes":
            return float(base_lambda * max(1.0, math.log2(max(n_model_classes, 2))))
        if rule == "sqrt_classes":
            return float(base_lambda * math.sqrt(max(n_model_classes - 1, 1)))
        if rule == "class_feature":
            class_term = max(1.0, math.log2(max(n_model_classes, 2)))
            feature_term = max(1.0, math.sqrt(math.log1p(n_features)))
            return float(base_lambda * class_term / feature_term)
        raise ValueError(f"Unsupported lambda rule '{rule}'")

    def local_path_dataframe(self, local_explanation: DPGLocalExplanation) -> pd.DataFrame:
        """Flatten local explanation paths into a compact long dataframe."""
        rows: List[Dict[str, Any]] = []
        for path in local_explanation.tree_paths:
            for step_idx, label in enumerate(path.labels):
                is_leaf = label.startswith("Class ") or label.startswith("Pred ")
                rows.append(
                    {
                        "sample_id": local_explanation.sample_id,
                        "tree_index": path.tree_index,
                        "step_index": step_idx,
                        "label": label,
                        "is_leaf": is_leaf,
                        "predicate_true": bool(path.predicate_truths[step_idx]),
                        "edge_exists_from_prev": True if step_idx == 0 else bool(path.edge_exists[step_idx - 1]),
                        "starts_from_root": path.starts_from_root,
                        "ends_in_leaf": path.ends_in_leaf,
                        "graph_path_valid": path.graph_path_valid,
                        "mean_lrc": path.mean_lrc,
                        "mean_bc": path.mean_bc,
                        "path_confidence": path.path_confidence,
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "sample_id",
                    "tree_index",
                    "step_index",
                    "label",
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
            )
        return pd.DataFrame(rows).sort_values(["tree_index", "step_index"]).reset_index(drop=True)

    def plot_local_on_dpg(
        self,
        plot_name: str,
        local_explanation: Optional[DPGLocalExplanation] = None,
        sample: Optional[np.ndarray] = None,
        sample_id: Union[int, str] = 0,
        true_class_label: Optional[Union[str, int]] = None,
        X: Optional[np.ndarray] = None,
        save_dir: str = "results/",
        path_indices: Optional[Iterable[int]] = None,
        layout_template: str = "wide",
        graph_width: Optional[float] = None,
        graph_height: Optional[float] = None,
        graph_size_lock: bool = False,
        graph_style: Optional[Dict[str, Any]] = None,
        node_style: Optional[Dict[str, Any]] = None,
        edge_style: Optional[Dict[str, Any]] = None,
        fig_size: Tuple[float, float] = (16, 8),
        dpi: int = 300,
        pdf_dpi: int = 600,
        show: bool = True,
        export_pdf: bool = False,
    ) -> Any:
        """
        Render one DPG plot aggregating selected local paths for one sample.

        Highlighted edges represent path frequency across selected local paths.
        """
        if local_explanation is None:
            if sample is None:
                raise ValueError("Provide either local_explanation or sample.")
            local_explanation = self.explain_local(
                sample=sample,
                sample_id=sample_id,
                X=X,
                validate_graph=True,
            )
        if not self._is_fitted:
            raise ValueError("DPGExplainer is not fitted. Call fit(X) first.")

        total_paths = len(local_explanation.tree_paths)
        if total_paths == 0:
            raise ValueError("local_explanation has no paths to plot.")

        if path_indices is None:
            selected = list(range(total_paths))
        else:
            selected = [int(i) for i in path_indices]
        bad = [i for i in selected if i < 0 or i >= total_paths]
        if bad:
            raise ValueError(f"path_indices out of range: {bad}; available range is [0, {total_paths - 1}]")

        node_metrics = self._node_metrics_df()
        edge_metrics = EdgeMetrics.extract_edge_metrics(self._graph, self._nodes)
        sample_token = str(local_explanation.sample_id).replace(" ", "_")
        out_plot_name = f"{plot_name}_sid_{sample_token}"
        selected_paths = [local_explanation.tree_paths[i] for i in selected]
        fig = plot_dpg_local_paths_aggregate(
            plot_name=out_plot_name,
            dot=self._dot,
            df=node_metrics,
            df_edges=edge_metrics,
            paths_node_ids=[[n for n in p.node_ids if n is not None] for p in selected_paths],
            path_confidences=[p.path_confidence for p in selected_paths if p.path_confidence is not None],
            sample_id=local_explanation.sample_id,
            true_class_label=true_class_label,
            obtained_class_label=local_explanation.majority_vote,
            sample_metrics=local_explanation.sample_confidence,
            save_dir=save_dir,
            class_flag=True,
            layout_template=layout_template,
            graph_width=graph_width,
            graph_height=graph_height,
            graph_size_lock=graph_size_lock,
            graph_style=graph_style,
            node_style=node_style,
            edge_style=edge_style,
            fig_size=fig_size,
            dpi=dpi,
            pdf_dpi=pdf_dpi,
            show=show,
            export_pdf=export_pdf,
        )
        return fig

    def plot(
        self,
        plot_name: str,
        explanation: Optional[DPGExplanation] = None,
        save_dir: str = "results/",
        attribute: Optional[str] = None,
        class_flag: bool = False,
        layout_template: str = "default",
        graph_style: Optional[Dict[str, Any]] = None,
        node_style: Optional[Dict[str, Any]] = None,
        edge_style: Optional[Dict[str, Any]] = None,
        fig_size: Tuple[float, float] = (16, 8),
        dpi: int = 300,
        pdf_dpi: int = 600,
        show: bool = True,
        export_pdf: bool = False,
    ) -> None:
        """Render a standard DPG plot."""
        if explanation is None:
            explanation = self.explain_global()
        plot_dpg(
            plot_name,
            explanation.dot,
            explanation.node_metrics,
            explanation.edge_metrics,
            save_dir=save_dir,
            attribute=attribute,
            class_flag=class_flag,
            layout_template=layout_template,
            graph_style=graph_style,
            node_style=node_style,
            edge_style=edge_style,
            fig_size=fig_size,
            dpi=dpi,
            pdf_dpi=pdf_dpi,
            show=show,
            export_pdf=export_pdf,
        )

    def plot_communities(
        self,
        plot_name: str,
        explanation: Optional[DPGExplanation] = None,
        save_dir: str = "results/",
        class_flag: bool = True,
        layout_template: str = "default",
        graph_style: Optional[Dict[str, Any]] = None,
        node_style: Optional[Dict[str, Any]] = None,
        edge_style: Optional[Dict[str, Any]] = None,
        fig_size: Tuple[float, float] = (16, 8),
        dpi: int = 300,
        pdf_dpi: int = 600,
        show: bool = True,
        export_pdf: bool = False,
        community_threshold: float = 0.2,
    ) -> None:
        """Render a community-colored DPG plot."""
        if explanation is None or explanation.communities is None:
            explanation = self.explain_global(
                communities=True,
                community_threshold=community_threshold,
            )
        plot_dpg_communities(
            plot_name,
            explanation.dot,
            explanation.node_metrics,
            explanation.communities,
            save_dir=save_dir,
            class_flag=class_flag,
            layout_template=layout_template,
            graph_style=graph_style,
            node_style=node_style,
            edge_style=edge_style,
            fig_size=fig_size,
            dpi=dpi,
            pdf_dpi=pdf_dpi,
            show=show,
            export_pdf=export_pdf,
        )

    def plot_lrc_importance(
        self,
        X_df: pd.DataFrame,
        explanation: Optional[DPGExplanation] = None,
        top_k: int = 10,
        dataset_name: str = "Dataset",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Any:
        """Plot top LRC predicates vs RF feature importances."""
        if explanation is None:
            explanation = self.explain_global()
        return plot_lrc_vs_rf_importance(
            explanation=explanation,
            model=self._builder.model,
            X_df=X_df,
            top_k=top_k,
            dataset_name=dataset_name,
            save_path=save_path,
            show=show,
        )

    def plot_top_lrc_splits(
        self,
        X_df: pd.DataFrame,
        y,
        explanation: Optional[DPGExplanation] = None,
        top_predicates: int = 5,
        top_features: int = 2,
        dataset_name: str = "Dataset",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Optional[Any]:
        """Plot top-LRC split lines over the top-2 LRC feature space."""
        if explanation is None:
            explanation = self.explain_global()
        return plot_top_lrc_predicate_splits(
            explanation=explanation,
            X_df=X_df,
            y=y,
            top_predicates=top_predicates,
            top_features=top_features,
            dataset_name=dataset_name,
            save_path=save_path,
            show=show,
        )

    def class_feature_predicate_counts(
        self,
        explanation: Optional[DPGExplanation] = None,
        community_threshold: float = 0.2,
    ) -> pd.DataFrame:
        """Return class-vs-feature predicate count matrix from communities."""
        if explanation is None or explanation.communities is None:
            explanation = self.explain_global(communities=True, community_threshold=community_threshold)
        return class_feature_predicate_counts(explanation)

    def plot_class_bounds_vs_dataset_ranges(
        self,
        X_df: pd.DataFrame,
        y,
        explanation: Optional[DPGExplanation] = None,
        dataset_name: str = "Dataset",
        top_features: int = 4,
        class_lookup: Optional[Dict[str, int]] = None,
        class_filter: Optional[List[str]] = None,
        density_tol_ratio: float = 0.03,
        predicate_alpha: float = 0.55,
        dataset_range_lw: float = 10,
        save_path: Optional[str] = None,
        show: bool = True,
        community_threshold: float = 0.2,
    ) -> Optional[Any]:
        """Plot DPG class bounds against empirical dataset feature ranges."""
        if explanation is None or explanation.communities is None:
            explanation = self.explain_global(communities=True, community_threshold=community_threshold)
        if class_lookup is None:
            class_lookup = class_lookup_from_target_names(self._builder.target_names)
        return plot_dpg_class_bounds_vs_dataset_feature_ranges(
            explanation=explanation,
            X_df=X_df,
            y=y,
            dataset_name=dataset_name,
            top_features=top_features,
            class_lookup=class_lookup,
            class_filter=class_filter,
            density_tol_ratio=density_tol_ratio,
            predicate_alpha=predicate_alpha,
            dataset_range_lw=dataset_range_lw,
            save_path=save_path,
            show=show,
        )
