from __future__ import annotations

from dataclasses import dataclass
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

    def __init__(
        self,
        model: Any,
        feature_names: Iterable[str],
        target_names: Optional[Iterable[str]] = None,
        config_file: str = "config.yaml",
        dpg_config: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        # Local edge weights are normalized inside the active graph so path supports
        # are comparable across paths for this sample.
        edge_weights = []
        for src, dst in active_graph.edges():
            w = active_graph.get_edge_data(src, dst, default={}).get("weight", 0.0)
            try:
                edge_weights.append(float(w))
            except (TypeError, ValueError):
                edge_weights.append(0.0)
        max_edge_weight = max(edge_weights) if edge_weights else 1.0
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

        tree_paths: List[DPGTreePathExplanation] = []
        path_records: List[Dict[str, Any]] = []
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

            tree_paths.append(
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
            path_records.append({"leaf_class": leaf_class, "path_confidence": path_confidence})

        leaf_classes = [
            p.labels[-1].replace("Class ", "", 1)
            for p in tree_paths
            if p.labels and p.labels[-1].startswith("Class ")
        ]
        class_votes: Dict[str, int] = (
            pd.Series(leaf_classes).value_counts().astype(int).to_dict() if leaf_classes else {}
        )

        # Class-level attraction/repulsion score.
        # For each class t:
        #   A_t = sum(path_support for paths ending in t)
        #   R_t = sum(path_support for paths ending in classes != t)
        #   score_t = A_t / (A_t + lambda * R_t + eps)
        # This rewards strong/short/high-LRC paths to the target class while
        # penalizing evidence that flows to competing classes.
        repulsion_lambda = 0.8
        eps = 1e-12
        class_support: Dict[str, float] = {}
        for rec in path_records:
            cls = rec["leaf_class"]
            if cls is None:
                continue
            class_support[cls] = class_support.get(cls, 0.0) + float(rec["path_confidence"])
        evidence_scores: Dict[str, float] = {}
        total_support = float(sum(class_support.values()))
        for cls, att in class_support.items():
            rep = max(0.0, total_support - att)
            evidence_scores[cls] = float(att / (att + repulsion_lambda * rep + eps))
        # Normalize evidence scores to sum to 1 for comparability/reporting.
        score_sum = float(sum(evidence_scores.values()))
        if score_sum > 0:
            evidence_scores = {k: float(v / score_sum) for k, v in evidence_scores.items()}
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
            "num_active_nodes": float(len(active_metric_nodes)),
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
