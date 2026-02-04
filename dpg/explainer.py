from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .core import DecisionPredicateGraph
from .visualizer import plot_dpg, plot_dpg_communities
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


class DPGExplainer:
    """
    High-level, user-friendly API for building and plotting DPG explanations.

    This class wraps DecisionPredicateGraph and the metrics/visualization utilities
    into a cohesive workflow.
    """

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

    @property
    def builder(self) -> DecisionPredicateGraph:
        return self._builder

    def fit(self, X: np.ndarray) -> "DPGExplainer":
        """Fit the DPG structure from training data."""
        self._dot = self._builder.fit(X)
        self._graph, self._nodes = self._builder.to_networkx(self._dot)
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

    def plot(
        self,
        plot_name: str,
        explanation: Optional[DPGExplanation] = None,
        save_dir: str = "results/",
        attribute: Optional[str] = None,
        class_flag: bool = False,
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
            show=show,
            export_pdf=export_pdf,
        )

    def plot_communities(
        self,
        plot_name: str,
        explanation: Optional[DPGExplanation] = None,
        save_dir: str = "results/",
        class_flag: bool = True,
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
            show=show,
            export_pdf=export_pdf,
        )
