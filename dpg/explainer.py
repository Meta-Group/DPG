from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .core import DecisionPredicateGraph
from .visualizer import (
    class_feature_predicate_counts,
    class_lookup_from_target_names,
    plot_sample_using_bc_weights,
    plot_dpg,
    plot_dpg_class_bounds_vs_dataset_feature_ranges,
    plot_dpg_communities,
    plot_lrc_vs_rf_importance,
    plot_top_lrc_predicate_splits,
    sample_bc_weights,
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
    node_metrics: Any
    edge_metrics: Any
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

    def fit(self, X: Any) -> "DPGExplainer":
        """Fit the DPG structure from training data."""
        self._dot = self._builder.fit(X)
        self._graph, self._nodes = self._builder.to_networkx(self._dot)
        self._is_fitted = True
        return self

    def explain_global(
        self,
        X: Optional[Any] = None,
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
        layout_template: str = "default",
        graph_style: Optional[Dict[str, Any]] = None,
        node_style: Optional[Dict[str, Any]] = None,
        edge_style: Optional[Dict[str, Any]] = None,
        fig_size: Tuple[float, float] = (16, 8),
        dpi: int = 300,
        pdf_dpi: int = 600,
        show: bool = True,
        export_pdf: bool = False,
        export_html: bool = False,
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
            export_html=export_html,
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
        export_html: bool = False,
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
            export_html=export_html,
        )

    def plot_lrc_importance(
        self,
        X_df: Any,
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
        X_df: Any,
        y,
        explanation: Optional[DPGExplanation] = None,
        top_predicates: int = 5,
        top_features: int = 2,
        dataset_name: str = "Dataset",
        class_names: Optional[Any] = None,
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
            class_names=class_names,
            save_path=save_path,
            show=show,
        )

    def class_feature_predicate_counts(
        self,
        explanation: Optional[DPGExplanation] = None,
        community_threshold: float = 0.2,
    ) -> Any:
        """Return class-vs-feature predicate count matrix from communities."""
        if explanation is None or explanation.communities is None:
            explanation = self.explain_global(communities=True, community_threshold=community_threshold)
        return class_feature_predicate_counts(explanation)

    def sample_bc_weights(
        self,
        X_df: Any,
        explanation: Optional[DPGExplanation] = None,
        top_k: int = 10,
    ) -> Any:
        """Return the per-sample BC-derived bottleneck exposure weights."""
        if explanation is None:
            explanation = self.explain_global()
        return sample_bc_weights(
            explanation=explanation,
            X_df=X_df,
            top_k=top_k,
        )

    def plot_sample_using_bc_weights(
        self,
        X_df: Any,
        y,
        explanation: Optional[DPGExplanation] = None,
        top_k: int = 10,
        dataset_name: str = "Dataset",
        class_names: Optional[Any] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Any:
        """Plot samples in PCA space with marker size set by BC-derived weights."""
        if explanation is None:
            explanation = self.explain_global()
        return plot_sample_using_bc_weights(
            explanation=explanation,
            X_df=X_df,
            y=y,
            top_k=top_k,
            dataset_name=dataset_name,
            class_names=class_names,
            save_path=save_path,
            show=show,
        )

    def plot_class_bounds_vs_dataset_ranges(
        self,
        X_df: Any,
        y,
        explanation: Optional[DPGExplanation] = None,
        dataset_name: str = "Dataset",
        top_features: int = 4,
        feature_cols_per_row: int = 4,
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
            feature_cols_per_row=feature_cols_per_row,
            class_lookup=class_lookup,
            class_filter=class_filter,
            density_tol_ratio=density_tol_ratio,
            predicate_alpha=predicate_alpha,
            dataset_range_lw=dataset_range_lw,
            save_path=save_path,
            show=show,
        )
