import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
)

from .core import DecisionPredicateGraph
from .sklearn_normalizer import SklearnEnsembleNormalizer
from .visualizer import (
    class_feature_predicate_counts,
    class_lookup_from_target_names,
    plot_class_feature_complexity,
    plot_dpg_local_paths_aggregate,
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


@dataclass
class DPGTreePathExplanation:
    tree_index: int
    tree_prefix: str
    labels: List[str]
    node_ids: List[Optional[str]]
    predicate_truths: List[bool]
    edge_exists: List[bool]
    starts_from_root: bool
    ends_in_leaf: bool
    graph_path_valid: bool
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
    sample_id: int
    sample: List[float]
    tree_paths: List[DPGTreePathExplanation]
    graph_validated: bool
    all_trees_valid: bool
    majority_vote: Optional[str]
    class_votes: Dict[str, int]
    path_mode: str
    sample_confidence: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sample": self.sample,
            "tree_paths": [path.as_dict() for path in self.tree_paths],
            "graph_validated": self.graph_validated,
            "all_trees_valid": self.all_trees_valid,
            "majority_vote": self.majority_vote,
            "class_votes": self.class_votes,
            "path_mode": self.path_mode,
            "sample_confidence": self.sample_confidence,
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
        self._node_metrics = None
        self._node_metrics_lookup = None
        self._edge_metrics = None

    @property
    def builder(self) -> DecisionPredicateGraph:
        return self._builder

    def fit(self, X: Any) -> "DPGExplainer":
        """Fit the DPG structure from training data."""
        self._dot = self._builder.fit(X)
        self._graph, self._nodes = self._builder.to_networkx(self._dot)
        self._node_metrics = None
        self._node_metrics_lookup = None
        self._edge_metrics = None
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

        node_metrics = self._get_node_metrics()
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

    def explain_local(
        self,
        sample: Any,
        sample_id: int = 0,
        X: Optional[Any] = None,
        validate_graph: bool = True,
    ) -> DPGLocalExplanation:
        """
        Trace a single sample through every estimator and map the executed path
        onto the fitted DPG graph.

        Args:
            sample: One sample with the same feature dimension used to fit the model.
            sample_id: Identifier to attach to the returned explanation.
            X: Optional training data. If provided, fit() is called before tracing.
            validate_graph: Whether to validate node and edge presence in the fitted graph.
        """
        if X is not None:
            self.fit(X)
        if not self._is_fitted:
            raise ValueError("DPGExplainer is not fitted. Call fit(X) or explain_local(X=...).")

        sample_array = np.asarray(sample).reshape(-1)
        expected_features = len(self._builder.feature_names)
        if sample_array.shape[0] != expected_features:
            raise ValueError(
                f"Sample has {sample_array.shape[0]} features, expected {expected_features}."
            )

        node_lookup = {label: node_id for node_id, label in self._nodes}
        node_metrics_lookup = self._get_node_metrics_lookup()

        tree_paths = []
        class_votes = Counter()
        for tree_index, tree in enumerate(self._builder.model.estimators_):
            path = self._trace_tree_path(
                tree=tree,
                sample=sample_array,
                sample_id=sample_id,
                tree_index=tree_index,
                node_lookup=node_lookup,
                node_metrics_lookup=node_metrics_lookup,
                validate_graph=validate_graph,
            )
            tree_paths.append(path)
            if path.labels and path.labels[-1].startswith("Class "):
                class_votes[self._normalize_class_vote_label(path.labels[-1])] += 1

        majority_vote = None
        if class_votes:
            majority_vote = class_votes.most_common(1)[0][0]

        sample_confidence = self._compute_sample_confidence(
            tree_paths,
            dict(class_votes),
            sample_array,
        )

        return DPGLocalExplanation(
            sample_id=sample_id,
            sample=sample_array.tolist(),
            tree_paths=tree_paths,
            graph_validated=validate_graph,
            all_trees_valid=all(path.graph_path_valid for path in tree_paths),
            majority_vote=majority_vote,
            class_votes=dict(class_votes),
            path_mode="execution_trace",
            sample_confidence=sample_confidence,
        )

    def plot_local_on_dpg(
        self,
        plot_name: str,
        local_explanation: Optional[DPGLocalExplanation] = None,
        sample: Optional[Any] = None,
        sample_id: int = 0,
        X: Optional[Any] = None,
        validate_graph: bool = True,
        path_indices: Optional[List[int]] = None,
        true_class_label: Optional[str] = None,
        obtained_class_label: Optional[str] = None,
        sample_metrics: Optional[Dict[str, Any]] = None,
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
        theme: str = "dpg",
        palette: str = "default",
        label_mode: str = "wrapped",
        readability: str = "presentation",
        title: Optional[str] = None,
    ) -> Any:
        if X is not None:
            self.fit(X)
        if not self._is_fitted:
            raise ValueError("DPGExplainer is not fitted. Call fit(X) or plot_local_on_dpg(X=...).")

        if local_explanation is None:
            if sample is None:
                raise ValueError("Either local_explanation or sample must be provided.")
            local_explanation = self.explain_local(
                sample=sample,
                sample_id=sample_id,
                X=None,
                validate_graph=validate_graph,
            )

        selected_paths = list(local_explanation.tree_paths)
        if path_indices is not None:
            selected_paths = []
            for idx in path_indices:
                if not isinstance(idx, int) or idx < 0 or idx >= len(local_explanation.tree_paths):
                    raise ValueError("path_indices must reference valid path indices.")
                selected_paths.append(local_explanation.tree_paths[idx])

        if obtained_class_label is None:
            obtained_class_label = local_explanation.majority_vote
        if sample_metrics is None:
            sample_metrics = local_explanation.sample_confidence

        return plot_dpg_local_paths_aggregate(
            plot_name=plot_name,
            dot=self._dot,
            df=self._get_node_metrics(),
            df_edges=self._get_edge_metrics(),
            paths_node_ids=[path.node_ids for path in selected_paths],
            path_confidences=[path.path_confidence for path in selected_paths],
            sample_id=local_explanation.sample_id,
            true_class_label=true_class_label,
            obtained_class_label=obtained_class_label,
            sample_metrics=sample_metrics,
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
            theme=theme,
            palette=palette,
            label_mode=label_mode,
            readability=readability,
            title=title,
        )

    def local_path_dataframe(self, local_explanation: DPGLocalExplanation) -> Any:
        """
        Flatten a local explanation into one row per path step.

        Args:
            local_explanation: Structured local explanation returned by explain_local().

        Returns:
            pd.DataFrame: Ordered path-step table.
        """
        columns = [
            "sample_id",
            "tree_index",
            "step_index",
            "label",
            "node_id",
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

        rows = []
        for path in sorted(local_explanation.tree_paths, key=lambda path: path.tree_index):
            for step_index, label in enumerate(path.labels):
                is_leaf = label.startswith("Class ") or label.startswith("Pred ")
                predicate_true = (
                    path.predicate_truths[step_index]
                    if step_index < len(path.predicate_truths)
                    else None
                )
                edge_exists_from_prev = (
                    True if step_index == 0 else path.edge_exists[step_index - 1]
                )
                node_id = path.node_ids[step_index] if step_index < len(path.node_ids) else None
                rows.append(
                    {
                        "sample_id": local_explanation.sample_id,
                        "tree_index": path.tree_index,
                        "step_index": step_index,
                        "label": label,
                        "node_id": node_id,
                        "is_leaf": is_leaf,
                        "predicate_true": predicate_true,
                        "edge_exists_from_prev": edge_exists_from_prev,
                        "starts_from_root": path.starts_from_root,
                        "ends_in_leaf": path.ends_in_leaf,
                        "graph_path_valid": path.graph_path_valid,
                        "mean_lrc": path.mean_lrc,
                        "mean_bc": path.mean_bc,
                        "path_confidence": path.path_confidence,
                    }
                )

        return pd.DataFrame(rows, columns=columns)

    def evaluate_faithfulness(
        self,
        X,
        y_true=None,
        max_samples=None,
        weights=None,
        return_details=False,
        sample_ids=None,
    ):
        """
        Evaluate local DPG explanations against the fitted black-box model.

        This measures output fidelity to the underlying model plus structural
        faithfulness diagnostics derived from local DPG traces. It does not
        measure ground-truth correctness unless ``y_true`` is provided, and the
        returned composite score is a heuristic summary, not a calibrated
        probability.
        """
        if not self._is_fitted:
            raise ValueError("DPGExplainer is not fitted. Call fit(X) before evaluate_faithfulness().")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be a positive integer when provided.")

        weights = self._validate_faithfulness_weights(weights)

        if isinstance(X, pd.DataFrame):
            X_eval = X.iloc[:max_samples].copy() if max_samples is not None else X.copy()
            row_iter = [(i, X_eval.iloc[i], X_eval.iloc[i].values) for i in range(len(X_eval))]
        else:
            X_array = np.asarray(X)
            X_eval = X_array[:max_samples] if max_samples is not None else X_array
            row_iter = [(i, X_eval[i], np.asarray(X_eval[i]).reshape(-1)) for i in range(len(X_eval))]

        n_samples = len(X_eval)
        if n_samples == 0:
            raise ValueError("X must contain at least one sample for faithfulness evaluation.")
        if y_true is not None:
            y_true_seq = list(y_true[:n_samples] if max_samples is not None else y_true)
            if len(y_true_seq) != n_samples:
                raise ValueError("y_true length must match the number of evaluated samples.")
        else:
            y_true_seq = None

        if sample_ids is not None:
            sample_ids_seq = list(sample_ids[:n_samples] if max_samples is not None else sample_ids)
            if len(sample_ids_seq) != n_samples:
                raise ValueError("sample_ids length must match the number of evaluated samples.")
        else:
            sample_ids_seq = list(range(n_samples))

        per_sample_records = []
        successful_records = []
        n_local_failures = 0

        for idx, row_for_predict, row_values in row_iter:
            sample_id = sample_ids_seq[idx]
            true_label_normalized = None
            if y_true_seq is not None:
                true_label_normalized = self._normalize_prediction_label(y_true_seq[idx])

            if isinstance(X_eval, pd.DataFrame):
                model_pred_raw = self._builder.model.predict(row_for_predict.to_frame().T)[0]
            else:
                model_pred_raw = self._builder.model.predict(np.asarray(row_values).reshape(1, -1))[0]
            model_pred = self._normalize_prediction_label(model_pred_raw)

            try:
                local = self.explain_local(sample=row_values, sample_id=sample_id)
                local_pred = local.majority_vote
                sample_confidence = local.sample_confidence or {}

                record = {
                    "sample_id": sample_id,
                    "model_pred": model_pred,
                    "local_pred": local_pred,
                    "matches_model": bool(local_pred == model_pred),
                    "vote_confidence": self._maybe_float(sample_confidence.get("vote_confidence")),
                    "evidence_score_pred": self._maybe_float(sample_confidence.get("evidence_score_pred")),
                    "evidence_score_margin": self._maybe_float(sample_confidence.get("evidence_score_margin")),
                    "trace_coverage_score": self._maybe_float(sample_confidence.get("trace_coverage_score")),
                    "recombination_rate": self._maybe_float(sample_confidence.get("recombination_rate")),
                    "graph_path_valid_rate": self._maybe_float(sample_confidence.get("graph_path_valid_rate")),
                    "node_recall": self._maybe_float(sample_confidence.get("node_recall")),
                    "node_precision": self._maybe_float(sample_confidence.get("node_precision")),
                    "edge_recall": self._maybe_float(sample_confidence.get("edge_recall")),
                    "edge_precision": self._maybe_float(sample_confidence.get("edge_precision")),
                    "evidence_margin_pred_vs_competitor": self._maybe_float(
                        sample_confidence.get("evidence_margin_pred_vs_competitor")
                    ),
                    "path_purity": self._maybe_float(sample_confidence.get("path_purity")),
                    "competitor_exposure": self._maybe_float(sample_confidence.get("competitor_exposure")),
                    "explanation_confidence": self._maybe_float(sample_confidence.get("explanation_confidence")),
                    "error": None,
                }
                if true_label_normalized is not None:
                    record["correct"] = bool(local_pred == true_label_normalized)

                per_sample_records.append(record)
                successful_records.append(record)
            except Exception as exc:
                n_local_failures += 1
                failure_record = {
                    "sample_id": sample_id,
                    "model_pred": model_pred,
                    "local_pred": None,
                    "matches_model": False,
                    "vote_confidence": None,
                    "evidence_score_pred": None,
                    "evidence_score_margin": None,
                    "trace_coverage_score": None,
                    "recombination_rate": None,
                    "graph_path_valid_rate": None,
                    "node_recall": None,
                    "node_precision": None,
                    "edge_recall": None,
                    "edge_precision": None,
                    "evidence_margin_pred_vs_competitor": None,
                    "path_purity": None,
                    "competitor_exposure": None,
                    "explanation_confidence": None,
                    "error": str(exc),
                }
                if y_true_seq is not None:
                    failure_record["correct"] = None
                per_sample_records.append(failure_record)

        if not successful_records:
            raise ValueError(
                "All local explanations failed during faithfulness evaluation; "
                "no faithfulness metrics could be computed."
            )

        output_fidelity = float(np.mean([record["matches_model"] for record in successful_records]))
        local_accuracy = None
        if y_true_seq is not None:
            local_accuracy = float(
                np.mean([record["correct"] for record in successful_records if record.get("correct") is not None])
            )

        mean_node_recall = self._mean_records(successful_records, "node_recall")
        mean_node_precision = self._mean_records(successful_records, "node_precision")
        mean_edge_recall = self._mean_records(successful_records, "edge_recall")
        mean_edge_precision = self._mean_records(successful_records, "edge_precision")
        mean_trace_coverage_score = self._mean_records(successful_records, "trace_coverage_score")
        mean_recombination_rate = self._mean_records(successful_records, "recombination_rate")

        mean_vote_confidence = self._mean_records(successful_records, "vote_confidence")
        mean_evidence_score_pred = self._mean_records(successful_records, "evidence_score_pred")
        mean_evidence_score_margin = self._mean_records(successful_records, "evidence_score_margin")
        mean_evidence_margin_pred_vs_competitor = self._mean_records(
            successful_records,
            "evidence_margin_pred_vs_competitor",
        )
        mean_path_purity = self._mean_records(successful_records, "path_purity")
        mean_competitor_exposure = self._mean_records(successful_records, "competitor_exposure")
        mean_explanation_confidence = self._mean_records(successful_records, "explanation_confidence")

        composite = (
            weights["output_fidelity"] * output_fidelity
            + weights["trace_coverage"] * mean_trace_coverage_score
            + weights["anti_recombination"] * (1.0 - mean_recombination_rate)
            + weights["evidence_margin"] * mean_evidence_score_margin
        )

        if not return_details:
            return float(composite)

        details = {
            "faithfulness_score": float(composite),
            "weights": weights,
            "n_samples": n_samples,
            "n_successful": len(successful_records),
            "n_local_failures": n_local_failures,
            "output_fidelity": output_fidelity,
            "mean_node_recall": mean_node_recall,
            "mean_node_precision": mean_node_precision,
            "mean_edge_recall": mean_edge_recall,
            "mean_edge_precision": mean_edge_precision,
            "mean_trace_coverage_score": mean_trace_coverage_score,
            "mean_recombination_rate": mean_recombination_rate,
            "mean_vote_confidence": mean_vote_confidence,
            "mean_evidence_score_pred": mean_evidence_score_pred,
            "mean_evidence_score_margin": mean_evidence_score_margin,
            "mean_evidence_margin_pred_vs_competitor": mean_evidence_margin_pred_vs_competitor,
            "mean_path_purity": mean_path_purity,
            "mean_competitor_exposure": mean_competitor_exposure,
            "mean_explanation_confidence": mean_explanation_confidence,
            "per_sample": pd.DataFrame(per_sample_records),
        }
        if local_accuracy is not None:
            details["local_accuracy"] = local_accuracy
        return details

    def _trace_tree_path(
        self,
        tree: Any,
        sample: np.ndarray,
        sample_id: int,
        tree_index: int,
        node_lookup: Dict[str, str],
        node_metrics_lookup: Dict[str, Dict[str, Any]],
        validate_graph: bool,
    ) -> DPGTreePathExplanation:
        is_regressor = isinstance(
            self._builder.model,
            (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor),
        )
        tree_ = tree.tree_
        node_index = 0
        tree_prefix = f"sample{sample_id}_dt{tree_index}"
        labels: List[str] = []
        predicate_truths: List[bool] = []

        while True:
            left = tree_.children_left[node_index]
            right = tree_.children_right[node_index]
            if left == right:
                if is_regressor:
                    pred = round(tree_.value[node_index][0][0], 2)
                    labels.append(f"Pred {pred}")
                else:
                    labels.append(self._leaf_class_label(tree_index, tree_, node_index))
                break

            feature_index = tree_.feature[node_index]
            threshold = round(tree_.threshold[node_index], self._builder.decimal_threshold)
            feature_name = self._builder.feature_names[feature_index]
            sample_val = sample[feature_index]
            if sample_val <= threshold:
                labels.append(f"{feature_name} <= {threshold}")
                predicate_truths.append(True)
                node_index = left
            else:
                labels.append(f"{feature_name} > {threshold}")
                predicate_truths.append(True)
                node_index = right

        native_node_ids = [self._label_to_node_id(label) for label in labels]
        node_ids = [
            node_lookup.get(label) if validate_graph else native_node_id
            for label, native_node_id in zip(labels, native_node_ids)
        ]

        edge_exists = []
        for i in range(len(native_node_ids) - 1):
            edge_exists.append(self._graph.has_edge(native_node_ids[i], native_node_ids[i + 1]))

        graph_path_valid = all(native_node_id in node_metrics_lookup for native_node_id in native_node_ids) and all(edge_exists)

        active_metric_rows = [
            node_metrics_lookup[native_node_id]
            for native_node_id in native_node_ids
            if native_node_id in node_metrics_lookup
        ]
        mean_lrc = None
        mean_bc = None
        if active_metric_rows:
            mean_lrc = float(np.mean([row["Local reaching centrality"] for row in active_metric_rows]))
            mean_bc = float(np.mean([row["Betweenness centrality"] for row in active_metric_rows]))

        node_coverage = (
            len(active_metric_rows) / len(labels)
            if labels
            else 0.0
        )
        edge_coverage = (
            sum(edge_exists) / len(edge_exists)
            if edge_exists
            else 1.0
        )
        path_confidence = float((node_coverage + edge_coverage) / 2.0) if labels else 0.0

        return DPGTreePathExplanation(
            tree_index=tree_index,
            tree_prefix=tree_prefix,
            labels=labels,
            node_ids=node_ids,
            predicate_truths=predicate_truths,
            edge_exists=edge_exists,
            starts_from_root=len(labels) > 0,
            ends_in_leaf=bool(labels and (labels[-1].startswith("Class ") or labels[-1].startswith("Pred "))),
            graph_path_valid=graph_path_valid,
            mean_lrc=mean_lrc,
            mean_bc=mean_bc,
            path_confidence=path_confidence,
        )

    @staticmethod
    def _label_to_node_id(label: str) -> str:
        return str(int(hashlib.sha1(label.encode()).hexdigest(), 16))

    def _leaf_class_label(self, tree_index: int, tree_: Any, node_index: int) -> str:
        """Return the class label for a classifier leaf node."""
        gb_class_index = SklearnEnsembleNormalizer.get_tree_class_index(
            self._builder.model,
            tree_index,
        )
        if gb_class_index is not None:
            pred_class = gb_class_index
        elif isinstance(self._builder.model, GradientBoostingClassifier) and getattr(self._builder.model, "n_classes_", 0) == 2:
            leaf_score = float(tree_.value[node_index][0][0])
            pred_class = 1 if leaf_score > 0 else 0
        else:
            pred_class = int(tree_.value[node_index].argmax())

        if self._builder.target_names is not None:
            pred_class = self._builder.target_names[pred_class]
        elif hasattr(self._builder.model, "classes_"):
            pred_class = self._builder.model.classes_[pred_class]
        return f"Class {pred_class}"

    @staticmethod
    def _normalize_class_vote_label(label: str) -> str:
        return label[len("Class ") :] if label.startswith("Class ") else label

    def _get_node_metrics(self) -> Any:
        if self._node_metrics is None:
            self._node_metrics = NodeMetrics.extract_node_metrics(self._graph, self._nodes)
        return self._node_metrics

    def _get_node_metrics_lookup(self) -> Dict[str, Dict[str, Any]]:
        if self._node_metrics_lookup is None:
            node_metrics = self._get_node_metrics()
            self._node_metrics_lookup = {
                row["Node"]: row
                for row in node_metrics.to_dict(orient="records")
            }
        return self._node_metrics_lookup

    def _get_edge_metrics(self) -> Any:
        if self._edge_metrics is None:
            self._edge_metrics = EdgeMetrics.extract_edge_metrics(self._graph, self._nodes)
        return self._edge_metrics

    def _compute_sample_confidence(
        self,
        tree_paths: List[DPGTreePathExplanation],
        class_votes: Dict[str, int],
        sample_array: np.ndarray,
    ) -> Dict[str, Any]:
        num_paths = len(tree_paths)
        num_valid_paths = sum(path.graph_path_valid for path in tree_paths)
        active_node_ids = []
        for path in tree_paths:
            for node_id in path.node_ids:
                if node_id is not None:
                    active_node_ids.append(node_id)
        unique_active_node_ids = list(dict.fromkeys(active_node_ids))

        node_metrics_lookup = self._get_node_metrics_lookup()
        active_metric_rows = [
            node_metrics_lookup[node_id]
            for node_id in unique_active_node_ids
            if node_id in node_metrics_lookup
        ]
        mean_lrc_active_nodes = (
            float(np.mean([row["Local reaching centrality"] for row in active_metric_rows]))
            if active_metric_rows
            else 0.0
        )
        mean_bc_active_nodes = (
            float(np.mean([row["Betweenness centrality"] for row in active_metric_rows]))
            if active_metric_rows
            else 0.0
        )

        total_votes = sum(class_votes.values())
        class_scores = (
            {label: votes / total_votes for label, votes in class_votes.items()}
            if total_votes > 0
            else {}
        )
        sorted_scores = sorted(class_scores.values(), reverse=True)
        if not sorted_scores:
            vote_confidence = 0.0
            score_margin = 0.0
        else:
            vote_confidence = float(sorted_scores[0])
            score_margin = float(
                sorted_scores[0] - sorted_scores[1]
                if len(sorted_scores) > 1
                else sorted_scores[0]
            )

        class_support, evidence_scores = self._compute_evidence_scores(
            tree_paths=tree_paths,
            class_scores=class_scores,
        )
        trace_diagnostics = self._compute_trace_diagnostics(tree_paths, sample_array)
        evidence_score_pred = None
        if class_votes:
            majority_vote = max(class_votes, key=class_votes.get)
            evidence_score_pred = evidence_scores.get(majority_vote)

        sorted_evidence = sorted(
            evidence_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if not sorted_evidence:
            evidence_score_margin = None
            top_competitor_class_pred = None
            evidence_score_competitor_pred = None
            evidence_margin_pred_vs_competitor = None
        else:
            top_score = float(sorted_evidence[0][1])
            second_item = sorted_evidence[1] if len(sorted_evidence) > 1 else None
            evidence_score_margin = float(
                top_score - second_item[1] if second_item is not None else top_score
            )
            top_competitor_class_pred = second_item[0] if second_item is not None else None
            evidence_score_competitor_pred = (
                float(second_item[1]) if second_item is not None else None
            )
            evidence_margin_pred_vs_competitor = (
                float(evidence_score_pred - evidence_score_competitor_pred)
                if evidence_score_pred is not None and evidence_score_competitor_pred is not None
                else float(evidence_score_pred) if evidence_score_pred is not None else None
            )

        return {
            "num_paths": num_paths,
            "num_valid_paths": num_valid_paths,
            "num_active_nodes": len(unique_active_node_ids),
            "mean_lrc_active_nodes": mean_lrc_active_nodes,
            "mean_bc_active_nodes": mean_bc_active_nodes,
            "graph_path_valid_rate": (num_valid_paths / num_paths) if num_paths > 0 else 0.0,
            "vote_confidence": vote_confidence,
            "class_scores": class_scores,
            "score_margin": score_margin,
            "class_support": class_support,
            "evidence_scores": evidence_scores,
            "evidence_score_pred": evidence_score_pred,
            "evidence_score_margin": evidence_score_margin,
            "top_competitor_class_pred": top_competitor_class_pred,
            "evidence_score_competitor_pred": evidence_score_competitor_pred,
            "evidence_margin_pred_vs_competitor": evidence_margin_pred_vs_competitor,
            **trace_diagnostics,
        }

    def _compute_evidence_scores(
        self,
        tree_paths: List[DPGTreePathExplanation],
        class_scores: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        class_support: Dict[str, float] = {}
        for path in tree_paths:
            if not path.labels:
                continue
            leaf_label = path.labels[-1]
            if not leaf_label.startswith("Class "):
                continue
            class_name = self._normalize_class_vote_label(leaf_label)
            support = float(path.path_confidence or 0.0)
            class_support[class_name] = class_support.get(class_name, 0.0) + support

        total_support = sum(class_support.values())
        if total_support > 0:
            evidence_scores = {
                class_name: support / total_support
                for class_name, support in class_support.items()
            }
        else:
            evidence_scores = dict(class_scores)

        return class_support, evidence_scores

    def _extract_execution_trace_labels(self, sample_arr: np.ndarray) -> List[List[str]]:
        traces = []
        for tree_index, tree in enumerate(self._builder.model.estimators_):
            traces.append(
                self._trace_execution_labels_for_tree(
                    tree,
                    sample_arr,
                    tree_index=tree_index,
                )
            )
        return traces

    def _trace_reference_sets(
        self,
        sample_arr: np.ndarray,
    ) -> Tuple[set, set]:
        trace_node_labels = set()
        trace_edge_labels = set()
        for labels in self._extract_execution_trace_labels(sample_arr):
            for label in labels:
                if not (label.startswith("Class ") or label.startswith("Pred ")):
                    trace_node_labels.add(label)
            for i in range(len(labels) - 1):
                trace_edge_labels.add((labels[i], labels[i + 1]))
        return trace_node_labels, trace_edge_labels

    def _compute_trace_diagnostics(
        self,
        tree_paths: List[DPGTreePathExplanation],
        sample_arr: np.ndarray,
    ) -> Dict[str, Any]:
        trace_node_labels, trace_edge_labels = self._trace_reference_sets(sample_arr)

        explanation_node_labels = set()
        explanation_edge_labels = set()
        for path in tree_paths:
            for label, node_id in zip(path.labels, path.node_ids):
                if node_id is not None and not (label.startswith("Class ") or label.startswith("Pred ")):
                    explanation_node_labels.add(label)
            for i in range(len(path.labels) - 1):
                src_id = path.node_ids[i] if i < len(path.node_ids) else None
                dst_id = path.node_ids[i + 1] if i + 1 < len(path.node_ids) else None
                edge_ok = path.edge_exists[i] if i < len(path.edge_exists) else False
                if src_id is not None and dst_id is not None and edge_ok:
                    explanation_edge_labels.add((path.labels[i], path.labels[i + 1]))

        node_overlap = len(trace_node_labels & explanation_node_labels)
        edge_overlap = len(trace_edge_labels & explanation_edge_labels)

        node_recall = (
            float(node_overlap / len(trace_node_labels))
            if trace_node_labels
            else 1.0
        )
        node_precision = (
            float(node_overlap / len(explanation_node_labels))
            if explanation_node_labels
            else 1.0
        )
        edge_recall = (
            float(edge_overlap / len(trace_edge_labels))
            if trace_edge_labels
            else 1.0
        )
        edge_precision = (
            float(edge_overlap / len(explanation_edge_labels))
            if explanation_edge_labels
            else 1.0
        )
        trace_coverage_score = float((node_recall + edge_recall) / 2.0)
        recombination_rate = (
            float(len(explanation_edge_labels - trace_edge_labels) / len(explanation_edge_labels))
            if explanation_edge_labels
            else 0.0
        )

        return {
            "trace_node_count_unique": int(len(trace_node_labels)),
            "trace_edge_count_unique": int(len(trace_edge_labels)),
            "explanation_node_count_unique": int(len(explanation_node_labels)),
            "explanation_edge_count_unique": int(len(explanation_edge_labels)),
            "node_recall": node_recall,
            "node_precision": node_precision,
            "edge_recall": edge_recall,
            "edge_precision": edge_precision,
            "trace_coverage_score": trace_coverage_score,
            "recombination_rate": recombination_rate,
        }

    def _trace_execution_labels_for_tree(
        self,
        tree: Any,
        sample: np.ndarray,
        tree_index: Optional[int] = None,
    ) -> List[str]:
        is_regressor = isinstance(
            self._builder.model,
            (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor),
        )
        tree_ = tree.tree_
        node_index = 0
        labels: List[str] = []

        while True:
            left = tree_.children_left[node_index]
            right = tree_.children_right[node_index]
            if left == right:
                if is_regressor:
                    pred = round(tree_.value[node_index][0][0], 2)
                    labels.append(f"Pred {pred}")
                else:
                    if tree_index is None:
                        pred_class = int(tree_.value[node_index].argmax())
                        if self._builder.target_names is not None:
                            pred_class = self._builder.target_names[pred_class]
                        elif hasattr(self._builder.model, "classes_"):
                            pred_class = self._builder.model.classes_[pred_class]
                        labels.append(f"Class {pred_class}")
                    else:
                        labels.append(self._leaf_class_label(tree_index, tree_, node_index))
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

        return labels

    def _normalize_prediction_label(self, value: Any) -> Any:
        if self._builder.target_names is not None and hasattr(self._builder.model, "classes_"):
            classes = list(self._builder.model.classes_)
            target_names = list(self._builder.target_names)
            if len(classes) == len(target_names):
                for class_value, target_name in zip(classes, target_names):
                    if value == class_value:
                        return str(target_name)
        if isinstance(value, str) and value.startswith("Class "):
            return value[len("Class ") :]
        return str(value)

    def _validate_faithfulness_weights(self, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        default_weights = {
            "output_fidelity": 0.35,
            "trace_coverage": 0.30,
            "anti_recombination": 0.20,
            "evidence_margin": 0.15,
        }
        if weights is None:
            return default_weights

        weights = dict(weights)
        unsupported = set(weights) - set(default_weights)
        if unsupported:
            raise ValueError(
                f"Unsupported faithfulness weight keys: {sorted(unsupported)}. "
                f"Supported keys are: {sorted(default_weights)}"
            )
        missing = set(default_weights) - set(weights)
        if missing:
            raise ValueError(
                f"Missing faithfulness weight keys: {sorted(missing)}. "
                f"Supported keys are: {sorted(default_weights)}"
            )
        total = float(sum(weights.values()))
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError("Faithfulness weights must sum to 1.0.")
        return {key: float(value) for key, value in weights.items()}

    @staticmethod
    def _mean_records(records: List[Dict[str, Any]], key: str) -> float:
        values = [
            float(record[key])
            for record in records
            if record.get(key) is not None and not pd.isna(record.get(key))
        ]
        if not values:
            return 0.0
        return float(np.mean(values))

    @staticmethod
    def _maybe_float(value: Any) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        return float(value)

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
        theme: str = "dpg",
        palette: str = "default",
        label_mode: str = "full",
        readability: str = "normal",
        title: Optional[str] = None,
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
            theme=theme,
            palette=palette,
            label_mode=label_mode,
            readability=readability,
            title=title,
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
        theme: str = "dpg",
        palette: str = "default",
        label_mode: str = "wrapped",
        readability: str = "presentation",
        title: Optional[str] = None,
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
            theme=theme,
            palette=palette,
            label_mode=label_mode,
            readability=readability,
            title=title,
        )

    def plot_lrc_importance(
        self,
        X_df: Any,
        explanation: Optional[DPGExplanation] = None,
        top_k: int = 10,
        dataset_name: str = "Dataset",
        save_path: Optional[str] = None,
        show: bool = True,
        theme: str = "dpg",
        palette: str = "default",
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
            theme=theme,
            palette=palette,
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
        theme: str = "dpg",
        palette: str = "default",
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
            theme=theme,
            palette=palette,
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

    def plot_class_feature_complexity(
        self,
        explanation: Optional[DPGExplanation] = None,
        dataset_name: str = "Dataset",
        top_n_features: int = 10,
        save_prefix: Optional[str] = None,
        show: bool = True,
        community_threshold: float = 0.2,
        theme: str = "dpg",
        palette: str = "default",
    ) -> Tuple[Any, Any]:
        """Plot community class-feature complexity using PCA-consistent class colors."""
        if explanation is None or explanation.communities is None:
            explanation = self.explain_global(communities=True, community_threshold=community_threshold)
        heat_df = class_feature_predicate_counts(explanation)
        return plot_class_feature_complexity(
            heat_df=heat_df,
            dataset_name=dataset_name,
            class_names=self._builder.target_names,
            top_n_features=top_n_features,
            save_prefix=save_prefix,
            show=show,
            theme=theme,
            palette=palette,
        )

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
        theme: str = "dpg",
        palette: str = "default",
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
            theme=theme,
            palette=palette,
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
        theme: str = "dpg",
        palette: str = "default",
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
            theme=theme,
            palette=palette,
        )
