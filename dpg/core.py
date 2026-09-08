import pandas as pd
pd.set_option("display.max_colwidth", 255)
import re
import math
import os
import warnings
import numpy as np

from tqdm import tqdm
import graphviz
import networkx as nx
import hashlib
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from joblib import Parallel, delayed

from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union
from sklearn.base import is_classifier, is_regressor

# Handle OmegaConf DictConfig if available
try:
    from omegaconf import DictConfig, OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from dpg.sklearn_normalizer import SklearnEnsembleNormalizer
from dpg.context_order import resolve_context_order

DEFAULT_DPG_CONFIG = {
    "dpg": {
        "default": {
            "perc_var": 0.000000001,
            "decimal_threshold": 6,
            "n_jobs": -1,
        },
        "graph_construction": {
            "mode": "aggregated_transitions",
            # Keep the 0.2.x graph as the default. DPG-k is opt-in through
            # context_order="auto" or an explicit order > 1.
            "context_order": 1,
        },
        "visualization": {},
    }
}

class DPGError(Exception):
    """Base exception class for DPG-specific errors"""
    pass


@dataclass(frozen=True)
class TraceSignature:
    """A single observed sample-tree execution trace.

    Attributes:
        signature: Canonical feature-occurrence signature of the trace
            (repeated feature splits within one tree are preserved).
        predicate_sequence: Ordered predicate/leaf labels as observed in the trace.
        path_count: Number of sample-tree executions that produced this exact
            predicate sequence.
    """

    signature: Tuple[str, ...]
    predicate_sequence: Tuple[str, ...]
    path_count: int


class DecisionPredicateGraph:
    SUPPORTED_GRAPH_CONSTRUCTION_MODES = {
        "aggregated_transitions",
        "execution_trace",
    }

    """
    Main class for converting tree-based ensemble models into interpretable graphs.

    Converts the internal decision paths of a tree-based ensemble (Random Forest,
    AdaBoost, Extra Trees, …) into a compact directed graph — the
    *Decision Predicate Graph* — that exposes which feature conditions the model
    uses, how often, and in what order.
    """
    def __init__(
        self,
        model: Any,
        feature_names: Iterable[str],
        target_names: Optional[Iterable[str]] = None,
        config_file: str = "config.yaml",
        dpg_config: Optional[Dict[str, Any]] = None,
    ) -> None:
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

        # Normalize sklearn ensemble models for consistent tree structure
        model = SklearnEnsembleNormalizer.normalize(model)

        # Initialize attributes
        self.model = model
        self.feature_names = list(feature_names)
        self.target_names = list(target_names) if target_names is not None else None
        
        # Get config values with defaults
        dpg_config_section = config.get('dpg', {})
        default_config = dpg_config_section.get('default', {})

        self.perc_var = default_config.get('perc_var', DEFAULT_DPG_CONFIG["dpg"]["default"]["perc_var"])
        self.decimal_threshold = default_config.get('decimal_threshold', DEFAULT_DPG_CONFIG["dpg"]["default"]["decimal_threshold"])
        self.n_jobs = default_config.get('n_jobs', DEFAULT_DPG_CONFIG["dpg"]["default"]["n_jobs"])
        graph_construction_config = dpg_config_section.get('graph_construction', {})
        self.graph_construction_mode = graph_construction_config.get(
            'mode',
            DEFAULT_DPG_CONFIG["dpg"]["graph_construction"]["mode"],
        )
        self.context_order = graph_construction_config.get(
            "context_order",
            DEFAULT_DPG_CONFIG["dpg"]["graph_construction"]["context_order"],
        )

        # Validate required config values
        if self.perc_var is None:
            raise DPGError("perc_var not found in DPG config")
        if self.decimal_threshold is None:
            raise DPGError("decimal_threshold not found in DPG config")
        if self.decimal_threshold != "auto" and (
            isinstance(self.decimal_threshold, bool)
            or not isinstance(self.decimal_threshold, int)
            or self.decimal_threshold < 0
        ):
            raise DPGError("decimal_threshold must be a non-negative integer or 'auto'")
        if self.n_jobs is None:
            raise DPGError("n_jobs not found in DPG config")
        if self.graph_construction_mode not in self.SUPPORTED_GRAPH_CONSTRUCTION_MODES:
            supported_modes = ", ".join(sorted(self.SUPPORTED_GRAPH_CONSTRUCTION_MODES))
            raise DPGError(
                f"Unsupported graph construction mode '{self.graph_construction_mode}'. "
                f"Supported modes are: {supported_modes}"
            )
        if self.context_order != "auto" and (
            isinstance(self.context_order, bool)
            or not isinstance(self.context_order, int)
            or self.context_order <= 0
        ):
            raise DPGError("context_order must be a positive integer or 'auto'")
        if (
            isinstance(self.context_order, int)
            and self.context_order > 1
            and self.graph_construction_mode != "execution_trace"
        ):
            raise DPGError(
                "context_order > 1 requires mode='execution_trace'"
            )

        print(
            "DPG initialized with "
            f"perc_var={self.perc_var}, "
            f"decimal_threshold={self.decimal_threshold}, "
            f"n_jobs={self.n_jobs}, "
            f"graph_construction_mode={self.graph_construction_mode}, "
            f"context_order={self.context_order}"
        )
        # Store visualization config for use by utils
        self.visualization_config = dpg_config_section.get('visualization', DEFAULT_DPG_CONFIG["dpg"]["visualization"])

        # Trace artefacts (populated only in "execution_trace" mode; reset on every fit())
        self._trace_consistent_lrc: Dict[str, float] = {}
        self._trace_consistent_trc: Dict[str, Set[str]] = {}
        self._trace_signatures: List[TraceSignature] = []
        self._resolved_decimal_threshold: Optional[int] = (
            self.decimal_threshold if isinstance(self.decimal_threshold, int) else None
        )
        self._resolved_context_order: int | float = 1
        self._context_order_history: Dict[int | float, int] = {}
        self._node_context_by_id: Dict[str, Tuple[str, ...]] = {}
        self._node_label_by_id: Dict[str, str] = {}

    def fit(self, X_train: Any) -> Any:
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

        # Reset trace artefacts on every fit so no state leaks across refits.
        self._trace_consistent_lrc = {}
        self._trace_consistent_trc = {}
        self._trace_signatures = []
        self._node_context_by_id = {}
        self._node_label_by_id = {}

        log_df = self._extract_trace_log(X_train)

        print(f'Total of paths: {len(log_df["case:concept:name"].unique())}')
        print('Building DPG...')
        if self.graph_construction_mode == "execution_trace":
            self._build_trace_artifacts(log_df)
            traces = self._trace_sequences(log_df)
            if self.context_order == "auto":
                self._resolved_context_order, self._context_order_history = resolve_context_order(traces)
            else:
                self._resolved_context_order = self.context_order
                self._context_order_history = {
                    k: self._local_context_violations(traces, k)
                    for k in range(1, int(self.context_order) + 1)
                }
            if self._resolved_context_order == 1:
                dfg = self.discover_dfg_execution_trace(log_df)
            else:
                dfg = self.discover_dfg_context(log_df, self._resolved_context_order)
        else:
            self._resolved_context_order = 1
            self._context_order_history = {1: 0}
            if self.perc_var > 0:
                log_df = self.filter_log(log_df)
            dfg = self.discover_dfg(log_df)

        print('Extracting graph...')
        return self.generate_dot(dfg)

    def _extract_trace_log(self, X_train: Any) -> pd.DataFrame:
        """
        Extract the raw execution trace log for all samples.

        Args:
            X_train: Training data (n_samples, n_features)

        Returns:
            pd.DataFrame: Raw trace log with case id and event columns
        """
        self._resolve_decimal_threshold(X_train)
        X_array = np.asarray(X_train)
        # Extract decision paths (parallel or sequential)
        if self.n_jobs == 1:
            log = Parallel(n_jobs=self.n_jobs)(
                delayed(self.tracing_ensemble)(i, sample) for i, sample in tqdm(list(enumerate(X_array)), total=len(X_array))
            )
        else:
            log = Parallel(n_jobs=self.n_jobs)(
                delayed(self.tracing_ensemble_parallel)(i, sample) for i, sample in tqdm(list(enumerate(X_array)), total=len(X_array))
            )

        log = [item for sublist in log for item in sublist]
        return pd.DataFrame(log, columns=["case:concept:name", "concept:name"])

    @staticmethod
    def _decimal_places(value: Any) -> int:
        """Return decimal places represented by a finite input value."""
        from decimal import Decimal

        numeric = float(value)
        if not np.isfinite(numeric):
            return 0
        if numeric.is_integer():
            return 0
        exponent = Decimal(str(numeric)).as_tuple().exponent
        return max(0, -int(exponent)) if isinstance(exponent, int) else 0

    def _resolve_decimal_threshold(self, X_train: Any) -> int:
        """Resolve ``decimal_threshold='auto'`` from data precision and audit it."""
        if self.decimal_threshold != "auto":
            self._resolved_decimal_threshold = int(self.decimal_threshold)
            return self._resolved_decimal_threshold

        values = np.asarray(X_train)
        precision = 0
        for value in values.reshape(-1):
            precision = max(precision, self._decimal_places(value))
        resolved = precision + 1
        self._resolved_decimal_threshold = resolved

        offending_features = set()
        for tree in self.model.estimators_:
            tree_ = tree.tree_
            for node, feature_index in enumerate(tree_.feature):
                if feature_index < 0:
                    continue
                threshold = float(tree_.threshold[node])
                if not np.isclose(threshold, round(threshold, resolved), rtol=0.0, atol=1e-12):
                    offending_features.add(self.feature_names[feature_index])
        if offending_features:
            names = ", ".join(sorted(offending_features))
            warnings.warn(
                "decimal_threshold='auto' found thresholds off the data-derived "
                f"{resolved}-decimal grid for feature(s): {names}; exact routing "
                "is retained and only predicate labels are rounded.",
                RuntimeWarning,
                stacklevel=2,
            )
        return resolved

    def get_decimal_threshold(self) -> int:
        """Return the effective precision used by the last trace extraction."""
        if self._resolved_decimal_threshold is None:
            raise DPGError("decimal_threshold='auto' is resolved when fit() is called")
        return self._resolved_decimal_threshold

    def _trace_sequences(self, log: Any) -> List[Tuple[str, ...]]:
        return [
            tuple(group["concept:name"].tolist())
            for _, group in log.groupby("case:concept:name", sort=False)
        ]

    @staticmethod
    def _local_context_violations(traces: List[Tuple[str, ...]], k: int) -> int:
        from dpg.context_order import local_violations

        return local_violations(traces, k)

    def _leaf_class_label(self, tree_index: int, tree_: Any, node_index: int) -> str:
        """Return the class label for a classifier leaf node."""
        gb_class_index = SklearnEnsembleNormalizer.get_tree_class_index(self.model, tree_index)
        if gb_class_index is not None:
            pred_class = gb_class_index
        elif isinstance(self.model, GradientBoostingClassifier) and getattr(self.model, "n_classes_", 0) == 2:
            leaf_score = float(tree_.value[node_index][0][0])
            pred_class = 1 if leaf_score > 0 else 0
        else:
            pred_class = int(tree_.value[node_index].argmax())

        if self.target_names is not None:
            pred_class = self.target_names[pred_class]
        elif hasattr(self.model, "classes_"):
            pred_class = self.model.classes_[pred_class]
        return f"Class {pred_class}"

    def tracing_ensemble(self, case_id: int, sample: Any) -> Generator[List[str], None, None]:
        """
        Extract decision path for a single sample (generator version).
        
        Args:
            case_id: Sample identifier
            sample: Feature values (1D array)
            
        Yields:
            List[str]: Path segments as [prefix, decision/prediction]
        """
        label_extractor = (
            self._trace_tree_labels_legacy
            if self.graph_construction_mode == "aggregated_transitions"
            else self._trace_tree_labels
        )
        for i, tree in enumerate(self.model.estimators_):
            prefix = f"sample{case_id}_dt{i}"
            for condition in label_extractor(i, tree, sample):
                yield [prefix, condition]

    def tracing_ensemble_parallel(self, case_id: int, sample: Any) -> List[List[str]]:
        """
        Extract decision path for a single sample (list version for parallel workers).

        Args:
            case_id: Sample identifier used to name each path prefix.
            sample: Feature values array, shape ``(n_features,)``.

        Returns:
            List of ``[prefix, event]`` pairs representing the full decision path
            across all trees in the ensemble.
        """
        label_extractor = (
            self._trace_tree_labels_legacy
            if self.graph_construction_mode == "aggregated_transitions"
            else self._trace_tree_labels
        )
        result = []
        for i, tree in enumerate(self.model.estimators_):
            prefix = f"sample{case_id}_dt{i}"
            for condition in label_extractor(i, tree, sample):
                result.append([prefix, condition])
        return result

    def _trace_tree_labels_legacy(
        self, tree_index: int, tree: Any, sample: Any
    ) -> List[str]:
        """Return labels using the pre-0.3.0 rounded-threshold traversal.

        The aggregated-transitions graph historically rounded each tree
        threshold before selecting a child.  Keep that behavior for the
        legacy path so its graph weights remain backward-compatible.  The
        execution-trace path uses :meth:`_trace_tree_labels`, which follows
        sklearn's native routing exactly.
        """
        sample_array = np.asarray(sample).reshape(-1)
        tree_ = tree.tree_
        node_index = 0
        labels: List[str] = []
        effective_decimal = self.get_decimal_threshold()

        while True:
            left = int(tree_.children_left[node_index])
            right = int(tree_.children_right[node_index])
            if left == right:
                if is_regressor(self.model):
                    pred = round(float(tree_.value[node_index][0][0]), 2)
                    labels.append(f"Pred {pred}")
                else:
                    labels.append(self._leaf_class_label(tree_index, tree_, node_index))
                return labels

            feature_index = int(tree_.feature[node_index])
            threshold = round(float(tree_.threshold[node_index]), effective_decimal)
            feature_name = self.feature_names[feature_index]
            if sample_array[feature_index] <= threshold:
                labels.append(f"{feature_name} <= {threshold}")
                node_index = left
            else:
                labels.append(f"{feature_name} > {threshold}")
                node_index = right

    def _trace_tree_labels(self, tree_index: int, tree: Any, sample: Any) -> List[str]:
        """Return the executed labels using sklearn's native routing decisions.

        Threshold rounding is deliberately applied only while formatting the
        predicate.  The child branch comes from ``decision_path`` so a rounded
        label can never change the recorded execution path.
        """
        sample_array = np.asarray(sample).reshape(1, -1)
        tree_ = tree.tree_
        decision_path = tree.decision_path(sample_array)
        path = decision_path.indices[decision_path.indptr[0] : decision_path.indptr[1]]
        leaf_id = int(tree.apply(sample_array)[0])
        labels: List[str] = []
        effective_decimal = self.get_decimal_threshold()

        for position, node_index in enumerate(path):
            if int(node_index) == leaf_id:
                break
            feature_index = int(tree_.feature[node_index])
            threshold = round(float(tree_.threshold[node_index]), effective_decimal)
            went_left = int(path[position + 1]) == int(tree_.children_left[node_index])
            operator = "<=" if went_left else ">"
            labels.append(f"{self.feature_names[feature_index]} {operator} {threshold}")

        if is_regressor(self.model):
            pred = round(float(tree_.value[leaf_id][0][0]), 2)
            labels.append(f"Pred {pred}")
        else:
            labels.append(self._leaf_class_label(tree_index, tree_, leaf_id))
        return labels
            

    def filter_log(self, log: Any) -> Any:
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

    def discover_dfg(self, log: Any) -> Dict[Tuple[str, str], int]:
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

    def discover_dfg_execution_trace(self, log: Any) -> Dict[Tuple[str, str], int]:
        """
        Build a directed frequency graph directly from the raw execution trace.

        If ``perc_var > 0``, infrequent edges are removed using a minimum edge
        count of ``total_cases * perc_var`` where ``total_cases`` is the number
        of unique case ids in the raw trace log.

        Args:
            log: Raw DataFrame of decision paths

        Returns:
            Dict[tuple, int]: Edge frequencies as {(source, target): count}
        """
        dfg = self.discover_dfg(log)
        if self.perc_var <= 0:
            return dfg

        min_count = log["case:concept:name"].nunique() * self.perc_var
        return {
            edge: count for edge, count in dfg.items()
            if count >= min_count
        }

    @staticmethod
    def _context_node(label_sequence: Tuple[str, ...], index: int, k: int) -> Any:
        label = label_sequence[index]
        if str(label).startswith(("Class ", "Pred ")):
            # Terminal outcomes are deliberately never contextualised.  A
            # shared sink cannot create a new path because it has no outgoing
            # edges, and keeps one sink per class/prediction invariant.
            return ("sink", str(label))
        return (
            "ctx",
            tuple(label_sequence[max(0, index - k + 1) : index + 1]),
        )

    @staticmethod
    def _context_node_info(node: Any) -> Tuple[str, Tuple[str, ...]]:
        kind, value = node
        if kind == "sink":
            return str(value), ()
        context = tuple(value)
        return context[-1], context

    def discover_dfg_context(self, log: Any, context_order: int) -> Dict[Tuple[Any, Any], int]:
        """Build a context-aware DFG from observed execution traces.

        The graph is still pooled into one DPG.  Only non-terminal predicate
        identity changes: it is the last ``context_order`` executed labels.
        Class outcomes remain one shared sink per class.
        """
        if context_order <= 1:
            return self.discover_dfg_execution_trace(log)

        dfg: Dict[Tuple[Any, Any], int] = {}
        for _, trace_df in log.groupby("case:concept:name", sort=False):
            labels = tuple(trace_df["concept:name"].tolist())
            nodes = [self._context_node(labels, i, context_order) for i in range(len(labels))]
            for source, target in zip(nodes, nodes[1:]):
                edge = (source, target)
                dfg[edge] = dfg.get(edge, 0) + 1

        if self.perc_var <= 0:
            return dfg
        min_count = log["case:concept:name"].nunique() * self.perc_var
        return {edge: count for edge, count in dfg.items() if count >= min_count}

    def get_context_order(self) -> int | float:
        """Return the effective context order from the last ``fit`` call."""
        return self._resolved_context_order

    def get_context_order_history(self) -> Dict[int | float, int]:
        """Return local recombination violations measured for each tested k."""
        return dict(self._context_order_history)

    def get_node_context(self, node: Any) -> Tuple[str, ...]:
        """Return the predicate context associated with a graph node.

        Sinks and all nodes in a k=1 graph return an empty tuple.  ``node`` is
        the node identifier returned by :meth:`to_networkx`.
        """
        if hasattr(node, "__str__"):
            return tuple(self._node_context_by_id.get(str(node), ()))
        return ()

    def get_node_ids_for_trace(self, labels: Iterable[str]) -> List[str]:
        """Map one executed label sequence to fitted graph node identifiers."""
        labels_tuple = tuple(labels)
        k = self.get_context_order()
        if k == 1:
            keys = list(labels_tuple)
        else:
            keys = [
                self._context_node(labels_tuple, index, int(k))
                for index in range(len(labels_tuple))
            ]
        return [self._node_id_for_key(key) for key in keys]

    def get_predicate_lrc(self, graph: Any) -> Dict[str, float]:
        """Aggregate unweighted node LRC scores back to predicate labels.

        At k>1 a predicate can occupy multiple contextual nodes.  The shipped
        aggregation is a sum, so predicates receive credit for every context
        in which they occur.
        """
        scores: Dict[str, float] = defaultdict(float)
        for node, data in graph.nodes(data=True):
            label = data.get("predicate")
            if label is None or not self._is_predicate_label(label):
                continue
            scores[label] += float(nx.local_reaching_centrality(graph, node, weight=None))
        return dict(scores)

    @staticmethod
    def _node_id_for_key(key: Any) -> str:
        stable_key = key if isinstance(key, str) else repr(key)
        return str(int(hashlib.sha1(stable_key.encode()).hexdigest(), 16))

    _PREDICATE_LABEL_RE = re.compile(
        r"^\s*(.+?)\s*(<=|>)\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$"
    )

    @classmethod
    def _is_predicate_label(cls, label: str) -> bool:
        """Return True if a trace label is a decision predicate (not a leaf/class label)."""
        return bool(cls._PREDICATE_LABEL_RE.match(label))

    @classmethod
    def _feature_signature_token(cls, label: str) -> str:
        """Return the feature-signature token for a trace label (feature name for
        predicates, the label itself for leaf/class labels)."""
        match = cls._PREDICATE_LABEL_RE.match(label)
        return match.group(1) if match else label

    def _build_trace_artifacts(self, log_df: Any) -> None:
        """
        Build trace-consistent artefacts from the raw execution-trace log.

        These artefacts are derived exclusively from single, observed
        sample-tree executions: a downstream relation or path is only recorded
        if it was witnessed within one same-trace sequence. This avoids the
        cross-trace "phantom path" issue that can arise when reading multi-hop
        paths off the pooled aggregated graph.

        Populates ``self._trace_signatures``, ``self._trace_consistent_trc``
        and ``self._trace_consistent_lrc``. Only raw sequences are consumed
        here; no raw per-trace records are retained afterwards.

        Note on ``perc_var``: this method always consumes the *unfiltered*
        raw trace log passed in from ``fit()``. ``perc_var`` filtering in
        ``execution_trace`` mode is applied afterwards, only to the pooled
        graph's edges (see ``discover_dfg_execution_trace``) — it never
        removes a trace artefact. A predicate can therefore appear in
        ``get_trace_consistent_lrc()`` / ``get_trace_signatures()`` even if
        every pooled edge it participates in was filtered out of the
        visualised graph. This is intentional: trace artefacts are meant to
        stay auditable evidence, independent of display-oriented filtering.
        """
        signature_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        signature_sequence: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
        downstream: Dict[str, Set[str]] = defaultdict(set)
        all_labels: Set[str] = set()

        grouped = log_df.groupby("case:concept:name", sort=False)
        for _, trace_df in grouped:
            sequence = tuple(trace_df["concept:name"].values)
            if not sequence:
                continue
            all_labels.update(sequence)

            signature = tuple(self._feature_signature_token(label) for label in sequence)
            signature_counts[signature] += 1
            signature_sequence.setdefault(signature, sequence)

            for i, label in enumerate(sequence):
                if not self._is_predicate_label(label):
                    continue
                downstream[label].update(sequence[i + 1:])

        self._trace_signatures = [
            TraceSignature(
                signature=sig,
                predicate_sequence=signature_sequence[sig],
                path_count=count,
            )
            for sig, count in signature_counts.items()
        ]
        self._trace_consistent_trc = dict(downstream)

        denom = max(len(all_labels) - 1, 1)
        self._trace_consistent_lrc = {
            label: len(downs) / denom for label, downs in downstream.items()
        }

    def get_trace_consistent_lrc(self) -> Dict[str, float]:
        """
        Return trace-consistent local-reaching-centrality-like scores.

        For each predicate label, the score is the fraction (over all distinct
        labels observed across every trace) of labels found downstream of it
        within at least one single observed sample-tree execution. Unlike the
        pooled-graph NetworkX local reaching centrality, this never credits a
        predicate with reach that only exists after pooling edges from
        different traces.

        Returns:
            Dict[str, float]: Mapping of predicate label to trace-consistent
            LRC score. Empty before ``fit()`` or outside
            ``graph_construction_mode="execution_trace"``.
        """
        return dict(self._trace_consistent_lrc)

    def get_trace_consistent_trc(self) -> Dict[str, Tuple[str, ...]]:
        """
        Return observed downstream predicate sets, keyed by predicate label.

        Each value is the sorted tuple of labels that were observed
        downstream of the key label within at least one single sample-tree
        trace. A tuple (rather than a ``set``) is returned so the result is
        directly JSON-serialisable and has a stable, deterministic order.

        Returns:
            Dict[str, Tuple[str, ...]]: Mapping of predicate label to its
            observed downstream labels. Empty before ``fit()`` or outside
            ``graph_construction_mode="execution_trace"``.
        """
        return {
            label: tuple(sorted(downs))
            for label, downs in self._trace_consistent_trc.items()
        }

    def get_trace_signatures(self) -> List[TraceSignature]:
        """
        Return the aggregated set of observed sample-tree trace signatures.

        Returns:
            List[TraceSignature]: One entry per distinct observed feature
            signature, with its predicate sequence and observed count. Empty
            before ``fit()`` or outside ``graph_construction_mode="execution_trace"``.
        """
        return list(self._trace_signatures)

    def generate_dot(self, dfg: Dict[Tuple[Any, Any], int]) -> Any:
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
        
        # Build graph_attr dict, dropping unset values
        final_graph_attr = {
            "bgcolor": graph_attrs.get('bgcolor'),
            "rankdir": graph_attrs.get('rankdir'),
            "overlap": "false",
            "fontsize": "20",
        }
        final_graph_attr = {k: v for k, v in final_graph_attr.items() if v is not None}

        # Build node_attr dict, dropping unset values
        final_node_attr = {
            "shape": node_attrs.get('shape'),
        }
        final_node_attr = {k: v for k, v in final_node_attr.items() if v is not None}
        
        # Get fillcolor for regular nodes
        default_fillcolor = node_attrs.get('fillcolor')
        
        dot = graphviz.Digraph(
            "dpg",
            engine="dot",
            graph_attr=final_graph_attr,
            node_attr=final_node_attr if final_node_attr else None,
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
        for edge, weight in sorted(dfg.items(), key=lambda item: item[1]):
            source, target = edge
            for node_key in (source, target):
                node_id = self._node_id_for_key(node_key)
                if node_id in added_nodes:
                    continue
                if isinstance(node_key, str):
                    label, context = node_key, ()
                else:
                    label, context = self._context_node_info(node_key)
                self._node_label_by_id[node_id] = label
                self._node_context_by_id[node_id] = context
                tooltip = " > ".join(context) if context else label
                dot.node(
                    node_id,
                    label=_escape_dot_label(label),
                    tooltip=_escape_dot_label(tooltip),
                    dpg_context_order=str(self.get_context_order()),
                    style="filled",
                    fontsize="20",
                    fillcolor=default_fillcolor,
                )
                added_nodes.add(node_id)
            dot.edge(
                self._node_id_for_key(source),
                self._node_id_for_key(target),
                label=str(weight),
                penwidth="1",
                fontsize="18"
            )
        return dot

    def to_networkx(self, graphviz_graph: Any) -> Tuple[Any, List[List[str]]]:
        """
        Convert Graphviz graph to NetworkX format.
        
        Args:
            graphviz_graph: Input graph
            
        Returns:
            Tuple[nx.DiGraph, List]: NetworkX graph and node metadata
        """
        networkx_graph = nx.DiGraph()
        nodes_list: List[List[str]] = []
        edges: List[Tuple[str, str]] = []
        weights: Dict[Tuple[str, str], float] = {}
        node_records: Dict[str, Tuple[str, Tuple[str, ...]]] = {}

        for line in graphviz_graph.body:
            if "->" in line:
                match = re.search(r"^\s*([^\s]+)\s*->\s*([^\s]+).*?label=([0-9.eE+-]+)", line)
                if match:
                    src, dest, weight = match.groups()
                    edges.append((src, dest))
                    weights[(src, dest)] = float(weight)
                continue

            match = re.search(r'^\s*([^\s]+)\s+\[label="(.*?)"(.*?)\]', line)
            if not match:
                continue
            node_id, label, attributes = match.groups()
            context = self._node_context_by_id.get(node_id, ())
            order_match = re.search(r"dpg_context_order=([^,\]]+)", attributes)
            order = self.get_context_order()
            if order_match:
                try:
                    order = float(order_match.group(1).strip('"'))
                    if order.is_integer():
                        order = int(order)
                except ValueError:
                    pass
            node_records[node_id] = (label, context)
            nodes_list.append([node_id, label])
            networkx_graph.add_node(
                node_id,
                predicate=label,
                context=context,
                context_order=order,
            )

        for src, dest in edges:
            networkx_graph.add_edge(src, dest, weight=weights[(src, dest)])
        return networkx_graph, sorted(nodes_list, key=lambda x: x[0])
