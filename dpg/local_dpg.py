"""Trace-indexed local Decision Predicate Graphs.

A local DPG is built only from the root-to-leaf paths that the ensemble executes
for one sample. Every edge stores the indices of the trees whose trace contains
it, so each route the explanation reports is a stored execution trace rather than
a recombination of edges observed in different trees.

When traces are merged on predicate labels (``context_order=1``), the drawn graph
can still admit source-to-sink routes that no tree executed. The graph can
therefore be resolved at the smallest local context order ``k*`` for which every
graph route is an observed trace (see :func:`dpg.context_order.resolve_context_order`).

Class support is the model's own decomposition of its output into per-tree
contributions, so the explained class is always ``model.predict(sample)``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from .context_order import _node_windows, path_violations, resolve_context_order

SOURCE = ("source",)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredicateStep:
    """One executed split test on a trace."""

    node_index: int
    feature_index: int
    feature: str
    threshold: float
    operator: str
    label: str
    value: float

    @property
    def slack(self) -> float:
        """Distance from the sample value to the split threshold, in feature units."""
        return abs(self.value - self.threshold)


@dataclass
class LocalTrace:
    """Executed root-to-leaf path of one base learner."""

    tree_index: int
    steps: List[PredicateStep]
    leaf_index: int
    outcome: str
    contribution: np.ndarray

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(step.label for step in self.steps) + (f"Class {self.outcome}",)


@dataclass
class PivotPredicate:
    """A predicate whose single-threshold crossing changes some trees' outputs.

    ``competitor_gain`` is the first-order change in the top competitor's support
    obtained by crossing the threshold in each tree that executed the predicate,
    one tree at a time, summed over those trees.
    """

    label: str
    feature_index: int
    feature: str
    threshold: float
    operator: str
    trees: List[int]
    slack: float
    competitor_gain: float
    predicted_loss: float
    crossing_value: float


@dataclass
class LocalDPG:
    """Trace-indexed local DPG for one sample."""

    sample_id: Any
    sample: np.ndarray
    model_family: str
    support_space: str
    class_names: List[str]
    traces: List[LocalTrace]
    class_support: Dict[str, float]
    predicted_class: str
    model_prediction: str
    top_competitor: Optional[str]
    support_margin: float
    alternative_mass: float
    competitor_support: float
    decomposition_residual: float
    context_order: int
    context_history: Dict[int, int]
    route_violations_k1: int
    k1_acyclic: bool
    k1_route_count: Optional[int]
    distinct_traces: int
    graph: nx.DiGraph
    pivots: List[PivotPredicate] = field(default_factory=list)

    @property
    def critical_predicate(self) -> Optional[PivotPredicate]:
        """Pivot with the largest competitor gain, if any pivot moves support."""
        return self.pivots[0] if self.pivots and self.pivots[0].competitor_gain > 0 else None

    @property
    def is_output_faithful(self) -> bool:
        return self.predicted_class == self.model_prediction

    @property
    def k1_route_precision(self) -> Optional[float]:
        """Share of source-to-sink routes of the k=1 merge that were executed.

        ``None`` when the k=1 merge contains a cycle, i.e. admits unbounded routes.
        """
        if not self.k1_route_count:
            return None
        return self.distinct_traces / self.k1_route_count

    def traces_for_class(self, class_name: str) -> List[LocalTrace]:
        return [trace for trace in self.traces if trace.outcome == class_name]

    def summary(self) -> Dict[str, Any]:
        critical = self.critical_predicate
        return {
            "sample_id": self.sample_id,
            "model_family": self.model_family,
            "predicted_class": self.predicted_class,
            "model_prediction": self.model_prediction,
            "is_output_faithful": self.is_output_faithful,
            "top_competitor": self.top_competitor,
            "predicted_support": self.class_support.get(self.predicted_class),
            "competitor_support": self.competitor_support,
            "support_margin": self.support_margin,
            "alternative_mass": self.alternative_mass,
            "decomposition_residual": self.decomposition_residual,
            "n_traces": len(self.traces),
            "distinct_traces": self.distinct_traces,
            "context_order": self.context_order,
            "route_violations_k1": self.route_violations_k1,
            "k1_acyclic": self.k1_acyclic,
            "k1_route_precision": self.k1_route_precision,
            "n_nodes": sum(1 for _, d in self.graph.nodes(data=True) if d.get("kind") == "predicate"),
            "n_edges": self.graph.number_of_edges(),
            "n_pivots": sum(1 for p in self.pivots if p.competitor_gain > 0),
            "critical_label": critical.label if critical else None,
            "critical_feature": critical.feature if critical else None,
            "critical_gain": critical.competitor_gain if critical else None,
            "critical_slack": critical.slack if critical else None,
        }

    def routes(self) -> List[Tuple[str, ...]]:
        """Distinct executed routes (predicate labels plus class sink)."""
        return sorted({trace.labels for trace in self.traces})

    def graph_routes(self) -> List[Tuple[str, ...]]:
        """Source-to-sink routes of ``graph`` expressed as predicate labels."""
        sinks = [n for n, d in self.graph.nodes(data=True) if d.get("kind") == "sink"]
        routes = set()
        for sink in sinks:
            for path in nx.all_simple_paths(self.graph, SOURCE, sink):
                routes.add(tuple(self.graph.nodes[n]["label"] for n in path[1:]))
        return sorted(routes)

    def to_graphviz(self, max_label_len: int = 40) -> Any:
        """Render the local graph with graphviz (requires the ``graphviz`` package)."""
        import graphviz

        palette = {"pred": "#b6d7a8", "comp": "#f4cccc", "other": "#eeeeee", "crit": "#ffd966"}
        critical = self.critical_predicate
        dot = graphviz.Digraph(graph_attr={"rankdir": "TB"}, node_attr={"shape": "box", "style": "rounded,filled", "fontsize": "10"})
        ids = {node: f"n{i}" for i, node in enumerate(self.graph.nodes)}
        for node, data in self.graph.nodes(data=True):
            if node == SOURCE:
                dot.node(ids[node], "x", shape="circle", fillcolor="white")
                continue
            label = data["label"][:max_label_len]
            if data["kind"] == "sink":
                cls = data["label"][len("Class "):]
                color = palette["pred"] if cls == self.predicted_class else palette["comp"] if cls == self.top_competitor else palette["other"]
                dot.node(ids[node], f"{label}\n{self.class_support.get(cls, 0.0):.2f}", fillcolor=color, shape="ellipse")
                continue
            counts = data["outcome_counts"]
            n_pred, n_comp = counts.get(self.predicted_class, 0), counts.get(self.top_competitor, 0)
            color = palette["pred"] if n_pred > n_comp else palette["comp"] if n_comp > n_pred else palette["other"]
            if critical is not None and data["predicate"] == critical.label:
                color = palette["crit"]
            dot.node(ids[node], f"{label}\n[{len(data['trees'])}]", fillcolor=color)
        for u, v, data in self.graph.edges(data=True):
            dot.edge(ids[u], ids[v], penwidth=str(0.5 + 4.0 * data["weight"]))
        return dot


# ---------------------------------------------------------------------------
# Model adapters: exact decomposition of the ensemble output per tree
# ---------------------------------------------------------------------------


def _class_index(classes: np.ndarray, values: np.ndarray) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(classes.tolist())}
    return np.array([lookup[v] for v in np.asarray(values).tolist()], dtype=int)


class _Adapter:
    """Per-tree decomposition of a fitted sklearn classifier ensemble.

    ``state`` is the additive quantity the ensemble aggregates: probabilities for
    forests and bagging, normalised weighted votes for SAMME AdaBoost, and raw
    scores for gradient boosting. ``support(state)`` maps it to class support
    (non-negative, summing to one) whose argmax is ``model.predict``.
    """

    family = "ensemble"
    space = "probability"

    def __init__(self, model: Any) -> None:
        self.model = model
        self.classes = np.asarray(model.classes_)
        self.n_classes = len(self.classes)

    def trees(self) -> List[Tuple[int, Any, Optional[np.ndarray]]]:
        return [(t, est, None) for t, est in enumerate(self.model.estimators_)]

    def state(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def support(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float)

    def contributions(self, t: int, estimator: Any, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def outcome(self, t: int, contribution: np.ndarray) -> int:
        return int(np.argmax(contribution))


class _ForestAdapter(_Adapter):
    space = "probability"

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.family = type(model).__name__
        self.n_trees = len(model.estimators_)

    def state(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x.reshape(1, -1))[0]

    def contributions(self, t: int, estimator: Any, X: np.ndarray) -> np.ndarray:
        proba = estimator.predict_proba(X)
        out = np.zeros((X.shape[0], self.n_classes))
        tree_classes = np.asarray(estimator.classes_).astype(int)
        out[:, tree_classes] = proba
        return out / self.n_trees


class _BaggingAdapter(_Adapter):
    space = "probability"
    family = "BaggingClassifier"

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.n_trees = len(model.estimators_)

    def trees(self) -> List[Tuple[int, Any, Optional[np.ndarray]]]:
        return [
            (t, est, np.asarray(features))
            for t, (est, features) in enumerate(zip(self.model.estimators_, self.model.estimators_features_))
        ]

    def state(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x.reshape(1, -1))[0]

    def contributions(self, t: int, estimator: Any, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], self.n_classes))
        tree_classes = np.asarray(estimator.classes_).astype(int)
        if hasattr(estimator, "predict_proba"):
            out[:, tree_classes] = estimator.predict_proba(X)
        else:
            out[np.arange(X.shape[0]), estimator.predict(X).astype(int)] = 1.0
        return out / self.n_trees


class _AdaBoostAdapter(_Adapter):
    """SAMME AdaBoost: class support is the weighted vote share.

    ``AdaBoostClassifier.decision_function`` is an increasing affine function of
    each class's weighted vote, so the argmax of the vote share equals
    ``model.predict``. The vote share is reported instead of ``predict_proba``.
    """

    space = "weighted_vote"
    family = "AdaBoostClassifier"

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.weights = np.asarray(model.estimator_weights_, dtype=float)
        self.total_weight = float(self.weights.sum())

    def state(self, x: np.ndarray) -> np.ndarray:
        votes = np.zeros(self.n_classes)
        for t, estimator, _ in self.trees():
            votes += self.contributions(t, estimator, x.reshape(1, -1))[0]
        return votes

    def contributions(self, t: int, estimator: Any, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], self.n_classes))
        out[np.arange(X.shape[0]), _class_index(self.classes, estimator.predict(X))] = (
            self.weights[t] / self.total_weight
        )
        return out


class _GradientBoostingAdapter(_Adapter):
    """Gradient boosting: additive raw scores mapped through the model's link."""

    space = "raw_score"
    family = "GradientBoostingClassifier"

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.binary = model.estimators_.shape[1] == 1
        self.learning_rate = float(model.learning_rate)

    def trees(self) -> List[Tuple[int, Any, Optional[np.ndarray]]]:
        stages, slots = self.model.estimators_.shape
        return [(stage * slots + slot, self.model.estimators_[stage, slot], None) for stage in range(stages) for slot in range(slots)]

    def _slot(self, t: int) -> int:
        return t % self.model.estimators_.shape[1]

    def state(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.decision_function(x.reshape(1, -1)), dtype=float).reshape(-1)

    def init_state(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model._raw_predict_init(x.reshape(1, -1)), dtype=float).reshape(-1)

    def support(self, state: np.ndarray) -> np.ndarray:
        raw = np.asarray(state, dtype=float)
        proba = self.model._loss.predict_proba(raw if self.binary else raw.reshape(1, -1))
        return np.asarray(proba, dtype=float).reshape(-1)

    def contributions(self, t: int, estimator: Any, X: np.ndarray) -> np.ndarray:
        values = self.learning_rate * estimator.predict(X)
        width = 1 if self.binary else self.n_classes
        out = np.zeros((X.shape[0], width))
        out[:, 0 if self.binary else self._slot(t)] = values
        return out

    def outcome(self, t: int, contribution: np.ndarray) -> int:
        if self.binary:
            return 1 if contribution[0] > 0 else 0
        return self._slot(t)


def _adapter_for(model: Any) -> _Adapter:
    if is_regressor(model):
        raise NotImplementedError("Local DPG class support is defined for classifiers only.")
    if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
        return _ForestAdapter(model)
    if isinstance(model, BaggingClassifier):
        return _BaggingAdapter(model)
    if isinstance(model, AdaBoostClassifier):
        return _AdaBoostAdapter(model)
    if isinstance(model, GradientBoostingClassifier):
        if isinstance(model.estimators_, list):
            raise ValueError("Pass the original GradientBoostingClassifier, not a DPG-normalised copy.")
        return _GradientBoostingAdapter(model)
    raise NotImplementedError(f"Unsupported model type for local DPG: {type(model).__name__}")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def crossing_value(value: float, threshold: float) -> float:
    """Closest float32 value on the other side of ``threshold`` from ``value``.

    sklearn trees route left when ``float32(x) <= threshold``; the returned value
    takes the opposite branch while staying as close to the threshold as float32
    allows.
    """
    # Compare in float64 as sklearn's Cython splitter does; under NEP 50 a bare
    # ``np.float32 <= float`` would be evaluated in float32.
    threshold = float(threshold)
    candidate = np.float32(threshold)
    if float(np.float32(value)) <= threshold:
        while float(candidate) <= threshold:
            candidate = np.nextafter(candidate, np.float32(np.inf))
    else:
        while float(candidate) > threshold:
            candidate = np.nextafter(candidate, np.float32(-np.inf))
    return float(candidate)


def _extract_trace(
    adapter: _Adapter,
    t: int,
    estimator: Any,
    features: Optional[np.ndarray],
    x: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    decimal_threshold: int,
) -> LocalTrace:
    x_tree = (x[features] if features is not None else x).reshape(1, -1)
    tree_ = estimator.tree_
    path = estimator.decision_path(x_tree).indices
    steps: List[PredicateStep] = []
    for node, child in zip(path, path[1:]):
        local_feature = int(tree_.feature[node])
        feature_index = int(features[local_feature]) if features is not None else local_feature
        threshold = float(tree_.threshold[node])
        operator = "<=" if int(child) == int(tree_.children_left[node]) else ">"
        name = feature_names[feature_index]
        steps.append(
            PredicateStep(
                node_index=int(node),
                feature_index=feature_index,
                feature=name,
                threshold=threshold,
                operator=operator,
                label=f"{name} {operator} {round(threshold, decimal_threshold)}",
                value=float(x[feature_index]),
            )
        )
    contribution = adapter.contributions(t, estimator, x_tree)[0]
    return LocalTrace(
        tree_index=t,
        steps=steps,
        leaf_index=int(path[-1]),
        outcome=class_names[adapter.outcome(t, contribution)],
        contribution=contribution,
    )


def _route_count(graph: nx.DiGraph) -> int:
    counts = {node: 0 for node in graph}
    counts[SOURCE] = 1
    for node in nx.topological_sort(graph):
        for successor in graph.successors(node):
            counts[successor] += counts[node]
    return int(sum(c for node, c in counts.items() if graph.out_degree(node) == 0 and node != SOURCE))


def merge_traces(traces: Sequence[LocalTrace], context_order: int, n_trees: int) -> nx.DiGraph:
    """Merge executed traces into a trace-indexed graph at a given context order.

    Nodes are contextual predicates (the last ``context_order`` executed labels)
    or class sinks; a virtual ``SOURCE`` node precedes every trace. Each node and
    edge stores the set of tree indices whose trace passes through it.
    """
    graph = nx.DiGraph()
    graph.add_node(SOURCE, kind="source", label="x", trees=set())
    for trace in traces:
        labels = trace.labels
        previous = SOURCE
        for key in _node_windows(labels, context_order):
            if not graph.has_node(key):
                if key[0] == "sink":
                    graph.add_node(key, kind="sink", label=key[1], predicate=None, context=(), trees=set(), outcome_counts=defaultdict(int))
                else:
                    context = key[1]
                    graph.add_node(key, kind="predicate", label=context[-1], predicate=context[-1], context=context, trees=set(), outcome_counts=defaultdict(int))
            node = graph.nodes[key]
            node["trees"].add(trace.tree_index)
            node["outcome_counts"][trace.outcome] += 1
            if graph.has_edge(previous, key):
                graph.edges[previous, key]["trees"].add(trace.tree_index)
            else:
                graph.add_edge(previous, key, trees={trace.tree_index})
            previous = key
    for _, _, data in graph.edges(data=True):
        data["weight"] = len(data["trees"]) / n_trees
    return graph


def _pivots(
    adapter: _Adapter,
    tree_index: Dict[int, Tuple[Any, Optional[np.ndarray]]],
    traces: Sequence[LocalTrace],
    x: np.ndarray,
    base_state: np.ndarray,
    base_support: np.ndarray,
    predicted: int,
    competitor: Optional[int],
) -> List[PivotPredicate]:
    if competitor is None:
        return []
    aggregated: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        if not trace.steps:
            continue
        estimator, features = tree_index[trace.tree_index]
        x_tree = x[features] if features is not None else x
        rows = np.repeat(x_tree.reshape(1, -1), len(trace.steps), axis=0)
        crossings = []
        for row, step in zip(rows, trace.steps):
            local_feature = int(estimator.tree_.feature[step.node_index])
            crossings.append(crossing_value(step.value, step.threshold))
            row[local_feature] = crossings[-1]
        delta = adapter.contributions(trace.tree_index, estimator, rows) - trace.contribution
        for step, change, crossed in zip(trace.steps, delta, crossings):
            support = adapter.support(base_state + change)
            entry = aggregated.setdefault(
                step.label,
                {"step": step, "trees": [], "slack": step.slack, "gain": 0.0, "loss": 0.0, "crossing": crossed},
            )
            entry["trees"].append(trace.tree_index)
            entry["gain"] += float(support[competitor] - base_support[competitor])
            entry["loss"] += float(base_support[predicted] - support[predicted])
            if step.slack < entry["slack"]:
                entry.update(slack=step.slack, crossing=crossed)
    pivots = [
        PivotPredicate(
            label=label,
            feature_index=entry["step"].feature_index,
            feature=entry["step"].feature,
            threshold=entry["step"].threshold,
            operator=entry["step"].operator,
            trees=sorted(entry["trees"]),
            slack=float(entry["slack"]),
            competitor_gain=float(entry["gain"]),
            predicted_loss=float(entry["loss"]),
            crossing_value=float(entry["crossing"]),
        )
        for label, entry in aggregated.items()
    ]
    pivots.sort(key=lambda p: (-p.competitor_gain, p.slack, p.label))
    return pivots


def build_local_dpg(
    model: Any,
    sample: Any,
    feature_names: Sequence[str],
    target_names: Optional[Sequence[str]] = None,
    context_order: Any = "auto",
    decimal_threshold: int = 6,
    sample_id: Any = 0,
    compute_pivots: bool = True,
) -> LocalDPG:
    """Build the trace-indexed local DPG of ``sample`` for a fitted classifier ensemble.

    Args:
        model: Fitted RandomForest, ExtraTrees, Bagging (tree base learners),
            AdaBoost (SAMME, tree base learners) or GradientBoosting classifier.
        sample: One feature vector.
        feature_names: Names used in predicate labels.
        target_names: Display names aligned with ``model.classes_``.
        context_order: ``"auto"`` resolves the smallest local order without
            unobserved graph routes; an integer fixes the order.
        decimal_threshold: Rounding used only to format predicate labels.
        sample_id: Identifier stored on the explanation.
        compute_pivots: Whether to evaluate single-threshold crossings.
    """
    x = np.asarray(sample, dtype=float).reshape(-1)
    if x.shape[0] != len(feature_names):
        raise ValueError(f"Sample has {x.shape[0]} features, expected {len(feature_names)}.")
    adapter = _adapter_for(model)
    class_names = [str(name) for name in (target_names if target_names is not None else adapter.classes)]
    if len(class_names) != adapter.n_classes:
        raise ValueError("target_names must align with model.classes_.")

    tree_index: Dict[int, Tuple[Any, Optional[np.ndarray]]] = {}
    traces = []
    for t, estimator, features in adapter.trees():
        tree_index[t] = (estimator, features)
        traces.append(_extract_trace(adapter, t, estimator, features, x, feature_names, class_names, decimal_threshold))

    base_state = adapter.state(x)
    support = adapter.support(base_state)
    decomposition = np.sum([trace.contribution for trace in traces], axis=0)
    if adapter.space == "raw_score":
        decomposition = decomposition + adapter.init_state(x)
    residual = float(np.max(np.abs(decomposition - base_state)))

    model_prediction = str(class_names[int(_class_index(adapter.classes, model.predict(x.reshape(1, -1)))[0])])
    predicted = class_names.index(model_prediction)
    order = [i for i in np.argsort(-support, kind="stable") if i != predicted]
    competitor = int(order[0]) if order else None

    label_traces = [trace.labels for trace in traces]
    if context_order == "auto":
        resolved, history = resolve_context_order(label_traces)
    else:
        if isinstance(context_order, bool) or not isinstance(context_order, int) or context_order < 1:
            raise ValueError("context_order must be a positive integer or 'auto'.")
        resolved = context_order
        history = {k: path_violations(label_traces, k) for k in sorted({1, resolved})}
    k1_violations = history.get(1)
    if k1_violations is None:
        k1_violations = path_violations(label_traces, 1)

    n_trees = len(traces)
    k1_graph = merge_traces(traces, 1, n_trees)
    k1_acyclic = nx.is_directed_acyclic_graph(k1_graph)
    graph = k1_graph if resolved == 1 else merge_traces(traces, int(resolved), n_trees)

    pivots = (
        _pivots(adapter, tree_index, traces, x, base_state, support, predicted, competitor)
        if compute_pivots
        else []
    )

    return LocalDPG(
        sample_id=sample_id,
        sample=x,
        model_family=adapter.family,
        support_space=adapter.space,
        class_names=class_names,
        traces=traces,
        class_support={name: float(value) for name, value in zip(class_names, support)},
        predicted_class=class_names[int(np.argmax(support))],
        model_prediction=model_prediction,
        top_competitor=class_names[competitor] if competitor is not None else None,
        support_margin=float(support[predicted] - (support[competitor] if competitor is not None else 0.0)),
        alternative_mass=float(1.0 - support[predicted]),
        competitor_support=float(support[competitor]) if competitor is not None else 0.0,
        decomposition_residual=residual,
        context_order=int(resolved),
        context_history={int(k): int(v) for k, v in history.items()},
        route_violations_k1=int(k1_violations),
        k1_acyclic=bool(k1_acyclic),
        k1_route_count=_route_count(k1_graph) if k1_acyclic else None,
        distinct_traces=len(set(label_traces)),
        graph=graph,
        pivots=pivots,
    )


def intervene(model: Any, sample: Any, pivot: PivotPredicate, target_class_index: Optional[int] = None) -> Dict[str, Any]:
    """Cross ``pivot``'s threshold in ``sample`` and re-evaluate the whole model.

    Unlike ``PivotPredicate.competitor_gain`` (first-order, trees that executed
    the predicate only), this includes every tree affected by the change.
    """
    adapter = _adapter_for(model)
    x = np.asarray(sample, dtype=float).reshape(-1)
    x_new = x.copy()
    x_new[pivot.feature_index] = pivot.crossing_value
    before = adapter.support(adapter.state(x))
    after = adapter.support(adapter.state(x_new))
    predicted_before = model.predict(x.reshape(1, -1))[0]
    predicted_after = model.predict(x_new.reshape(1, -1))[0]
    result = {
        "support_before": before,
        "support_after": after,
        "prediction_before": predicted_before,
        "prediction_after": predicted_after,
        "flipped": bool(predicted_before != predicted_after),
        "perturbation": float(abs(x_new[pivot.feature_index] - x[pivot.feature_index])),
    }
    if target_class_index is not None:
        result["target_delta"] = float(after[target_class_index] - before[target_class_index])
    return result
