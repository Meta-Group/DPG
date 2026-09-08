"""Analysis-only LRC aggregation variants, for E5's alignment comparison.

`DecisionPredicateGraph.get_predicate_lrc()` (dpg/core.py) ships the
*production* aggregation documented in CHANGELOG 0.3.0: an unweighted sum of
each contextual node's local reaching centrality, grouped by predicate label.
At `context_order=1` a predicate maps to exactly one node, so the choice is
moot there; at k>1 a predicate can occupy several contextual nodes and the
aggregation becomes a real modelling decision.

`max` and `weighted_sum` below exist only to support the E5 comparison in
`scripts/run_dpg030_e5_lrc.py`. They are deliberately NOT added to the public
`dpg` package: shipping a second aggregation is a semantic decision about
what "predicate LRC" means, not an implementation detail, and picking one by
which correlates best with `feature_importances_` would make the comparison
circular (see the K8 note in `.agents/DPG_0.3.0_execution_plan.md` / the
attached experimental plan: aggregation must be fixed by principle, then
measured -- not chosen post hoc). `sum` is reproduced here only so all three
can be computed from one fitted graph without divergence from the production
function; `test_e5_lrc_aggregation.py` asserts it matches
`get_predicate_lrc()` exactly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import networkx as nx

from dpg.core import DecisionPredicateGraph

AGGREGATIONS = ("sum", "max", "weighted_sum")


def _node_traffic(graph: nx.DiGraph, node: str) -> float:
    """Execution frequency reaching a node: in-edge weight, or out-edge
    weight for a root with no incoming edges."""
    weight = sum(data.get("weight", 0.0) for _, _, data in graph.in_edges(node, data=True))
    if weight == 0.0:
        weight = sum(data.get("weight", 0.0) for _, _, data in graph.out_edges(node, data=True))
    return weight


def predicate_lrc(graph: nx.DiGraph, how: str) -> Dict[str, float]:
    """Aggregate unweighted node LRC to predicate labels using ``how``.

    ``how="sum"`` matches ``DecisionPredicateGraph.get_predicate_lrc()``.
    """
    if how not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation {how!r}; expected one of {AGGREGATIONS}")

    sums: Dict[str, float] = defaultdict(float)
    maxima: Dict[str, float] = {}
    for node, data in graph.nodes(data=True):
        label = data.get("predicate")
        if label is None or not DecisionPredicateGraph._is_predicate_label(label):
            continue
        value = float(nx.local_reaching_centrality(graph, node, weight=None))
        if how == "weighted_sum":
            sums[label] += value * _node_traffic(graph, node)
        elif how == "sum":
            sums[label] += value
        elif how == "max":
            if label not in maxima or value > maxima[label]:
                maxima[label] = value

    return dict(maxima) if how == "max" else dict(sums)


def feature_scores(predicate_scores: Dict[str, float], feature_names: Iterable[str]) -> Dict[str, float]:
    """Sum predicate-level scores onto their parsed feature name.

    Mirrors ``feature_importances_``'s shape: one score per feature, summed
    over every predicate (contextual node, in the sum/weighted_sum case)
    that splits on it.
    """
    totals = {name: 0.0 for name in feature_names}
    for label, value in predicate_scores.items():
        feature = DecisionPredicateGraph._feature_signature_token(label)
        if feature in totals:
            totals[feature] += value
    return totals
