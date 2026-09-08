"""Fast resolution of the smallest context order without path enumeration."""

from collections import defaultdict
import math
from typing import Iterable, Sequence


def _node_windows(sequence: Sequence[str], k: int | float) -> list[object]:
    """Represent a trace as contextual predicate nodes and class sinks."""
    nodes: list[object] = []
    for index, label in enumerate(sequence):
        if str(label).startswith(("Class ", "Pred ")):
            nodes.append(("sink", str(label)))
            continue
        if math.isinf(k):
            context = tuple(sequence[: index + 1])
        else:
            context = tuple(sequence[max(0, index - int(k) + 1) : index + 1])
        nodes.append(("ctx", context))
    return nodes


def local_violations(traces: Iterable[Sequence[str]], k: int | float) -> int:
    """Count successor combinations implied by pooling but never observed.

    The check is local: for each contextual node, every predecessor history
    must expose the same successor set as the node globally.  This is linear
    in the number of observed trace events and avoids exponential path
    enumeration in the recombining case.
    """
    successors: defaultdict[object, set[object]] = defaultdict(set)
    pair_successors: defaultdict[tuple[object, object], set[object]] = defaultdict(set)

    for sequence in traces:
        nodes = _node_windows(sequence, k)
        for index in range(len(nodes) - 1):
            source, target = nodes[index], nodes[index + 1]
            successors[source].add(target)
            if index:
                pair_successors[(nodes[index - 1], source)].add(target)

    return sum(
        len(successors[node] - seen)
        for (_, node), seen in pair_successors.items()
    )


def resolve_context_order(
    traces: Iterable[Sequence[str]], max_k: int | None = None
) -> tuple[int | float, dict[int | float, int]]:
    """Return the smallest order with no local recombination.

    ``max_k`` defaults to the longest observed trace.  At that order every
    observed prefix is distinct, which is a proof-based bound rather than a
    user-facing cap.
    """
    materialized = [tuple(trace) for trace in traces if trace]
    if not materialized:
        return 1, {1: 0}
    if max_k is None:
        max_k = max(len(trace) for trace in materialized)

    history: dict[int | float, int] = {}
    for k in range(1, max_k + 1):
        violations = local_violations(materialized, k)
        history[k] = violations
        if violations == 0:
            return k, history
    return max_k, history
