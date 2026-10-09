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


def path_violations(traces: Iterable[Sequence[str]], k: int | float) -> int:
    """Count pooled-graph paths that are not observed trace prefixes.

    A pooled DFG can be locally consistent while still recombining after
    several hops.  For example, observing ``A-B-C-D`` and ``X-B-C-E`` gives
    every adjacent transition a witness, but also creates the unobserved
    path ``A-B-C-E``.  This check builds a trie of observed contextual traces,
    computes each prefix state's future signature bottom-up, and verifies
    that states merged by the same contextual node have the same future
    language.  It avoids both exponential simple-path enumeration and the
    potentially large graph-by-trie product.

    The returned value is a diagnostic count, not a count of all phantom
    paths.  A value of zero is the useful acceptance condition: every graph
    path is represented by an observed trace.
    """
    materialized = [tuple(trace) for trace in traces if trace]
    if not materialized:
        return 0

    trie_children: list[dict[object, int]] = [{}]
    trie_terminal: set[int] = set()
    trie_context: list[object | None] = [None]

    for sequence in materialized:
        nodes = _node_windows(sequence, k)
        trie_node = 0
        for node in nodes:
            next_node = trie_children[trie_node].get(node)
            if next_node is None:
                next_node = len(trie_children)
                trie_children[trie_node][node] = next_node
                trie_children.append({})
                trie_context.append(node)
            trie_node = next_node
        trie_terminal.add(trie_node)

    signatures: list[tuple[bool, tuple[tuple[object, object], ...]]] = [
        (False, ())
    ] * len(trie_children)
    for trie_node in range(len(trie_children) - 1, -1, -1):
        children = tuple(
            sorted(
                (
                    (child_label, signatures[child_node])
                    for child_label, child_node in trie_children[trie_node].items()
                ),
                key=repr,
            )
        )
        signatures[trie_node] = (trie_node in trie_terminal, children)

    future_by_context: defaultdict[object, set[tuple[bool, tuple]]] = defaultdict(set)
    for trie_node in range(1, len(trie_children)):
        context = trie_context[trie_node]
        assert context is not None
        future_by_context[context].add(signatures[trie_node])

    return sum(max(0, len(futures) - 1) for futures in future_by_context.values())


def resolve_context_order(
    traces: Iterable[Sequence[str]], max_k: int | None = None
) -> tuple[int | float, dict[int | float, int]]:
    """Return the smallest order with no global trace recombination.

    ``max_k`` defaults to the longest observed trace.  At that order every
    observed prefix is distinct, which is a proof-based bound rather than a
    user-facing cap.  When a caller supplies a smaller cap, resolution must
    still succeed within that cap; otherwise ``ValueError`` is raised instead
    of returning an order whose history still contains violations.
    """
    if max_k is not None and (
        isinstance(max_k, bool) or not isinstance(max_k, int) or max_k < 1
    ):
        raise ValueError("max_k must be a positive integer or None")

    materialized = [tuple(trace) for trace in traces if trace]
    if not materialized:
        return 1, {1: 0}
    if max_k is None:
        max_k = max(len(trace) for trace in materialized)

    history: dict[int | float, int] = {}
    for k in range(1, max_k + 1):
        violations = path_violations(materialized, k)
        history[k] = violations
        if violations == 0:
            return k, history
    raise ValueError(
        f"no context order <= max_k={max_k} eliminates all global trace violations"
    )
