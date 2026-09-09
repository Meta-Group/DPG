"""Independent edge-case review tests for the DPG-k context resolver."""

import pytest

import dpg.context_order as context_order_module
from dpg.context_order import path_violations, resolve_context_order


def test_empty_and_duplicate_traces_resolve_at_k_one():
    assert resolve_context_order([]) == (1, {1: 0})
    assert resolve_context_order(
        [("A", "Class 0"), ("A", "Class 0")]
    ) == (1, {1: 0})


@pytest.mark.parametrize("invalid_max_k", [0, -1, True, 1.5])
def test_max_k_requires_a_positive_integer(invalid_max_k):
    with pytest.raises(ValueError, match="max_k must be a positive integer"):
        resolve_context_order([("A", "Class 0")], max_k=invalid_max_k)


def test_insufficient_max_k_does_not_return_unresolved_order():
    traces = [
        ("A", "B", "C", "D", "Class 0"),
        ("X", "B", "C", "E", "Class 1"),
    ]

    with pytest.raises(ValueError, match="eliminates all global trace violations"):
        resolve_context_order(traces, max_k=2)


def test_sufficient_max_k_returns_zero_violation_order_and_history():
    traces = [
        ("A", "B", "C", "D", "Class 0"),
        ("X", "B", "C", "E", "Class 1"),
    ]

    resolved, history = resolve_context_order(traces, max_k=3)

    assert resolved == 3
    assert history[resolved] == 0


# ---------------------------------------------------------------------------
# path_violations -- the diagnostic at the heart of ``auto`` resolution
# ---------------------------------------------------------------------------


class TestPathViolations:
    """Edge-case coverage for ``path_violations``.

    The motivating example from the docstring is also exercised end-to-end
    via ``resolve_context_order`` -- here we test the diagnostic primitive
    in isolation so a regression can point at the right call site.
    """

    def test_empty_traces_return_zero(self):
        assert path_violations([], 1) == 0
        # Falsy traces are dropped before the trie is built.
        assert path_violations([(), ("a",)], 1) == 0

    def test_single_trace_has_no_violations_at_any_k(self):
        # A single trace has no recombination to detect.
        assert path_violations([("A", "B", "C", "Class 0")], 1) == 0
        assert path_violations([("A", "B", "C", "Class 0")], 3) == 0

    def test_pairwise_consistent_traces_become_consistent_at_higher_k(self):
        """The motivating phantom-path example from the docstring.

        ``(B)`` and ``(C)`` appear in both traces with *different* futures
        (``D`` vs ``E``), so even at k=1 there are violations: the pooled
        graph lets a sample reach ``D`` from a state that was observed to
        lead only to ``E``.  k=3 disambiguates by prefix and the trace
        language is recovered exactly.
        """
        traces = [
            ("A", "B", "C", "D", "Class 0"),
            ("X", "B", "C", "E", "Class 1"),
        ]
        # k=1: (B) and (C) each merge across traces with different futures.
        assert path_violations(traces, 1) > 0
        # k=2: only the (B,C) context still merges with diverging futures.
        assert path_violations(traces, 2) == 1
        # k=3 keys every branch by its full prefix and the language is exact.
        assert path_violations(traces, 3) == 0

    def test_infinite_k_uses_full_history_so_no_violations(self):
        """``k=math.inf`` is the documented identity bound -- it should
        recover every observed prefix as a distinct context and so never
        merge states with different futures."""
        traces = [
            ("A", "B", "C", "D", "Class 0"),
            ("X", "B", "C", "E", "Class 1"),
        ]
        assert path_violations(traces, float("inf")) == 0

    def test_sink_nodes_are_never_contextualised(self):
        """Sinks collapse to a single shared node, which cannot create a
        new path (it has no outgoing edges)."""
        traces = [
            ("A", "B", "Class 0"),
            ("X", "B", "Class 0"),
        ]
        # Both traces terminate in the same Class 0 sink, so no recombination.
        assert path_violations(traces, 1) == 0

    def test_shared_prefix_branch_is_seen_by_the_trie(self):
        # Both traces share the (A,B,C) prefix, so the trie at ctx(C) has
        # *both* ctx(D) and ctx(E) as children.  The contextual signature
        # of (C) therefore captures both children as a single future
        # language -- there is no separate contextual node with two
        # futures, hence path_violations sees zero violations.
        assert path_violations(
            [("A", "B", "C", "D", "Class 0"), ("A", "B", "C", "E", "Class 1")], 1
        ) == 0
        assert path_violations(
            [("A", "B", "C", "D", "Class 0"), ("A", "B", "C", "E", "Class 1")], 3
        ) == 0

    def test_resolved_by_distinguishing_first_symbol(self):
        # The first symbol (A vs X) splits (B) into two distinct contexts
        # at k=2; only the (B,C) window still merges and the violation
        # drops to 1, then to 0 at k=3 once every node carries a unique
        # context.
        assert path_violations(
            [("A", "B", "C", "D", "Class 0"), ("X", "B", "C", "E", "Class 1")], 2
        ) == 1
        assert path_violations(
            [("A", "B", "C", "D", "Class 0"), ("X", "B", "C", "E", "Class 1")], 3
        ) == 0


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestContextOrderModule:
    """The module surface is intentionally small but stable."""

    def test_public_exports_are_callable(self):
        # Module surface is small but stable: both helpers are reachable.
        assert callable(context_order_module.path_violations)
        assert callable(context_order_module.resolve_context_order)

    def test_history_is_a_dict_not_a_list(self):
        resolved, history = resolve_context_order([("A", "Class 0")])
        assert isinstance(history, dict)

    def test_long_history_traces_walk_k_until_zero_violations(self):
        """``resolve_context_order`` walks k=1..max_k until violations vanish,
        so the returned history contains every k actually evaluated."""
        traces = [
            ("A", "B", "C", "D", "Class 0"),
            ("X", "B", "C", "E", "Class 1"),
        ]
        resolved, history = resolve_context_order(traces)
        assert resolved == 3
        assert history[resolved] == 0
        # Every intermediate k that we evaluated is recorded.
        assert sorted(history) == [1, 2, 3]

    def test_explicit_max_k_caps_the_evaluation_window(self):
        """``max_k`` is the upper bound *and* the upper bound on the
        recorded history; k is walked only up to the supplied cap."""
        traces = [
            ("A", "B", "C", "D", "Class 0"),
            ("X", "B", "C", "E", "Class 1"),
        ]
        resolved, history = resolve_context_order(traces, max_k=5)
        assert resolved == 3
        # The function stops as soon as a k yields zero violations, so
        # the resolved key's value is zero and no larger k is recorded.
        assert history[resolved] == 0
        assert max(history.keys()) == resolved
        assert max(history.keys()) <= 5
