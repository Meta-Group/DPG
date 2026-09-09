"""Independent edge-case review tests for the DPG-k context resolver."""

import pytest

from dpg.context_order import resolve_context_order


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
