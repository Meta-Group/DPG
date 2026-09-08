"""Broader, cell-level tests for scripts/run_dpg030_e5_lrc.py.

Run after the focused tests in test_e5_lrc_aggregation.py. Exercises
run_task() directly (not the CLI/process pool) on tiny cells so this stays
fast; the full 225-cell grid is run separately, on a dedicated output path,
once the machine is free (see the E5 section of the 0.3.0 execution plan).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_dpg030_e5_lrc import AGGREGATIONS, run_task


def _row_fields(row):
    fields = {"dataset", "model", "n_estimators", "seed", "status", "error",
              "has_feature_importances", "top_k", "k1_seconds", "kauto_seconds",
              "kauto", "kauto_violations", "k1_nodes", "k1_edges", "kauto_nodes",
              "kauto_edges", "node_ratio", "git_commit"}
    for how in AGGREGATIONS:
        fields |= {f"spearman_k1_{how}", f"spearman_kauto_{how}", f"overlap_k1_{how}", f"overlap_kauto_{how}"}
    assert fields <= row.keys()


def test_random_forest_cell_is_ok_and_well_formed():
    row = run_task(("iris", "RandomForestClassifier", 5, 0, "test-commit"))
    _row_fields(row)
    assert row["status"] == "ok"
    assert row["error"] == ""
    assert row["has_feature_importances"] is True
    assert row["k1_nodes"] > 0 and row["kauto_nodes"] > 0
    assert row["kauto"] != ""
    for how in AGGREGATIONS:
        for prefix in ("k1", "kauto"):
            rho = row[f"spearman_{prefix}_{how}"]
            overlap = row[f"overlap_{prefix}_{how}"]
            assert rho == "" or -1.0 <= rho <= 1.0
            assert 0.0 <= overlap <= 1.0


def test_bagging_classifier_is_skipped_not_dropped():
    """BaggingClassifier has no feature_importances_: the row must still be
    written (never silently omitted), with graph metrics preserved and the
    alignment columns left blank."""
    row = run_task(("iris", "BaggingClassifier", 5, 0, "test-commit"))
    _row_fields(row)
    assert row["status"] == "skipped_no_feature_importances"
    assert row["has_feature_importances"] is False
    assert row["k1_nodes"] > 0 and row["kauto_nodes"] > 0
    for how in AGGREGATIONS:
        assert row[f"spearman_k1_{how}"] == ""
        assert row[f"overlap_k1_{how}"] == ""


def test_unknown_dataset_or_model_is_recorded_as_error_row():
    """run_task must never raise: every cell becomes a row, per the CSV
    failure-preservation contract used across the E-series runners."""
    row = run_task(("not-a-real-dataset", "RandomForestClassifier", 5, 0, "test-commit"))
    _row_fields(row)
    assert row["status"] == "error"
    assert row["error"] != ""
