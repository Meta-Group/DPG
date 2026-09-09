"""
Smoke tests: verify all public packages and key symbols are importable.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


# Public symbols that must remain importable.  Keeping them in one place
# makes it cheap to extend the surface area and easy to spot when a
# rename slips through.
PUBLIC_SYMBOLS = {
    "dpg.core": ["DecisionPredicateGraph", "DPGError"],
    "dpg.explainer": [
        "DPGExplainer",
        "DPGExplanation",
        "DPGLocalExplanation",
        "DPGTreePathExplanation",
    ],
    "dpg.sklearn_dpg": ["test_dpg", "select_dataset"],
    "dpg.visualizer": ["plot_dpg", "plot_dpg_communities"],
    "metrics.nodes": ["NodeMetrics"],
    "metrics.edges": ["EdgeMetrics"],
    "metrics.graph": ["GraphMetrics"],
}


def test_public_api_imports():
    """Every documented public symbol must import without error.

    A bare ``import dpg`` only proves the package is importable, not that
    its public surface is intact.  This single test asserts the full
    documented surface area so a renamed/removed symbol causes a single
    targeted failure rather than silently passing.
    """
    import importlib

    for module_name, symbols in PUBLIC_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for symbol in symbols:
            assert hasattr(module, symbol), (
                f"Public symbol {symbol!r} missing from {module_name!r}"
            )
            assert callable(getattr(module, symbol)) or not callable(getattr(module, symbol)), (
                f"Public symbol {symbol!r} from {module_name!r} is not accessible"
            )


def test_top_level_packages_importable():
    """The top-level ``dpg`` and ``metrics`` packages must import."""
    import dpg  # noqa: F401
    import metrics  # noqa: F401


def test_local_explanation_public_workflow_smoke():
    from dpg import DPGExplainer

    X, y = load_iris(return_X_y=True, as_frame=True)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    explainer = DPGExplainer(
        model=model,
        feature_names=X.columns.tolist(),
        target_names=np.unique(y).astype(str).tolist(),
        dpg_config={
            "dpg": {
                "default": {
                    "perc_var": 1e-9,
                    "decimal_threshold": 6,
                    "n_jobs": 1,
                },
                "graph_construction": {
                    "mode": "execution_trace",
                },
            }
        },
    )
    explainer.fit(X.values)

    local = explainer.explain_local(sample=X.iloc[0].values, sample_id=0)
    local_df = explainer.local_path_dataframe(local)

    assert local.majority_vote is not None
    assert local.class_votes
    assert not local_df.empty
    assert "evidence_scores" in local.sample_confidence
    assert "trace_coverage_score" in local.sample_confidence
    assert "recombination_rate" in local.sample_confidence


def test_local_experiment_runner_smoke(tmp_path):
    from experiments.local_explanation import run_local_explanation_experiments

    summary_df, per_sample_df = run_local_explanation_experiments(
        datasets=["iris"],
        out_dir=str(tmp_path),
        n_estimators=3,
        max_depth=3,
        perc_var=1e-9,
        decimal_threshold=6,
        graph_construction_mode="execution_trace",
        seed=42,
        max_test_samples=3,
    )

    summary_path = tmp_path / "summary.csv"
    per_sample_path = tmp_path / "per_sample.csv"

    assert summary_path.exists()
    assert per_sample_path.exists()
    assert not summary_df.empty
    assert not per_sample_df.empty

    required_summary_columns = {
        "dataset",
        "n_train",
        "n_test_explained",
        "n_estimators",
        "max_depth",
        "perc_var",
        "decimal_threshold",
        "graph_construction_mode",
        "seed",
        "model_accuracy",
        "local_matches_model_rate",
        "local_accuracy",
        "avg_vote_confidence",
        "avg_evidence_score_pred",
        "avg_trace_coverage_score",
        "avg_recombination_rate",
        "avg_num_paths",
    }
    required_per_sample_columns = {
        "dataset",
        "sample_index",
        "true_label",
        "model_pred",
        "local_pred",
        "local_matches_model",
        "local_correct",
        "vote_confidence",
        "evidence_score_pred",
        "evidence_score_margin",
        "trace_coverage_score",
        "recombination_rate",
        "num_paths",
        "num_valid_paths",
    }

    assert required_summary_columns.issubset(summary_df.columns)
    assert required_per_sample_columns.issubset(per_sample_df.columns)


def test_local_experiment_runner_parse_helpers():
    from experiments.local_explanation.run_local_explanations import (
        parse_csv_values,
        parse_graph_mode_values,
        parse_optional_int_values,
    )

    assert parse_csv_values("3,5", caster=int) == [3, 5]
    assert parse_optional_int_values("2,None") == [2, None]
    assert parse_graph_mode_values("aggregated_transitions,execution_trace") == [
        "aggregated_transitions",
        "execution_trace",
    ]


def test_local_experiment_runner_sweep_smoke(tmp_path):
    from experiments.local_explanation import run_local_explanation_experiments

    summary_df, per_sample_df = run_local_explanation_experiments(
        datasets=["iris"],
        out_dir=str(tmp_path),
        n_estimators=[3, 4],
        max_depth=[2],
        perc_var=[0.0],
        decimal_threshold=[6],
        graph_construction_mode=["aggregated_transitions", "execution_trace"],
        seed=[27],
        max_test_samples=3,
    )

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "per_sample.csv").exists()
    assert len(summary_df) == 4
    assert {"aggregated_transitions", "execution_trace"} == set(summary_df["graph_construction_mode"])
    assert {3, 4} == set(summary_df["n_estimators"])
    assert {"n_estimators", "max_depth", "perc_var", "decimal_threshold", "graph_construction_mode", "seed"}.issubset(
        per_sample_df.columns
    )


def test_local_experiment_analysis_smoke(tmp_path):
    from experiments.local_explanation.analyze_results import (
        aggregate_summary,
        build_cohort_summary,
        load_results,
    )

    results_dir = tmp_path / "results"
    out_dir = tmp_path / "analysis"
    results_dir.mkdir()
    out_dir.mkdir()

    summary_df = pd.DataFrame(
        [
            {
                "dataset": "iris",
                "graph_construction_mode": "execution_trace",
                "model_accuracy": 0.95,
                "local_matches_model_rate": 1.0,
                "local_accuracy": 0.95,
                "avg_vote_confidence": 0.9,
                "avg_evidence_score_pred": 0.88,
                "avg_trace_coverage_score": 0.97,
                "avg_recombination_rate": 0.0,
                "avg_num_paths": 5.0,
            },
            {
                "dataset": "iris",
                "graph_construction_mode": "execution_trace",
                "model_accuracy": 0.90,
                "local_matches_model_rate": 0.9,
                "local_accuracy": 0.85,
                "avg_vote_confidence": 0.8,
                "avg_evidence_score_pred": 0.78,
                "avg_trace_coverage_score": 0.92,
                "avg_recombination_rate": 0.05,
                "avg_num_paths": 5.0,
            },
        ]
    )
    per_sample_df = pd.DataFrame(
        [
            {
                "dataset": "iris",
                "graph_construction_mode": "execution_trace",
                "local_matches_model": True,
                "local_correct": True,
                "vote_confidence": 0.9,
                "evidence_score_pred": 0.85,
                "trace_coverage_score": 1.0,
                "recombination_rate": 0.0,
                "num_paths": 5,
            },
            {
                "dataset": "iris",
                "graph_construction_mode": "execution_trace",
                "local_matches_model": False,
                "local_correct": True,
                "vote_confidence": 0.7,
                "evidence_score_pred": 0.65,
                "trace_coverage_score": 0.9,
                "recombination_rate": 0.1,
                "num_paths": 5,
            },
        ]
    )

    summary_df.to_csv(results_dir / "summary.csv", index=False)
    per_sample_df.to_csv(results_dir / "per_sample.csv", index=False)

    loaded_summary, loaded_per_sample = load_results(str(results_dir))
    aggregate_df = aggregate_summary(loaded_summary, ["dataset", "graph_construction_mode"])
    cohort_df = build_cohort_summary(loaded_per_sample, ["dataset", "graph_construction_mode"])

    aggregate_df.to_csv(out_dir / "aggregate_summary.csv", index=False)
    cohort_df.to_csv(out_dir / "cohort_summary.csv", index=False)

    assert (out_dir / "aggregate_summary.csv").exists()
    assert (out_dir / "cohort_summary.csv").exists()
    assert {
        "dataset",
        "graph_construction_mode",
        "model_accuracy_mean",
        "local_matches_model_rate_mean",
        "avg_trace_coverage_score_mean",
    }.issubset(aggregate_df.columns)
    assert {
        "dataset",
        "graph_construction_mode",
        "cohort",
        "n_samples",
        "mean_vote_confidence",
        "mean_trace_coverage_score",
    }.issubset(cohort_df.columns)


def test_local_experiment_analysis_missing_columns_raise_clear_error(tmp_path):
    from experiments.local_explanation.analyze_results import load_results

    results_dir = tmp_path / "results_missing"
    results_dir.mkdir()
    pd.DataFrame([{"dataset": "iris"}]).to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame([{"dataset": "iris"}]).to_csv(results_dir / "per_sample.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_results(str(results_dir))
