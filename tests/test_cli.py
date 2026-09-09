"""Tests for the packaged command-line entry point."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dpg.cli import build_parser, main


def test_cli_parser_matches_documented_defaults():
    args = build_parser().parse_args([])

    assert args.dataset == "iris"
    assert args.n_learners == 5
    assert args.model_name == "RandomForestClassifier"
    assert args.seed == 160898
    assert args.pv == 1e-9


def test_cli_reports_failure_on_insufficient_nodes(tmp_path):
    """``test_dpg`` signals this case with ``(None, None)``, not a bare ``None``."""
    with patch("dpg.cli.test_dpg", return_value=(None, None)):
        assert main(["--dataset", "iris", "--dir", str(tmp_path)]) == 1


# ---------------------------------------------------------------------------
# CLI parser coverage
# ---------------------------------------------------------------------------


class TestCLIParser:
    """Cover aliases and defaults the public CLI exposes to users."""

    def test_long_and_short_aliases_resolve_to_same_dest(self):
        long_args = build_parser().parse_args(
            ["--dataset", "iris", "--n_learners", "7", "--pv", "0.5"]
        )
        short_args = build_parser().parse_args(
            ["--ds", "iris", "--l", "7", "--pv", "0.5"]
        )
        assert long_args.dataset == short_args.dataset == "iris"
        assert long_args.n_learners == short_args.n_learners == 7
        assert long_args.pv == short_args.pv == 0.5

    def test_store_true_flags_default_to_false(self):
        args = build_parser().parse_args([])
        for store_true_field in ("plot", "communities", "clusters", "class_flag"):
            assert getattr(args, store_true_field) is False

    def test_optional_fields_default_to_none(self):
        args = build_parser().parse_args([])
        assert args.target_column is None
        assert args.attribute is None
        assert args.threshold_clusters is None


# ---------------------------------------------------------------------------
# CLI main() -- happy path
# ---------------------------------------------------------------------------


class TestCLIMainSuccess:
    """When ``test_dpg`` returns a valid 6-tuple, ``main`` persists it."""

    def test_main_writes_node_edge_and_graph_metric_files(self, tmp_path: Path):
        node_df = pd.DataFrame({"Node": ["n1"], "Label": ["Class 0"]})
        edge_df = pd.DataFrame({"Source": ["a"], "Target": ["b"], "Frequency": [1]})
        graph_metrics = {"nodes": 1, "edges": 1}

        with patch(
            "dpg.cli.test_dpg",
            return_value=(node_df, edge_df, graph_metrics, None, None, None),
        ):
            exit_code = main(["--dataset", "iris", "--dir", str(tmp_path)])

        assert exit_code == 0
        node_csv = tmp_path / "iris_seed160898_node_metrics.csv"
        edge_csv = tmp_path / "iris_seed160898_edge_metrics.csv"
        dpg_txt = tmp_path / "iris_seed160898_dpg_metrics.txt"
        assert node_csv.exists()
        assert edge_csv.exists()
        assert dpg_txt.exists()
        assert "nodes: 1" in dpg_txt.read_text()

    def test_main_writes_clusters_file_when_provided(self, tmp_path: Path):
        node_df = pd.DataFrame({"Node": ["n1"], "Label": ["Class 0"]})
        edge_df = pd.DataFrame({"Source": ["a"], "Target": ["b"], "Frequency": [1]})
        graph_metrics = {"nodes": 1}
        clusters = {"Clusters": {"c0": ["n1"]}}
        node_prob = {"n1": 1.0}
        confidence = {"n1": 0.5}

        with patch(
            "dpg.cli.test_dpg",
            return_value=(node_df, edge_df, graph_metrics, clusters, node_prob, confidence),
        ):
            exit_code = main(["--dataset", "iris", "--dir", str(tmp_path)])

        assert exit_code == 0
        clusters_file = tmp_path / "iris_seed160898_clusters.txt"
        assert clusters_file.exists()
        text = clusters_file.read_text()
        assert "Clusters" in text
        assert "Probability" in text
        assert "Confidence" in text

    def test_main_creates_missing_output_directory(self, tmp_path: Path):
        out = tmp_path / "nested" / "deep" / "results"
        node_df = pd.DataFrame({"Node": ["n1"], "Label": ["Class 0"]})
        edge_df = pd.DataFrame({"Source": ["a"], "Target": ["b"], "Frequency": [1]})

        with patch(
            "dpg.cli.test_dpg", return_value=(node_df, edge_df, {}, None, None, None)
        ):
            exit_code = main(["--dataset", "iris", "--dir", str(out)])

        assert exit_code == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# CLI main() -- failure handling
# ---------------------------------------------------------------------------


class TestCLIMainFailure:
    """When ``test_dpg`` signals an error, ``main`` returns 1 and writes nothing."""

    def test_main_returns_one_on_none_result(self, tmp_path: Path):
        with patch("dpg.cli.test_dpg", return_value=None):
            assert main(["--dataset", "iris", "--dir", str(tmp_path)]) == 1

    def test_main_returns_one_on_wrong_shape_result(self, tmp_path: Path):
        # 5-tuple instead of the expected 6-tuple.
        with patch("dpg.cli.test_dpg", return_value=(None, None, None, None, None)):
            assert main(["--dataset", "iris", "--dir", str(tmp_path)]) == 1

    def test_main_does_not_write_files_on_failure(self, tmp_path: Path):
        with patch("dpg.cli.test_dpg", return_value=(None, None)):
            main(["--dataset", "iris", "--dir", str(tmp_path)])
        assert list(tmp_path.iterdir()) == []
