"""Tests for the packaged command-line entry point."""

from unittest.mock import patch

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
