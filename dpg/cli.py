"""Packaged command-line interface for the standard DPG demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .sklearn_dpg import test_dpg


def build_parser() -> argparse.ArgumentParser:
    """Build the parser used by the installed ``dpg`` command."""
    parser = argparse.ArgumentParser(
        description="Train a sklearn tree ensemble and export its DPG metrics."
    )
    parser.add_argument(
        "--dataset",
        "--ds",
        dest="dataset",
        default="iris",
        help="sklearn dataset name or path to a CSV file",
    )
    parser.add_argument(
        "--target_column",
        default=None,
        help="target column when --dataset points to a CSV file",
    )
    parser.add_argument("--n_learners", "--l", dest="n_learners", type=int, default=5)
    parser.add_argument("--model_name", default="RandomForestClassifier")
    parser.add_argument("--dir", default="examples/results", help="output directory")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save_plot_dir", default="examples/results")
    parser.add_argument("--attribute", default=None)
    parser.add_argument("--communities", action="store_true")
    parser.add_argument("--clusters", action="store_true")
    parser.add_argument("--threshold_clusters", type=float, default=None)
    parser.add_argument("--t", type=int, default=3, help="threshold decimal precision")
    parser.add_argument("--class_flag", action="store_true")
    parser.add_argument("--seed", type=int, default=160898)
    parser.add_argument(
        "--pv",
        type=float,
        default=1e-9,
        help="minimum path frequency proportion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standard DPG workflow and write metrics to disk."""
    args = build_parser().parse_args(argv)
    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = test_dpg(
        datasets=args.dataset,
        target_column=args.target_column,
        n_learners=args.n_learners,
        perc_var=args.pv,
        decimal_threshold=args.t,
        n_jobs=-1,
        model_name=args.model_name,
        file_name=str(output_dir / f"{Path(args.dataset).stem}_seed{args.seed}_stats.txt"),
        plot=args.plot,
        save_plot_dir=args.save_plot_dir,
        attribute=args.attribute,
        communities=args.communities,
        clusters_flag=args.clusters,
        threshold_clusters=args.threshold_clusters,
        class_flag=args.class_flag,
        seed=args.seed,
    )
    # ``test_dpg`` signals "insufficient nodes for DPG analysis" with
    # ``(None, None)`` rather than a bare ``None``; guard on shape, not identity.
    if result is None or len(result) != 6:
        return 1

    df, df_edges, graph_metrics, clusters, node_prob, confidence = result
    stem = Path(args.dataset).stem
    df.to_csv(output_dir / f"{stem}_seed{args.seed}_node_metrics.csv", index=False)
    df_edges.to_csv(output_dir / f"{stem}_seed{args.seed}_edge_metrics.csv", index=False)
    with (output_dir / f"{stem}_seed{args.seed}_dpg_metrics.txt").open("w", encoding="utf-8") as handle:
        for key, value in graph_metrics.items():
            handle.write(f"{key}: {value}\n")

    if clusters is not None:
        with (output_dir / f"{stem}_seed{args.seed}_clusters.txt").open("w", encoding="utf-8") as handle:
            handle.write(f"Clusters: {clusters}\n")
            handle.write(f"Probability: {node_prob}\n")
            handle.write(f"Confidence: {confidence}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
