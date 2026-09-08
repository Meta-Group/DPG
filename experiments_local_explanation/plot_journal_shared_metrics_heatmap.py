#!/usr/bin/env python3
"""Generate the journal per-dataset shared-metrics heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/summary_selected_test.csv"
OUTPUT = ROOT / "JOURNAL_paper/per_dataset_shared_metrics_heatmap.pdf"
OUTPUTS_BY_METRIC = {
    "local_matches_model_rate": ROOT / "JOURNAL_paper/per_dataset_fidelity_heatmap.pdf",
    "local_accuracy": ROOT / "JOURNAL_paper/per_dataset_local_accuracy_heatmap.pdf",
    "shared_margin": ROOT / "JOURNAL_paper/per_dataset_margin_heatmap.pdf",
}

METHOD_ORDER = [
    ("dpg_execution_trace", "DPG-local"),
    ("dpg", "DPG-global"),
    ("shap", "TreeSHAP"),
    ("lime", "LIME"),
    ("anchors", "Anchors"),
    ("lore", "LORE"),
    ("tree_path", "Tree-path"),
]

DATASET_ORDER = [
    "banknote-authentication",
    "breast_cancer",
    "diabetes",
    "digits",
    "ionosphere",
    "iris",
    "isolet",
    "madelon",
    "phoneme",
    "qsar-biodeg",
    "segment",
    "spambase",
    "vehicle",
    "wdbc",
    "wine",
]


def method_label(row: pd.Series) -> str | None:
    for method, label in METHOD_ORDER:
        if row["method"] == method:
            return label
    return None


def metric_frame(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    plot_df = df.pivot_table(index="dataset", columns="method_label", values=value_col, aggfunc="mean")
    return plot_df.reindex(index=DATASET_ORDER, columns=[label for _, label in METHOD_ORDER])


def draw_heatmap(ax: plt.Axes, data: pd.DataFrame, title: str, annot_size: float) -> None:
    vmax = float(data.max().max())
    vmin = float(data.min().min())
    span = vmax - vmin if vmax > vmin else 1.0
    sns.heatmap(
        data,
        ax=ax,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        linewidths=0.25,
        linecolor="white",
        annot=False,
        cbar_kws={"shrink": 0.82, "pad": 0.015},
    )
    for y, dataset in enumerate(data.index):
        for x, method in enumerate(data.columns):
            value = data.loc[dataset, method]
            normalized = (float(value) - vmin) / span
            text_color = "white" if normalized < 0.48 else "black"
            ax.text(
                x + 0.5,
                y + 0.5,
                f"{float(value):.2f}",
                ha="center",
                va="center",
                fontsize=annot_size,
                color=text_color,
            )
    ax.set_title(title, fontsize=11, pad=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8.5)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8.5)


def main() -> None:
    df = pd.read_csv(INPUT)
    df = df[df["method"].isin([method for method, _ in METHOD_ORDER])].copy()
    df["method_label"] = df.apply(method_label, axis=1)

    df["shared_margin"] = df["avg_score_margin_pred_vs_competitor"]
    is_dpg = df["method"].isin({"dpg", "dpg_execution_trace"})
    df.loc[is_dpg, "shared_margin"] = df.loc[is_dpg, "avg_support_margin"]

    metrics = [
        ("local_matches_model_rate", "Fidelity"),
        ("local_accuracy", "Local accuracy"),
        ("shared_margin", "Margin / separation"),
    ]

    sns.set_theme(style="white", font_scale=0.95)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for col, title in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.9), constrained_layout=True)
        draw_heatmap(ax, metric_frame(df, col), title, annot_size=7.6)
        fig.savefig(OUTPUTS_BY_METRIC[col], bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 15.6), constrained_layout=True)

    for ax, (col, title) in zip(axes, metrics):
        draw_heatmap(ax, metric_frame(df, col), title, annot_size=8.2)

    fig.suptitle("Per-dataset shared metrics for route, rule, surrogate, and graph explainers", fontsize=13)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
