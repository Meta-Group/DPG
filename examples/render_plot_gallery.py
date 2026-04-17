from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from dpg import (
    DPGExplainer,
    classwise_feature_bounds_from_communities,
    plot_dpg,
    plot_dpg_class_bounds_vs_dataset_feature_ranges,
    plot_dpg_constraints_overview,
    plot_lec_vs_rf_importance,
    plot_lrc_vs_rf_importance,
    plot_sample_using_bc_weights,
    plot_top_lrc_predicate_splits,
)
from dpg.visualizer import plot_dpg_communities, plot_dpg_reg


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dpg_image_examples" / "plot_gallery"


def build_explainer() -> tuple[DPGExplainer, object, object, object, List[str], RandomForestClassifier]:
    X, y = load_iris(return_X_y=True, as_frame=True)
    target_names = ["setosa", "versicolor", "virginica"]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    explainer = DPGExplainer(
        model,
        feature_names=X.columns.tolist(),
        target_names=target_names,
    )
    explanation = explainer.explain_global(X.values, communities=True)
    return explainer, explanation, X, y, target_names, model


def build_constraints_dict(explanation) -> Dict[str, Dict[str, Dict[str, float | None]]]:
    class_bounds = classwise_feature_bounds_from_communities(explanation)
    if class_bounds.empty:
        return {}

    normalized: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    grouped = class_bounds.groupby(["class_name", "feature"], as_index=False).agg(
        lower_bound=("lower_bound", "min"),
        upper_bound=("upper_bound", "max"),
    )

    for _, row in grouped.iterrows():
        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])
        cls = str(row["class_name"])
        feature = str(row["feature"])
        normalized.setdefault(cls, {})[feature] = {
            "min": lower if math.isfinite(lower) else None,
            "max": upper if math.isfinite(upper) else None,
        }
    return normalized


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    explainer, explanation, X, y, target_names, model = build_explainer()
    base_dot = copy.deepcopy(explanation.dot)
    normalized_constraints = build_constraints_dict(explanation)

    renderers: Dict[str, Callable[[], None]] = {
        "explainer_plot": lambda: explainer.plot(
            "explainer_plot",
            explanation=type(explanation)(
                graph=explanation.graph,
                nodes=explanation.nodes,
                dot=copy.deepcopy(base_dot),
                node_metrics=explanation.node_metrics,
                edge_metrics=explanation.edge_metrics,
                class_boundaries=explanation.class_boundaries,
                communities=explanation.communities,
                community_threshold=explanation.community_threshold,
            ),
            save_dir=str(OUT_DIR),
            attribute="Local reaching centrality",
            class_flag=True,
            show=False,
        ),
        "explainer_plot_communities": lambda: explainer.plot_communities(
            "explainer_plot_communities",
            explanation=type(explanation)(
                graph=explanation.graph,
                nodes=explanation.nodes,
                dot=copy.deepcopy(base_dot),
                node_metrics=explanation.node_metrics,
                edge_metrics=explanation.edge_metrics,
                class_boundaries=explanation.class_boundaries,
                communities=explanation.communities,
                community_threshold=explanation.community_threshold,
            ),
            save_dir=str(OUT_DIR),
            class_flag=True,
            show=False,
        ),
        "explainer_plot_lrc_importance": lambda: explainer.plot_lrc_importance(
            X,
            explanation=explanation,
            top_k=10,
            dataset_name="Iris",
            save_path=str(OUT_DIR / "explainer_plot_lrc_importance.png"),
            show=False,
        ),
        "explainer_plot_top_lrc_splits": lambda: explainer.plot_top_lrc_splits(
            X,
            y,
            explanation=explanation,
            top_predicates=5,
            top_features=2,
            dataset_name="Iris",
            class_names=target_names,
            save_path=str(OUT_DIR / "explainer_plot_top_lrc_splits.png"),
            show=False,
        ),
        "explainer_plot_sample_using_bc_weights": lambda: explainer.plot_sample_using_bc_weights(
            X,
            y,
            explanation=explanation,
            top_k=10,
            dataset_name="Iris",
            class_names=target_names,
            save_path=str(OUT_DIR / "explainer_plot_sample_using_bc_weights.png"),
            show=False,
        ),
        "explainer_plot_class_bounds_vs_dataset_ranges": lambda: explainer.plot_class_bounds_vs_dataset_ranges(
            X,
            y,
            explanation=explanation,
            dataset_name="Iris",
            top_features=4,
            feature_cols_per_row=2,
            save_path=str(OUT_DIR / "explainer_plot_class_bounds_vs_dataset_ranges.png"),
            show=False,
        ),
        "plot_dpg": lambda: plot_dpg(
            "plot_dpg",
            copy.deepcopy(base_dot),
            explanation.node_metrics,
            explanation.edge_metrics,
            save_dir=str(OUT_DIR),
            attribute="Local reaching centrality",
            class_flag=True,
            show=False,
        ),
        "plot_dpg_communities": lambda: plot_dpg_communities(
            "plot_dpg_communities",
            copy.deepcopy(base_dot),
            explanation.node_metrics,
            explanation.communities,
            save_dir=str(OUT_DIR),
            class_flag=True,
            show=False,
        ),
        "plot_lrc_vs_rf_importance": lambda: plot_lrc_vs_rf_importance(
            explanation,
            model,
            X,
            top_k=10,
            dataset_name="Iris",
            save_path=str(OUT_DIR / "plot_lrc_vs_rf_importance.png"),
            show=False,
        ),
        "plot_lec_vs_rf_importance": lambda: plot_lec_vs_rf_importance(
            explanation,
            model,
            X,
            top_k=10,
            dataset_name="Iris",
            save_path=str(OUT_DIR / "plot_lec_vs_rf_importance.png"),
            show=False,
        ),
        "plot_top_lrc_predicate_splits": lambda: plot_top_lrc_predicate_splits(
            explanation,
            X,
            y,
            top_predicates=5,
            top_features=2,
            dataset_name="Iris",
            class_names=target_names,
            save_path=str(OUT_DIR / "plot_top_lrc_predicate_splits.png"),
            show=False,
        ),
        "plot_sample_using_bc_weights": lambda: plot_sample_using_bc_weights(
            explanation,
            X,
            y,
            top_k=10,
            dataset_name="Iris",
            class_names=target_names,
            save_path=str(OUT_DIR / "plot_sample_using_bc_weights.png"),
            show=False,
        ),
        "plot_dpg_class_bounds_vs_dataset_feature_ranges": lambda: plot_dpg_class_bounds_vs_dataset_feature_ranges(
            explanation,
            X,
            y,
            dataset_name="Iris",
            top_features=4,
            feature_cols_per_row=2,
            save_path=str(OUT_DIR / "plot_dpg_class_bounds_vs_dataset_feature_ranges.png"),
            show=False,
        ),
        "plot_dpg_reg": lambda: plot_dpg_reg(
            "plot_dpg_reg",
            copy.deepcopy(base_dot),
            explanation.node_metrics,
            explanation.communities,
            save_dir=str(OUT_DIR),
            attribute="Local reaching centrality",
        ),
        "plot_dpg_constraints_overview": lambda: plot_dpg_constraints_overview(
            normalized_constraints=normalized_constraints,
            feature_names=X.columns.tolist(),
            class_colors_list=["#E3C800", "#F0A30A", "#FA6800"],
            output_path=str(OUT_DIR / "plot_dpg_constraints_overview.png"),
            title="Iris constraints overview",
        ),
    }

    results: List[str] = []
    failures: List[str] = []

    for name, render in renderers.items():
        try:
            render()
            results.append(name)
            print(f"OK  {name}")
        except Exception as exc:  # pragma: no cover - helper script
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"ERR {name}: {type(exc).__name__}: {exc}")

    report_path = OUT_DIR / "render_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("Rendered plots\n")
        for item in results:
            fh.write(f"OK  {item}\n")
        if failures:
            fh.write("\nFailures\n")
            for item in failures:
                fh.write(f"{item}\n")

    print(f"Saved outputs to {OUT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
