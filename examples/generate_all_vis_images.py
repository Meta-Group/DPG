"""Generate all visualization documentation images."""
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from dpg import (
    DPGExplainer,
    plot_lrc_vs_rf_importance,
    plot_top_lrc_predicate_splits,
    plot_dpg_class_bounds_vs_dataset_feature_ranges,
    classwise_feature_bounds_from_communities,
)
from dpg.visualizer import plot_sample_using_bc_weights

# Setup paths
VISUALIZATION_DIR = PROJECT_ROOT / "docs" / "_static" / "visualization"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


def generate_all_visualizations():
    """Generate all visualization documentation images."""
    print("Generating all visualization images...")

    # Load and train model
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

    # 1. iris_dpg.png
    print("  - iris_dpg.png")
    explainer.plot(
        "iris_dpg",
        explanation=explanation,
        save_dir=str(VISUALIZATION_DIR),
        class_flag=False,
        show=False,
    )

    # 2. iris_dpg_communities.png
    print("  - iris_dpg_communities.png")
    explainer.plot_communities(
        "iris_dpg_communities",
        explanation=explanation,
        save_dir=str(VISUALIZATION_DIR),
        class_flag=True,
        show=False,
    )

    # 3. lrc_vs_rf_importance.png
    print("  - lrc_vs_rf_importance.png")
    plot_lrc_vs_rf_importance(
        explanation,
        model,
        X,
        top_k=10,
        dataset_name="Iris",
        save_path=str(VISUALIZATION_DIR / "lrc_vs_rf_importance.png"),
        show=False,
    )

    # 4. top_lrc_predicate_splits.png
    print("  - top_lrc_predicate_splits.png")
    plot_top_lrc_predicate_splits(
        explanation,
        X,
        y,
        top_predicates=5,
        top_features=2,
        dataset_name="Iris",
        class_names=target_names,
        save_path=str(VISUALIZATION_DIR / "top_lrc_predicate_splits.png"),
        show=False,
    )

    # 5. bc_bottleneck_pca_cloud.png
    print("  - bc_bottleneck_pca_cloud.png")
    plot_sample_using_bc_weights(
        explanation,
        X,
        y,
        top_k=10,
        dataset_name="Iris",
        class_names=target_names,
        save_path=str(VISUALIZATION_DIR / "bc_bottleneck_pca_cloud.png"),
        show=False,
    )

    # 6. dpg_vs_dataset_feature_ranges.png
    print("  - dpg_vs_dataset_feature_ranges.png")
    plot_dpg_class_bounds_vs_dataset_feature_ranges(
        explanation,
        X,
        y,
        dataset_name="Iris",
        top_features=4,
        feature_cols_per_row=2,
        save_path=str(VISUALIZATION_DIR / "dpg_vs_dataset_feature_ranges.png"),
        show=False,
    )

    print("\nAll visualization images generated successfully!")


if __name__ == "__main__":
    try:
        generate_all_visualizations()
    except Exception as e:
        print(f"Error generating images: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
