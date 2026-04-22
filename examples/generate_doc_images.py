"""Generate documentation images for quickstart and visualization guides."""
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from dpg import DPGExplainer, plot_class_feature_complexity, plot_lrc_vs_rf_importance, plot_top_lrc_predicate_splits

# Setup paths
QUICKSTART_DIR = PROJECT_ROOT / "docs" / "_static" / "quickstart"
VISUALIZATION_DIR = PROJECT_ROOT / "docs" / "_static" / "visualization"

QUICKSTART_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


def generate_quickstart_images():
    """Generate images for quickstart documentation."""
    print("Generating quickstart images...")

    X, y = load_iris(return_X_y=True, as_frame=True)
    target_names = ["setosa", "versicolor", "virginica"]

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    explainer = DPGExplainer(
        model,
        feature_names=X.columns.tolist(),
        target_names=target_names,
        dpg_config={
            "dpg": {
                "default": {
                    "perc_var": 1e-9,
                    "decimal_threshold": 2,
                    "n_jobs": -1,
                }
            }
        },
    )
    explanation = explainer.explain_global(X.values, communities=True)

    # Generate iris_dpg
    print("  - iris_dpg.png")
    explainer.plot(
        "iris_dpg",
        explanation=explanation,
        save_dir=str(QUICKSTART_DIR),
        class_flag=False,
        layout_template="vertical",
        label_mode="wrapped",
        readability="presentation",
        fig_size=(14, 14),
        title="Iris Decision Predicate Graph by Local Reaching Centrality",
        show=False,
    )

    # Generate lrc_vs_rf_importance
    print("  - lrc_vs_rf_importance.png")
    plot_lrc_vs_rf_importance(
        explanation,
        model,
        X,
        top_k=10,
        dataset_name="Iris",
        save_path=str(QUICKSTART_DIR / "lrc_vs_rf_importance.png"),
        show=False,
    )

    # Generate top_lrc_predicate_splits
    print("  - top_lrc_predicate_splits.png")
    plot_top_lrc_predicate_splits(
        explanation,
        X,
        y,
        top_predicates=5,
        top_features=2,
        dataset_name="Iris",
        class_names=target_names,
        save_path=str(QUICKSTART_DIR / "top_lrc_predicate_splits.png"),
        show=False,
    )


def generate_visualization_images():
    """Generate images for visualization documentation."""
    print("Generating visualization images...")

    X, y = load_iris(return_X_y=True, as_frame=True)
    target_names = ["setosa", "versicolor", "virginica"]

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    explainer = DPGExplainer(
        model,
        feature_names=X.columns.tolist(),
        target_names=target_names,
    )
    explanation = explainer.explain_global(X.values, communities=True)

    # Generate iris_dpg
    print("  - iris_dpg.png")
    explainer.plot(
        "iris_dpg",
        explanation=explanation,
        save_dir=str(VISUALIZATION_DIR),
        class_flag=False,
        layout_template="vertical",
        label_mode="wrapped",
        readability="presentation",
        fig_size=(14, 14),
        title="Iris Decision Predicate Graph by Local Reaching Centrality",
        show=False,
    )

    # Generate iris_dpg_communities
    print("  - iris_dpg_communities.png")
    explainer.plot_communities(
        "iris_dpg",
        explanation=explanation,
        save_dir=str(VISUALIZATION_DIR),
        class_flag=True,
        layout_template="vertical",
        label_mode="wrapped",
        readability="presentation",
        fig_size=(14, 14),
        title="Iris Decision Predicate Graph by Community Assignment",
        show=False,
    )

    print("  - iris_dpg_communities_communities.png")
    explainer.plot_communities(
        "iris_dpg_communities",
        explanation=explanation,
        save_dir=str(VISUALIZATION_DIR),
        class_flag=True,
        layout_template="vertical",
        label_mode="wrapped",
        readability="presentation",
        fig_size=(14, 14),
        title="Iris Decision Predicate Graph by Community Assignment",
        show=False,
    )

    # Generate lrc_vs_rf_importance
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

    # Generate top_lrc_predicate_splits
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

    # Generate bc_bottleneck_pca_cloud
    print("  - bc_bottleneck_pca_cloud.png")
    explainer.plot_sample_using_bc_weights(
        X,
        y,
        explanation=explanation,
        top_k=10,
        dataset_name="Iris",
        class_names=target_names,
        save_path=str(VISUALIZATION_DIR / "bc_bottleneck_pca_cloud.png"),
        show=False,
    )

    print("  - communities_class_feature_complexity_heatmap.png")
    print("  - communities_class_feature_complexity_bars.png")
    heat_df = explainer.class_feature_predicate_counts(explanation=explanation)
    plot_class_feature_complexity(
        heat_df=heat_df,
        dataset_name="Iris",
        class_names=target_names,
        top_n_features=4,
        save_prefix=str(VISUALIZATION_DIR / "communities_class_feature_complexity"),
        show=False,
    )


if __name__ == "__main__":
    try:
        generate_quickstart_images()
        generate_visualization_images()
        print("\nAll images generated successfully!")
    except Exception as e:
        print(f"Error generating images: {e}", file=sys.stderr)
        sys.exit(1)
