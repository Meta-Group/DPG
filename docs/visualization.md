# Visualization

DPG includes graph renderers, feature-importance comparisons, predicate-space plots,
and class-boundary views.

This page collects the available visualization entry points in one place so the
Read the Docs site has a dedicated reference for plotting.

## Overview

For most projects, the easiest workflow is:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from dpg import DPGExplainer

X, y = load_iris(return_X_y=True, as_frame=True)
model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

explainer = DPGExplainer(
    model,
    feature_names=X.columns.tolist(),
    target_names=["setosa", "versicolor", "virginica"],
)
explanation = explainer.explain_global(X.values, communities=True)
```

## Main chart types

### 1. Standard DPG graph

Use `explainer.plot(...)` or `dpg.plot_dpg(...)` to render the Decision Predicate Graph.

```python
explainer.plot(
    "iris_dpg",
    explanation,
    save_dir="results/",
    attribute="Local reaching centrality",
)
```

```{figure} _static/visualization/iris_dpg.png
:alt: Decision Predicate Graph for Iris with node coloring by Local Reaching Centrality
:width: 100%

Standard DPG rendering colored by Local Reaching Centrality.
```

Useful options:

- `attribute`: color nodes by a metric such as `Local reaching centrality`
- `class_flag=True`: highlight class nodes
- `layout_template`: choose from `default`, `compact`, `vertical`, or `wide`
- `export_pdf=True`: save a PDF next to the PNG

### 2. Community-colored DPG graph

Use `explainer.plot_communities(...)` when you want clusters or communities highlighted.

```python
explainer.plot_communities(
    "iris_dpg",
    explanation,
    save_dir="results/",
    class_flag=True,
)
```

```{figure} _static/visualization/iris_dpg_communities.png
:alt: Decision Predicate Graph for Iris with nodes colored by community assignment
:width: 100%

DPG rendering with community-based coloring.
```

This plot requires an explanation built with `communities=True`.

### 3. LRC vs Random Forest importance

Use `explainer.plot_lrc_importance(...)` or `dpg.plot_lrc_vs_rf_importance(...)` to
compare DPG predicate importance with model feature importance.

```python
explainer.plot_lrc_importance(
    X,
    explanation=explanation,
    dataset_name="Iris",
    top_k=10,
)
```

```{figure} _static/visualization/lrc_vs_rf_importance.png
:alt: Side by side comparison of top LRC predicates and Random Forest feature importances
:width: 100%

Top DPG predicates compared with Random Forest feature importance.
```

### 4. Top predicate split lines in feature space

Use `explainer.plot_top_lrc_splits(...)` or `dpg.plot_top_lrc_predicate_splits(...)`
to overlay important split thresholds on the most relevant feature pair.

```python
explainer.plot_top_lrc_splits(
    X,
    y,
    explanation=explanation,
    dataset_name="Iris",
    top_predicates=5,
)
```

```{figure} _static/visualization/top_lrc_predicate_splits.png
:alt: Scatter plot with top LRC predicate split lines overlaid in feature space
:width: 100%

Top LRC predicate thresholds drawn over the selected feature space.
```

### 5. Bottleneck-centrality sample cloud

Use `explainer.plot_sample_using_bc_weights(...)` or
`dpg.plot_sample_using_bc_weights(...)` to project samples into PCA space and scale
point size by BC-derived predicate exposure.

```python
explainer.plot_sample_using_bc_weights(
    X,
    y,
    explanation=explanation,
    dataset_name="Iris",
    top_k=10,
)
```

```{figure} _static/visualization/bc_bottleneck_pca_cloud.png
:alt: PCA scatter plot where point size reflects bottleneck centrality derived weight
:width: 75%

Samples in PCA space sized by BC-derived bottleneck weight.
```

### 6. DPG class bounds vs dataset feature ranges

Use `explainer.plot_class_bounds_vs_dataset_ranges(...)` or
`dpg.plot_dpg_class_bounds_vs_dataset_feature_ranges(...)` to compare DPG-derived
constraints against empirical per-class feature ranges.

```python
explainer.plot_class_bounds_vs_dataset_ranges(
    X,
    y,
    explanation=explanation,
    dataset_name="Iris",
    top_features=4,
)
```

```{figure} _static/visualization/dpg_vs_dataset_feature_ranges.png
:alt: Comparison view between DPG class bounds and observed dataset feature ranges
:width: 100%

DPG class bounds compared with empirical dataset ranges.
```

This view is especially useful when you want to inspect which feature ranges are
well separated by the graph structure.

## Derived community summaries

DPG also exposes helper functions that return tables you can plot with pandas,
Matplotlib, or seaborn.

### Class-feature predicate counts

`explainer.class_feature_predicate_counts(...)` returns a class-by-feature count matrix
derived from the community structure.

```python
heat_df = explainer.class_feature_predicate_counts(explanation=explanation)
print(heat_df.head())
```

This is commonly turned into a heatmap or a per-class bar chart:

```{figure} _static/visualization/communities_class_feature_complexity_heatmap.png
:alt: Heatmap showing class-wise feature predicate complexity derived from DPG communities
:width: 100%

Class-feature predicate count heatmap built from community summaries.
```

```{figure} _static/visualization/communities_class_feature_complexity_bars.png
:alt: Bar chart showing class-wise feature predicate complexity derived from DPG communities
:width: 100%

Alternative bar-chart view of the same class-feature complexity summary.
```

### Bounds and predicate helpers

These helpers are useful when building custom charts:

- `dpg.classwise_feature_bounds_from_communities(explanation)`
- `dpg.class_lookup_from_target_names(target_names)`
- `dpg.sample_bc_weights(explanation, X_df, top_k=10)`

## Low-level plotting utilities

Most users can stay with `DPGExplainer`, but two lower-level utilities are also
available:

### `plot_dpg_reg`

Regression-oriented DPG renderer with optional node coloring by metric or community.
Use this when working directly with a regression-style DPG pipeline.

### `plot_dpg_constraints_overview`

Standalone constraint overview chart for visualizing normalized class constraints
across features, optionally including the original sample and class transition.

```python
from dpg import plot_dpg_constraints_overview

fig = plot_dpg_constraints_overview(
    normalized_constraints=normalized_constraints,
    feature_names=feature_names,
    class_colors_list=["#4c78a8", "#f58518", "#54a24b"],
    title="DPG Constraints Overview",
)
```

## Which plot should I use?

- Use `plot` for the main graph structure.
- Use `plot_communities` when cluster assignments matter.
- Use `plot_lrc_importance` to compare DPG signals with the underlying model.
- Use `plot_top_lrc_splits` to inspect important threshold predicates visually.
- Use `plot_sample_using_bc_weights` to see which samples activate bottleneck predicates.
- Use `plot_class_bounds_vs_dataset_ranges` to compare learned constraints with real data ranges.

## See also

- [Quickstart](quickstart.md)
- [API reference](api_reference.md)
