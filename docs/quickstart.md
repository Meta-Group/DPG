# Quickstart

## Installation

```bash
pip install dpg
```

DPG requires Python 3.10+ and a working [Graphviz](https://graphviz.org/download/) installation
(the `dot` executable must be on your `PATH`).

## Minimal example

The simplest way to use DPG is through `DPGExplainer`:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from dpg import DPGExplainer

# 1. Train any tree-based ensemble
X, y = load_iris(return_X_y=True, as_frame=True)
model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

# 2. Create the explainer
explainer = DPGExplainer(
    model,
    feature_names=X.columns.tolist(),
    target_names=["setosa", "versicolor", "virginica"],
)

# 3. Fit (extract the graph from training paths)
explanation = explainer.fit(X.values)

# 4. Inspect metrics
print(explanation.node_metrics.head())
print(explanation.edge_metrics.head())

# 5. Visualise
explainer.plot(explanation, save_dir="results/")
```

## What `DPGExplainer` returns

`explainer.fit()` returns a {class}`dpg.DPGExplanation` dataclass with:

| Attribute | Type | Description |
|---|---|---|
| `graph` | `nx.DiGraph` | NetworkX directed graph |
| `dot` | `graphviz.Digraph` | Graphviz rendering object |
| `node_metrics` | `pd.DataFrame` | Per-node betweenness, LRC, degree, … |
| `edge_metrics` | `pd.DataFrame` | Per-edge weight, source/target labels |
| `class_boundaries` | `dict` | Per-class feature constraint ranges |
| `communities` | `dict` | Optional community assignments |

## Configuration

DPG can be configured via a YAML file or a dict:

```python
explainer = DPGExplainer(
    model,
    feature_names=X.columns.tolist(),
    dpg_config={
        "dpg": {
            "default": {
                "perc_var": 0.001,      # minimum path frequency (0-1)
                "decimal_threshold": 2, # rounding for thresholds
                "n_jobs": -1,           # -1 = all CPU cores
            }
        }
    },
)
```

## Visualisation options

```python
from dpg import plot_dpg, plot_lrc_vs_rf_importance, plot_top_lrc_predicate_splits

# Basic DPG plot
plot_dpg(
    "iris_dpg",
    explanation.dot,
    explanation.node_metrics,
    explanation.edge_metrics,
    save_dir="results/",
    attribute="Local reaching centrality",  # color by LRC
)

# Compare DPG importance vs Random Forest importance
plot_lrc_vs_rf_importance(explanation, model, X, dataset_name="Iris")

# Visualise top predicate split lines in feature space
plot_top_lrc_predicate_splits(explanation, X, y, dataset_name="Iris")
```

## scikit-learn compatible pipeline

DPG also ships a scikit-learn `Transformer` wrapper:

```python
from dpg.sklearn_dpg import DPGTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("dpg", DPGTransformer(model, feature_names=X.columns.tolist())),
])
```
