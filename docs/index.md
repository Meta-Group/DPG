# DPG — Decision Predicate Graphs

**DPG** is a model-agnostic tool to provide a global interpretation of tree-based ensemble models, addressing transparency and explainability challenges.

DPG is a graph structure that captures the tree-based ensemble model and learned dataset details, preserving the relations among features, logical decisions, and predictions towards emphasising insightful points.
DPG enables graph-based evaluations and the identification of model decisions towards facilitating comparisons between features and their associated values while offering insights into the entire model.
DPG provides descriptive metrics that enhance the understanding of the decisions inherent in the model, offering valuable insights

::::{grid} 4
:::{grid-item-card} Getting Started
:link: quickstart
:link-type: doc

Install DPG and run your first explanation in minutes.
:::
:::{grid-item-card} API Reference
:link: api_reference
:link-type: doc

Full API documentation for dpg, metrics, and counterfactual.
:::
:::{grid-item-card} Visualization
:link: visualization
:link-type: doc

Browse all graph and chart outputs available in DPG.
:::
:::{grid-item-card} Counterfactual
:link: api/counterfactual/index
:link-type: doc

Counterfactual generation and constraint explanation on DPG models.
:::
:::{grid-item-card} Development
:link: development/index
:link-type: doc

Internal change logs, design decisions, and migration notes.
:::
::::

## Quick example

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
explanation = explainer.fit(X.values)
explainer.plot(explanation)
```

## Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

quickstart
```

```{toctree}
:maxdepth: 2
:caption: Visualization

visualization
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api_reference
```

```{toctree}
:maxdepth: 2
:caption: Development

development/index
```
