# DPGExplainer Saga Benchmarks — Episode 1: Iris

A practitioner-friendly walkthrough of Decision Predicate Graphs (DPG) using the classic Iris dataset. We train a small RandomForest, build a DPG, and interpret three key signals: Local Reaching Centrality (LRC), Betweenness Centrality (BC), and Communities.

---

## 1. Why DPG (in one minute)
Tree ensembles can be accurate but hard to interpret globally. DPG converts the ensemble into a graph where:
- Nodes are predicates like `petal length <= 2.45`
- Edges capture how often training samples traverse those predicates
- Metrics quantify how predicates structure the model’s global reasoning

This gives a global map of decision logic, complementing local explainers.

---

## 2. Setup (Iris + Random Forest + DPG)

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from dpg import DPGExplainer

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

model = RandomForestClassifier(n_estimators=10, random_state=27)
model.fit(X, y)

explainer = DPGExplainer(
    model=model,
    feature_names=X.columns,
    target_names=iris.target_names.tolist(),
    config_file="config.yaml",  # optional if present
)

explanation = explainer.explain_global(
    X.values,
    communities=True,
    community_threshold=0.2,
)
```

---

## 3. Read the DPG Metrics

```python
explanation.node_metrics.head()
```

**Local Reaching Centrality (LRC)**
- High LRC nodes can reach many other nodes downstream.
- These predicates often act early, framing large portions of the model’s logic.

**Betweenness Centrality (BC)**
- High BC nodes lie on many shortest paths between other nodes.
- These predicates are “bottlenecks” that connect major decision flows.

---

## 4. Find the Top LRC and BC Predicates

```python
explanation.node_metrics.sort_values(
    "Local reaching centrality", ascending=False
).head(10)
```

```python
explanation.node_metrics.sort_values(
    "Betweenness centrality", ascending=False
).head(10)
```

Interpretation guide:
- If a predicate has **high LRC**, it likely sets an early rule that shapes many later decisions.
- If a predicate has **high BC**, it likely separates multiple alternative paths in the model.

---

## 5. Communities (Decision Themes)
Communities group predicates that are tightly connected. For Iris, you often see groups that align with:
- Short petal rules (often Setosa)
- Longer petal or wider sepal rules (often Versicolor/Virginica)

```python
explanation.communities.keys()
explanation.communities.get("Communities", [])[:3]
```

---

## 6. Visualize the Story

```python
run_name = "iris_dpg"
explainer.plot(run_name, explanation, save_dir="results", class_flag=True, export_pdf=True)
explainer.plot_communities(run_name, explanation, save_dir="results", class_flag=True, export_pdf=True)
```

---

## 7. What to Say in the Story
Use these 3 points for a quick practitioner summary:
- **LRC:** Which predicate most strongly frames the model’s logic?
- **BC:** Which predicate acts as a bottleneck between key decision paths?
- **Communities:** Which predicate groups define the “themes” of each class?

---

## Next Episode
We will move to a more complex dataset (UCI) and compare how DPG signatures change as the decision space grows.
