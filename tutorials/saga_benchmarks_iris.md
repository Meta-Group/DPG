# DPGExplainer Saga Benchmarks — Episode 1: Iris

A practitioner-friendly walkthrough of Decision Predicate Graphs (DPG) using the classic Iris dataset. We train a small Random Forest (RF), build a DPG to map the model’s global behavior using Explainable AI (XAI), and interpret three key properties to explain the model: Local Reaching Centrality (LRC), Betweenness Centrality (BC), and node communities.

---

## 1. What is Explainable AI (XAI)
Explainable AI (XAI) focuses on making model behavior understandable to people. It helps answer questions like why a prediction was made, what features mattered most, and whether the model behaves as intended.

Common motivations for XAI include:
- Explain to justify: Provide evidence for decisions in high-stakes contexts.
- Explain to discover: Surface patterns, biases, or unexpected signals in the data.
- Explain to improve: Debug models, features, and data issues.
- Explain to control: Support monitoring, governance, and compliance.

XAI methods are often grouped into:
- Global explanations: Summarize how the model behaves overall.
- Local explanations: Explain a single prediction or a small region of the feature space.

SHAP is a popular local method, while DPG provides a global view by turning an ensemble into a predicate graph and analyzing its structure.


## 2. Why DPG (in one minute)
Tree ensembles, such as RF, can be accurate but hard to interpret globally. DPG converts the ensemble into a graph where:
- Nodes are predicates like `petal length <= 2.45`, in the iris case.
- Edges capture how often training samples traverse those predicates
- Metrics quantify how predicates structure the model’s global reasoning

This gives a global map of decision logic and allows the use of graph metrics to capture the model’s rationale.

In the next steps, we create a Random Forest model of the Iris dataset and explain it with DPG.

---

## 3. Setup (Iris + Random Forest + DPG)

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from dpg import DPGExplainer

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=27, stratify=y
)

model = RandomForestClassifier(n_estimators=10, random_state=27)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot()
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

## 4. Extracting DPG from RF

Next, we extract the DPG from our RF model. The parameters `feature_names` and `target_names` provide readable output for the mapped scenarios.
```python

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

## 5. Read the DPG Metrics
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

## 6. Find the Top LRC and BC Predicates

```python
import matplotlib.pyplot as plt
import re

def parse_feature_from_predicate(label: str) -> str:
    """
    Extract feature name from labels like:
    - petal length (cm) <= 2.45
    - sepal width (cm) > 3.1
    """
    m = re.search(r"(.+?)\s*(<=|>)\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(label))
    return m.group(1).strip() if m else str(label)

nm = explanation.node_metrics.copy()
nm = nm[nm["Label"].str.contains(r"(<=|>)", regex=True, na=False)]

top_lrc = nm.sort_values("Local reaching centrality", ascending=False).head(10).copy()
top_bc = nm.sort_values("Betweenness centrality", ascending=False).head(10).copy()

top_lrc["Feature"] = top_lrc["Label"].apply(parse_feature_from_predicate)
top_bc["Feature"] = top_bc["Label"].apply(parse_feature_from_predicate)

# One shared color map so the same feature has the same color in both charts
all_features = list(dict.fromkeys(top_lrc["Feature"].tolist() + top_bc["Feature"].tolist()))
cmap = plt.cm.tab20
feature_to_color = {
    f: cmap(i / max(1, len(all_features) - 1))
    for i, f in enumerate(all_features)
}

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# LRC chart
lrc_plot = top_lrc.sort_values("Local reaching centrality", ascending=True)
axes[0].barh(
    lrc_plot["Label"],
    lrc_plot["Local reaching centrality"],
    color=[feature_to_color[f] for f in lrc_plot["Feature"]],
    edgecolor="black",
    linewidth=0.4,
)
axes[0].set_title("Top 10 LRC Predicates")
axes[0].set_xlabel("Local Reaching Centrality")
axes[0].set_ylabel("Predicate")

# BC chart
bc_plot = top_bc.sort_values("Betweenness centrality", ascending=True)
axes[1].barh(
    bc_plot["Label"],
    bc_plot["Betweenness centrality"],
    color=[feature_to_color[f] for f in bc_plot["Feature"]],
    edgecolor="black",
    linewidth=0.4,
)
axes[1].set_title("Top 10 BC Predicates")
axes[1].set_xlabel("Betweenness Centrality")
axes[1].set_ylabel("Predicate")

# Shared legend (feature -> color)
legend_handles = [
    plt.Line2D([0], [0], marker='s', color='w', label=f,
               markerfacecolor=feature_to_color[f], markeredgecolor='black', markersize=8)
    for f in all_features
]
fig.legend(handles=legend_handles, title="Feature colors",
           loc="lower center", ncol=min(4, max(1, len(legend_handles))), frameon=True)

plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()
```

Interpretation guide:
- If a predicate has **high LRC**, it likely sets an early rule that shapes many later decisions.
- If a predicate has **high BC**, it likely separates multiple alternative paths in the model.

---

## 7. Communities (Decision Themes)
Communities group predicates that are tightly connected. For Iris, you often see groups that align with:
- Short petal rules (often Setosa)
- Longer petal or wider sepal rules (often Versicolor/Virginica)

```python
explanation.communities.keys()
explanation.communities.get("Communities", [])[:3]
```

---

## 8. Visualize the Story

```python
run_name = "iris_dpg"
explainer.plot(run_name, explanation, save_dir="results", class_flag=True, export_pdf=True)
explainer.plot_communities(run_name, explanation, save_dir="results", class_flag=True, export_pdf=True)
```

---

## 9. What to Say in the Story
Use these three points for a quick practitioner summary:
- **LRC:** Which predicate most strongly frames the model’s logic?
- **BC:** Which predicate acts as a bottleneck between key decision paths?
- **Communities:** Which predicate groups define the “themes” of each class?

---

## Next Episode
We will move to another scikit-learn benchmark dataset.
