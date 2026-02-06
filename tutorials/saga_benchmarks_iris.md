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
