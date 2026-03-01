# Communities and Constraints Update

In February 2026 the way DPG computes **communities** (node groupings) and
**class boundaries** (per-class feature constraints) was redesigned.
This page documents the before/after outputs produced by the
[`export_iris_communities.py`](https://github.com/Meta-Group/DPG/blob/main/counterfactual/scripts/export_iris_communities.py)
script on the Iris dataset so the differences can be inspected side-by-side.

## What changed

| Aspect | Old (pre-Feb 2026) | New (current) |
|---|---|---|
| **Community detection** | NetworkX `asyn_lpa_communities` (Label Propagation) — non-deterministic | Absorbing Markov-chain: class nodes are absorbing states, each predicate node gets a probability distribution over classes |
| **Node assignment** | Flat sets of node labels, one set per community | Deterministic per-class clusters with an explicit **Ambiguous** bucket for low-confidence nodes |
| **Class boundaries** | Derived from graph predecessors of each class node | Derived from community membership: only predicates that belong to a class cluster contribute to that class's bounds |
| **Extra outputs** | None | **Probability** (per-node absorption probabilities) and **Confidence** (margin between top-two class probabilities) |

Key commits:

* [`a906954`](https://github.com/Meta-Group/DPG/commit/a906954) — initial community visualization support and clustering utility (Feb 3 2026)
* [`fea1df7`](https://github.com/Meta-Group/DPG/commit/fea1df7) — edge metrics extraction; `extract_graph_metrics_lpa` refactor (Feb 5 2026)
* [`d7a4e6f`](https://github.com/Meta-Group/DPG/commit/d7a4e6f) — `extract_class_boundaries` using absorbing Markov chain (Feb 12 2026)

## Reproduction

Both outputs below were generated with the same parameters
(5 estimators, `random_state=42`, default config) using the script at
[`counterfactual/scripts/export_iris_communities.py`](https://github.com/Meta-Group/DPG/blob/main/counterfactual/scripts/export_iris_communities.py).

**Old output** — extracted at commit
[`8bb3583`](https://github.com/Meta-Group/DPG/commit/8bb3583) (Jan 28 2026, last
commit before the rework):

```json
{
  "api": "old (LPA-based)",
  "communities": [
    [
      "Class 1",
      "petal length (cm) <= 4.45",
      "petal length (cm) <= 4.85",
      "petal length (cm) <= 4.95",
      "petal length (cm) <= 5.05",
      "petal length (cm) <= 5.2",
      "petal length (cm) <= 5.35",
      "petal length (cm) > 2.45",
      "petal length (cm) > 4.45",
      "petal width (cm) <= 1.35",
      "petal width (cm) <= 1.55",
      "petal width (cm) <= 1.65",
      "petal width (cm) <= 1.7",
      "petal width (cm) <= 1.75",
      "petal width (cm) > 0.8",
      "petal width (cm) > 1.35",
      "sepal length (cm) <= 5.75",
      "sepal length (cm) > 5.25",
      "sepal width (cm) <= 2.75",
      "sepal width (cm) > 2.9",
      "sepal width (cm) > 3.1"
    ],
    [
      "Class 2",
      "petal length (cm) <= 4.65",
      "petal length (cm) > 2.5",
      "petal length (cm) > 4.65",
      "petal length (cm) > 4.85",
      "petal length (cm) > 4.95",
      "petal length (cm) > 5.05",
      "petal length (cm) > 5.2",
      "petal length (cm) > 5.35",
      "petal width (cm) > 1.55",
      "petal width (cm) > 1.65",
      "petal width (cm) > 1.7",
      "petal width (cm) > 1.75",
      "sepal length (cm) <= 5.25",
      "sepal length (cm) <= 6.05",
      "sepal length (cm) > 5.75",
      "sepal length (cm) > 6.05",
      "sepal width (cm) <= 2.9",
      "sepal width (cm) <= 3.1",
      "sepal width (cm) > 2.75"
    ],
    [
      "Class 0",
      "petal length (cm) <= 2.45",
      "petal length (cm) <= 2.5",
      "petal width (cm) <= 0.8"
    ]
  ],
  "class_boundaries": {
    "Class 0": [
      "petal width (cm) <= 1.65",
      "petal length (cm) <= 2.5"
    ],
    "Class 1": [
      "2.45 < petal length (cm) <= 5.35",
      "0.8 < petal width (cm) <= 1.75",
      "2.75 < sepal width (cm) <= 2.9",
      "5.25 < sepal length (cm) <= 6.05"
    ],
    "Class 2": [
      "2.45 < petal length (cm) <= 5.35",
      "0.8 < petal width (cm) <= 1.75",
      "2.75 < sepal width (cm) <= 3.1",
      "5.75 < sepal length (cm) <= 6.05"
    ]
  },
  "clusters": null
}
```

**New output** — extracted at commit
[`730dabd`](https://github.com/Meta-Group/DPG/commit/730dabd) (current `main`):

```json
{
  "api": "new (absorbing Markov chain)",
  "communities": null,
  "class_boundaries": {
    "Class Bounds": {
      "Class 0": [
        "petal width (cm) <= 1.65",
        "petal length (cm) <= 2.5"
      ],
      "Class 1": [
        "0.8 < petal width (cm) <= 1.75",
        "2.75 < sepal width (cm) <= 2.75",
        "2.45 < petal length (cm) <= 5.35",
        "5.25 < sepal length (cm) <= 6.05"
      ],
      "Class 2": [
        "petal width (cm) > 1.55",
        "4.65 < petal length (cm) <= 4.65",
        "5.25 < sepal length (cm) <= 6.05",
        "sepal width (cm) <= 3.1"
      ]
    }
  },
  "clusters": {
    "Clusters": {
      "Class 0": [
        "petal width (cm) <= 1.65", "petal width (cm) <= 0.8",
        "petal length (cm) <= 2.45", "petal length (cm) <= 2.5", "Class 0"
      ],
      "Class 1": [
        "petal width (cm) <= 1.7", "sepal width (cm) > 2.75",
        "petal width (cm) <= 1.75", "Class 1",
        "petal length (cm) <= 4.85", "sepal width (cm) > 3.1",
        "sepal length (cm) <= 6.05", "petal length (cm) <= 5.05",
        "sepal width (cm) <= 2.75", "petal length (cm) > 4.45",
        "petal length (cm) > 2.45", "petal length (cm) <= 5.35",
        "petal width (cm) > 1.35", "sepal length (cm) > 5.25",
        "sepal width (cm) > 2.9", "petal width (cm) <= 1.55",
        "sepal length (cm) <= 5.75", "petal width (cm) > 0.8",
        "petal length (cm) <= 4.95", "petal length (cm) <= 5.2",
        "petal length (cm) <= 4.45", "petal length (cm) > 2.5",
        "petal width (cm) <= 1.35"
      ],
      "Class 2": [
        "Class 2", "petal width (cm) > 1.75",
        "petal length (cm) > 4.95", "petal length (cm) > 4.65",
        "sepal length (cm) > 6.05", "petal length (cm) <= 4.65",
        "sepal length (cm) <= 5.25", "petal length (cm) > 5.05",
        "petal width (cm) > 1.65", "petal length (cm) > 4.85",
        "petal width (cm) > 1.55", "petal length (cm) > 5.35",
        "sepal width (cm) <= 3.1", "sepal width (cm) <= 2.9",
        "petal width (cm) > 1.7", "sepal length (cm) > 5.75",
        "petal length (cm) > 5.2"
      ],
      "Ambiguous": []
    },
    "Probability": {
      "petal width (cm) <= 1.7":  {"Class 0": 0.0, "Class 1": 0.69, "Class 2": 0.31},
      "petal width (cm) <= 1.75": {"Class 0": 0.0, "Class 1": 0.96, "Class 2": 0.04},
      "petal width (cm) > 1.75":  {"Class 0": 0.0, "Class 1": 0.02, "Class 2": 0.98},
      "petal length (cm) <= 2.45": {"Class 0": 1.0, "Class 1": 0.0, "Class 2": 0.0},
      "..."
    },
    "Confidence Interval": {
      "petal width (cm) <= 1.7": 0.38,
      "petal width (cm) <= 1.75": 0.92,
      "petal width (cm) > 1.75": 0.96,
      "petal length (cm) <= 2.45": 1.0,
      "..."
    }
  }
}
```

*(The full JSON files live in `outputs/iris_dpg_communities_OLD.json` and
`outputs/iris_dpg_communities_NEW.json`.)*

## Analysis

1. **Class 0 boundaries are identical** in both versions — this class is
   well-separated by `petal width <= 1.65` and `petal length <= 2.5`.

2. **Class 1 boundaries** are nearly the same, but the new version derives them
   only from nodes whose absorption probability favours Class 1.  In the old
   version the boundaries came from *all* predecessors of the Class 1 terminal
   node, regardless of how strongly they were associated.

3. **Class 2 boundaries changed the most.**  The old version produced ranges
   identical to Class 1 (`2.45 < petal length <= 5.35`, `0.8 < petal width <= 1.75`)
   because both classes shared many predecessor nodes.  The new version narrows
   these to `petal width > 1.55` and `4.65 < petal length <= 4.65`, reflecting
   only the nodes that are probabilistically absorbed by Class 2.

4. **Community structure moved from non-deterministic to deterministic.**
   LPA can return different groupings on each run; the absorbing-chain method
   always produces the same clusters for the same graph.

5. **Richer per-node information.**  The new output includes absorption
   probabilities (e.g., `petal width <= 1.75` → 96 % Class 1, 4 % Class 2) and
   confidence margins, enabling finer-grained analysis of borderline nodes.
