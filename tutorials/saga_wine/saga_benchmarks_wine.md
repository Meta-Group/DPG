# DPGExplainer Saga Benchmarks — Episode 2: Wine

Episode 1 (Iris) established the baseline idea: Random Forest can classify well, but DPG reveals the *decision program* behind that accuracy.

Episode 2 keeps the same analysis flow, but moves to a harder landscape. Wine has more features, denser interactions, and class regions that demand richer predicate logic.

The workflow remains unchanged:
1. Train a Random Forest baseline.
2. Extract DPG from the trained model.
3. Analyze LRC, BC, communities, overlap, and class complexity.
4. Cross-check DPG boundaries against dataset feature ranges.

---

## 1. Wine dataset and data visualization

Like Iris in Episode 1, Wine is a classic benchmark. The difference is complexity: Wine introduces 13 continuous chemical features and 3 classes, with stronger multivariate coupling and less obvious low-dimensional separation.

A pairplot gives the first geometric sanity check:

![Wine pairplot](images/pairplot.png)

Interpretation before modeling:
- Some class structure is visible, but separation is less clean than Iris.
- Multiple features co-move, so the model will likely rely on combinations of thresholds.
- We should expect broader overlap regions and a larger rule budget than Episode 1.

This sets the hypothesis for DPG: in Wine, communities and bottlenecks should matter even more than in Iris.

---

## 2. Model creation

We keep the same baseline strategy from Episode 1: a compact `RandomForestClassifier` (`n_estimators=10`, `random_state=27`) trained with stratified split (`test_size=0.2`, `random_state=42`).

Baseline performance check:

![RF confusion matrix](images/rf_confusion_matrix.png)

Confusion matrix (`rows=true`, `cols=predicted`):

```text
[[12  0  0]
 [ 1 13  0]
 [ 0  0 10]]
```

Classification report:

```text
              precision    recall  f1-score   support

class_0           0.92      1.00      0.96        12
class_1           1.00      0.93      0.96        14
class_2           1.00      1.00      1.00        10

accuracy                               0.97        36
macro avg         0.97      0.98      0.97        36
weighted avg      0.97      0.97      0.97        36
```

As in Iris, predictive quality is strong. The key question is now structural: *how* is this performance achieved over a more complex feature space?

---

## 3. Why DPG on top of Random Forest

Episode 1 already showed the core limitation: RF importance is useful, but it is not a blueprint of decision flow.

DPG converts the forest into a graph where:
- Nodes are concrete predicates (`feature <= threshold` or `feature > threshold`).
- Edges represent transitions observed across tree decision paths.
- Graph metrics expose routing role, bottlenecks, modularity, and class-level structure.

In Wine, this matters more because global feature relevance alone cannot explain how threshold interactions separate partially overlapping classes.

---

## 4. LRC vs RF importance (complementary views)

As in Iris, RF importance and LRC answer different questions:
- RF importance: which features reduce impurity globally.
- LRC: which *specific predicates* are structurally upstream routers.

![LRC vs RF importance](images/lrc_vs_rf_importance.png)

Top-10 LRC predicates from the notebook:

| Predicate | LRC |
|---|---:|
| `color_intensity > 3.46` | 0.504830 |
| `flavanoids > 0.91` | 0.502640 |
| `od280/od315_of_diluted_wines > 1.965` | 0.463310 |
| `flavanoids > 1.655` | 0.377340 |
| `color_intensity > 3.82` | 0.302024 |
| `hue > 0.745` | 0.281635 |
| `alcohol > 13.04` | 0.212821 |
| `proline > 724.5` | 0.211581 |
| `od280/od315_of_diluted_wines > 2.005` | 0.204453 |
| `total_phenols > 2.335` | 0.199512 |

Top RF features:

| Feature | RF importance |
|---|---:|
| `od280/od315_of_diluted_wines` | 0.164883 |
| `proline` | 0.161166 |
| `color_intensity` | 0.158853 |
| `alcohol` | 0.147452 |
| `flavanoids` | 0.087611 |
| `hue` | 0.079530 |
| `magnesium` | 0.067015 |
| `total_phenols` | 0.055643 |
| `malic_acid` | 0.042399 |
| `proanthocyanins` | 0.015424 |

Compared with Episode 1, the same complementarity appears, but with richer threshold reuse across more features.

Projected split view:

![Top LRC predicate splits](images/top_lrc_predicate_splits.png)

This projection shows where high-LRC rules cut the data manifold and where routing pressure accumulates.

---

## 5. BC as bottleneck decision logic

BC captures bridge predicates that connect major decision regions.

![BC bottleneck PCA cloud](images/bc_bottleneck_pca_cloud.png)

Top BC predicates:
- `hue <= 0.83` (0.002359)
- `alcohol > 13.04` (0.001993)
- `ash <= 2.57` (0.001927)
- `alcohol <= 13.04` (0.001894)
- `magnesium > 94.5` (0.001761)

Narrative link to Iris: in Episode 1, BC concentrated around the versicolor/virginica transition. In Wine, BC again localizes in overlap-sensitive regions, but now across a broader chemical feature set.

---

## 6. Global DPG and communities

Global graph:

![DPG graph](images/wine_dpg.png)

Community-colored graph:

![DPG communities](images/wine_dpg_communities.png)

Community reading follows the same logic as Episode 1:
- communities behave as decision submodules,
- cross-community links reveal handoff points,
- larger communities indicate heavier rule allocation.

In Wine, these modules are denser and more numerous than Iris, consistent with higher feature dimensionality.

---

## 7. Communities, overlap, and class complexity

Class-feature predicate concentration:

![Community class-feature heatmap](images/communities_class_feature_complexity_heatmap.png)

Class-level complexity summary:

![Community class complexity bars](images/communities_class_feature_complexity_bars.png)

From notebook counts:
- `class_1`: `80` predicates across `13` features.
- `class_0`: `47` predicates across `13` features.
- `class_2`: `45` predicates across `11` features.

Key reading:
- `class_1` receives the largest rule budget, so it is structurally the most complex region in this benchmark.
- `class_0` and `class_2` require fewer predicates but still much more than the simplest Iris class from Episode 1.
- Overlap is visible through shared high-use features (`od280/od315_of_diluted_wines`, `color_intensity`, `malic_acid`, `proline`) with class-specific predicate densities.

Implementation note:
- community analysis uses DPGExplainer cluster output directly (`Clusters`),
- `community_id` remains tied to a unique DPG community,
- class association uses cluster labels when present.

---

## 8. DPG community ranges vs dataset ranges

![DPG vs dataset feature ranges](images/dpg_vs_dataset_feature_ranges.png)

Following the Episode 1 boundary-validation logic, this view compares:
- dataset class ranges (gray),
- DPG community ranges (blue),
- predicate-threshold density by operator (`>` in green, `<=` in red),
- unbounded-side markers (`-inf`, `+inf`) when the model uses one-sided constraints.

Notebook-derived boundary summary:
- `class_0`: 13 modeled features, 12 finite lower bounds, 5 finite upper bounds.
- `class_1`: 13 modeled features, 9 finite lower bounds, 13 finite upper bounds.
- `class_2`: 11 modeled features, 7 finite lower bounds, 6 finite upper bounds.

Interpretation:
- Wine ranges are often asymmetric and partially open-ended, unlike the cleaner intervals seen for easier regions in Iris.
- This pattern supports a realistic model behavior: classes are carved by mixed one-sided and bounded rules across interacting chemistry dimensions.

---

## 9. Main DPG contributions in this benchmark

DPG extends standard RF interpretation with the same seven benefits established in Episode 1, now stress-tested on a harder dataset:

1. Global rule topology.
2. Predicate-level influence via LRC.
3. Bottleneck routing via BC.
4. Community-level class semantics.
5. Overlap diagnostics.
6. Class complexity profiling.
7. Boundary validation against dataset statistics.

What changes from Episode 1 to Episode 2 is not the method, but the evidence scale: Wine shows how the same DPG toolkit remains interpretable when rule interactions become denser and less visually obvious.

---

## 10. References and related work

### Original DPG proposal
- Arrighi, L., Pennella, L., Marques Tavares, G., Barbon Junior, S.
  **Decision Predicate Graphs: Enhancing Interpretability in Tree Ensembles**.
  *World Conference on Explainable Artificial Intelligence*, 311-332.
  https://link.springer.com/chapter/10.1007/978-3-031-63797-1_16

### Extended DPG (Isolation Forest)
- Ceschin, M., Arrighi, L., Longo, L., Barbon Junior, S.
  **Extending Decision Predicate Graphs for Comprehensive Explanation of Isolation Forest**.
  *World Conference on Explainable Artificial Intelligence*, 271-293.
  https://link.springer.com/chapter/10.1007/978-3-032-08324-1_12

### Real-life applications
- Systems:
  https://www.mdpi.com/2079-8954/13/11/935
- Computers and Electronics in Agriculture:
  https://www.sciencedirect.com/science/article/pii/S0168169926000979

### Saga context
- Episode 1 (Iris):
  https://medium.com/@sbarbonjr/dpgexplainer-saga-benchmarks-episode-1-iris-c8816db2857d
