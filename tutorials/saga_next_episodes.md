# DPGExplainer Saga — Next Episodes: Suggestions and Roadmap

This document outlines suggested future episodes for the DPGExplainer Saga Benchmarks series.
Each proposal includes the target dataset, the analytical sections to develop, and the key discussions to drive.

---

## Context: What the Saga Has Covered

| Episode | Dataset | Features | Classes | Key additions |
|---|---|---|---|---|
| 1 | Iris | 4 | 3 | Full baseline pipeline: LRC, BC, communities, class bounds vs dataset ranges |
| 2 | Wine | 13 | 3 | Same pipeline on denser feature space, richer overlap analysis |
| 3 | Breast Cancer | 30 | 2 | TreeSHAP comparison, binary classification, high-dimensional BC |

The escalation across episodes has been intentional: more features, denser threshold interactions, and progressively harder class geometry. The workflow (train RF → extract DPG → analyze metrics → validate boundaries) remains consistent.

What has not yet been shown:
- DPG applied to **local/actionable explanations** (counterfactuals)
- DPG applied to **anomaly detection** (Isolation Forest)
- DPG **sensitivity to configuration** (threshold rounding, tree depth)
- DPG on datasets with **strong domain semantics** (clinical, financial)

---

## Episode 4 — DPG + Counterfactual Explanations

### Motivation

Episodes 1–3 build global interpretability: predicate topology, routing centrality, community structure, class boundaries. The natural next step is to show how those global structures enable **local, actionable explanations**.

A counterfactual explanation answers a concrete question: *"what would need to change in this sample for the model to predict a different class?"* DPG class boundaries encode exactly the threshold constraints that define class regions, making them a principled source of counterfactual generation.

### Suggested Dataset: Diabetes (Pima Indians Diabetes)

**Why this dataset:**
- Binary outcome (diabetic vs non-diabetic): clear actionability narrative.
- 8 clinical features (glucose, BMI, insulin, age, etc.) with direct human meaning.
- Well-known benchmark, easy to motivate for a general audience.
- DPG boundaries on features like `glucose` or `BMI` map directly to clinically meaningful thresholds.

**Alternative datasets already configured in the project:**
- `heart_disease_uci` — similar binary clinical setup, strong domain semantics.
- `german_credit` — financial risk, high actionability relevance.
- `telco_customer_churn` — business context (churn prevention).

### Sections

#### 1. Baseline Model and DPG Extraction
- Train a compact Random Forest on Diabetes.
- Extract DPG with `decimal_threshold=2`.
- Sanity-check: confusion matrix, classification report.

#### 2. Global DPG Review (brief)
- LRC top predicates mapped to clinical features.
- Communities as class-region clusters.
- Class boundary plot (DPG ranges vs empirical ranges) — emphasize `glucose` and `BMI` intervals.

#### 3. From Global Boundaries to Local Constraints
- Discussion: DPG class boundaries are not just visualization — they encode the rule envelope for each class.
- Show how a borderline or misclassified sample sits relative to DPG class bounds.
- Introduce the concept of a *constraint region*: the intersection of predicates a sample must satisfy to be assigned to a given class.

#### 4. Counterfactual Generation via DPG
- Use `DPGCounterfactualExplainer` to generate a counterfactual for a selected sample (e.g., a false negative: predicted non-diabetic, true diabetic).
- Show the generated counterfactual: which features changed, by how much, and which predicates were crossed.
- Overlay the CF path on the DPG boundary plot.

#### 5. CF Quality Metrics
- Proximity: how far is the counterfactual from the original sample?
- Plausibility: does the CF stay within empirical data ranges?
- Sparsity: how many features changed?
- Compare DPG-guided CF vs random perturbation baseline.

#### 6. Interpreting the CF Through the Graph
- Which high-LRC predicates were crossed to reach the target class?
- Did the path go through a high-BC bottleneck predicate?
- Discussion: bottleneck predicates as "decision gates" — small changes near a high-BC node can flip routing.

#### 7. Takeaway: Global → Local
- DPG global structure (communities, boundaries) informs local intervention (counterfactuals).
- LRC identifies which predicates are structurally influential; BC identifies which are routing-sensitive.
- Counterfactuals derived from DPG constraints are inherently model-faithful (they respect the actual decision logic, not a surrogate).

### Key Discussions
- **XAI levels**: global interpretability (what does the model do?) vs local interpretability (why this prediction?) vs actionable explanations (how to change this prediction?).
- **DPG as a constraint system**: communities and class boundaries as feasibility regions for CF search.
- **Clinical framing**: what does it mean to suggest "lower glucose by X units" — does the CF comply with physiologically plausible ranges?
- **Comparison with SHAP**: TreeSHAP gives attribution, not actionability; DPG-CF gives a concrete path from one class region to another.

---

## Episode 5 — DPG for Anomaly Detection (Isolation Forest)

### Motivation

Episodes 1–4 all use supervised classification. The DPG framework has been formally extended to Isolation Forest (see references), but no saga episode demonstrates this. This episode expands the narrative: *DPG works for any tree ensemble where decision paths carry meaning.*

In Isolation Forest, tree paths isolate anomalies. DPG captures which predicates appear most frequently in short isolation paths — these are structurally the most anomaly-discriminating rules.

### Suggested Dataset: Credit Card Fraud Detection

**Why this dataset:**
- Canonical anomaly detection benchmark (highly imbalanced: ~0.17% fraud).
- Features are PCA-transformed transaction components — abstract but interpretable in terms of "deviation from normal behavior."
- High stakes domain: false negatives (missed fraud) have direct financial cost, motivating explainability.

**Alternative datasets:**
- `mammography` — medical anomaly detection (microcalcification vs normal tissue).
- `ozone_level` — environmental anomaly detection.
- Any dataset from `counterfactual/configs/` with strong class imbalance.

### Sections

#### 1. Isolation Forest Basics and Dataset Overview
- Brief recap of Isolation Forest: anomalies are isolated faster (shorter paths) because they are rare and different.
- Dataset statistics: class imbalance, feature distributions (normal vs anomalous).
- Baseline model: Isolation Forest with standard contamination estimate.

#### 2. Extending DPG to Isolation Forest
- How DPG is adapted: instead of class labels at leaves, path length (anomaly score) drives edge weights.
- Predicate nodes now represent isolation logic: a high-LRC predicate is one that consistently appears early in short isolation paths.
- Brief alignment with the published extension paper.

#### 3. LRC in the Anomaly Detection Context
- Top-LRC predicates are the rules that most efficiently isolate anomalies.
- Visualization: LRC bar chart with feature labels.
- Discussion: compare LRC ranking vs feature importance from a supervised RF trained on the same data — do they agree? Where do they diverge?

#### 4. BC as Routing Ambiguity in Anomaly Space
- High-BC predicates: bridging rules between normal and anomalous regions.
- PCA projection: where do high-BC predicate thresholds land relative to anomalous samples?
- Discussion: BC in the anomaly context marks "borderline" predicates — regions where the model is uncertain whether to isolate or not.

#### 5. Global DPG and Communities
- Full DPG graph of the Isolation Forest.
- Community structure: do communities segment by anomaly type or by feature cluster?
- Community complexity: how many predicates are devoted to anomaly isolation vs normal routing?

#### 6. Explaining a Flagged Sample
- Select a true positive (correctly flagged fraud/anomaly).
- Trace its active DPG predicates: which nodes it passes through, which are high-LRC.
- Narrative: "This transaction was flagged because it triggered these three high-LRC isolation rules, all concentrated in the short-path anomaly community."

#### 7. Explaining a False Positive
- Select a false positive (normal flagged as anomaly).
- Show which borderline predicates (high-BC) caused the misrouting.
- Discussion: BC as a diagnostic for model uncertainty — false positives tend to concentrate near high-BC predicates.

### Key Discussions
- **Supervised vs unsupervised DPG**: what changes in interpretation when there are no class labels.
- **Anomaly score as a continuous label**: how DPG aggregates paths by isolation depth rather than leaf class.
- **LRC as an anomaly feature selector**: an alternative to permutation importance for unsupervised models.
- **Operationalizing explanations**: can a high-LRC anomaly predicate be turned into a human-readable alert rule?

---

## Episode 6 — DPG Configuration Sensitivity

### Motivation

The previous episodes all use DPG with fixed parameters (`decimal_threshold=2`, compact RF). Practitioners need to understand how configuration choices affect the graph structure and metric rankings. This episode is more technical but fills a critical gap: without sensitivity analysis, DPG results can appear opaque or arbitrary.

### Suggested Dataset: Wine (reuse from Episode 2)

**Why reuse Wine:**
- Familiar to readers who followed the saga — comparison is straightforward.
- 13 features produce visible structural changes under different threshold resolutions.
- Multi-class with overlap: effects of configuration on community structure are non-trivial.

### Sections

#### 1. Baseline Recall
- Brief restate of Episode 2 results (Wine RF + DPG with `decimal_threshold=2`) as the reference configuration.

#### 2. Effect of `decimal_threshold` on Predicate Count and Graph Size
- Run DPG with `decimal_threshold` ∈ {1, 2, 3}.
- Table: number of unique predicates, number of edges, number of communities per setting.
- Visualization: side-by-side DPG graphs for each setting (or community-colored versions).
- Discussion: coarser rounding merges nearby thresholds → simpler graph, less resolution. Finer rounding separates them → more granular routing, harder to interpret visually.

#### 3. Effect of `decimal_threshold` on LRC Rankings
- Top-10 LRC predicates for each threshold setting.
- Do feature rankings stay stable? Do specific predicates appear, disappear, or merge?
- Discussion: LRC stability as a proxy for robustness — if rankings change dramatically under mild rounding, the explanation is threshold-sensitive.

#### 4. Effect of RF Size (n_estimators) on DPG
- Fix `decimal_threshold=2`, vary `n_estimators` ∈ {5, 10, 50, 100}.
- Metrics: graph density, community count, top-LRC predicate stability.
- Discussion: richer forests produce denser graphs — at what point does the DPG become too large to read without community aggregation?

#### 5. Effect of Tree Depth (max_depth) on DPG
- Shallow trees → few predicates per path → sparse DPG.
- Deep trees → many predicates per path → dense, potentially noisy DPG.
- Show how BC distribution shifts: shallow forests tend to concentrate BC near the root predicates.

#### 6. Practical Configuration Guide
- Recommended starting point: `decimal_threshold=2`, moderate forest (10–50 trees), `max_depth` unconstrained or moderate.
- When to use coarser thresholds: fast exploration, high-dimensional datasets, early-stage analysis.
- When to use finer thresholds: final reporting, low-dimensional datasets, when specific threshold values matter to stakeholders.

### Key Discussions
- **Threshold quantization as an interpretability trade-off**: more precise thresholds → more faithful to the model, but harder to read and more sensitive to rounding choice.
- **Graph complexity vs interpretability**: there is a practical upper limit on DPG size before community aggregation becomes mandatory.
- **Stability as a quality signal**: an LRC ranking that is robust to `decimal_threshold` variation is more trustworthy than one that shifts heavily.
- **Connection to the literature**: how threshold discretization in DPG relates to predicate merging in rule-based explanation methods.

---

## Summary Table

| Episode | Dataset | Task | New analytical layer |
|---|---|---|---|
| 4 | Diabetes (or Heart Disease) | Binary classification | Counterfactual explanations via DPG boundaries |
| 5 | Credit Card Fraud (or Mammography) | Anomaly detection | DPG on Isolation Forest |
| 6 | Wine (reuse) | Classification | Configuration sensitivity: `decimal_threshold`, forest size, depth |

### Recommended Episode Order

**Episode 4 first**: it directly extends the existing classification pipeline with actionable explanations, and the counterfactual module is already implemented. Low setup cost, high narrative payoff.

**Episode 5 second**: introduces a new model family (Isolation Forest), expands the audience to anomaly detection practitioners, and is backed by a published paper — good for academic credibility.

**Episode 6 third**: a technical deep dive best appreciated after readers have built intuition from Episodes 1–5. Reusing Wine makes it accessible without dataset fatigue.

---

## References

### DPG (Classification)
- Arrighi, L., Pennella, L., Marques Tavares, G., Barbon Junior, S.
  **Decision Predicate Graphs: Enhancing Interpretability in Tree Ensembles**.
  *World Conference on Explainable Artificial Intelligence*, 311-332.
  https://link.springer.com/chapter/10.1007/978-3-031-63797-1_16

### DPG (Isolation Forest Extension)
- Ceschin, M., Arrighi, L., Longo, L., Barbon Junior, S.
  **Extending Decision Predicate Graphs for Comprehensive Explanation of Isolation Forest**.
  *World Conference on Explainable Artificial Intelligence*, 271-293.
  https://link.springer.com/chapter/10.1007/978-3-032-08324-1_12

### Counterfactual Explanations
- Wachter, S., Mittelstadt, B., Russell, C.
  **Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR**.
  *Harvard Journal of Law & Technology*, 31(2), 2018.

### Isolation Forest
- Liu, F. T., Ting, K. M., Zhou, Z.-H.
  **Isolation Forest**.
  *IEEE International Conference on Data Mining*, 2008.

### SHAP / TreeSHAP
- Lundberg, S. M., Lee, S.-I.
  **A Unified Approach to Interpreting Model Predictions**.
  *NeurIPS 2017*.
