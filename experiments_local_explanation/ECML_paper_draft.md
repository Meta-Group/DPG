# Semantic Local Decision Graphs for Tree-Ensemble Explanations

## Working title

Semantic Local Decision Graphs for Tree-Ensemble Explanations:
Execution-Trace Construction, Structural Faithfulness, and Critical Decision Points

## One-sentence pitch

We introduce an execution-trace DPG for local explanations of tree ensembles and show that, while it does not dominate model-direct attribution baselines on raw fidelity, it provides substantially stronger structural faithfulness and a semantic explanation layer based on competitor paths, explanation confidence, and critical decision nodes.

## Abstract

Local explanations for tree ensembles are often evaluated only through output-level agreement with the underlying model. This is insufficient when the explanation object is structured, because two explanations can have similar prediction-level fidelity while differing substantially in path-level faithfulness and diagnostic value. We study this issue in Decision Predicate Graphs (DPGs), where local explanations are represented as sample-specific subgraphs over canonicalized tree predicates. We identify a key source of local unfaithfulness in prior DPG constructions: aggregated predicate transitions can recombine fragments from different execution contexts, producing locally plausible but structurally invalid paths. To address this, we propose an execution-trace graph construction in which local graphs are built from exact tree paths executed by the ensemble. On 15 numeric tabular datasets, execution-trace DPGs achieve near-perfect edge precision and recall while nearly eliminating recombination. We further introduce a semantic layer for local graph explanations, including explanation confidence, competitor-path exposure, and critical nodes defined as the last shared non-root predicate between the strongest predicted and competitor paths. These quantities support disagreement-aware analysis and structural case studies that standard attribution baselines do not naturally provide. Our results suggest that DPG for local explanation should be understood not as a raw-fidelity winner over TreeSHAP or ICE, but as a method for structurally faithful and diagnostically rich local decision explanations for tree ensembles.

## 1. Introduction

### Problem

Post-hoc local explanations for tree ensembles are typically assessed by output-level criteria:

- whether the explanation matches the model prediction
- whether it matches the ground-truth label
- how strongly it separates the predicted class from alternatives

These metrics are necessary but incomplete for structured explanation objects. In particular, for graph- or path-based explanations, a method can preserve prediction-level agreement while distorting the actual local decision structure used by the model.

### Motivation

DPGs provide graph-based explanations over canonical predicates extracted from tree ensembles. They are useful because they expose predicate connectivity, support graph-level summarization, and can produce human-interpretable local subgraphs. However, prior compact graph constructions merged transitions across many trees and many execution contexts. This compactness comes at a cost: local explanation paths can be recombined from edges that were never jointly executed for the current sample.

This observation leads to the central question of the paper:

> Can a graph-based local explanation for tree ensembles be made structurally faithful enough to support semantic local analysis, rather than only output-level fidelity?

### Main idea

We answer this question by using an execution-trace construction that preserves exact root-to-leaf traces. On top of this more faithful substrate, we define semantic local explanation quantities:

- predicted and competitor support paths
- explanation confidence
- path purity and competitor exposure
- critical nodes and branch contrast

These quantities allow us to analyze not only correct predictions, but also disagreement, ambiguity, and model errors.

### Contributions

This paper makes four contributions.

1. We introduce an execution-trace local DPG construction for tree ensembles, designed to reduce path recombination and improve local structural faithfulness.
2. We define a semantic layer for local graph explanations, including explanation confidence, competitor-path analysis, and critical decision nodes.
3. We propose path-level faithfulness and disagreement-aware analyses that go beyond standard output-level fidelity.
4. We provide an empirical study over 15 tabular datasets showing that execution-trace DPGs yield structurally faithful local explanations, even when raw fidelity remains lower than TreeSHAP and ICE.

## 2. Related Positioning

### Explanation families in this paper

We compare against two kinds of baselines.

- Model-direct baselines:
  - TreeSHAP
  - tree-path decomposition
  - anchor rules evaluated directly on the trained forest
  - ICE as a local response-profile diagnostic
- Local surrogate baseline:
  - LIME

This distinction matters because DPG for local explanation is not a surrogate explainer. It should primarily be compared to methods that also interrogate the trained model directly.

### Scope of claims

This paper does **not** claim that DPG for local explanation is the strongest method on raw fidelity across all local explanation paradigms.

The intended claim is narrower:

- DPG for local explanation improves the structural faithfulness of graph-based local explanations for tree ensembles
- DPG for local explanation exposes semantic local decision information that attribution and response-profile baselines do not naturally provide in the same form

## 3. Method

### 3.1 Execution-trace graph construction

Let `T_b` denote the `b`-th tree and `pi_b(x)` the exact root-to-leaf predicate sequence executed by sample `x` in `T_b`.

Earlier compact DPG constructions merged observed predicate transitions into a global graph. This can produce recombined local paths when local explanations traverse aggregated connectivity.

In execution-trace DPGs, the local graph for `x` is built from:

- the union of executed predicate nodes across trees
- the union of executed directed edges across those paths

This removes the need to reconstruct local paths from aggregated connectivity.

### 3.2 Local support and competitor semantics

Each local path contributes support to a leaf class. From these path supports we derive:

- support-predicted class
- top competitor class
- support margin
- predicted-class concentration
- model-vote agreement
- trace coverage score

These are combined into an explanation confidence score.

### 3.3 Critical node

Let the strongest predicted-class path and strongest competitor-class path be compared by longest common prefix over predicate labels.

The critical node is the last shared **non-root** predicate before the two paths diverge. This excludes root-only agreement, because the root is too generic to represent a meaningful local decision bottleneck.

We also define:

- critical split depth
- predicted and competitor critical successors
- critical-node branch contrast

### 3.4 Path-level faithfulness

We evaluate local graph explanations using:

- node recall
- node precision
- edge recall
- edge precision
- path purity
- competitor exposure
- recombination rate

These metrics are specific to structured local explanations and are central to our DPG comparison.

## 4. Experimental Setup

### Datasets

We use 15 curated numeric tabular datasets from the local-explanation benchmark suite.

### Model class

Random forest classifiers are used throughout.

### DPG configuration

The paper presents only the execution-trace DPG configuration, while holding fixed:

- local evidence variant: `top_competitor`
- `base_lambda = 0.8`

### Hyperparameter grid

The study uses:

- `n_estimators in {10, 20}`
- `max_depth = 4`
- `perc_var in {1e-4, 1e-5}`
- `decimal_threshold = 6`
- `seed in {27, 42}`

### Baselines

The broader comparison includes:

- SHAP
- ICE
- LIME
- Anchors
- tree-path decomposition
- legacy DPG variants from earlier studies

### Shared metrics

Across methods we compare:

- fidelity to the model prediction
- local accuracy
- class-separation margin
- runtime where relevant
- sufficiency and comprehensiveness

### DPG-only structural metrics

Only DPG variants are evaluated on:

- node/edge recall and precision
- recombination rate
- explanation confidence
- critical-node metrics
- disagreement and cohort diagnostics

## 5. Main Results

### 5.1 Main quantitative result

For the presented DPG method, execution-trace DPG achieves:

- mean fidelity `0.8960`
- mean local accuracy `0.8021`
- mean margin `0.7284`

These values establish the output-level performance of the method presented in this paper. The motivating ablation against earlier compact DPG constructions is part of the development history, but it is not a main comparison in the paper narrative.

### 5.2 Structural faithfulness

For the best next-phase configuration per dataset:

- execution-trace DPG: edge precision `1.0000`, edge recall `0.9999`, recombination `0.0000`, explanation confidence `0.6872`

Interpretation:

- execution-trace preserves executed structure almost exactly
- recombination is effectively removed
- explanation confidence is high when the local graph is structurally faithful

### 5.3 Comparison to prior DPG variants and baselines

Current consolidated results show:

- SHAP remains strongest on mean raw fidelity
- ICE also exceeds DPG for local explanation on raw fidelity

This leads to the correct paper framing:

- DPG for local explanation should not be sold as a raw-fidelity winner over TreeSHAP
- DPG for local explanation should be sold as a structurally faithful and semantically rich local graph explainer

## 6. Failure And Disagreement Analysis

### Cohorts

We partition samples into:

- `MC-EC`
- `MC-EW`
- `MW-EM`
- `MW-EC`
- `MW-EW`

This allows explanation quality to be studied separately under correct predictions, disagreements, and model errors.

### Effect-size result

Execution-trace DPG shows clear separation between:

- `DISAGREE` vs `AGREE`
- `MW-EM` vs `MC-EC`

For example:

- `DISAGREE` vs `AGREE` shows a large drop in explanation confidence and path purity
- competitor exposure rises strongly in disagreement cohorts

This supports the claim that the new semantic metrics are not only decorative; they track meaningful failure modes.

## 7. Semantic Case Studies

### What Phase C should demonstrate

Case studies should show:

- wide local subgraph
- predicted path
- competitor path
- critical node and branch successors
- explanation confidence and support margin

### Strong current recovery/disagreement case

`vehicle`, sample `56`, execution-trace:

- true label `2`
- model prediction `1`
- local explanation prediction `2`
- critical node: `MAX.LENGTH_ASPECT_RATIO > -0.231807`

This is a strong paper case because the explanation recovers the true class while the model is wrong, and the critical split localizes where the predicted and competitor paths separate.

### Important caveat about the previously selected spambase case

`spambase`, sample `859` remains a strong high-confidence error case, but under the stricter non-root rule its critical node is absent. Therefore it should be used as an error example, but not as the main critical-node example.

We should select a different misclassification case with a valid non-root critical node for the core critical-node narrative.

## 8. What We Can Claim

### Safe claims

1. Execution-trace graph construction yields structurally faithful local explanations for tree ensembles.
2. Graph-construction semantics materially affect local explanation fidelity.
3. Execution-trace DPGs support semantic local analysis through competitor paths, explanation confidence, and critical nodes.
4. Structural metrics and disagreement cohorts reveal information that is invisible from raw fidelity alone.

### Claims to avoid

1. DPG for local explanation is the best overall explainer for tree ensembles.
2. DPG for local explanation beats SHAP on raw fidelity.
3. Critical nodes always exist or always identify a single causal reason for model error.

## 9. Missing Pieces Before Submission

1. Finish the new baseline sweep with Anchors and tree-path decomposition across all 15 datasets.
2. Finish the semantic-faithfulness sweep and merge the resulting tables.
3. Replace `spambase 859` with a stronger non-root critical-node error example.
4. Convert the current working draft into a venue-specific manuscript format.

## 10. Immediate Writing Plan

1. Write Introduction and Contributions from this draft nearly as-is.
2. Turn the Method section into a concise ECML-style formal section using the exact formulas already defined in `DPG2.0.md`.
3. Build the main results section around three messages:
   - execution-trace DPG is competitive on shared output-level metrics
   - execution-trace DPG is strongly faithful structurally
   - DPG for local explanation provides semantic diagnostics beyond attribution methods
4. Reserve Phase C for two to four strong figures, not a large gallery.
