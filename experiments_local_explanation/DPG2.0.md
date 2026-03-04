# DPG 2.0: Graph Construction via `aggregated_transitions` vs `execution_trace`

## Goal

Improve `DPGExplainer.explain_local` fidelity by changing how the DPG graph is constructed, without changing the local evidence formulation yet.

The working hypothesis is:

- the main fidelity gap is caused by graph-construction semantics
- changing local evidence before fixing graph construction mixes causes
- fidelity may improve substantially by replacing recombined graph structure with exact execution traces

So DPG 2.0 should first isolate one design axis:

- `graph_construction.mode = aggregated_transitions`
- `graph_construction.mode = execution_trace`

## Proposed API

```python
explainer = DPGExplainer(
    model=rf,
    feature_names=feature_names,
    target_names=class_names,
    dpg_config={
        "dpg": {
            "default": {
                "perc_var": 0.0001,
                "decimal_threshold": 6,
                "n_jobs": 1,
            },
            "graph_construction": {
                "mode": "aggregated_transitions",   # or "execution_trace"
            },
        }
    },
)
```

This mode should affect:

- global graph construction
- local subgraph extraction
- path extraction used by `explain_local`

It should not change:

- local evidence formula
- attraction/repulsion variant
- lambda policy

Those stay fixed during the first DPG 2.0 experiments.

## Why the Current Construction Can Hurt Fidelity

### Current issue

The current graph is built from aggregated predicate transitions. This is compact, but it can encode transitions that are only locally valid in the union of many paths.

When local explanation reuses this graph, it may recover predicate chains that were not jointly executed for the current sample.

This produces a structural artifact:

- path recombination

The problem is not that individual predicates or edges are false in isolation. The problem is that a recovered local path may be valid only as a recombination of pieces observed in different execution contexts.

### Why this matters

Path recombination can:

- lower `local_matches_model_rate`
- inflate competitor-class support
- distort evidence margins
- increase explanation complexity without increasing faithfulness

This explains why previous score and pruning changes could alter margins but not materially improve fidelity.

## Two Graph Construction Modes

### 1. `aggregated_transitions`

This is the current DPG-style construction.

Semantics:

- nodes are canonical predicates
- edges represent observed predicate transitions aggregated across the ensemble
- the graph is compact and useful for high-level structural summaries

Strengths:

- compact
- visually interpretable
- good for global predicate-transition analysis

Weaknesses:

- local paths may be reconstructed from aggregated connectivity rather than exact execution
- can support path recombination

### 2. `execution_trace`

This is the DPG 2.0 proposal.

Semantics:

- nodes are still canonical predicates
- edges are inserted from exact root-to-leaf tree execution traces
- local explanations are derived from paths that were actually executed by the ensemble for the sample

Strengths:

- faithful by construction at the local path level
- reduces recombined paths
- cleaner support aggregation

Tradeoff:

- graph may be less compact than the current aggregated version
- some global summaries may become more fragmented

## Exact Meaning of `execution_trace`

For a sample `x` and tree `T_b`, compute the exact path:

\[
\pi_b(x) = [p_1, p_2, \ldots, p_{L_b}]
\]

where each `p_i` is the actual predicate tested along the tree path from root to leaf.

Then canonicalize predicates using the same DPG conventions:

- feature naming
- operator convention
- threshold rounding with `decimal_threshold`

The local activated graph for `x` is:

\[
V_x = \bigcup_b \pi_b(x)
\]

\[
E_x = \bigcup_b \{(p_i \rightarrow p_{i+1})\}_{i=1}^{L_b-1}
\]

So the local explanation graph is the union of exact executed transitions, not a traversal of aggregated connectivity.

## Handling `perc_var`

`perc_var` stays part of the construction pipeline and must be studied explicitly because it changes the predicate vocabulary.

This creates an important diagnostic case:

- a predicate may be executed in a tree trace but removed from the DPG vocabulary by `perc_var`

So `execution_trace` mode should report:

- number of executed predicates
- number of executed predicates mapped into the DPG
- number of missing predicates after filtering
- number of executed edges
- number of executed edges preserved after mapping/filtering

This is essential for interpreting fidelity changes. If fidelity stays low, the reason may be the filtering policy rather than the trace idea itself.

## Recommended Experimental Scope

Do not change local evidence yet.

Hold fixed:

- local evidence variant
- lambda
- scoring rule

Only change:

- `graph_construction.mode`

This is the cleanest causal test.

## Experimental Design

### Main comparison

Compare:

1. `graph_construction.mode = aggregated_transitions`
2. `graph_construction.mode = execution_trace`

Use the same evaluation pipeline and the same 15 curated datasets.

### Fixed settings for the first study

Keep a single local evidence setup, preferably the current strongest default:

- `variant = top_competitor`
- `base_lambda = 0.8`

This avoids confounding graph construction with other design choices.

### Hyperparameter grid

Use the same grid style already used in the local studies:

- `n_estimators = 10, 20`
- `max_depth = 4`
- `perc_var = 0.0001, 0.00001`
- `decimal_threshold = 6`
- `seeds = 27, 42`

Optional second wave:

- add one more `decimal_threshold`
- add one more `perc_var`

But do not broaden the grid before the first signal is clear.

## Primary Metrics

The primary target is:

- `local_matches_model_rate`

This is the main success criterion because the change is motivated by local faithfulness to the model.

## Secondary Metrics

- `local_accuracy`
- `avg_evidence_margin_pred_vs_competitor`
- `avg_num_paths`
- `avg_num_active_nodes`
- `avg_num_active_edges`
- `local_failure_rate`

## New Diagnostics Needed for DPG 2.0

For `execution_trace`, log:

- `num_executed_paths`
- `num_executed_predicates`
- `num_executed_edges`
- `num_trace_predicates_missing_from_dpg`
- `num_trace_edges_missing_from_dpg`
- `trace_node_coverage`
- `trace_edge_coverage`

For `aggregated_transitions`, log at least:

- `num_local_paths_recovered`
- `num_active_nodes`
- `num_active_edges`

These diagnostics will help explain whether fidelity differences come from:

- better path semantics
- better edge semantics
- reduced structural ambiguity
- or filtering losses from `perc_var`

## Dataset-Level Analysis

For each dataset, compare the best config under both modes using:

1. fidelity
2. local accuracy
3. margin
4. complexity

Also track whether the winning mode changes with `perc_var`.

This is important because `execution_trace` may help most on:

- multiclass datasets
- datasets with more predicate reuse
- settings where the aggregated graph is denser

## Expected Outcomes

### Best-case outcome

`execution_trace` improves fidelity clearly across most datasets, with stable or slightly improved local accuracy.

This would support the claim that graph-construction semantics were the main bottleneck.

### Moderate outcome

`execution_trace` improves fidelity mostly on harder multiclass datasets, while binary datasets remain similar.

This is still a good result and would fit the recombination hypothesis.

### Negative outcome

`execution_trace` gives little improvement.

Then the main issue is likely not graph construction alone, but:

- predicate filtering
- local evidence aggregation
- or the final decision rule

Even this is useful, because it cleanly rules out one major hypothesis.

## Paper Positioning if the Result Is Positive

If `execution_trace` improves fidelity, the contribution becomes much stronger:

- DPG 2.0 replaces recombination-prone graph construction with execution-trace projection
- local explanations become faithful by construction
- previous score variants become secondary refinements, not the core fix

That is a more compelling methodological contribution than another score tweak.

## Observed Results So Far

The implementation work described above is now complete:

- `execution_trace` graph construction is implemented in the global DPG builder and in `DPGExplainer.explain_local`
- local explanation confidence is implemented
- critical node extraction is implemented
- path-level faithfulness metrics are implemented
- misclassification and disagreement diagnostics are implemented
- next-phase analysis notebooks and CSV summaries are available

### Controlled graph-construction result

Using the fixed DPG 2.0 graph-construction protocol on the curated 15 datasets:

- `aggregated_transitions`: mean fidelity `0.8813`, mean local accuracy `0.7910`, mean margin `0.6367`
- `execution_trace`: mean fidelity `0.8960`, mean local accuracy `0.8021`, mean margin `0.7284`

So the controlled comparison supports the main DPG 2.0 hypothesis:

- `execution_trace` improves local fidelity
- `execution_trace` improves local accuracy
- `execution_trace` improves evidence separation

### Next-phase path-faithfulness result

For the best next-phase configuration per dataset and graph mode:

- `aggregated_transitions`: mean edge precision `0.8732`, mean edge recall `0.8613`, mean recombination rate `0.1268`, mean explanation confidence `0.5047`
- `execution_trace`: mean edge precision `1.0000`, mean edge recall `0.9999`, mean recombination rate `0.0000`, mean explanation confidence `0.6872`

This confirms the structural claim behind DPG 2.0:

- recombination is a real artifact in aggregated local paths
- `execution_trace` removes that artifact almost completely
- confidence improves when the explanation is faithful at the path level

### Consolidated comparison against previous DPG variants and baselines

The current cross-experiment summary over the same 15 datasets is:

- `SHAP`: mean fidelity `1.0000`
- `ICE`: mean fidelity `0.9620`
- legacy best DPG sweep: mean fidelity `0.9074`
- DPG 2.0 `execution_trace`: mean fidelity `0.8956`
- best pruning result: mean fidelity `0.8801`
- DPG 2.0 `aggregated_transitions`: mean fidelity `0.8797`
- `LIME`: mean fidelity `0.4793`

Interpretation:

- DPG 2.0 `execution_trace` clearly improves over the controlled aggregated DPG 2.0 baseline
- DPG 2.0 `execution_trace` also improves over the pruning study in mean fidelity
- the broader legacy DPG sweep still has slightly higher raw fidelity than the restricted DPG 2.0 grid
- `SHAP` and `ICE` remain stronger than DPG on raw fidelity alone
- DPG 2.0's strongest advantage is now structural faithfulness and failure diagnosis, not raw fidelity leadership

### Misclassification and disagreement separation

The new effect-size analysis shows that `execution_trace` produces clearer cohort separation than aggregated DPG.

Examples from the best-config next-phase analysis:

- `execution_trace`, `DISAGREE` vs `AGREE`: explanation confidence drops from `0.6878` to `0.4164` with Cohen's `d = -1.9692`
- `execution_trace`, `DISAGREE` vs `AGREE`: path purity drops from `0.8246` to `0.3864` with Cohen's `d = -2.1752`
- `execution_trace`, `DISAGREE` vs `AGREE`: competitor exposure rises from `0.1754` to `0.6136` with Cohen's `d = 2.1752`
- `execution_trace`, `MW-EM` vs `MC-EC`: explanation confidence drops from `0.7061` to `0.5932` with Cohen's `d = -0.7711`

So DPG 2.0 now has a concrete diagnostic story:

- disagreement cases are structurally lower-confidence
- disagreement cases are more competitor-exposed
- model-error cases also show weaker path purity and weaker explanation confidence

### Artifact locations

The main result artifacts for this phase are:

- `experiments_local_explanation/experiment_dpg2_next_phase/_analysis/best_configs.csv`
- `experiments_local_explanation/experiment_dpg2_next_phase/_analysis/best_cohort_summary.csv`
- `experiments_local_explanation/experiment_dpg2_next_phase/_analysis/best_agreement_summary.csv`
- `experiments_local_explanation/experiment_dpg2_next_phase/_analysis/comparison_story/consolidated_comparison_summary.csv`
- `experiments_local_explanation/experiment_dpg2_next_phase/_analysis/comparison_story/cohort_effect_sizes.csv`
- `experiments_local_explanation/experiment_dpg2_next_phase/_analysis/comparison_story/selected_case_rows.csv`
- `experiments_local_explanation/RQ_dpg2_next_phase.executed.ipynb`

## Next Research Phase

The next phase should stop focusing only on output-level fidelity and start measuring:

- path-level faithfulness
- structural interpretability
- explanation confidence
- error diagnosis on misclassified samples

This is the most natural way to show what DPG 2.0 provides beyond SHAP.

## Exact Local Explanation Confidence

Define the local explanation for sample `x` as a weighted path set:

\[
\mathcal{P}_x = \{(p, w_x(p), y(p))\}
\]

where:

- `p` is a local explanation path
- `w_x(p) \ge 0` is its local contribution weight
- `y(p)` is the class label at the path endpoint

Let the total explained support be:

\[
W_x = \sum_{p \in \mathcal{P}_x} w_x(p)
\]

and define normalized path weights:

\[
\tilde{w}_x(p) = \frac{w_x(p)}{W_x + \varepsilon}
\]

Define class support:

\[
S_x(y) = \sum_{p \in \mathcal{P}_x,\; y(p)=y} \tilde{w}_x(p)
\]

Let:

\[
\hat{y}^{exp}_x = \arg\max_y S_x(y)
\]

be the explanation-predicted class, and let:

\[
y^{(2)}_x = \arg\max_{y \neq \hat{y}^{exp}_x} S_x(y)
\]

be the top competitor.

### 1. Support margin

\[
M_x = S_x(\hat{y}^{exp}_x) - S_x(y^{(2)}_x)
\]

Since the supports are normalized, `M_x \in [0, 1]`.

### 2. Predicted-class concentration

Let `\mathcal{P}_{x,\hat{y}} = \{p \in \mathcal{P}_x : y(p)=\hat{y}^{exp}_x\}` and let `TopK_x(\hat{y}^{exp}_x)` be the `K` highest-weight paths inside that set.

\[
C_x^{(K)} = \frac{\sum_{p \in TopK_x(\hat{y}^{exp}_x)} \tilde{w}_x(p)}{S_x(\hat{y}^{exp}_x) + \varepsilon}
\]

Use `K = 3` by default. High values mean that the predicted-class explanation is carried by a small number of dominant paths rather than diffuse low-weight fragments.

### 3. Model-vote agreement

Let `B` be the number of trees and let:

\[
V_x(y) = \sum_{b=1}^{B} \mathbf{1}[\text{tree } b \text{ predicts } y]
\]

Then define:

\[
A_x = \frac{V_x(\hat{y}^{exp}_x)}{B}
\]

This measures whether the explanation's winning class is also strongly supported by the forest vote.

### 4. Trace coverage

For `execution_trace`, let:

\[
R_x^{node} = \frac{|E_x^{node} \cap T_x^{node}|}{|T_x^{node}| + \varepsilon}
\]

\[
R_x^{edge} = \frac{|E_x^{edge} \cap T_x^{edge}|}{|T_x^{edge}| + \varepsilon}
\]

where:

- `T_x^{node}` and `T_x^{edge}` are the exact executed trace nodes and edges
- `E_x^{node}` and `E_x^{edge}` are the nodes and edges recovered by the local explanation

Define:

\[
G_x = \frac{R_x^{node} + R_x^{edge}}{2}
\]

For `aggregated_transitions`, keep `G_x` undefined and do not report the final confidence as a directly comparable quantity unless an equivalent coverage notion is implemented.

### Final confidence score

\[
\mathrm{Conf}_x = G_x \cdot \frac{M_x + C_x^{(K)} + A_x}{3}
\]

Interpretation:

- `M_x` captures class separation inside the explanation
- `C_x^{(K)}` captures structural concentration
- `A_x` captures agreement with the forest vote
- `G_x` discounts confidence when the explanation misses executed trace structure

This is explicitly a **confidence score for the explanation object**, not model calibration confidence.

## Exact Critical Node

Define the strongest predicted-class path:

\[
p_x^{*} = \arg\max_{p \in \mathcal{P}_x,\; y(p)=\hat{y}^{exp}_x} \tilde{w}_x(p)
\]

and the strongest competitor-class path:

\[
q_x^{*} = \arg\max_{p \in \mathcal{P}_x,\; y(p)=y^{(2)}_x} \tilde{w}_x(p)
\]

Write them as ordered node sequences:

\[
p_x^{*} = [u_1, u_2, \ldots, u_m]
\]

\[
q_x^{*} = [v_1, v_2, \ldots, v_n]
\]

Define the longest common prefix length:

\[
\ell_x = \max \{ t \ge 0 : u_i = v_i \;\; \forall i \le t \}
\]

### Critical node

If `\ell_x \ge 1`, define the critical node as:

\[
c_x = u_{\ell_x} = v_{\ell_x}
\]

If `\ell_x = 0`, there is no shared predicate prefix and the explanation has no critical node under this definition.

### Critical split depth

\[
d_x = \ell_x
\]

This is the depth at which the strongest predicted-class path and strongest competitor path stop sharing structure.

### Critical successors

If both paths continue beyond the shared prefix, define the diverging successors:

\[
s_x^{pred} = u_{\ell_x + 1}, \qquad s_x^{comp} = v_{\ell_x + 1}
\]

### Critical-node branch support

Let `Desc_x(z)` be the set of explanation paths whose node sequence contains node `z` immediately after the common prefix branch. Define downstream branch support:

\[
B_x(z) = \sum_{p \in Desc_x(z)} \tilde{w}_x(p)
\]

Then define critical-node contrast:

\[
\Delta_x^{crit} = |B_x(s_x^{pred}) - B_x(s_x^{comp})|
\]

Large `\Delta_x^{crit}` means that once the explanation diverges at the critical node, one branch quickly dominates the competing branch.

## Path-Level Faithfulness Metrics

The next experiment should compare the explanation directly against the exact executed tree traces, not only against the final model label.

For sample `x`, let:

- `T_x^{node}` be the set of executed trace nodes
- `T_x^{edge}` be the set of executed trace edges
- `E_x^{node}` be the nodes used by the local explanation
- `E_x^{edge}` be the edges used by the local explanation

Define:

### Node recall

\[
\mathrm{NodeRecall}_x = \frac{|E_x^{node} \cap T_x^{node}|}{|T_x^{node}| + \varepsilon}
\]

### Node precision

\[
\mathrm{NodePrecision}_x = \frac{|E_x^{node} \cap T_x^{node}|}{|E_x^{node}| + \varepsilon}
\]

### Edge recall

\[
\mathrm{EdgeRecall}_x = \frac{|E_x^{edge} \cap T_x^{edge}|}{|T_x^{edge}| + \varepsilon}
\]

### Edge precision

\[
\mathrm{EdgePrecision}_x = \frac{|E_x^{edge} \cap T_x^{edge}|}{|E_x^{edge}| + \varepsilon}
\]

### Path purity

\[
\mathrm{PathPurity}_x = \sum_{p \in \mathcal{P}_x,\; y(p)=\hat{y}^{exp}_x} \tilde{w}_x(p)
\]

### Competitor exposure

\[
\mathrm{CompetitorExposure}_x = \sum_{p \in \mathcal{P}_x,\; y(p)\neq \hat{y}^{exp}_x} \tilde{w}_x(p)
\]

### Recombination rate

Let `\Gamma_x` be the set of explanation edges not present in the executed trace:

\[
\Gamma_x = E_x^{edge} \setminus T_x^{edge}
\]

Define:

\[
\mathrm{RecombinationRate}_x = \frac{|\Gamma_x|}{|E_x^{edge}| + \varepsilon}
\]

This metric is central for the `aggregated_transitions` versus `execution_trace` comparison. In `execution_trace`, it should be near zero except for losses induced by filtering or canonicalization.

## Misclassification Analysis Plan

This should become a dedicated experiment rather than a few aggregate tables.

### Cohorts

For each sample `x`, record:

- model prediction `\hat{y}^{model}_x`
- explanation prediction `\hat{y}^{exp}_x`
- ground-truth label `y_x`

Then partition the test set into:

1. `MC-EC`: model correct, explanation correct
2. `MC-EW`: model correct, explanation wrong
3. `MW-EM`: model wrong, explanation matches model
4. `MW-EC`: model wrong, explanation recovers the true label
5. `DISAGREE`: `\hat{y}^{exp}_x \neq \hat{y}^{model}_x`

`DISAGREE` overlaps with the previous groups and should be analyzed separately as a failure-diagnosis bucket.

### Per-sample metrics to log

For every sample in every cohort, log:

- `Conf_x`
- `M_x`
- `C_x^{(K)}`
- `A_x`
- `G_x`
- `NodeRecall_x`
- `NodePrecision_x`
- `EdgeRecall_x`
- `EdgePrecision_x`
- `RecombinationRate_x`
- `PathPurity_x`
- `CompetitorExposure_x`
- critical split depth `d_x`
- critical-node contrast `\Delta_x^{crit}`

### Dataset-level summaries

For each dataset and configuration, report:

- cohort counts and cohort rates
- mean and median of every per-sample metric by cohort
- effect size between `MC-EC` and `MW-EM`
- effect size between `AGREE` and `DISAGREE`

Primary comparisons:

1. `MW-EM` vs `MC-EC`
2. `DISAGREE` vs agreement cases
3. `aggregated_transitions` vs `execution_trace`

### Main hypotheses

1. `MW-EM` samples have lower `Conf_x`, lower `M_x`, and lower `A_x`.
2. `DISAGREE` samples have lower node/edge precision and higher recombination rate.
3. Model errors with high competitor exposure correspond to small `\Delta_x^{crit}` or shallow `d_x`.
4. Low `G_x` predicts explanation unreliability even when output-level fidelity appears acceptable.

## Next Experiment Design

The next phase should be a three-part study run on the same curated 15 datasets.

### Phase A: Path-level faithfulness benchmark

Goal:

- quantify how much of the exact executed trace is preserved by the explanation
- directly measure recombination artifacts

Protocol:

1. Run both `aggregated_transitions` and `execution_trace`.
2. Keep `variant = top_competitor` and fixed `base_lambda = 0.8`.
3. Use the same RF grid as the current DPG 2.0 study.
4. Log node/edge recall, node/edge precision, recombination rate, path purity, competitor exposure, and `Conf_x`.

Primary success criterion:

- `execution_trace` should materially improve edge precision and reduce recombination rate without hurting `local_matches_model_rate`.

### Phase B: Misclassification and disagreement analysis

Goal:

- determine whether explanation failures and model failures have distinct structural signatures

Protocol:

1. Focus on the best one or two configs per dataset from Phase A.
2. Build the five cohorts above.
3. Summarize per-cohort distributions for confidence, faithfulness, and critical-node metrics.
4. Inspect whether disagreement cases are concentrated in low-coverage or high-ambiguity regions.

Primary success criterion:

- the metrics separate correct, wrong, and disagreement cases in a consistent direction across datasets.

### Phase C: Structural case studies

Goal:

- produce interpretable qualitative evidence for the paper

Select four representative samples per dataset family:

1. high-confidence correct prediction
2. low-confidence but correct ambiguous prediction
3. model misclassification with high competitor exposure
4. explanation/model disagreement

For each selected sample, visualize:

- local subgraph
- top predicted-class path
- top competitor-class path
- critical node `c_x`
- critical successors `s_x^{pred}` and `s_x^{comp}`
- the scalar metrics `Conf_x`, `M_x`, `G_x`, `d_x`, and `\Delta_x^{crit}`

## Structural Interpretability Questions To Answer

For each local explanation subgraph, extract:

- active classes in the subgraph
- top competitor class
- top-k predicates by local reaching centrality
- top-k predicted-class paths
- top-k competitor-class paths
- critical node and critical split depth
- explanation confidence `Conf_x`

This supports a paper claim that DPG 2.0 provides:

- class-competitive subgraph explanations
- structurally localized decision points
- explanation-level confidence
- interpretable failure analysis for hard or misclassified samples

## Current Status

The implementation phase is complete and the remaining work is now mainly presentation and paper framing:

- finalize Phase C exemplar figures and curate the strongest qualitative examples
- decide whether to compare DPG 2.0 against the broader legacy DPG search space or only against the controlled DPG 2.0 aggregated baseline
- write the paper narrative so that raw fidelity and structural faithfulness are discussed separately

At this point, the evidence supports the following position:

- DPG 2.0 `execution_trace` is the best DPG variant for structurally faithful local explanations
- DPG still does not dominate SHAP/ICE on raw fidelity alone
- DPG 2.0 is strongest when evaluated as a faithful path-based explanation framework with explicit diagnostic structure
