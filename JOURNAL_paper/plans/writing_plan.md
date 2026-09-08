# Journal Writing Plan

## Core Reframing

The ECML version argued that output-level agreement is insufficient and that DPG-local provides structurally faithful local explanations. The journal version should make a sharper and safer claim:

> DPG-local is a structure-aware diagnostic complement for tree ensembles. It is useful when the analyst wants to inspect executed predicate transitions, class competition, and local decision gates, not when the only goal is output agreement.

This directly answers the reviewers who objected that TreeSHAP, Anchors, ICE, and DPG-local optimize different explanation objects.

## Title Direction

Avoid sounding like the paper invents path explanations generally. Candidate title pattern:

> Execution-Trace Decision Predicate Graphs for Structure-Aware Diagnostics of Random Forest Predictions

If boosted-tree experiments succeed, use "tree ensembles"; otherwise use "random forests" or "tree ensembles with extractable execution traces."

## Abstract Rewrite

The abstract should:

- Acknowledge that path-based explanations have a long history in rule systems and decision trees.
- State the actual gap: local, contrastive, auditable trace graphs for ensembles.
- Separate construction consistency from diagnostic value.
- Name the new validation experiments: path baselines, critical-node interventions, confidence-score sensitivity, scalability, and corrected statistics.
- Avoid saying DPG-local "competes" with output-aligned explainers as if they solve the same task.

## Introduction Rewrite

New structure:

1. Local explanation methods answer different questions.
2. Output agreement answers "does the explanation reproduce the model's prediction?"
3. Structural diagnosis asks "which executed predicates and branches carried support, where did alternatives diverge, and when is the decision contested?"
4. Historical path explanations exist, but ensemble-level trace aggregation creates new issues: many independently trained trees, overlapping predicates, contradictory paths, and class competition.
5. DPG-local's contribution is not displaying a path; it is a canonical, sample-specific predicate-transition graph plus contrastive diagnostics and validation checks.

Tone change:

- Replace broad novelty claims with precise contribution claims.
- Say "diagnostic complement" early and repeatedly.
- State the method is not causal explanation; it is structural faithfulness to the model's executed trace.

## Related Work Expansion

Add or strengthen subsections:

- Historical rule-base and decision-tree path explanations.
- Rule extraction and rule ensembles.
- Tree ensemble explanations and path-based feature contributions.
- Feature attribution and local surrogate methods.
- Contrastive explanations and explanation disagreement.
- Structural faithfulness versus output fidelity.

The key paragraph should explicitly say:

- Single-tree path explanations are old and useful.
- DPG-local differs because it handles ensembles by canonicalizing predicates, preserving transition topology across executed traces, quantifying class-contrastive support, and exposing recombination/coverage diagnostics.

## Method Rewrite

Clarify three layers:

1. Execution trace extraction: exact root-to-leaf paths from each tree.
2. DPG-local graph construction: canonical predicate-transition graph for one sample.
3. Semantic Graph View: diagnostic summaries over the graph.

Important edits:

- Mark edge precision and zero recombination as construction-consistency checks, not standalone evidence of usefulness.
- Describe explanation confidence as a heuristic diagnostic index unless Phase 6 validates or recalibrates it.
- Add a subsection "What structural faithfulness does and does not mean." State that it is faithful to the model execution trace, not necessarily causal or semantically human-faithful.
- Add complexity and readability controls, including top-k pruning or support thresholds if implemented.

## Evaluation Rewrite

Replace the current single benchmark framing with research questions.

Suggested RQs:

- RQ1: Does execution-trace construction preserve the executed predicate-transition structure better than aggregated DPG construction?
- RQ2: Does the Semantic Graph View provide diagnostic signal for model errors, low-margin decisions, and explanation/model disagreement beyond simpler path baselines?
- RQ3: When non-root critical nodes exist, do they identify meaningful contrastive decision gates compared with controls?
- RQ4: How do graph size, runtime, and readability scale with tree depth, ensemble size, dataset dimension, and number of classes?
- RQ5: How does DPG-local relate to output-aligned, surrogate, rule, and response-profile explainers under a documented common interface?

Baseline grouping:

- Structure-aware methods: DPG-local, DPG-global, raw path union, path bag.
- Tree-native output/path methods: TreeSHAP, tree-path feature contribution, Anchors.
- Local surrogate methods: LIME.
- Response-profile methods: ICE, sensitivity-only.

The evaluation section must include a table mapping each native explanation object to shared fields and unsupported fields.

## Results Rewrite

Recommended order:

1. Protocol and selected configurations.
2. Structural construction checks: trace recall, precision, recombination.
3. Diagnostic value: disagreement/error detection and effect sizes.
4. Critical-node validation: eligibility, probability shifts, flips conditional on eligibility, controls.
5. Confidence-score sensitivity.
6. Scalability: depth, number of trees, graph size, runtime, readability.
7. Shared output-level metrics as context, not as the primary contest.
8. Failure modes: isolet and madelon.

This ordering prevents reviewers from reading the paper as "we lost to TreeSHAP on fidelity but still claim superiority."

## Discussion Rewrite

Address Reviewer 2's conceptual critique directly:

- A random forest is not a single coherent reasoning chain.
- DPG-local does not claim to recover a human-semantic or causal reason.
- It recovers and summarizes the model's executed predicate-transition evidence.
- This is valuable for audit/debug tasks where the analyst needs to know how support was distributed across tree paths and where alternatives diverged.

Also discuss:

- When DPG-local is useful: auditing, debugging, disagreement analysis, contested predictions, model inspection.
- When it is less useful: very high-dimensional/many-class settings, very deep forests without pruning, tasks needing causal explanations, tasks needing only output agreement.
- How DPG-local should be paired with TreeSHAP or Anchors in practice.

## Figures and Tables

Main paper:

- One clean conceptual diagram with larger labels.
- One baseline mapping/protocol table.
- One structural-vs-diagnostic results table.
- One corrected-statistics table with confidence intervals.
- One critical-node validation table with eligibility and conditional effects.
- One scalability table or line plot for depth/trees versus graph size/runtime.
- Two readable case-study figures max.

Appendix/supplement:

- Full per-dataset heatmaps.
- Full case bank.
- Full hyperparameter grid and selected configs.
- Detailed baseline mappings.
- Environment and run manifest.

Figure requirements:

- Increase font sizes.
- Avoid dense full-graph figures in the main text unless pruned.
- Prefer top-k or focused subgraphs with a companion table of omitted context.
- Report graph size and pruning threshold in every graph caption.

## Claim Edits

Replace:

- "DPG-local achieves competitive results against TreeSHAP, ICE, LIME, and Anchors"

With:

- "DPG-local is not optimized for output agreement; it provides complementary structure-aware diagnostics. Output-aligned methods remain preferable when reproducing the final prediction is the sole objective."

Replace:

- "near-complete trace recovery confirms structural faithfulness"

With:

- "near-complete trace recovery validates the construction; diagnostic utility is assessed separately through disagreement, error, and critical-node analyses."

Replace:

- "critical node identifies where predicted and competing paths diverge"

With:

- "when a non-root shared prefix exists, the critical node identifies the last shared predicate between the strongest predicted and competitor routes; its practical value is evaluated conditionally."

## Journal Submission Strategy

Do the venue-choice pass after Phase 4 or Phase 5, when the strength of the added evidence is clear. The safest target is a journal that welcomes longer empirical validation and reproducibility artifacts in interpretable machine learning or data mining.

Before submission:

- Decide whether the scope is "random forests" or broader "tree ensembles."
- Ensure code and splits are public or anonymized according to the journal's review model.
- Prepare a cover letter that frames the ECML rejection as useful feedback already addressed by new experiments.
