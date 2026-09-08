# Journal Readiness Memo

Date: 2026-06-19

## Current Position

The journal version is now substantially stronger than the ECML submission. The contribution should be framed as:

> Decision Predicate Graphs provide a structure-aware local diagnostic representation for vote-based tree-ensemble classifiers.

The safest claim is not that DPG-local is a better local explainer than TreeSHAP, LIME, or Anchors. The stronger and more defensible claim is that DPG-local exposes additional local diagnostic information that feature-attribution and rule summaries do not provide: executed predicate-transition recovery, predicted-versus-competitor support, vote agreement, competitor exposure, disagreement detection, stability over neighbouring samples, and clear regime/failure signals.

## Strongest Evidence Now Available

- Full locked final-test evaluation on 15 datasets and 6,079 test samples per method.
- Path-control baselines are included, addressing the concern that DPG-local should be compared against simpler path-based representations.
- ICE is appendix-only, addressing the concern that ICE is not a suitable main local route/path comparator.
- Confidence sensitivity is reported, including concentration-only failure, model-error detection, and ranking stability across formula variants.
- The model-family experiment covers RandomForest, ExtraTrees, AdaBoost, and Bagging over all 15 datasets and 10 seeds.
- kNN local-stability analysis adds a formal robustness view using naturally neighbouring samples.
- Critical-node claims are now bounded and no longer presented as a central execution-trace DPG-local result.
- The common scoring interface is documented in an appendix table.
- Readability/cost and scalability are now surfaced.

## Remaining Risks

- The contribution is still incremental if described as merely "showing tree paths"; the writing must keep emphasizing graph diagnostics and class-contrastive analysis.
- The current evaluated model scope excludes gradient-boosted trees. This is acceptable only because the manuscript explicitly frames them as out of scope due to additive path semantics.
- Critical nodes are a weak empirical contribution under the current execution-trace definition; they should remain secondary.
- The current tables are dense because the paper is still in an LNCS-style template. A journal template may allow better table placement or supplementary material.
- User-oriented validation is still absent. This should be future work, not claimed as demonstrated utility.

## Best-Fit Journal Direction

Information Sciences is currently the safer Q1 target than Information Fusion. The Information Fusion route is possible only if the framing emphasizes fused path evidence and local decision-evidence aggregation, but the present evidence reads more naturally as graph-based XAI and diagnostic analysis for intelligent systems.

## Next Editorial Tasks

1. Perform a full language pass for formal journal tone and remove remaining conference-style compression.
2. Move dense audit/support material to appendix or supplementary form when the final target journal template is selected.
3. Re-check every numeric result against its source CSV immediately before submission.
4. Prepare a reproducibility appendix listing scripts, data splits, environment assumptions, and exact output directories.
5. Decide target journal and adapt structure, length, citation style, and figure/table placement accordingly.
