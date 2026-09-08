# Concerns Audit for `main_journal.tex`

Date: 2026-06-19

This audit checks the current journal manuscript against `CONCERNS.md` after the latest experiment and writing passes.

## Status Summary

| # | Concern | Status | Current manuscript handling |
|---|---|---|---|
| 1 | Critical node occurs zero times in execution-trace DPG-local | Resolved with limitation | Critical nodes are now described as conditional descriptors. The old vehicle case-study text was removed, and the semantic-faithfulness table no longer reports a DPG-local critical-flip column. |
| 2 | Top-K concentration is anti-predictive but remains in confidence formula | Resolved with qualification | Section `Confidence Sensitivity and Diagnostic Value` reports concentration-only AUROC 0.0876 and states it should be interpreted as an evidence-distribution descriptor, not a disagreement detector. |
| 3 | Model-error detection absent | Resolved | Section `Confidence Sensitivity and Diagnostic Value` reports execution-trace model-error AUROC/AUPRC for low confidence and vote agreement. |
| 4 | Confidence ranking stability not reported | Resolved | Section `Confidence Sensitivity and Diagnostic Value` reports Spearman rank correlations for alternative confidence variants. |
| 5 | Gradient boosting exclusion not disclosed | Resolved as scope boundary | Experimental setup, model-family results, and limitations now state that additive gradient-boosted models require a separate adapter and are out of scope. |
| 6 | AdaBoost vs Bagging disparity not discussed | Resolved | Section `Supported Model Families and Scalability` discusses the disparity and names `phoneme`, `spambase`, and `isolet`. |
| 7 | Confidence calibration non-monotonicity not disclosed | Resolved with qualification | Section `Confidence Sensitivity and Diagnostic Value` states that the score is ordinal and not perfectly monotonic in the highest-risk bins. |
| 8 | Vote agreement missing from regime table | Resolved | The regime table now includes `Vote agr.`. |
| 9 | Common scoring interface not documented | Resolved | Appendix `Common Scoring Interface` documents native object, explained class, score fields, and size proxy. |

## Additional Completed Gap

The manuscript now includes a nearest-neighbour local-stability analysis. It uses 30,395 kNN pairs for execution-trace DPGs and reports label agreement plus median changes in confidence, support margin, competitor exposure, and vote agreement.

## Remaining Deliberate Limitations

- The confidence formula is retained for continuity and interpretability, but the manuscript no longer treats the exact arithmetic form as the main contribution.
- Critical nodes are not a central DPG-local contribution under the current execution-trace definition.
- Gradient-boosted ensembles remain out of scope until additive path semantics are implemented and validated.
- The current journal version still uses compact tables in LNCS style; a target journal template may allow wider tables or supplementary material.
