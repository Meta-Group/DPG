# Review Concerns for main_journal.tex

This document lists specific concerns identified through analysis of the paper text and experimental results.
Each concern includes: the claim made in the paper, the contradicting or missing evidence, the data source, and the suggested fix.

---

## CONCERN 1 — Critical Node Occurs Zero Times in DPG-local (Severity: Critical)

**Claim in paper (Section 3.3 and 4.3 and 4.5):**
The critical node is presented as a DPG-local (execution-trace) diagnostic. Section 4.3 states: "In vehicle, eight samples admit a non-root critical node and the intervention flips toward the competitor branch in 37.5% of those cases." The case study (Section 4.5) shows a critical node in the DPG-local graph of vehicle sample 56. Table 3 reports a pooled "Critical Flip Rate" of 0.025 for DPG-local.

**Contradicting evidence:**
The systematic validation (`critical_node_validation_final/critical_node_dataset_summary.csv`) shows `critical_node_present_rate = 0.000` and `critical_node_present_count = 0` for `dpg_execution_trace` across ALL 15 datasets. Critical nodes only occur for `dpg` (aggregated transitions), where they appear in several datasets (diabetes: 78.6%, digits: 68.3%, banknote: 23.6%, etc.).

**Root cause:**
In DPG-local, each tree executes one independent path to a leaf. Paths from different trees do not share a standardized common prefix in the local graph, so the "longest common prefix between strongest predicted-class path and strongest competitor-class path" is typically the root or empty. Critical nodes emerge in the aggregated DPG because global transition merging creates shared subgraph structure across samples.

**Suggested fix:**
Either (a) explicitly state in Section 3.3 and Section 4.3 that the critical node is currently applicable only to DPG-global and is presented as a prospective structural concept for DPG-local, or (b) redefine the critical node for DPG-local in terms of predicate nodes explicitly shared between the highest-weight predicted-class path and the highest-weight competitor-class path within the local graph, and re-run the validation.

**Check question for LLM:**
Does the paper claim that critical nodes are a validated, empirically demonstrated feature of DPG-local (execution trace), or does it clearly restrict the critical node validation to DPG-global? Does Section 4.3 clearly distinguish which construction mode was used for the "eight samples in vehicle" result?

---

## CONCERN 2 — Top-K Concentration Is Anti-Predictive but Remains in the Confidence Formula (Severity: High)

**Claim in paper (Section 3.2):**
The confidence formula `Conf_x = G_x^cov * (M_x + C_x^(K) + A_x) / 3` assigns equal weight to margin (`M_x`), Top-K concentration (`C_x^(K)`), and vote agreement (`A_x`).

**Contradicting evidence:**
From `confidence_sensitivity/confidence_sensitivity_summary.csv`, for `dpg_execution_trace` predicting local disagreement:
- `risk_low_concentration_only`: AUROC = 0.0876 (essentially random, anti-predictive)
- `risk_low_vote_agreement_only`: AUROC = 0.9808 (strongest signal)
- `risk_low_reported` (full formula): AUROC = 0.9046

From `confidence_sensitivity/confidence_variant_stability.csv`:
- Spearman ρ between reported confidence and concentration-only = -0.6715 (negative correlation)

The Top-K concentration term (`C_x^(K)`) actively works against the diagnostic accuracy of the full formula and is the only component with near-zero predictive value for disagreement detection.

**Suggested fix:**
Section 4.4 should explicitly report the AUROC of concentration-only (0.0876) to justify the statement that "concentration-only performs poorly." The Discussion should explain why: in execution-trace DPGs, evidence concentration reflects ensemble training properties (well-separated classes concentrate paths), not whether the local decision is contested. Consider dropping `C_x^(K)` from the formula or reframing it as a coverage descriptor rather than a disagreement predictor.

**Check question for LLM:**
Does the paper report the AUROC for concentration-only as a standalone diagnostic? Does Section 4.4 explain why Top-K concentration does not carry disagreement signal in DPG-local, or does it only state a general "concentration-only performs poorly" conclusion without the supporting AUROC number?

---

## CONCERN 3 — Model Error Detection Is Entirely Absent from the Paper (Severity: High)

**Claim in paper:**
Section 4.4 evaluates confidence sensitivity only for predicting local disagreement (explanation-model mismatch). No result for predicting model error (model misclassification) appears in the paper text or tables.

**Missing evidence:**
From `confidence_sensitivity/confidence_sensitivity_summary.csv`, for `dpg_execution_trace` predicting **model error**:
- `vote_agreement_only`: AUROC 0.764, AUPRC 0.342
- `reported_confidence`: AUROC 0.752, AUPRC 0.322
- `competitor_exposure`: AUROC 0.747, AUPRC 0.326
- `margin_only`: AUROC 0.737, AUPRC 0.315
- Base rate (mean_positive_rate): 0.144

This is a meaningful second diagnostic task. AUROC 0.75 on a 14% base rate is a useful ranking signal.

**Suggested fix:**
Add a second sub-section or table row in Section 4.4 reporting model error detection AUROCs alongside the local disagreement AUROCs. This directly answers one of the most important XAI questions — "Can this explainer flag when the model is likely to be wrong?" — and the data already exists.

**Check question for LLM:**
Does Section 4.4 report AUROC or AUPRC values for DPG diagnostics predicting model misclassification, separately from the disagreement-detection results? Does Table in Section 4.4 (Diagnostic Value) include a "model_error" target row?

---

## CONCERN 4 — Confidence Ranking Stability (ρ = 1.0) Not Reported (Severity: Medium)

**Claim in paper (Section 4.4):**
Section 4.4 states the confidence score is robust to alternative formulas but does not provide quantitative ranking-stability evidence.

**Missing evidence:**
From `confidence_sensitivity/confidence_variant_stability.csv`, for `dpg_execution_trace` (n = 6079 samples):
- Spearman ρ between reported confidence and multiplicative_gate variant: 1.000
- Spearman ρ between reported confidence and additive_coverage variant: 1.000
- Spearman ρ between reported confidence and no_gate variant: 1.000
- Spearman ρ between reported confidence and margin_only: 0.954
- Spearman ρ between reported confidence and vote_agreement_only: 0.919

For DPG-local, the sample ranking produced by the reported formula is identical to the ranking from three alternative formulations (all ρ = 1.0). This is strong evidence against the reviewer concern (all three ECML reviewers) that the formula is an unjustified heuristic.

**Suggested fix:**
Add a sentence or table to Section 4.4 reporting these Spearman correlations. The claim should be: "the sample ranking produced by the reported formula is essentially invariant to gate choice for DPG-local (ρ = 1.0 for multiplicative, additive, and no-gate variants), confirming that the diagnostic conclusion does not depend on the specific formula."

**Check question for LLM:**
Does Section 4.4 report Spearman rank correlations between the reported confidence score and alternative confidence formula variants? Does the paper quantify the ranking stability of the confidence score across formula alternatives?

---

## CONCERN 5 — GradientBoosting Exclusion Not Disclosed in Paper (Severity: High)

**Claim in paper (Section 4.4, Supported Model Families):**
The paper evaluates DPG-local across four model families: RandomForest, ExtraTrees, AdaBoost, and Bagging. The paper does not mention GradientBoosting, XGBoost, or LightGBM.

**Missing disclosure:**
The model family summary (`model_family_all15_10seeds_test_report/summary.md`) explicitly states: "GradientBoosting is intentionally excluded until a dedicated adapter handles its 2D estimator layout and additive class-contribution semantics."

Gradient-boosted trees (XGBoost, LightGBM, sklearn GradientBoostingClassifier) are the most widely used tree ensembles in practice. All three ECML reviewers noted the limited model scope. Without explicit disclosure, a Q1 reviewer will assume this is an oversight rather than a conscious boundary.

**Suggested fix:**
Add a sentence to Section 4.4 and to the Limitations (Section 5) stating explicitly: "The current execution-trace DPG applies to vote-based tree ensembles where each base learner contributes a class vote; additive gradient-boosted models (XGBoost, LightGBM, sklearn GradientBoostingClassifier) use a different path-aggregation semantics that requires a separate adapter and is out of scope for this evaluation."

**Check question for LLM:**
Does the paper explicitly state that gradient-boosted tree ensembles (XGBoost, LightGBM) are out of scope and explain why? Does Section 4.4 or the Limitations section contain a statement about the boundary between vote-based and additive ensembles?

---

## CONCERN 6 — AdaBoost vs Bagging Disparity Not Discussed (Severity: Medium)

**Claim in paper (Section 4.4, Table 4):**
Table 4 reports mean local match for AdaBoost (0.725) and Bagging (0.899) but does not discuss the practical severity of this difference or where it originates.

**Missing evidence:**
From `model_family_all15_10seeds_test_report/model_family_by_dataset_family.csv`:
- `phoneme`, AdaBoost: local match = 0.369 vs Bagging: 0.923
- `spambase`, AdaBoost: local match = 0.675 vs Bagging: 0.981
- `isolet`, AdaBoost: local match = 0.153 (lowest across all families and datasets)

The difference is not uniform noise — it is concentrated on specific datasets and reflects how AdaBoost's sample-weighted path distribution creates more diffuse local evidence than bootstrap-aggregated ensembles.

**Suggested fix:**
Section 4.4 should note that model-family variation is not uniform: on binary datasets with concentrated class boundaries (phoneme, spambase), Bagging produces substantially higher local match than AdaBoost. The practical implication is that the choice of ensemble family affects DPG diagnostic quality, not just model accuracy.

**Check question for LLM:**
Does Section 4.4 discuss which datasets drive the largest difference between AdaBoost and Bagging in local match rate? Does the paper explain the mechanism behind AdaBoost's lower local match on certain datasets?

---

## CONCERN 7 — Confidence Calibration Non-Monotonicity at Tails Not Disclosed (Severity: Medium)

**Claim in paper (Section 3.2 and 4.4):**
The confidence score is described as a "diagnostic ranking index, not a calibrated probability." No calibration evidence is presented.

**Missing evidence:**
From `confidence_sensitivity/confidence_calibration_bins.csv` for DPG (aggregated):
- Bin 1 (risk 0.19–0.27): observed disagreement rate = 0.0%
- Bin 7 (risk 0.61–0.64): observed rate = 67.2%
- Bin 8 (risk 0.64–0.66): observed rate = 73.4%
- Bin 9 (risk 0.66–0.74): observed rate = 45.6%  ← drops
- Bin 10 (risk 0.74–0.97): observed rate = 42.3% ← drops further

The score correctly separates low-risk from moderate-risk samples, but the highest risk scores (bins 9–10) do not have the highest observed disagreement rates. Calibration is not monotonic at the tails.

**Suggested fix:**
Add a brief discussion noting that the score functions as a reliable ordinal ranker in the middle range but is not monotonically calibrated at the extremes. This is consistent with the paper's stated claim that confidence is a ranking index, but the evidence should be presented rather than asserted.

**Check question for LLM:**
Does the paper present any empirical evidence about the calibration behavior of the confidence score across its value range? Does Section 4.4 or the Discussion note that calibration is non-monotonic at high confidence-risk values?

---

## CONCERN 8 — Vote Agreement Column Missing from Regime Table (Severity: Low)

**Claim in paper (Table 5, Section 4.4):**
Table 5 (Dataset/model regimes) shows: Local match, Confidence, and Competitor exposure. Vote agreement is not shown.

**Missing evidence:**
Vote agreement is the single strongest diagnostic signal for disagreement detection (AUROC 0.9808 for DPG-local). It is directly computable from the model family results (`avg_model_vote_agreement` column). Example values across regimes:
- `banknote-authentication` (low-dim binary): ~0.94
- `isolet` (high-dim many-class, RF): 0.316
- `isolet` (high-dim many-class, AdaBoost): 0.156

Adding this column to Table 5 would show that the regime difficulty is directly visible in vote agreement, making the regime characterization more actionable.

**Suggested fix:**
Add `avg_vote_agreement` as a column to Table 5 alongside confidence and competitor exposure, using the final-test summary values per dataset.

**Check question for LLM:**
Does Table 5 (regime summary) include a column for mean vote agreement alongside confidence and competitor exposure? If not, is vote agreement discussed in the regime analysis section as a per-regime characterizer?

---

## CONCERN 9 — Common Scoring Interface Never Referenced in Paper (Severity: Medium)

**Claim in paper (Section 4.1 and Section 4.2 — Compared Methods, Metrics):**
The paper describes how all methods are compared on shared metrics (fidelity, local accuracy, margin) but does not explain how fundamentally different explanation objects (SHAP attribution vectors, Anchor rules, DPG graphs) are mapped to a common scoring interface.

**Missing disclosure:**
The file `journal_report_final_test/common_explainer_interface.csv` documents exactly this mapping (native object, explained class field, score field, size field, limitations per method). This information exists but does not appear anywhere in the paper.

ECML Reviewer 3 said: "it's completely unclear how you mapped these distinct objects into a common scoring interface. Please add a detailed appendix or table explaining exactly how you extracted class-level scores, competitors, and feature subsets for each baseline. Without this, the benchmark isn't fully auditable."

**Suggested fix:**
Add an appendix table derived from `common_explainer_interface.csv` showing: method, native object type, how the explained class is determined, what score field is used for margin computation, and which limitations apply. This directly resolves the auditing concern.

**Check question for LLM:**
Does the paper include a table or appendix section explaining how each baseline method's native output (SHAP vector, Anchor rule, LORE rule, tree-path ranking, DPG graph) was mapped to the common scoring interface for the shared metrics comparison? Is this documented clearly enough for an independent researcher to reproduce the baseline comparisons?

---

## Summary Table

| # | Concern | Severity | Data source |
|---|---|---|---|
| 1 | Critical node = 0 for DPG-local in systematic validation | **Critical** | `critical_node_dataset_summary.csv` |
| 2 | Top-K concentration anti-predictive (AUROC 0.088) but in formula | **High** | `confidence_sensitivity_summary.csv` |
| 3 | Model error detection results (AUROC 0.75) absent from paper | **High** | `confidence_sensitivity_summary.csv` |
| 4 | Confidence ranking stability (ρ=1.0) not reported | **Medium** | `confidence_variant_stability.csv` |
| 5 | GradientBoosting exclusion not disclosed | **High** | `model_family summary.md` |
| 6 | AdaBoost vs Bagging per-dataset disparity not discussed | **Medium** | `model_family_by_dataset_family.csv` |
| 7 | Confidence calibration non-monotonic at tails not disclosed | **Medium** | `confidence_calibration_bins.csv` |
| 8 | Vote agreement missing from regime Table 5 | **Low** | `model_family_by_dataset_family.csv` |
| 9 | Common scoring interface not documented in paper | **Medium** | `common_explainer_interface.csv` |
