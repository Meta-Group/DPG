# Reviewer Issue Matrix

This matrix converts the ECML rejection comments into concrete revision obligations for the journal paper.

## Highest Priority Issues

| Reviewer concern | Evidence needed | Writing response | Planned artifact |
| --- | --- | --- | --- |
| Path explanations are old; novelty is modest | Explicit baselines against simple path union, tree-path feature ranking, DPG-global, and DPG-local ablations | Reframe novelty as an ensemble trace-graph diagnostic layer, not as "showing paths" | New path-baseline experiment, ablation table, expanded related work |
| Structural metrics are partly tautological | Diagnostic tasks where DPG-local must predict or explain something not guaranteed by construction | Separate construction checks from diagnostic value; stop using edge precision/recombination as headline empirical wins | Error/disagreement detection, critical-node intervention, control baselines |
| Baseline comparisons are semantically unclear | Formal mapping table from each explainer object to common fields: explained class, score, selected features, competitor, size, runtime | Move shared metrics into a "comparison under constrained interface" subsection; avoid claiming a single winner | Baseline protocol appendix/table and code-level interface |
| ICE comparison seems unjustified | Either remove ICE from main ranking or label it as response-profile/sensitivity context only | Explain ICE is not a local route explainer; use it only for perturbation diagnostics | Revised baseline grouping and tables |
| Small/shallow RFs weaken claims | Deeper RFs, larger ensembles, and at least one boosted-tree family if feasible | Narrow claims if only RFs remain; broaden only after boosted-tree evidence exists | Scalability/depth experiment, optional XGBoost/LightGBM adapters |
| Missing variance and statistical rigor | Multiple seeds, confidence intervals, corrected paired tests, effect sizes | Report mean plus 95% CI and corrected p-values; avoid over-reading n=15 | Statistics script and journal tables |
| Per-dataset tuning may be unfair | Train/validation/test split or nested selection protocol applied equally to DPG and baselines | State selection happens on validation data only, final test untouched | Split registry and config-selection manifest |
| Critical node evidence is weak | Eligibility rate, non-root occurrence rate, probability shifts, label flips conditional on eligibility, controls | Present critical node as conditional diagnostic, not universal explanation primitive | Critical-node validation table and case bank |
| Isolet failure under-discussed | Failure-mode analysis by class count, feature count, graph size, vote entropy, margin, and competitor exposure | Use isolet as an honest boundary case and design lesson | Isolet/madelon failure subsection |
| Reproducibility insufficient | Anonymous repository, environment, exact data splits, scripts, configs, manifests | Add reproducibility section and artifact checklist | Release package and `RUN_MANIFEST.md` |
| Figures hard to read | Larger fonts, simplified top-k views, multi-panel case-study figures | Use full-width journal figures and move dense graphs to appendix | Revised figures and figure-generation scripts |

## Reviewer-Specific Notes

### Reviewer 1

Supportive, but asked for variance, clearer figures, and code. This reviewer can likely be satisfied by:

- 95% confidence intervals across seeds and/or bootstrap samples.
- Bigger, cleaner figures.
- Public package and exact run instructions.

### Reviewer 2

Most philosophical and novelty-focused. The revision must directly address:

- Historical rule-base and decision-tree path explanations.
- Why an ensemble trace graph is not just a decision-tree path printout.
- Why structural faithfulness is useful but not causal explanation.
- Why TreeSHAP, Anchors, LIME, ICE, and DPG-local should not be ranked as if they optimize the same object.
- Why critical nodes are useful only when eligible.

### Reviewer 3

Most experiment-protocol-focused. The revision must add:

- A simple path-union baseline.
- Auditable common-interface mapping for all baselines.
- Larger/deeper model settings or narrowed claims.
- Clear validation/test selection protocol.
- Corrected statistics and confidence intervals.
- Visible runtime and explanation-size results.
- Reproducible code, splits, preprocessing, and environment.
