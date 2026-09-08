# Critical Nodes And Dataset Properties

## Analysis Added

Script:

- `experiments_local_explanation/analyze_critical_node_dataset_properties.py`

Outputs:

- `experiments_local_explanation/results_journal_v1/critical_node_property_analysis/summary.md`
- `critical_node_dataset_property_table.csv`
- `critical_node_property_correlations.csv`
- `critical_node_favorable_contexts.csv`
- `critical_node_class_level_summary.csv`
- `critical_node_minority_majority_cohorts.csv`

## Main Finding

The current data do not support the hypothesis that critical nodes become more useful mainly under imbalanced datasets/classes.

Dataset-level signals:

- Critical-node occurrence is most strongly associated with DPG recombination rate, not imbalance.
- Critical-node eligibility is lower in high feature-per-sample / high-dimensional settings.
- Advantage over the random-path control is more associated with model accuracy than class imbalance.
- Imbalance features are present in the analysis, but they are not the strongest predictors of critical-node usefulness.

Class-level signal:

- Minority target classes do not show a positive advantage over the random-path control.
- For target minority classes, critical changed-to-competitor rate is about 0.47%, while random-path control is about 6.37%.
- The most favorable single contexts are isolated cases such as banknote-authentication and some diabetes/digits classes, not a broad imbalance pattern.

## Recommended Claim

Do not claim that critical nodes are especially important in imbalanced scenarios.

Safer journal wording:

> We additionally explored whether critical-node behavior is associated with dataset-level imbalance, class prevalence, dimensionality, and model difficulty. In this exploratory analysis, imbalance was not the dominant factor. Critical-node occurrence was more closely tied to recombination structure, and intervention advantage over controls was not consistently stronger for minority classes. We therefore treat critical nodes as conditional structural diagnostics rather than as imbalance-specific explanatory mechanisms.

## Paper Placement

Main paper:

- Mention briefly as a negative/clarifying analysis if space allows.

Appendix:

- Include the property correlation table and minority/majority cohort table.

Discussion:

- Use this to show that the journal revision tested a plausible reviewer concern and bounded the claim.
