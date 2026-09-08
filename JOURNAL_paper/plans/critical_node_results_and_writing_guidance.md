# Critical Node Validation: Journal Writing Guidance

## Current Result

The corrected final-test critical-node validation is available at:

- `experiments_local_explanation/results_journal_v1/critical_node_validation_final_fixed/`
- integrated report: `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/summary.md`

Main aggregated-DPG result over 15 datasets and 6079 held-out samples:

- Mean critical-node present rate: 19.6%.
- Mean intervention-eligible rate: 6.0%.
- Mean label-changed rate after competitor-branch intervention: 2.6%.
- Mean changed-to-competitor rate: 2.5%.
- Same-depth control changed-to-competitor rate: 1.7%.
- Random-path control changed-to-competitor rate: 4.8%.
- Execution-trace DPG has 0 critical nodes under the current definition.

## Interpretation

The result should not be framed as strong causal evidence that critical nodes drive the model decision. The intervention signal is mixed: the random-path control changes to the competitor more often than the critical-node intervention on average.

The result is still useful because it clarifies the scope of the construct. Critical nodes are a conditional structural diagnostic that appears only when the aggregated DPG creates a non-root divergence between the predicted class path and the strongest competitor path. This makes them a secondary interpretability object rather than the main empirical claim.

## Recommended Manuscript Framing

Use confidence, support margin, and competitor exposure as the main diagnostic-value evidence, especially because they predict local disagreement well in the final test.

Use critical nodes as a descriptive structural analysis:

> Critical nodes identify a subset of cases where the aggregated DPG exposes a non-root divergence between predicted-class and competitor-class evidence. In our final-test validation, these nodes are present in a minority of held-out cases and are intervention-eligible in a smaller subset. Intervention outcomes are mixed relative to controls, so we report them as conditional diagnostic evidence rather than as causal proof.

## Placement

Main paper:

- One paragraph in the DPG diagnostics/results section.
- One compact table row or small table with present rate, eligible rate, changed-to-competitor rate, and controls.
- Explicit limitation sentence.

Appendix:

- Full dataset-level critical-node table.
- Intervention protocol details.
- Explanation that execution-trace DPG has no critical nodes under this recombination-based definition.

## Claim Boundary

Avoid:

- "Critical nodes causally explain model decisions."
- "Critical-node interventions outperform controls."
- "Critical nodes are universal DPG objects."

Prefer:

- "Critical nodes provide conditional structural diagnostics."
- "The strongest empirical diagnostic evidence comes from DPG confidence, support margin, and competitor exposure."
- "The critical-node experiment is included to test, and bound, a stronger causal interpretation."
