# Numerical Consistency Audit for `main_journal.tex`

Date: 2026-06-19

This audit checks the main numerical claims in the current journal manuscript against the available experiment artifacts.

## Source Artifacts Checked

- Main final-test report: `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/summary.md`
- Selected final-test tables: `experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/summary_selected_test.csv` and `per_sample_selected_test.csv`
- Confidence sensitivity: `experiments_local_explanation/results_journal_v1/confidence_sensitivity/summary.md`
- Local stability: `experiments_local_explanation/results_journal_v1/local_stability_knn/summary.md`
- Readability/cost: `experiments_local_explanation/results_journal_v1/readability_cost_analysis/summary.md`
- Model-family experiment: `experiments_local_explanation/results_journal_v1/model_family_all15_10seeds_test_report/summary.md`
- Scalability: `experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_report/summary.md`
- Critical-node validation: `experiments_local_explanation/results_journal_v1/critical_node_validation_final_fixed/summary.md`
- Semantic faithfulness: `experiments_local_explanation/results_journal_v1/semantic_faithfulness/semantic_faithfulness_summary.csv`

## Verified Claims

| Manuscript topic | Status | Notes |
| --- | --- | --- |
| Main final-test sample count | Verified | Each method has 6,079 final-test explanations; 10 methods give 60,790 per-sample rows. |
| Dataset test counts | Verified | Counts sum to 6,079: 275, 114, 154, 360, 71, 30, 1,560, 520, 1,081, 211, 462, 921, 170, 114, 36. |
| DPG ablation means | Verified | Dataset-level means: local match 0.825 to 0.860, local accuracy 0.752 to 0.783, confidence 0.477 to 0.654, support margin 0.617 to 0.689, recombination 0.134 to 0.000. |
| Structural summary | Verified | Dataset-level execution-trace DPG: 20.00 paths, 63.96 active nodes, edge precision 1.00, edge recall 0.9999, recombination 0.00. |
| Cohort paragraph | Verified | Per-sample execution-trace DPG: MC-EC 61.3%, confidence 0.676, path purity 0.839, exposure 0.161; MW-EM confidence 0.546, path purity 0.678. |
| Agreement/disagreement split | Verified | Per-sample execution-trace DPG: agreement confidence 0.656, margin 0.691, vote agreement 0.813, exposure 0.186; disagreement confidence 0.399, margin 0.112, vote agreement 0.255, exposure 0.622. |
| Confidence sensitivity | Verified | Execution-trace DPG: low confidence AUROC 0.9046/AUPRC 0.6509 for disagreement; vote agreement AUROC 0.9808/AUPRC 0.9104; concentration-only AUROC 0.0876. |
| Model-error diagnostic | Verified | Execution-trace DPG: low confidence AUROC 0.7523/AUPRC 0.3216; vote agreement AUROC 0.7638/AUPRC 0.3417; positive rate about 0.144. |
| Local stability | Verified | 30,395 kNN pairs; explained-label agreement 0.785; model-label agreement 0.795; median absolute changes 0.024, 0.044, 0.029, 0.050. |
| Regime examples | Verified | `isolet` and `madelon` values match the final-test/regime artifacts. |
| Model-family experiment | Verified | 15 datasets x 4 families x 10 seeds = 600 runs; family means in the manuscript match the report. |
| Readability table | Verified | Values come from the sample-level readability analysis: DPG-local 72.5 active nodes, path controls 666.4 predicates, TreeSHAP 55.0, LORE 3.2, Tree-path 3.9. |
| Scalability paragraph | Verified | Depth means are about 119 ms, 4.38 s, and 14.63 s for depths 4, 8, and 12. |
| Critical-node limitation | Verified | Execution-trace DPG has zero eligible critical nodes; aggregated DPG present rate is 0.1962 and changed-to-competitor rate is 0.02463. |
| Semantic-faithfulness table | Verified with scope note | Values match the focused semantic-faithfulness analysis: DPG-local 0.535/0.130, LIME 0.548/0.166, ICE 0.476/0.069, rounded in the manuscript. |

## Manuscript Changes Made During Audit

- Added an explicit aggregation-basis statement in the statistical-analysis subsection: main results are dataset-level unless stated otherwise; readability and kNN stability are sample-level.
- Clarified that the readability comparison is sample-level.
- Added the semantic-faithfulness sample count: 2,975 evaluated instances across the 15 datasets.
- Mirrored the semantic-faithfulness artifacts into `experiments_local_explanation/results_journal_v1/semantic_faithfulness/` and added a provenance README with SHA-256 checksums.

## Residual Risks

- The current manuscript mixes dataset-level and sample-level summaries, but now states this explicitly. A journal template with more space should move some sample-level details to supplementary material.
- The semantic-faithfulness files are now mirrored under `results_journal_v1` and match the original focused analysis byte-for-byte. For a final public reproducibility package, rerunning the semantic-faithfulness script directly into the journal root would provide an even cleaner execution trail.
- `main_journal.tex` remains in LNCS format; some table and float warnings are layout-related rather than numerical issues.
