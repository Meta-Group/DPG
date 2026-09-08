# Journal Next Steps Handoff

Date: 2026-06-19

Use this as the single active memory file for the next work session. The other Markdown files in `JOURNAL_paper/plans/` are now archival background unless a specific detail is needed.

## Current Objective

Prepare a stronger Q1 journal version of the rejected ECML DPG-local paper.

Update after latest manuscript pass:

- `JOURNAL_paper/main_journal.tex` now incorporates the main `CONCERNS.md` fixes already supported by the existing artifacts.
- A targeted journal-style language pass has been applied to `JOURNAL_paper/main_journal.tex`. The edits remove informal project-language, clarify the local graph construction, tighten the common-metrics description, shorten compact table labels, and avoid unsupported overgeneral wording.
- `JOURNAL_paper/main_journal.pdf` was rebuilt successfully after the language pass with `latexmk -pdf -interaction=nonstopmode -halt-on-error main_journal.tex`; the PDF remains 21 pages. Remaining warnings are minor underfull/overfull layout warnings in compact LNCS tables/paragraphs, not compilation errors.
- A numerical consistency audit has been completed and saved at `JOURNAL_paper/NUMERICAL_CONSISTENCY_AUDIT_2026-06-19.md`. Main numerical claims were checked against the available final-test, confidence, stability, readability, model-family, scalability, critical-node, and semantic-faithfulness artifacts.
- The audit found no required numerical corrections. It added manuscript clarifications that main aggregates are dataset-level unless otherwise stated, that readability and kNN stability are sample-level analyses, and that semantic faithfulness used 2,975 evaluated instances.
- Semantic-faithfulness artifacts have been mirrored into `experiments_local_explanation/results_journal_v1/semantic_faithfulness/` with a provenance README and SHA-256 checksums, so the manuscript values now have a journal-root artifact path.
- A compact reproducibility appendix has been added to `JOURNAL_paper/main_journal.tex`. It summarises datasets, splits, seeds, baselines, model-family scope, semantic-faithfulness scope, kNN stability, statistical tests, and the main journal-root artifact paths.
- `JOURNAL_paper/main_journal.pdf` was rebuilt successfully after adding the reproducibility appendix; the PDF is now 22 pages. Remaining warnings are layout/float warnings in compact LNCS formatting, not compilation errors.
- The per-dataset shared-metrics heatmap has been regenerated from `final_test_main_with_lore_path_controls/summary_selected_test.csv` using `experiments_local_explanation/plot_journal_shared_metrics_heatmap.py`. The updated `JOURNAL_paper/per_dataset_shared_metrics_heatmap.pdf` excludes ICE and includes LORE-style in the main route/path comparison.
- The execution-trace critical-node claim has been downgraded to a bounded/conditional descriptor, consistent with the validation showing zero eligible cases under the current non-root shared-prefix definition.
- The paper now reports model-error diagnostic value, confidence ranking stability, concentration-only anti-predictiveness, explicit gradient-boosting exclusion, AdaBoost-vs-Bagging disparity, vote agreement in the regime table, and an appendix common-scoring-interface table.
- The old `vehicle` critical-node case-study figure was removed from the main text to avoid overstating unsupported execution-trace evidence.
- A kNN local-stability experiment has been implemented and added to the manuscript. It compares each final-test sample with its five nearest test neighbours in standardized input space and reports explained-label agreement plus absolute diagnostic changes.
- A concern-by-concern audit has been written at `JOURNAL_paper/CONCERNS_AUDIT_2026-06-19.md`. All nine tracked concerns are now either resolved in the manuscript or explicitly framed as deliberate limitations.
- The semantic-faithfulness table was corrected to remove the misleading DPG-local critical-flip column; critical nodes are now consistently described as conditional descriptors rather than a core execution-trace result.
- A journal-readiness memo has been written at `JOURNAL_paper/JOURNAL_READINESS_MEMO_2026-06-19.md`. It summarizes the current contribution framing, strongest evidence, remaining risks, and next editorial tasks.
- The regime table and common-scoring-interface appendix table have been constrained to text width to reduce the most visible LaTeX layout issues in the current LNCS-style draft.

The strongest framing is no longer:

> DPG-local is another local explainer.

Use instead:

> Decision Predicate Graphs enable local diagnostic analysis of tree-ensemble classifiers by converting executed tree paths into canonical predicate-transition graphs. This supports trace recovery checks, class-contrastive support analysis, competitor exposure, disagreement detection, critical-node boundary analysis, and scalability/readability diagnostics.

This framing keeps Information Fusion possible because tree paths can be described as decision evidence and the DPG as the fused local representation. Information Sciences remains the safer target if the paper is framed as graph-based XAI for intelligent systems.

## Repository State

Main paper directories:

- Submitted/rejected snapshot: `paper_ecml_rejected/`
- Journal working paper: `JOURNAL_paper/`
- Current manuscript file: `JOURNAL_paper/main_journal.tex`
- Do not edit `JOURNAL_paper/ECML26_DPG.tex`; it is kept as a restored ECML reference copy.

Important review file:

- `JOURNAL_paper/reviews/ecml_rejection_comments.txt`

Active experiment root:

- `experiments_local_explanation/results_journal_v1/`

## Main Reviewer Concerns And Current Status

| Concern | Current status |
| --- | --- |
| ICE is not a suitable main comparator | Addressed. ICE is appendix-only in the current manuscript. |
| Need simple path baselines | Addressed experimentally with `raw_path_union`, `path_bag`, and `random_same_size_path`. |
| Heterogeneous explainer comparison unclear | Addressed as an artifact via `common_explainer_interface.csv`; still needs manuscript appendix table. |
| Structural metrics are tautological | Partially addressed. Present edge precision/recombination as construction checks; use diagnostic AUROC/AUPRC as value evidence. |
| Critical nodes weak | Addressed as a bounded/conditional diagnostic, not a main contribution. |
| Missing uncertainty/corrected tests | Addressed experimentally with CIs and Holm-corrected paired tests. |
| Shallow RandomForest only | Partially addressed with depth/ensemble scalability and new model-family experiment. |
| Runtime/size invisible | Addressed experimentally; must be surfaced in the manuscript. |
| Isolet/madelon failure under-discussed | Addressed experimentally; needs manuscript failure-mode discussion. |
| Confidence formula heuristic | Addressed with confidence sensitivity analysis. |
| Broad tree-ensemble claim | Still requires careful wording. Use "tested DPG-compatible sklearn classification ensembles": RandomForest, ExtraTrees, AdaBoost, and Bagging. |

## Completed Key Analyses

### 1. Main Final-Test Report

Path:

- `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/summary.md`

Important outputs:

- `method_summary_ci.csv`
- `size_runtime_summary_ci.csv`
- `paired_tests_holm.csv`
- `diagnostic_value_summary.csv`
- `common_explainer_interface.csv`
- `failure_focus_dataset_method.csv`

Key result:

- DPG low-confidence predicts local disagreement well.
- `dpg`: AUROC about 0.900.
- `dpg_execution_trace`: AUROC about 0.905.
- Path controls match output but are much larger and lack DPG semantic diagnostics.

### 2. Critical-Node Validation

Path:

- `experiments_local_explanation/results_journal_v1/critical_node_validation_final_fixed/summary.md`

Key result:

- Aggregated DPG critical nodes present in about 19.6% of held-out cases.
- Intervention-eligible about 6.0%.
- Changed-to-competitor about 2.5%.
- Random-path control changed-to-competitor about 4.8%.
- Execution-trace DPG has no critical nodes under the current recombination-based definition.

Writing stance:

- Critical nodes are conditional structural diagnostics.
- Do not claim causal decision explanation.
- Do not present critical nodes as a central contribution.

### 3. Critical-Node Dataset Property Analysis

Path:

- `experiments_local_explanation/results_journal_v1/critical_node_property_analysis/summary.md`

Key result:

- Imbalance is not the main driver.
- Critical-node occurrence is more associated with recombination structure.
- Minority-class scenarios do not show a reliable critical-node advantage over controls.

Writing stance:

- Report as exploratory boundary analysis.

### 4. Depth/Ensemble Scalability

Path:

- `experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_report/summary.md`

Scope:

- Datasets: `iris`, `banknote-authentication`, `diabetes`, `vehicle`, `madelon`, `isolet`
- Execution-trace DPG
- `n_estimators`: 20, 50
- `max_depth`: 4, 8, 12

Key result:

- Trace faithfulness remains high and recombination remains zero.
- Runtime and graph size grow sharply with depth and forest size.
- Depth 12 / 50 trees can be expensive, especially on high-dimensional datasets.

Writing stance:

- Use as bounded scalability evidence.
- State practical deployment may need pruning/sampling.

### 5. Confidence-Score Sensitivity

Script:

- `experiments_local_explanation/analyze_confidence_sensitivity.py`

Output:

- `experiments_local_explanation/results_journal_v1/confidence_sensitivity/summary.md`

Key result:

- Reported low-confidence remains strong for local-disagreement detection:
  - `dpg`: AUROC 0.9001.
  - `dpg_execution_trace`: AUROC 0.9046.
- Low vote agreement is even stronger:
  - `dpg`: AUROC 0.9237.
  - `dpg_execution_trace`: AUROC 0.9808.
- Concentration-only is weak and should not be emphasized alone.

Writing stance:

- Confidence is a diagnostic ranking index, not a calibrated probability.
- Emphasize the broader DPG diagnostic family: vote agreement, support margin, competitor exposure, and confidence.

### 6. DPG-Supported Model-Family Experiment

Implementation:

- New model factory: `experiments_local_explanation/model_factory.py`
- Updated runner: `experiments_local_explanation/run_local_experiments.py`
- New CLI option: `--model_families`

Smoke output:

- `experiments_local_explanation/results_journal_v1/model_family_smoke/`

Focused model-family output:

- `experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset/summary.csv`
- `experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset/per_sample.csv`
- `experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset/dataset_overview.csv`
- `experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset_report/summary.md`

Scope:

- Datasets: `banknote-authentication`, `iris`, `vehicle`, `madelon`, `isolet`
- Model families: `random_forest`, `extra_trees`, `adaboost`, `bagging`
- `n_estimators`: 20
- `max_depth`: 4
- Seed: 27
- Graph mode: `execution_trace`
- Max test samples: 50

Run status:

- Completed successfully.
- 20/20 runs completed.
- No local failures.
- Recombination rate is 0.0 for all runs.
- Edge recall is approximately 1.0 across runs.

Important model-family result:

- `isolet` remains hard across model families, with low local match and high competitor exposure.
- Bagging tends to produce strong local match on several datasets.
- ExtraTrees can produce weaker local match on some datasets despite high trace recovery.
- This supports a model-family-dependent diagnostic story.
- By model-family mean over the five-dataset subset, Bagging has the highest local match rate (0.88), followed by RandomForest (0.749), AdaBoost (0.705), and ExtraTrees (0.672).
- All four families have zero recombination and near-perfect edge recall in execution-trace mode, so construction robustness is strong while diagnostic behavior differs.

Model-family scope:

- Keep the journal experiment focused on RandomForest, ExtraTrees, AdaBoost with tree base learners, and Bagging with tree base learners.
- Do not introduce untested model families in the manuscript; this keeps the experimental story cohesive.

### 7. Readability/Cost Analysis

Script:

- `experiments_local_explanation/analyze_readability_costs.py`

Output:

- `experiments_local_explanation/results_journal_v1/readability_cost_analysis/summary.md`
- `experiments_local_explanation/results_journal_v1/readability_cost_analysis/readability_by_method.csv`
- `experiments_local_explanation/results_journal_v1/readability_cost_analysis/readability_by_group.csv`
- `experiments_local_explanation/results_journal_v1/readability_cost_analysis/readability_dpg_ratios.csv`
- `experiments_local_explanation/results_journal_v1/readability_cost_analysis/readability_dpg_dataset_pressure.csv`

Scope:

- Reuses `final_test_main_with_lore_path_controls/per_sample_selected_test.csv`.
- Measures display-size proxies and runtime/size trade-offs across the locked final-test methods.
- For DPG, display size is active graph nodes.
- For path controls, display size is unique predicate count.
- For SHAP/LIME, display size is nonzero feature contribution count.
- For rule/surrogate methods, display size is rule length when available.

Key result:

- Execution-trace DPG averages about 72.5 active nodes, with p90 about 88.
- Path controls average about 666 unique predicates, with p90 about 1194.
- Path controls are about 9.2x larger than execution-trace DPG by this display-size proxy.
- Tree-path and LORE-style summaries are much smaller, but they discard predicate-transition topology and predicted-versus-competitor branch structure.
- Highest-cost DPG datasets by graph size are `isolet`, `madelon`, and `phoneme`; `isolet` is also the clearest hard diagnostic regime because local match is low and competitor exposure is high.

Writing stance:

- Use this as readability/cost evidence, not as a new fidelity benchmark.
- Do not claim universal readability superiority.
- Claim that DPG-local provides a structured middle ground: much more compact than raw path controls while retaining topology and class-contrast diagnostics absent from tree-path feature summaries.
- This supports the need for top-k/support-threshold views in journal figures and as future work.

### 8. Local Stability Analysis

Script:

- `experiments_local_explanation/analyze_local_stability.py`

Output:

- `experiments_local_explanation/results_journal_v1/local_stability_knn/summary.md`
- `experiments_local_explanation/results_journal_v1/local_stability_knn/local_stability_summary.csv`
- `experiments_local_explanation/results_journal_v1/local_stability_knn/local_stability_by_dataset.csv`
- `experiments_local_explanation/results_journal_v1/local_stability_knn/local_stability_correlations.csv`
- `experiments_local_explanation/results_journal_v1/local_stability_knn/local_stability_pairs.csv`

Scope:

- Reuses the locked final-test per-sample artifact.
- For each dataset and method, compares each final-test sample with its five nearest final-test neighbours in standardized input space.
- Covers all 15 datasets and all final-test methods.

Key result for execution-trace DPG:

- 30,395 nearest-neighbour pairs.
- Explained-label agreement: 0.785.
- Model-label agreement on the same pairs: 0.795.
- Median absolute diagnostic changes:
  - confidence: 0.024
  - support margin: 0.044
  - competitor exposure: 0.029
  - vote agreement: 0.050
- Spearman correlations between input distance and diagnostic change:
  - confidence: 0.234
  - support margin: 0.291
  - competitor exposure: 0.326
  - vote agreement: 0.249

Writing stance:

- Use this as local robustness/stability evidence.
- Do not claim DPG is more stable than feature-ranking baselines in all senses.
- Claim that DPG diagnostic quantities vary smoothly for nearby samples when the local model decision is stable, while larger diagnostic shifts help identify locally contested regions.

## Current Best Contribution Set

1. Execution-trace DPG construction for local explanations.
2. DPG-based local diagnostic framework: support margin, path purity, competitor exposure, confidence, trace coverage, recombination, and critical-node descriptors.
3. Auditable comparison protocol with heterogeneous explainers and path controls.
4. Diagnostic-value evidence for disagreement/model-error regimes.
5. Boundary analysis by dataset type, model family, depth, graph size, runtime, and critical-node eligibility.

## Claims To Use

Use:

- DPG-local is a structure-aware diagnostic complement, not a replacement for SHAP/LIME/Anchors.
- DPGs turn raw executed paths into compact analyzable graph objects.
- DPG diagnostics provide measurable signal for contested local decisions.
- Path controls show that raw path dumps can be output-faithful but large and semantically poorer.
- Critical nodes are conditional diagnostics, not causal proof.
- DPG analysis is model-family and dataset-regime dependent.

Avoid:

- "DPG-local is better than SHAP."
- "DPG-local replaces output-aligned explainers."
- "Critical nodes causally explain decisions."
- "All tree ensembles are supported."
- Unsupported model-family claims beyond RandomForest, ExtraTrees, AdaBoost, and Bagging.
- "Zero recombination proves explanatory value" without diagnostic evidence.

## Manuscript Tasks

High priority:

1. Rewrite Experimental Setup around locked validation/test protocol.
2. Replace old ECML numbers in `JOURNAL_paper/main_journal.tex` with final-test results.
3. Add common explainer interface appendix from `common_explainer_interface.csv`.
4. Add path-control subsection.
5. Add diagnostic-value subsection centered on disagreement AUROC/AUPRC.
6. Add confidence sensitivity paragraph.
7. Add model-family generalization subsection.
8. Add scalability subsection.
9. Rewrite critical-node discussion as bounded/conditional.
10. Add failure-mode discussion for `isolet` and `madelon`.

Status update:

- `JOURNAL_paper/main_journal.tex` now includes the completed 15-dataset, 4-family, 10-seed model-family experiment.
- The old five-dataset, one-seed model-family wording has been removed from the manuscript.
- A compact model-family table was added with results for RandomForest, ExtraTrees, AdaBoost, and Bagging.
- A readability/cost table was added comparing DPG-local, path controls, TreeSHAP, tree-path, and LORE-style surrogates.
- The prose was revised to avoid implementation-style wording and to keep the model-family discussion focused on the tested families.
- `JOURNAL_paper/main_journal.pdf` was rebuilt successfully with `latexmk -pdf -interaction=nonstopmode -halt-on-error main_journal.tex`.
- A numerical consistency pass was performed after the 10-seed model-family run. The manuscript now uses the locked final-test values for execution-trace DPG structural summaries: local match 0.860, local accuracy 0.783, confidence 0.654, support margin 0.689, competitor exposure 0.185, edge recall about 0.9999, active paths 20.00, active nodes 63.96, and recombination 0.000.
- Older cohort/disagreement values were replaced with per-sample values from `final_test_main_with_lore_path_controls/per_sample_selected_test.csv`: agreement confidence 0.656 vs disagreement confidence 0.399; agreement margin 0.691 vs disagreement margin 0.112; competitor exposure 0.186 vs 0.622.
- The baseline-comparison wording was made more conservative: DPG-local is no longer described as competitive with output-aligned methods on raw fidelity, and margin claims against heterogeneous baselines were removed unless directly supported by current reports.
- The current semantic-faithfulness table no longer reports a DPG-local critical-flip column. The semantic-faithfulness artifacts are mirrored under `experiments_local_explanation/results_journal_v1/semantic_faithfulness/` with a provenance README.

## Next Analysis Tasks

### Current Gap Audit After Reviewer Recheck

Status on 2026-06-19:

- The core reviewer-requested experiments are mostly complete: path controls, LORE-style baseline, ICE appendix treatment, confidence sensitivity, critical-node validation with controls, failure/regime analysis, scalability/depth stress, uncertainty summaries, corrected tests, and model-family support for RandomForest, ExtraTrees, AdaBoost, and Bagging.
- A full 10-seed model-family robustness run is currently active under `experiments_local_explanation/results_journal_v1/model_family_all15_10seeds_test/`. This is the main remaining experiment already in progress.
- The main final-test baseline comparison covers all 15 datasets and all main baselines, but selected final-test configurations currently use seeds 27 and 42 rather than 10 seeds for every baseline. Do not state that every baseline experiment has 10 repetitions unless a new full 10-seed baseline run is executed.
- Keep untested model families out of the claims.
- Critical nodes are not a main contribution under the current evidence. They are useful as conditional structural descriptors, but execution-trace DPGs have zero critical nodes under the current recombination-based definition and aggregated-DPG interventions are weak versus controls.
- Recombination/edge precision should be described as construction-consistency checks, not as standalone proof of explanatory utility.
- ICE should remain appendix-only or response-profile context, not a main route/path comparator.

Remaining details that can still cause journal-review gaps:

1. Finish and analyze the active 10-seed model-family run.
2. Add a manuscript appendix table for the common explainer interface.
3. Surface runtime and explanation-size results clearly in the manuscript, not only in Markdown artifacts.
4. Improve figure readability with top-k/focused case views and larger journal-ready typography.
5. Make the reproducibility package submission-ready: exact commands per table/figure, environment freeze, split registry, selected-config manifest, and anonymized release notes.
6. Decide whether to run a lightweight readability/cost analysis from existing per-sample metrics. This can address figure/readability concerns without implementing true graph pruning.
7. Treat true graph-pruning and soft critical-node probability-shift validation as optional extensions, not required unless targeting a venue that demands stronger usability or intervention evidence.

### A. Generate Model-Family Report

Status: completed.

Script:

- `experiments_local_explanation/analyze_model_family_experiment.py`

Output:

- `experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset_report/summary.md`

Create a compact report from:

- `experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset/summary.csv`

Suggested contents:

- Summary by `model_family`.
- Summary by `dataset`.
- Worst cases by local match and competitor exposure.
- Runtime and graph size by family.
- Writing guidance.

Suggested output path:

- `experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset_report/`

### B. Dataset/Model-Regime Analysis

Status: completed.

Script:

- `experiments_local_explanation/analyze_dataset_model_regimes.py`

Output:

- `experiments_local_explanation/results_journal_v1/dataset_model_regime_synthesis/summary.md`
- `experiments_local_explanation/results_journal_v1/dataset_model_regime_synthesis/dataset_model_regime_table.csv`
- `experiments_local_explanation/results_journal_v1/dataset_model_regime_synthesis/regime_group_summary.csv`

Combine:

- final-test report,
- confidence sensitivity,
- critical-node property analysis,
- scalability report,
- model-family report.

Question:

> Which dataset/model regimes make DPG diagnostics more or less useful?

Report dimensions:

- binary vs multiclass,
- low-dimensional vs high-dimensional,
- many-class datasets,
- low-margin/high-competitor-exposure decisions,
- model family,
- graph size/runtime.

Key result:

- Strong regimes: low- to moderate-dimensional binary and small multiclass datasets with high support margin and low competitor exposure, e.g. `banknote-authentication`, `iris`, `wdbc`, `breast_cancer`, `phoneme`, `spambase`, `qsar-biodeg`, `segment`, and `wine`.
- Hard/contested regimes: many-class datasets, especially `isolet` and `digits`, where competitor exposure is high and local agreement is lower.
- Boundary/high-cost regimes: `madelon` and `vehicle`; they are not total failures, but they show lower confidence, higher graph/runtime cost, or both.
- Writing implication: separate construction robustness from diagnostic usefulness. Edge recall and recombination indicate faithful trace construction, while support margin, competitor exposure, confidence, active nodes, and runtime determine how useful the graph is as a local diagnostic object.

### C. Reproducibility Package

Create:

- `RUN_MANIFEST.md`
- exact commands per table/figure
- split registry documentation
- selected config manifest
- environment freeze

## Useful Commands

Check active full model-family run launched on 2026-06-19:

```bash
ps -p 1965917 -o pid,ppid,sid,stat,etime,cmd
tail -f experiments_local_explanation/results_journal_v1/logs/model_family_all15_test_20260619_103222.log
```

Check active 10-seed model-family robustness run launched on 2026-06-19:

```bash
ps -p 1968207 -o pid,ppid,sid,stat,etime,cmd
tail -f experiments_local_explanation/results_journal_v1/logs/model_family_all15_10seeds_test_20260619_105526.log
```

Run details:

- Output root: `experiments_local_explanation/results_journal_v1/model_family_all15_10seeds_test/`
- PID file: `experiments_local_explanation/results_journal_v1/logs/model_family_all15_10seeds_test_20260619_105526.pid`
- Log file: `experiments_local_explanation/results_journal_v1/logs/model_family_all15_10seeds_test_20260619_105526.log`
- Scope: 15 curated datasets, final test split, `random_forest`, `extra_trees`, `adaboost`, `bagging`, execution-trace DPG, `n_estimators=20`, `max_depth=4`.
- Seeds: `27,42,100,101,102,103,104,105,106,107`.
- Parallelism: 12 dataset workers, `rf_n_jobs=1`, native thread caps set by the journal launcher.

When the 10-seed model-family run completes, generate the report:

```bash
./.venv/bin/python experiments_local_explanation/analyze_model_family_experiment.py \
  --input_root experiments_local_explanation/results_journal_v1/model_family_all15_10seeds_test \
  --out_dir experiments_local_explanation/results_journal_v1/model_family_all15_10seeds_test_report
```

Run details:

- Output root: `experiments_local_explanation/results_journal_v1/model_family_all15_test/`
- PID file: `experiments_local_explanation/results_journal_v1/logs/model_family_all15_test_20260619_103222.pid`
- Log file: `experiments_local_explanation/results_journal_v1/logs/model_family_all15_test_20260619_103222.log`
- Scope: 15 curated datasets, final test split, `random_forest`, `extra_trees`, `adaboost`, `bagging`, execution-trace DPG, `n_estimators=20`, `max_depth=4`, seed `27`.
- Parallelism: 12 dataset workers, `rf_n_jobs=1`, native thread caps set by the journal launcher.

When the full model-family run completes, generate the report:

```bash
./.venv/bin/python experiments_local_explanation/analyze_model_family_experiment.py \
  --input_root experiments_local_explanation/results_journal_v1/model_family_all15_test \
  --out_dir experiments_local_explanation/results_journal_v1/model_family_all15_test_report
```

Check model-family result:

```bash
./.venv/bin/python -c "import pandas as pd; df=pd.read_csv('experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset/summary.csv'); print(df[['dataset','model_family','model_accuracy','local_matches_model_rate','avg_explanation_confidence','avg_support_margin','avg_competitor_exposure','avg_edge_recall','avg_recombination_rate','avg_num_active_nodes','avg_runtime_ms']].to_string(index=False))"
```

Re-run confidence sensitivity:

```bash
./.venv/bin/python experiments_local_explanation/analyze_confidence_sensitivity.py \
  --input experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/per_sample_selected_test.csv \
  --out_dir experiments_local_explanation/results_journal_v1/confidence_sensitivity
```

Re-run model-family subset if needed:

```bash
experiments_local_explanation/launch_nohup.sh \
  --name model_family_dpg_supported_subset \
  -- ./.venv/bin/python experiments_local_explanation/run_local_experiments.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset \
  --datasets banknote-authentication,iris,vehicle,madelon,isolet \
  --model_families random_forest,extra_trees,adaboost,bagging \
  --n_estimators 20 \
  --max_depth 4 \
  --perc_var 0.0001 \
  --decimal_threshold 6 \
  --seeds 27 \
  --graph_construction_modes execution_trace \
  --max_test_samples 50 \
  --rf_n_jobs 1 \
  --progress_every 10
```

Re-run dataset/model-regime synthesis:

```bash
./.venv/bin/python experiments_local_explanation/analyze_dataset_model_regimes.py \
  --out_dir experiments_local_explanation/results_journal_v1/dataset_model_regime_synthesis
```

## Archival Markdown Files

Keep these for provenance, but the next session should start from this handoff:

- `JOURNAL_paper/plans/NEXT_STEPS_2026-06-19.md`
- `JOURNAL_paper/plans/critical_node_property_analysis_note.md`
- `JOURNAL_paper/plans/critical_node_results_and_writing_guidance.md`
- `JOURNAL_paper/plans/implementation_plan.md`
- `JOURNAL_paper/plans/model_family_experiment_note.md`
- `JOURNAL_paper/plans/q1_journal_readiness_assessment.md`
- `JOURNAL_paper/plans/q1_open_issues_and_contribution_strategy.md`
- `JOURNAL_paper/plans/reviewer_issue_matrix.md`
- `JOURNAL_paper/plans/scalability_depth_experiment_note.md`
- `JOURNAL_paper/plans/writing_plan.md`

## Bottom Line

The project now has enough experimental material for a serious journal rewrite. The next best step is not another broad experiment; it is to create the reproducibility package and rewrite the manuscript around DPG-enabled local diagnostic analysis.

## Figure Readability Update

The per-dataset shared-metrics heatmap in `JOURNAL_paper/main_journal.tex` was split into three full-width figures for fidelity, local accuracy, and margin/separation. The regenerated figures remove ICE from the main comparison and include LORE. The plotting script now writes explicit contrast-aware numeric annotations for every cell, avoiding the missing-cell-label problem observed in the earlier compressed multi-panel figure.

Updated files:

- `experiments_local_explanation/plot_journal_shared_metrics_heatmap.py`
- `JOURNAL_paper/per_dataset_fidelity_heatmap.pdf`
- `JOURNAL_paper/per_dataset_local_accuracy_heatmap.pdf`
- `JOURNAL_paper/per_dataset_margin_heatmap.pdf`
- `JOURNAL_paper/main_journal.tex`

Verification completed:

```bash
python3 experiments_local_explanation/plot_journal_shared_metrics_heatmap.py
python3 -m py_compile experiments_local_explanation/plot_journal_shared_metrics_heatmap.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main_journal.tex
pdftoppm -f 12 -l 14 -png -r 180 JOURNAL_paper/main_journal.pdf /tmp/main_journal_heatmap_page_v2
```

Visual inspection of pages 12--14 confirmed that all heatmap cells now have readable numeric annotations.
