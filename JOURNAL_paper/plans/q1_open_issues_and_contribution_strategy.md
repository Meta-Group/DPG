# Q1 Open Issues And Contribution Strategy

Date: 2026-06-19

This note updates the next-step plan after checking `NEXT_STEPS_2026-06-19.md`, the Q1 readiness audit, the implementation plan, and the available final-test outputs.

## Short Answer

We do not need another broad experiment immediately. The most valuable next step is a lightweight analysis over existing final-test CSVs:

1. Confidence-score sensitivity/calibration.
2. Dataset/model-property analysis for when DPG diagnostics are useful or weak.
3. Manuscript reframing from "DPG-local as another local explainer" to "Decision Predicate Graphs as an analysis framework for local explanation diagnostics."

Only after these should we decide whether to run a supported-model-family experiment across all classification models currently documented by DPG.

## What Is Already Covered

The overnight/focused scalability item in `NEXT_STEPS_2026-06-19.md` has been completed and summarized in:

- `experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_report/summary.md`

This answers the shallow-depth reviewer concern partially:

- Depths 4, 8, and 12 were tested.
- Forest sizes 20 and 50 were tested.
- Six representative datasets were tested.
- Execution-trace DPG remains recombination-free and keeps high edge recall.
- Runtime grows sharply, so the correct claim is bounded scalability, not unrestricted deployment readiness.

The final-test report already covers:

- ICE moved to appendix-only.
- LORE-style rule/counterfactual baseline.
- Path controls: `raw_path_union`, `path_bag`, `random_same_size_path`.
- Common explainer interface.
- Confidence intervals.
- Holm-corrected paired tests.
- Critical-node validation and controls.
- Isolet/madelon failure focus.
- Size/runtime tables.

## Still-Open Q1 Issues

### 0. DPG-Supported Classification Model Families

The DPG documentation lists the following supported sklearn classification ensembles:

- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `ExtraTreesClassifier`
- `AdaBoostClassifier`
- `BaggingClassifier`

It also states that DPGExplainer works with sklearn ensemble models exposing an `estimators_` attribute, while XGBoost, LightGBM, and CatBoost are listed as future integrations rather than currently supported models.

This is a strong idea for the journal paper because it changes the model-scope question from "only shallow random forests" to:

> DPG-based local diagnostics apply across the classification model families currently supported by the DPG framework.

This is cleaner than adding unsupported external libraries. It also fits the broader contribution framing: DPGs enable local diagnostic analysis for supported tree-ensemble classifiers, not only RandomForest.

Implementation status:

- A model-family factory has been added in `experiments_local_explanation/model_factory.py`.
- `run_local_experiments.py` now accepts `--model_families` and writes `model_family` to the output CSVs.
- A smoke test passed for `random_forest`, `extra_trees`, `adaboost`, and `bagging`.
- The smoke outputs are under `experiments_local_explanation/results_journal_v1/model_family_smoke/`.
- `GradientBoostingClassifier` currently fails in this checkout because its `estimators_` is a 2D array; it also requires a careful class-contribution interpretation because boosted trees are additive rather than majority-vote trees.
- Full note: `JOURNAL_paper/plans/model_family_experiment_note.md`.

Recommended model-family experiment:

- Scope: DPG methods first, baselines second if feasible.
- Datasets: representative subset rather than all 15 at first:
  - `banknote-authentication`: low-dimensional binary.
  - `iris` or `segment`: multiclass.
  - `madelon`: high-dimensional binary.
  - `isolet`: high-dimensional many-class.
  - `vehicle`: contested multiclass case.
- Models:
  - RandomForest.
  - ExtraTrees.
  - AdaBoost.
  - Bagging.
- GradientBoosting after an explicit adapter is implemented and validated.
- Metrics:
  - trace coverage / edge recall.
  - recombination rate.
  - graph size.
  - runtime.
  - support margin.
  - competitor exposure.
  - confidence/disagreement AUROC where enough samples exist.

Claim if successful:

> Across the sklearn classification ensembles supported by DPGExplainer, DPG-based local diagnostics preserve a common predicate-transition representation while exposing model-family differences in graph size, support concentration, competitor exposure, and runtime.

Claim if mixed:

> DPG-local is technically applicable to the supported sklearn ensemble families, but diagnostic behavior and computational cost are model-family dependent. Random forests and ExtraTrees provide the most direct path-level interpretation; boosting models require more careful discussion because their additive ensemble semantics differ from majority-vote forests.

Important writing caution:

- Do not imply all tree ensembles are supported.
- Say "supported sklearn classification ensembles" unless XGBoost/LightGBM/CatBoost adapters are implemented later.
- Explain that boosting ensembles have different decision semantics from bagging/forest models; a DPG can still represent executed predicates, but support aggregation and competitor interpretation must be written carefully.

### 1. Confidence Score Sensitivity

This is the highest-priority remaining analysis because a reviewer explicitly questioned the equal-weight confidence formula and structural coverage gate.

Status: completed as an analysis over locked final-test CSVs.

Outputs:

- `experiments_local_explanation/analyze_confidence_sensitivity.py`
- `experiments_local_explanation/results_journal_v1/confidence_sensitivity/summary.md`
- `confidence_sensitivity_summary.csv`
- `confidence_sensitivity_per_dataset.csv`
- `confidence_variant_stability.csv`
- `confidence_calibration_bins.csv`

Main result:

- For local disagreement, the reported low-confidence score remains strong:
  - `dpg`: AUROC 0.9001.
  - `dpg_execution_trace`: AUROC 0.9046.
- Vote-agreement alone is even stronger:
  - `dpg`: AUROC 0.9237.
  - `dpg_execution_trace`: AUROC 0.9808.
- No-gate/additive variants remain competitive, especially for execution-trace DPG.
- Concentration-only is poor, so it should not be emphasized as a standalone diagnostic.

Writing implication:

> The confidence score is robust enough as a diagnostic ranking index, but the paper should emphasize the broader DPG diagnostic family rather than claiming that the aggregate confidence formula is uniquely optimal. Vote agreement, support margin, and competitor exposure are the more interpretable class-contrastive signals.

Good news: this can be done from existing per-sample outputs. The final-test CSV already contains:

- `support_margin`
- `predicted_class_concentration_top3`
- `model_vote_agreement`
- `trace_coverage_score`
- `explanation_confidence`
- `disagree_with_model`
- `model_correct`
- `local_matches_model`

Recommended variants:

- Current multiplicative gate:
  - `trace_coverage_score * mean(support_margin, predicted_class_concentration_top3, model_vote_agreement)`
- No gate:
  - `mean(support_margin, predicted_class_concentration_top3, model_vote_agreement)`
- Additive coverage:
  - `mean(trace_coverage_score, support_margin, predicted_class_concentration_top3, model_vote_agreement)`
- Margin-only:
  - `support_margin`
- Competition-only:
  - `1 - competitor_exposure` or `path_purity`
- Validation-learned logistic score if validation data are aligned.

Report:

- AUROC/AUPRC for local disagreement.
- AUROC/AUPRC for model error.
- Spearman correlation with the current confidence score.
- Per-dataset robustness.
- Calibration/reliability bins if useful.

Desired claim:

> The confidence index is a diagnostic ranking score, not a calibrated probability. Its disagreement-detection behavior is stable across reasonable component choices.

If the result is not stable:

> We replace the original confidence formula with the best validation-supported variant, or downgrade confidence to a heuristic descriptive score.

### 2. Dataset And Model Regime Analysis

The paper should discuss which kinds of datasets and models benefit from DPG-style local analysis.

Already partly available:

- Critical-node property analysis shows imbalance is not the main driver.
- Failure focus shows `isolet` and `madelon` are hard regimes.
- Scalability shows depth and number of trees increase graph size/runtime.

Recommended additional analysis:

- Correlate DPG diagnostic performance with:
  - number of classes
  - number of features
  - features per training sample
  - model accuracy
  - support margin
  - competitor exposure
  - graph size
  - tree depth / forest size from scalability run
- Separate regimes:
  - low-dimensional binary
  - low-dimensional multiclass
  - high-dimensional binary
  - high-dimensional many-class
  - low-accuracy/ambiguous model regimes
  - high-depth/high-runtime regimes

Expected discussion:

- DPG analysis is strongest when the forest decision process has reusable predicate structure and meaningful class competition.
- DPG analysis is weakest when high-dimensional/many-class data creates diffuse support, low margins, large graphs, or high extraction cost.
- Class imbalance alone is not the best explanation for critical-node usefulness.

### 3. Model Family Scope

Current evidence is strong for random forests and compatible forest-style ensembles with extractable root-to-leaf traces. The DPG documentation provides a better intermediate scope than external XGBoost/LightGBM experiments: all DPG-supported sklearn classification ensembles.

Options:

1. Narrow claim now:
   - "random forests and tree ensembles with extractable execution traces"
   - fastest and safest for Information Sciences.

2. Add a supported sklearn model-family experiment:
   - RandomForest, ExtraTrees, GradientBoosting, AdaBoost, and Bagging.
   - Use only a representative subset.
   - This is useful if targeting Information Fusion or keeping broader "tree ensembles with DPG-supported execution traces" language.

3. Add external boosted-tree adapters later:
   - XGBoost, LightGBM, or CatBoost only if implemented and tested separately.
   - The DPG documentation lists these as future integrations, so they should not be promised in the current paper.

Recommendation:

- Do not block manuscript rewriting on external boosted-tree libraries.
- First run confidence sensitivity and dataset-regime analysis.
- Then implement the supported sklearn model-family experiment if the selected journal/title needs broader model-scope evidence.

### 4. Reproducibility Package

Not a new experiment, but mandatory for Q1 polish.

Needed:

- `RUN_MANIFEST.md`
- exact commands per table/figure
- environment freeze
- split registry documentation
- selected configuration manifest
- clear final-test artifact paths

### 5. Manuscript Integration

The manuscript is currently behind the experiments. The latest report artifacts must be moved into the paper.

High-priority edits:

- Replace shallow-only setup wording.
- Add final validation/test protocol.
- Add path-control results.
- Add LORE result.
- Add diagnostic AUROC/AUPRC.
- Add scalability paragraph/table.
- Add critical-node bounded interpretation.
- Add dataset-regime/failure-mode discussion.
- Move ICE to appendix only and keep it there.

## Should The Paper Focus On "DPG-local"?

The stronger framing is not simply "DPG-local is a new local explainer."

A stronger Q1 framing is:

> Decision Predicate Graphs provide a local diagnostic analysis layer for tree ensembles. By converting executed tree paths into a canonical predicate-transition graph, they enable analyses that standard local explainers and raw path dumps do not directly provide: trace recovery checks, recombination diagnostics, class-contrastive support, competitor exposure, disagreement detection, conditional critical-node analysis, and scalability/readability inspection.

In this framing, DPG-local is the construction mechanism, but the contribution is broader:

- DPGs enable local structural auditing.
- DPGs expose class competition inside an ensemble.
- DPGs turn raw paths into analyzable graph objects.
- DPGs create diagnostics for contested or unstable predictions.
- DPGs show when path-level explanations are compact, large, reliable, or weak.

This also helps with Information Fusion:

- Each tree path can be treated as a source of decision evidence.
- The DPG is the fused local representation.
- Semantic Graph View summarizes the fused evidence.
- Competitor exposure and support margin are decision-fusion diagnostics.
- Scalability results address computational demands of the fusion process.

## Proposed Contribution Set

### Main Contribution 1: Execution-Trace DPG Construction

Build a sample-specific predicate-transition graph from exact root-to-leaf paths. This is the technical construction.

### Main Contribution 2: DPG-Based Local Diagnostic Framework

Use the graph to compute support margin, path purity, competitor exposure, confidence, recombination, trace coverage, and critical-node descriptors.

This is more important than the graph alone.

### Main Contribution 3: Auditable Evaluation Protocol

Compare DPG diagnostics with output-aligned, rule/surrogate, path-control, and response-profile methods under a documented common interface.

### Main Contribution 4: Diagnostic-Value Evidence

Show that DPG-derived signals identify local disagreement and model-error regimes, with uncertainty intervals and corrected tests.

### Main Contribution 5: Scope And Boundary Analysis

Report when DPG analysis works well and when it becomes costly or weak:

- high-dimensional data
- many-class data
- low-margin model behavior
- deeper/larger forests
- critical-node eligibility limitations

## Recommended Immediate Order

1. Implement confidence-score sensitivity analysis from existing final-test per-sample CSVs.
2. Implement dataset/model-regime analysis from existing reports and scalability outputs.
3. Add generated Markdown/CSV summaries for both.
4. Rewrite the paper contribution framing around "DPG-enabled local diagnostic analysis."
5. Only then decide whether to run the supported sklearn model-family experiment.

## Decision

For now, do not launch another expensive full experiment. Run focused analyses first. They directly answer reviewer concerns, strengthen Q1 readiness, and clarify the right scale for a supported-model-family experiment.
