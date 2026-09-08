# Journal Implementation Plan

## Objective

Build a stronger journal version of DPG-local by converting the ECML paper from a mainly construction-and-comparison study into a reproducible, statistically supported, diagnostic validation study.

The central claim should become:

> DPG-local is a structure-aware diagnostic complement for tree ensembles. It preserves executed predicate-transition traces and adds contrastive diagnostics that help identify contested, unstable, or erroneous local decisions.

## Phase 0: Baseline Preservation

Status: done for workspace setup.

- `paper_ecml_rejected/` remains the rejected/submitted snapshot.
- `JOURNAL_paper/` has been initialized from the submitted ZIP.
- `JOURNAL_paper/journal_paper.tex` is the editable journal manuscript.
- `JOURNAL_paper/reviews/ecml_rejection_comments.txt` stores the reviewer comments.

Acceptance criteria:

- The submitted TeX and ZIP remain available.
- All journal changes happen in `JOURNAL_paper/` and experiment outputs under a new journal-specific run directory.

## Phase 1: Reproducibility and Protocol Hardening

Goal: eliminate reviewer concerns about leakage, missing settings, and unauditable baselines.

Implementation tasks:

- Add a journal experiment config, for example `experiments_local_explanation/configs/journal_v1.yaml`.
- Define a split registry with fixed train/validation/test indices per dataset.
- Use validation data for all configuration selection, including DPG-local and baselines.
- Evaluate exactly once on the final test set after selection.
- Write a `RUN_MANIFEST.md` with git commit, package versions, seeds, datasets, commands, output paths, and hardware notes.
- Store environment files, ideally `requirements.txt` plus a frozen `pip freeze` artifact.

Current code to extend:

- `experiments_local_explanation/prepare_datasets.py`
- `experiments_local_explanation/run_local_experiments.py`
- `experiments_local_explanation/run_baseline_experiments.py`
- `experiments_local_explanation/run_per_dataset.py`
- `experiments_local_explanation/run_per_dataset_baselines.py`

Acceptance criteria:

- Every table in the journal paper can be regenerated from a documented command.
- The selected configuration for each method is traceable to validation metrics, not the test set.
- The final test CSVs include `split`, `selection_metric`, `selected_on`, and `selection_rank` fields.

## Phase 2: Common Explainer Interface and Baseline Audit

Goal: make comparisons semantically auditable.

Implementation tasks:

- Create a common output schema for every explainer:
  - `method`
  - `native_object_type`
  - `model_access`
  - `explained_class`
  - `class_score`
  - `competitor_class`
  - `competitor_score`
  - `selected_features`
  - `structural_object`
  - `size`
  - `runtime_ms`
  - `unsupported_fields`
- Add a protocol table that states exactly how each method is mapped:
  - DPG-local: support class from executed trace graph; selected features from strongest predicted path.
  - DPG-global: same DPG scoring, but graph from aggregated transitions.
  - TreeSHAP: predicted model class and absolute SHAP ranking for that class.
  - LIME: local surrogate class weights and top weighted features.
  - Anchors: target predicted class, rule precision, coverage, rule features.
  - Tree-path: per-feature probability changes along executed paths.
  - ICE: response-profile sensitivity only; do not present as a route/path explainer.
- Move ICE out of the main "local explanation ranking" table unless there is a carefully scoped sensitivity-only analysis.

Current code already contains useful pieces:

- `run_baseline_experiments.py` implements `shap`, `lime`, `ice`, `anchors`, and `tree_path`.
- `analyze_semantic_faithfulness.py` already maps DPG, SHAP, LIME, tree-path, Anchors, and ICE into selected feature sets for sufficiency/comprehensiveness.

Acceptance criteria:

- A generated CSV/Markdown table describes the native object and mapping for each method.
- Unsupported or non-comparable fields are explicitly marked instead of filled with misleading values.
- Main comparison tables separate output-aligned methods, rule/surrogate methods, response-profile methods, and structure-aware methods.

## Phase 3: Add Stronger Path-Level Baselines and Ablations

Goal: answer the "basic path union" and "tautology" critiques.

New baselines:

- `raw_path_union`: union of executed root-to-leaf predicates and transitions with no DPG canonical aggregation or Semantic Graph View.
- `path_bag`: multiset of executed predicates, ignoring transition order.
- `tree_path_feature`: existing feature-level tree-path baseline.
- `dpg_global`: existing aggregated-transition graph.
- `random_same_size_path`: control baseline selecting same-size path predicates at random from the forest path vocabulary.
- `majority_vote_only`: control that reports the forest vote and no structural evidence.

Ablations:

- DPG-local without competitor exposure.
- DPG-local without critical node reporting.
- DPG-local with raw support only, no confidence score.
- DPG-local with top-k graph pruning thresholds.
- DPG-local confidence score variants from Phase 6.

Metrics:

- Output agreement and local accuracy as context only.
- Trace node/edge recall as construction checks.
- Graph size, runtime, and readability proxies.
- Diagnostic metrics from Phase 4 as the primary value evidence.

Acceptance criteria:

- DPG-local beats raw path union/path bag on at least one diagnostic task, not merely on trace recovery.
- The paper can state which value comes from exact traces and which value comes from the Semantic Graph View.

## Phase 4: Diagnostic-Value Experiments

Goal: prove that DPG-local helps diagnose model behavior better than simpler baselines.

### 4.1 Disagreement and Error Detection

Question:

Can DPG-local metrics identify contested or wrong predictions?

Implementation:

- Use per-sample fields already produced by `run_local_experiments.py`:
  - `explanation_confidence`
  - `support_margin`
  - `competitor_exposure`
  - `model_vote_agreement`
  - `path_purity`
  - `critical_node_contrast`
  - `num_active_nodes`
  - `num_paths`
- Predict:
  - model error: `y_model_pred != y_true`
  - explanation/model disagreement: `y_local_pred != y_model_pred`
  - low-margin cases: bottom quantile of model probability margin
- Compare against baselines using their available margins, rule precision/coverage, SHAP concentration, LIME concentration, tree-path concentration, and random controls.

Report:

- AUROC/AUPRC with 95% bootstrap CI.
- Calibration curves or reliability bins for `explanation_confidence`.
- Per-dataset and pooled results.

Acceptance criteria:

- DPG-local diagnostics provide statistically supported signal for contested/model-error cases.
- Negative or mixed results are reported as scope limits.

### 4.2 Critical-Node Validation

Question:

When a non-root critical node exists, does it identify a meaningful local decision gate?

Implementation:

- Report eligibility:
  - fraction with any critical node
  - fraction with non-root critical node
  - fraction with both predicted and competitor successors parseable
- Replace the current pooled-only flip rate with conditional metrics:
  - `P(flip to competitor | eligible)`
  - mean competitor probability increase after competitor-branch intervention
  - mean predicted-class probability decrease after competitor-branch intervention
  - branch margin delta
- Add controls:
  - random same-depth split
  - random path split
  - sibling branch not on strongest competitor route
- Test paired differences between critical-node intervention and controls.

Current code to extend:

- `experiments_local_explanation/analyze_semantic_faithfulness.py`

Acceptance criteria:

- The paper reports critical nodes as conditional diagnostics with occurrence rates.
- If label flips remain rare, probability-shift evidence becomes the main result.
- Controls show whether critical nodes are better than arbitrary path predicates.

### 4.3 Case Bank Instead of One Anecdote

Question:

Do qualitative examples represent recurring patterns?

Implementation:

- Automatically select cases for:
  - model correct / explanation correct
  - model wrong / explanation matches model
  - model wrong / explanation recovers true class
  - explanation disagrees with model
  - high competitor exposure
  - isolet or madelon failure
- Export one compact graph and one table per case.

Acceptance criteria:

- The journal paper includes 2-3 main cases and moves the full case bank to appendix/supplement.
- Vehicle sample 56 remains only if it is representative of a larger cohort.

## Phase 5: Statistical Rigor

Goal: replace single averages and uncorrected tests.

Implementation tasks:

- Increase seeds from 2 to at least 10 for the main RF study.
- Report per-dataset mean plus 95% CI across seeds or bootstrap over samples.
- Use Friedman tests only as global omnibus checks.
- Use paired Wilcoxon signed-rank tests with Holm correction for planned pairwise comparisons.
- Report effect sizes:
  - rank-biserial correlation for Wilcoxon
  - Cliff's delta where appropriate
  - Cohen's d only for approximately continuous paired summaries
- Avoid overclaiming from n=15 datasets.

Acceptance criteria:

- Every main table has uncertainty intervals.
- All p-values in method comparisons are corrected.
- Claims are phrased around effect sizes and consistency, not p-values alone.

## Phase 6: Confidence Score Sensitivity

Goal: answer the critique that equal weighting and multiplicative coverage gate are heuristic.

Implementation tasks:

- Evaluate alternative score forms:
  - current: `coverage * mean(margin, concentration, vote_agreement)`
  - additive coverage: `mean(coverage, margin, concentration, vote_agreement)`
  - no gate: `mean(margin, concentration, vote_agreement)`
  - learned validation weights using logistic regression for error/disagreement detection
  - grid weights over margin/concentration/vote agreement
- Compare:
  - ranking stability by Spearman correlation
  - AUROC for error/disagreement detection
  - calibration by reliability bins
  - dataset-level sensitivity of conclusions

Acceptance criteria:

- The paper either justifies the simple default as robust or replaces it with a validated formula.
- The confidence score is described as a diagnostic index, not a calibrated probability.

## Phase 7: Scalability and Broader Model Scope

Goal: address "depth 4 shallow RF" and practical unreadability concerns.

RF scalability grid:

- `n_estimators`: 20, 50, 100
- `max_depth`: 4, 8, 12, None
- seeds: at least 5 for the scalability grid, 10 if feasible

Metrics:

- runtime per explanation
- graph nodes/edges/paths
- memory proxy: serialized graph size or edge count
- fidelity/context metrics
- diagnostic metrics
- top-k pruned graph readability: number of visible nodes/edges after pruning

Boosted-tree scope:

- First implement an adapter abstraction for path extraction:
  - sklearn RandomForest
  - sklearn ExtraTrees or HistGradientBoosting if available
  - XGBoost and LightGBM if dependencies and tree dumps are available
- If XGBoost/LightGBM support is not completed, narrow the journal claim to random forests and explicitly mark boosted trees as future work.

Acceptance criteria:

- The paper no longer claims broad "tree ensembles" from shallow RFs alone.
- Either boosted-tree evidence is added, or the scope is explicitly "random forests and compatible tree ensembles with extractable root-to-leaf traces."
- Runtime and graph-size results are visible in the main paper, not hidden.

## Phase 8: Failure-Mode Analysis

Goal: turn isolet and madelon from liabilities into honest boundary evidence.

Implementation tasks:

- For isolet and madelon, report:
  - class count
  - feature count
  - model accuracy
  - model probability margin
  - vote entropy
  - explanation confidence
  - competitor exposure
  - graph size
  - trace coverage
  - runtime
- Regress or correlate failure/disagreement with:
  - number of classes
  - number of features
  - graph size
  - model vote entropy
  - support margin
- Add a short "when DPG-local is least useful" subsection.

Acceptance criteria:

- The journal paper explicitly explains isolet's low output fidelity and what it implies.
- The limitations section names expected failure regimes.

## Phase 9: Reproducible Release Package

Goal: make the paper citable and auditable.

Implementation tasks:

- Add a top-level experiment README for the journal study.
- Add one command per table/figure.
- Add smoke tests for:
  - dataset loading
  - one DPG explanation
  - one baseline explanation
  - one semantic-faithfulness run
- Add a packaging checklist:
  - anonymized repository if double blind
  - exact splits
  - preprocessing
  - environment
  - result manifests
  - generated figures

Acceptance criteria:

- A fresh checkout can reproduce smoke results quickly.
- Full runs can be launched with documented commands.
- The manuscript points to code and data artifacts.

## Suggested Milestones

1. Week 1: protocol hardening, split registry, baseline mapping table.
2. Week 2: path-union baselines and ablations.
3. Week 3: diagnostic-value and critical-node validation scripts.
4. Week 4: 10-seed RF rerun and corrected statistics.
5. Week 5: scalability grid and optional boosted-tree adapter.
6. Week 6: figure refresh, reproducibility package, first journal manuscript rewrite.

## Immediate Next Commands To Prepare

These are not run yet; they are the next implementation targets.

```bash
PYTHONPATH=. python3 experiments_local_explanation/run_local_experiments.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results_journal_v1/dpg \
  --n_estimators 20,50,100 \
  --max_depth 4,8,12,None \
  --perc_var 0.0001,0.00001 \
  --decimal_threshold 6 \
  --seeds 27,42,100,101,102,103,104,105,106,107 \
  --graph_construction_modes aggregated_transitions,execution_trace
```

```bash
PYTHONPATH=. python3 experiments_local_explanation/run_baseline_experiments.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results_journal_v1/baselines \
  --methods shap,lime,anchors,tree_path,ice
```

These commands should be adjusted after the split registry and journal config are implemented.
