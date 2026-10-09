# KBS Revision Plan: DPG-local after the AAAI-27 Rejection

Date: 2026-10-09
Target: Knowledge-Based Systems (Elsevier, `elsarticle`)
Inputs: AAAI-27 reviews (y8XZ, kpr7, AI reviewer), `local_explanation` branch (AAAI code),
`origin/main` = DPG v0.3.4, `JOURNAL_paper/main_journal.tex`, July 10-seed results.

**Not yet read:** the AAAI-27 source repo (`sbarbonjr/AAAI27_DPG_LocalExplanation`) is private and
could not be cloned from this machine. Section and figure references below follow the reviews.

---

## 1. Pilot evidence (2026-10-09)

Scripts: `experiments_local_explanation/pilots_kbs_2026-10-09/`. Data: the prepared
`data_numeric` splits, 14 datasets (wdbc dropped, see W3). All test samples were used unless noted.

### Pilot A: AAAI DPG-local vs forest-native uncertainty (RF 20 trees, depth 4, seeds 27/42)

The AAAI-branch `DPGExplainer` was run in execution_trace mode with the `top_competitor`
evidence variant. Scores are mean AUROC over datasets; higher score means riskier.

| Score | Model error | DPG class ≠ forest class |
|---|---|---|
| forest 1 − max prob | **0.836** | 0.835 |
| forest −prob margin | 0.832 | 0.838 |
| forest entropy | 0.817 | 0.827 |
| forest −hard-vote margin | 0.780 | 0.921 |
| DPG 1 − confidence (AAAI) | 0.735 | 0.905 |
| DPG confidence **without** vote agreement | 0.640 | 0.726 |
| DPG 1 − vote agreement | 0.754 | **0.981** |
| DPG −support margin | 0.720 | 0.883 |
| DPG competitor exposure | 0.730 | 0.882 |
| DPG −top-3 concentration | 0.354 (anti-predictive) | 0.082 |

- **Incremental value.** Out-of-fold logistic regression on 9 datasets with at least 10 errors:
  forest features alone 0.810, forest + DPG features 0.810 (Δ = −0.0003, Wilcoxon p = 0.65).
- **Circularity.** The disagreement target is close to circular. Vote agreement alone gives 0.98,
  and removing it from the confidence score drops 0.905 to 0.726.
- **Fidelity is lost by the weighting rule, not by necessity.** The explained class differs from
  `RandomForest.predict` in 14.1% of samples (isolet 68%). Hard vote ≠ soft vote in 5.1% of
  samples (isolet 31%).
- **Edge recall.** Mean edge recall is 0.9994, and 1.8% of samples have recall < 1.
- **Critical nodes.** Zero critical nodes in every dataset under the AAAI definition.

### Pilot B: structural fragility and a redefined critical predicate (same models)

- **Fragility as an error detector fails.** Fragility features (threshold slack of pivotal predicates,
  fraction of trees flippable within δ) reach standalone model-error AUROC of about 0.54. Their
  incremental value over forest probability is Δ = −0.002 (p = 1.0). **Structural signals do not
  improve error detection beyond forest probability.**
- **The redefined critical predicate works.** It is the predicate (feature, threshold) on the
  executed traces whose single-threshold crossing moves the most tree probability toward the
  forest's top competitor.
  - Coverage: 100% of samples (vs 0% for the AAAI definition).
  - Whole-forest intervention, crossing one threshold:
    - Δ P(competitor): +0.156 vs +0.049 for a random executed predicate.
    - Forest label flip: 26.9% vs 9.8%.
    - Critical predicate better on 14/14 datasets (Wilcoxon p = 1.2e-4).
  - The control is weak. The full study needs nearest-threshold, SHAP-top-feature (matched
    magnitude) and Tolomei-style feature-tweaking comparators.

### Pilot C: phantom routes in the merged local graph (seed 27, ≤300 samples/dataset)

| | k = 1 (AAAI merge) | k = k* (DPG-k, v0.3.x `resolve_context_order`) |
|---|---|---|
| Samples with ≥1 phantom route | **91.2%** | 0% |
| Samples with a cycle (infinite routes) | 23.7% | 0% |
| Route precision (acyclic cases) | 0.554 | 1.000 |
| Mean nodes | 63.6 | 72.7 (+14%) |
| k* | – | mean 1.99, per-dataset max ≈ 2–3 |

Reviewer y8XZ's W1 is therefore correct about the merged graph as drawn. In the AAAI code, the
Semantic Graph View quantities were computed over the stored per-tree traces (`tree_paths` keep
`tree_index`), so class support and the critical node never used synthetic routes. Algorithm 1
and the figure, however, presented the merged graph as if it were route-faithful.

### Pilot D: RF 100 trees, depth 8, seed 27 (AAAI implementation)

- **Same conclusion at scale.**
  - Model-error AUROC: forest −margin 0.856, 1 − max prob 0.855, DPG confidence 0.817, confidence
    without vote agreement 0.764.
  - Incremental value Δ = −0.0008 (p = 0.91).
- **Edge recall collapses.** Mean 0.953, and 79% of samples have recall < 1. The AAAI code maps
  test-time traces onto the training-fitted, `perc_var`-pruned global DPG, so deeper forests lose
  edges. The reported "≈ 0.99" was a property of depth-4 forests. A local-only construction makes
  recall 1 at any depth.
- **Cost and critical nodes.** 546 ms per sample, 383 active nodes. Critical nodes (AAAI
  definition) appear in about 9 samples in total across 14 datasets.

---

## 2. Consequences for the paper

The AAAI claim "DPG confidence has diagnostic value (AUROC ≈ 0.90)" **does not survive** the
controls the reviewers asked for. Resubmitting with added baselines but the same claims would
reproduce the rejection. Three claims the evidence does support:

1. **Route faithfulness is non-trivial and measurable.**
   - Merging executed traces into one graph invents routes in about 91% of local explanations.
   - The trace-indexed representation plus local context order (DPG-k) removes them with a ~14% size cost.
   - This makes y8XZ's objection a contribution and uses what DPG ≥ 0.3.0 already ships.
2. **Exact output fidelity is free.** Define class support from leaf class distributions, which is
   what sklearn's RF `predict_proba` averages. Fidelity is then 1 by construction, and the
   "disagreement" target disappears. This answers kpr7 Q2 and the AI reviewer's "lossy" point.
3. **The value is contrastive localisation, not uncertainty estimation.**
   - DPG-local can say *which executed predicate tests carry the competitor* and *where* the
     decision would change.
   - Validate this interventionally against controls and against feature-attribution-guided
     perturbations. This is the task-level evaluation the AI reviewer suggested.
   - Report the negative result honestly: for error detection, forest probability is sufficient.

---

## 3. Point-by-point response matrix

| # | Reviewer point | Verdict | Action (code / experiment / text) |
|---|---|---|---|
| y8XZ-W1 | Edge-faithful ≠ path-faithful; merged routes may be synthetic | **Correct for the drawn graph**; incorrect for computed quantities | Formalise G_x as **trace-indexed** (each edge stores the tree ids that executed it). Prove that every route used for support, competitor and critical predicate is a stored trace. Report route precision at k=1 vs k* (Pilot C). Render the local graph at k*. |
| y8XZ-Q1 | Are tree/path ids retained? | Yes (`DPGTreePathExplanation.tree_index`) | State it in the method and the algorithm. |
| y8XZ-W2 / kpr7 / AI | Confidence includes vote agreement, so disagreement AUROC is circular; no forest-uncertainty baselines | **Correct** (Pilot A) | Drop the composite confidence as a headline. Report forest max-prob, margin, entropy and vote margin as baselines. Add the component ablation and incremental CV test. State the negative result. Make the class support exact. |
| y8XZ-W2 | `predict` uses mean leaf probabilities, V_x uses hard votes | Correct | Class support = mean leaf distribution (RF/ET/Bagging), SAMME weights (AdaBoost), additive raw score (GBM). |
| kpr7-Q2 | Why can the explanation class differ from the forest? | Due to path weighting (global edge weight × global LRC × length penalty) | Explain in the response. The new definition removes the discrepancy. |
| y8XZ-W3 | breast_cancer = wdbc | **Correct** (569×30, 212/357, OpenML 1510 = sklearn copy) | Keep one. Add optdigits, satimage and wine_quality (already prepared) plus 1–3 OpenML numeric sets to keep ≥15. |
| y8XZ-W3 / kpr7 | Only 10–20 trees, depth 4 | Correct | Grid: RF/ET 100 trees × depth {4, 8, unlimited}. Add GBM (supported in 0.3.x), AdaBoost and Bagging. Report runtime and graph-size curves. |
| y8XZ-W3 / kpr7-Q3 | DPG tuned per dataset, baselines fixed | **Worse than stated** | `select_journal_configs.py` selected the RF *per method*. Path controls (fidelity 1) won the local-accuracy tie-break, so they explained more accurate forests (0.905 vs 0.858; isolet 0.939 vs 0.745). **New protocol:** one black box per dataset, chosen on validation accuracy only; every explainer explains it; DPG uses `decimal_threshold="auto"` with nothing tuned. |
| y8XZ-W3 / kpr7 | "LORE-style" is not LORE | Correct | Use the official `lore_sa` package, or drop LORE and say so. |
| y8XZ-W4 | w(p), top-K, branch support B_x, class terminals and pruning undefined; Algorithm 1 ≠ code | **Correct**. The paper said ω = c_e/B; the code weights traces by the *global* DPG (edge weight, LRC) | Full formal definitions in the method and the algorithm. The local graph uses only local traces (no global-graph dependence). |
| y8XZ-W4 | Why edge recall ≈ 0.99, not 1? | Traces were mapped onto the training-fitted global DPG, so some test-time edges were absent | Under the new definition recall is 1 by construction. Report it as a unit-test invariant, not a result. |
| y8XZ-W4 / AI | AUPRC promised but missing; prevalence absent | Correct | Report AUPRC with prevalence and no-skill line. |
| y8XZ-W4 / AI | Unresolved refs, Figure 3 legend vs caption, section pointers | Correct | Fix in the new manuscript. Generate figures by script with one palette definition. |
| y8XZ-W4 / AI | Missing related work | Correct | Add CHIRPS, LionForests, Izza & Marques-Silva 2021, Audemard et al. 2022, Potyka et al. 2023, Tolomei et al. 2017 (feature tweaking, closest to the critical predicate), inTrees, SIRUS, born-again ensembles, FOCUS, and DSAF (companion KBS paper). |
| kpr7 | Critical node valid in 2/15 datasets; weak AUROC | **Correct** (0 in Pilot A) | Redefine it as the counterfactual pivot predicate on executed traces (Pilot B: 100% coverage). Validate by intervention against stronger controls. Report per-dataset counts. |
| AI | Local accuracy 0.905 for controls ≠ mean RF accuracy | **Correct**; protocol bug (see W3 row) | Fixed by the single-black-box protocol. |
| AI | Competitor exposure = total alternative support, not strongest competitor | Correct (`1 − support_pred`) | Define both: alternative mass and top-competitor support. Keep the names honest. |
| AI | Exact traces can reproduce the forest; low fidelity not necessary | Correct | Exact fidelity (Section 2, point 2). |
| AI | Novelty vs forest-level and graph-based explanations | Fair | Position the work as sample-specific, route-faithful, contrastive localisation. Add a comparison table vs CHIRPS / LionForests / abductive and majoritary reasons / argumentation graphs / feature tweaking. |
| AI (suggestion) | Task-level evaluation vs trace trie and forest summaries | Adopt | Counterfactual-localisation task (Pilot B design) with a trie/trace-list baseline and attribution-guided baselines. |

---

## 4. Work packages

**WP1 – Library (DPG ≥ 0.4.0, branched from `origin/main` v0.3.4, not from `local_explanation`)**
- `explain_local` already traces each tree with exact `decision_path` routing.
- Add:
  - Leaf class distributions per trace and exact class support (RF/ET/Bagging/AdaBoost/GBM adapters).
  - Trace-indexed local graph (edge → tree ids), with optional local `context_order` (k* via `resolve_context_order`) and route-precision diagnostics.
  - Contrastive view: top competitor, competitor traces, alternative mass, pivot predicates with slack and competitor gain, critical predicate.
  - `intervene_on_predicate()` validation helper.
- Port the useful AAAI code (local plotting, path dataframe). Retire the composite confidence or keep it as legacy.
- Tests: fidelity == predict for each family, route precision == 1 at k*, every reported route ∈ traces, critical-predicate determinism.

**WP2 – Protocol and experiments**
- One black box per dataset, validated on accuracy. 15+ deduplicated datasets.
- Families: RF, ET, GBM, AdaBoost, Bagging. RF/ET at 100 trees × depth {4, 8, None}. 10 seeds.
- RQ1 route faithfulness: phantom/cycle rate at k=1, k*, size overhead, runtime vs trees/depth.
- RQ2 contrastive localisation: intervention study (critical predicate vs random, nearest-threshold, SHAP-top, LIME-top, Anchors-rule feature, feature-tweaking), as Δ P(competitor), flip rate and perturbation size.
- RQ3 error-detection honesty check: forest baselines vs DPG components, incremental CV test, AUROC/AUPRC.
- RQ4 cost and readability: size and runtime against TreeSHAP, LIME, Anchors, LORE (official) and path controls.
- Compute: about 15 datasets × 5 families × 3 depths × 10 seeds. Feasible on the 36-core machine; days on this 8-core one.

**WP3 – Manuscript (KBS, `elsarticle`, ~25–30 pp)**
- Start from the AAAI-27 source (needs repo access), merging `main_journal.tex` material.
- New title direction: *Route-Faithful Decision Predicate Graphs for Contrastive Local Explanation of Tree Ensembles*.
- Contributions: (i) trace-indexed DPG-local with route-faithfulness guarantee (DPG-k); (ii) exact-fidelity contrastive view and critical predicate; (iii) interventional evaluation protocol; (iv) negative result on error detection.

**WP4 – Submission package**
- Cover letter noting the AAAI-27 history (optional for KBS) and the companion DSAF paper.
- Highlights, graphical abstract, data/code availability (DPG release + Zenodo DOI), CRediT, declaration of generative-AI use.
