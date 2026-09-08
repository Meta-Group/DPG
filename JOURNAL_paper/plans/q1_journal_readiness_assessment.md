# Q1 Journal Readiness Assessment

Date: 2026-06-19

This note audits the ECML reviewer concerns, the submitted manuscript, the current journal manuscript, and the available experimental artifacts for a possible Q1 journal submission such as Information Sciences or Information Fusion.

## Sources Checked

- ECML rejection comments: `JOURNAL_paper/reviews/ecml_rejection_comments.txt`
- Submitted ECML manuscript: `paper_ecml_rejected/ECML26_DPG.tex`
- Current working manuscript: `JOURNAL_paper/ECML26_DPG.tex`
- Main final-test experimental report: `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/summary.md`
- Common explainer interface: `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/common_explainer_interface.csv`
- Critical-node validation: `experiments_local_explanation/results_journal_v1/critical_node_validation_final_fixed/summary.md`
- Critical-node property analysis: `experiments_local_explanation/results_journal_v1/critical_node_property_analysis/summary.md`
- Focused deeper-RF scalability report: `experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_report/summary.md`
- Official journal scope pages checked online:
  - Information Fusion, ScienceDirect/Elsevier: https://www.sciencedirect.com/journal/information-fusion
  - Information Sciences, ScienceDirect/Elsevier: https://www.sciencedirect.com/journal/information-sciences

## Executive Verdict

The project is no longer in the same state as the rejected ECML submission. The experimental base is now much stronger: ICE has been moved out of the main comparison, LORE-style and path-control baselines have been added, confidence intervals and Holm-corrected paired tests are available, critical-node claims have been empirically bounded, and a focused deeper-random-forest stress test has been completed.

However, the current manuscript is not yet ready for a Q1 journal submission. The main blocker is not only missing experiments; it is alignment. The latest experimental evidence is mostly in result artifacts, while `JOURNAL_paper/ECML26_DPG.tex` still contains ECML-era framing, older result numbers, a shallow-RF setup description, and only partial treatment of the reviewer concerns.

Current readiness:

- Experimental readiness for an Information Sciences-style submission: moderate to good, after manuscript integration.
- Experimental readiness for Information Fusion: moderate at best unless the paper is reframed explicitly as decision/process fusion inside tree ensembles.
- Manuscript readiness today: not submission-ready.

## Journal Fit

Information Sciences is the stronger target. Its official scope includes intelligent systems, artificial intelligence, decision support systems, learning/evolutionary computing, data fusion, information and knowledge, pattern recognition, and a balance of theory and practice. DPG-local can fit as a graph-based, structure-aware XAI method for intelligent systems if the contribution is written as a diagnostic method with rigorous comparative evaluation.

Information Fusion is harder. Its official scope is centered on multi-sensor, multi-source, multi-process information fusion, including data/feature/decision/multilevel fusion, multi-classifier/decision systems, intelligent fusion processing, and computational demands in fusion systems. DPG-local can fit only if the journal version reframes the method as fusing tree-level executed decision evidence into a local graph with class-contrastive diagnostic signals. Without that fusion framing, the paper may be viewed as XAI for tree ensembles rather than an information-fusion contribution.

Recommendation: target Information Sciences first unless the manuscript is substantially rewritten around an information-fusion narrative.

## Reviewer Concern Coverage

| Reviewer concern | Submitted ECML weakness | Current evidence | Status | Required journal action |
| --- | --- | --- | --- | --- |
| ICE comparison is unjustified | ICE appeared in the abstract/contributions and main baseline list | ICE is now appendix-only in the working manuscript | Mostly addressed | Keep ICE out of main rankings and explain it only as a response-profile diagnostic |
| Need simple path-union/path baselines | Original comparison did not include raw path controls | `raw_path_union`, `path_bag`, and `random_same_size_path` are in the final report | Experimentally addressed | Add a main/appendix table showing DPG compactness versus raw path controls |
| Common interface unclear | Original paper did not audit how heterogeneous explainers map to shared metrics | `common_explainer_interface.csv` exists and includes DPG, SHAP, LIME, Anchors, LORE, tree-path, path controls, and ICE | Artifact addressed, manuscript missing | Add an appendix table and reference it from Experimental Setup |
| Structural metrics are tautological | Edge precision/recombination were too prominent as wins | Final report separates construction checks from diagnostic AUROC/AUPRC | Partially addressed | Rewrite Results so edge precision/recombination are checks, while diagnostic AUROC/confidence/competitor exposure are the empirical value claim |
| Critical nodes weak | Original framing made critical nodes look like a major contribution despite near-zero pooled flip rate | Final validation: present 19.6%, eligible 6.0%, changed-to-competitor 2.5%, random-path control 4.8%; execution-trace DPG has 0 under this definition | Addressed as a bounded negative/conditional result | Present critical nodes as secondary conditional diagnostics, not a main contribution |
| Need variance and corrected tests | Original paper emphasized averages and uncorrected comparisons | Final report includes 95% CIs and Holm-corrected paired tests | Addressed experimentally | Replace current statistical paragraph with the corrected table and use cautious wording |
| Per-dataset tuning may be unfair | Current manuscript still says best DPG configuration is selected per dataset without a clear locked final test protocol | Final report says it uses validation-selected, locked final-test outputs | Partially addressed | Rewrite the protocol around train/validation/test splits and include the selection manifest in appendix |
| Shallow random forests only | Submitted/current manuscript still describes depth-4 forests | Focused stress test completed: depth 4/8/12 and 20/50 trees on six datasets | Partially addressed | Add a scalability subsection and narrow claims to random forests unless boosted-tree evidence is added |
| Runtime/size invisible | Original paper promised size/runtime but underreported them | Final report has native size/runtime; scalability report has depth/runtime/size | Addressed experimentally | Add one compact main table and one appendix table |
| Isolet/madelon failure under-discussed | Original paper did not engage deeply with failures | Final report has an isolet/madelon failure-focus section | Partially addressed | Add a limitations/failure-mode paragraph with concrete numbers |
| Confidence score weights unjustified | Equal-weight confidence and structural gate lacked sensitivity analysis | No full confidence-weight sensitivity experiment is visible in the current artifacts | Still open | Add a small sensitivity/calibration experiment or explicitly downgrade confidence to a heuristic diagnostic |
| Broad tree-ensemble claims | Original phrasing suggests broad tree ensembles | Current evidence is still RandomForest-centered, with deeper RF stress test | Still open if claiming broad tree ensembles | Either add XGBoost/LightGBM/sklearn boosting evidence or narrow the title/abstract/claims |
| Reproducibility/code | Original review asked for code/splits/parameters | Scripts and artifacts exist locally | Partially addressed | Package release, environment, manifest, exact commands, and data split registry |

## Current Experimental Evidence: Strengths

1. Diagnostic value is now the strongest empirical contribution.

Final-test diagnostic rows show that DPG low-confidence and competitor-exposure scores predict local disagreement well:

- DPG low-confidence AUROC: 0.9001 with CI [0.8528, 0.9431].
- Execution-trace DPG low-confidence AUROC: 0.9046 with CI [0.8416, 0.9538].
- Competitor exposure also performs strongly for disagreement detection.

This directly answers the tautology criticism: the value is not that the graph recovers its own traces, but that DPG-derived signals identify disagreement/error regimes.

2. Path controls now directly address the simple path-union critique.

The final report shows that path controls can match model output, but they are much larger and do not provide semantic graph diagnostics:

- `raw_path_union`: native size about 375 predicates.
- `path_bag`: native size about 501 predicates.
- `dpg_execution_trace`: native size about 64 nodes.
- `dpg`: native size about 58 nodes.

This supports the claim that DPG is not merely a raw path dump; it is a compact diagnostic representation.

3. LORE-style rule/counterfactual baseline improves relevance.

The final report includes `lore` as a rule-surrogate baseline with strong output agreement and local accuracy. This is more suitable than ICE for the main comparison.

4. Critical-node claims have been made honest.

The corrected critical-node validation shows weak causal-intervention evidence and no execution-trace critical nodes under the current recombination-based definition. This is not a failure if the manuscript treats critical nodes as conditional structural diagnostics.

5. Scalability/depth evidence exists.

The focused stress test shows DPG execution-trace construction remains recombination-free at depths 4, 8, and 12, with high edge recall:

- Depth 4: edge recall 0.9978, average runtime 118.9 ms, average active nodes 96.5.
- Depth 8: edge recall 0.9849, average runtime 4379 ms, average active nodes 161.8.
- Depth 12: edge recall 0.9687, average runtime 14630 ms, average active nodes 192.7.

This supports a bounded claim: DPG-local remains structurally trace-faithful as depth grows, but runtime can become substantial.

## Current Experimental Evidence: Weaknesses

1. The current manuscript does not yet use the strongest results.

`JOURNAL_paper/ECML26_DPG.tex` still reports older ECML-era values in the Results section and still describes the main model setup as depth-4 random forests. The final report has better and more defensible numbers, but they are not yet integrated.

2. Broad "tree ensemble" claims remain risky.

The new deeper-RF stress test helps, but it does not establish generality for boosted trees. For Information Sciences, this can be acceptable if the claims are narrowed to random forests. For Information Fusion, a broader fusion-system claim may need either boosted-tree evidence or a stronger formal framing.

3. Confidence-score sensitivity remains open.

Reviewer 2 specifically criticized equal weighting and the multiplicative structural coverage gate. The current artifacts do not show a confidence-weight/gate sensitivity analysis. For a Q1 journal, this is one of the most important remaining experiments because it directly tests the robustness of the main diagnostic score.

4. Runtime is a double-edged result.

The deeper-RF run is valuable but shows that runtime can grow sharply: depth-12/50-tree runs average about 21.7 seconds, with `isolet` reaching about 72 seconds in the worst-case table. This is acceptable only if written as a deployment boundary and paired with pruning/sampling as future or optional work.

5. Critical nodes should not be claimed as a central contribution.

The data do not support critical nodes as a strong causal or intervention primitive. They can remain as a secondary diagnostic and case-study tool.

## Recommended Claims

Use these claims:

- DPG-local is a structure-aware diagnostic complement for random forest explanations.
- The main contribution is the execution-trace graph plus class-contrastive diagnostics, not merely "showing paths."
- DPG-derived confidence, support margin, and competitor exposure provide measurable diagnostic value for local disagreement and model-error regimes.
- Raw path controls show why simple executed-path dumping is not enough: it is output-faithful but large and semantically poorer.
- Critical nodes are conditional structural diagnostics whose causal interpretation is limited.
- Deeper-RF stress tests bound scalability: structural trace faithfulness remains high, but graph size and runtime grow.

Avoid these claims:

- DPG-local replaces SHAP/LIME/Anchors.
- DPG-local is generally superior on output fidelity.
- Critical nodes causally explain decisions.
- The method is validated for all tree ensembles unless boosted-tree evidence is added.
- Zero recombination alone proves explanatory value.

## Acceptance Readiness By Target

### Information Sciences

Fit: good, if framed as graph-based XAI for intelligent systems.

Adherence after integrating current results: plausible.

Main remaining requirements:

- Rewrite the manuscript in journal format.
- Replace ECML results with final-test results.
- Add common-interface appendix.
- Add path-control and LORE tables.
- Add confidence sensitivity or explicitly downgrade confidence as heuristic.
- Add scalability section and bounded claims.
- Add reproducibility package/manifest.

### Information Fusion

Fit: possible but riskier.

The paper must be reframed as a method for fusing tree-level decision-process evidence into a class-contrastive local graph. The current XAI framing alone may not be enough for Information Fusion, whose scope is explicitly about multi-source/process fusion, fusion architectures, fusion algorithms, applications, and computational demands.

Additional requirements for stronger fit:

- Use "decision/process fusion" language carefully and formally.
- Explain each tree path as a source of decision evidence and DPG-local as the fusion object.
- Emphasize multi-classifier/decision-system relevance.
- Surface computational-demands analysis from the depth experiment.
- Consider a stronger application/auditing case study or a human-grounded diagnostic task.

## Remaining Experiments

Minimum remaining experiment for a strong journal revision:

1. Confidence-score sensitivity/calibration.
   - Vary the confidence components or the structural gate.
   - Report whether disagreement AUROC remains stable.
   - This directly answers the reviewer concern about equal weighting.

Strongly recommended if keeping broad "tree ensemble" language:

2. Boosted-tree transfer experiment.
   - XGBoost or LightGBM if available; otherwise explicitly narrow to random forests.
   - Use a small representative subset: low-dimensional binary, multiclass, high-dimensional binary, high-dimensional multiclass.
   - Report whether trace extraction, graph size, runtime, and diagnostic AUROC remain meaningful.

Recommended writing-side experiment/table:

3. Full common-interface appendix.
   - Convert `common_explainer_interface.csv` into a LaTeX table.
   - Add fields for explained class, score, competitor, selected features, native size, runtime, and limitations.

Optional, high-effort, high-reward:

4. Human-grounded diagnostic task.
   - Compare whether users can identify disagreement/error cases faster or more accurately using DPG diagnostics versus path union/SHAP.
   - This would help Information Fusion more than Information Sciences, but it is not strictly necessary for a solid Information Sciences submission.

## Immediate Next Writing Steps

1. Rewrite the Experimental Setup around the final-test protocol.
2. Replace the main Results section with the final-test report numbers.
3. Add a path-control subsection.
4. Add a diagnostic-value subsection centered on disagreement AUROC/AUPRC.
5. Add a scalability subsection using the depth-focused report.
6. Move critical-node details to a small bounded paragraph plus appendix.
7. Add a limitations paragraph on `isolet`, `madelon`, and depth-12 runtime.
8. Add the common-interface appendix.

## Bottom Line

The current experiments are much closer to journal quality than the ECML submission, but the paper is not Q1-ready yet. The strongest route is to target Information Sciences with a conservative, rigorous story: DPG-local is a compact structure-aware diagnostic layer for random forests, supported by path-control baselines, corrected statistics, disagreement-detection evidence, critical-node boundary analysis, and scalability stress tests.

Information Fusion remains possible, but only after a stronger fusion-centered rewrite.
