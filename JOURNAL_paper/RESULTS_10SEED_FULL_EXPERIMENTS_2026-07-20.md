# Full 10-Seed Experimental Results for the Journal Revision

Date: 2026-07-20

This note summarizes the completed full experiment requested for the journal revision. It is intended as a paper-update source, not as a replacement for the manuscript text. The wording below keeps the results explicit and conservative, because several baselines are output-aligned or path-control procedures and therefore should not be interpreted as direct evidence against the structural contribution of Decision Predicate Graphs.

## Source Files

Final run directory:

```text
experiments_local_explanation/results_journal_v1/final_test_all_methods_10seeds
```

Generated report directory:

```text
experiments_local_explanation/results_journal_v1/journal_report_all_methods_10seeds
```

Main files used here:

- `experiments_local_explanation/results_journal_v1/final_test_all_methods_10seeds/summary_selected_test.csv`
- `experiments_local_explanation/results_journal_v1/final_test_all_methods_10seeds/per_sample_selected_test.csv`
- `experiments_local_explanation/results_journal_v1/journal_report_all_methods_10seeds/method_summary_ci.csv`
- `experiments_local_explanation/results_journal_v1/journal_report_all_methods_10seeds/paired_tests_holm.csv`
- `experiments_local_explanation/results_journal_v1/journal_report_all_methods_10seeds/diagnostic_value_summary.csv`
- `experiments_local_explanation/results_journal_v1/journal_report_all_methods_10seeds/size_runtime_summary_ci.csv`
- `experiments_local_explanation/results_journal_v1/journal_report_all_methods_10seeds/common_explainer_interface.csv`

## Experimental Coverage

The completed run contains 1500 summary rows: 15 datasets, 10 methods or controls, and 10 seeds per dataset-method pair. The seeds are: 27, 42, 100, 101, 102, 103, 104, 105, 106, 107.

Completed runs by method:

```text
               method  completed_runs
              anchors             150
                  dpg             150
  dpg_execution_trace             150
                 lime             150
                 lore             150
             path_bag             150
random_same_size_path             150
       raw_path_union             150
                 shap             150
            tree_path             150
```

Test samples evaluated per seed and method:

```text
                dataset  n_test_evaluated
banknote-authentication               275
          breast_cancer               114
               diabetes               154
                 digits               360
             ionosphere                71
                   iris                30
                 isolet              1560
                madelon               520
                phoneme              1081
            qsar-biodeg               211
                segment               462
               spambase               921
                vehicle               170
                   wdbc               114
                   wine                36
```

Important paper detail: the local explanation evaluation was performed on the full test split for each dataset, not on a small subsample. Therefore each method-seed pair evaluates 30 to 1560 test instances depending on the dataset, and every dataset-method pair has 10 repetitions.

## Method Roles

The comparison should be framed by explainer role. Some methods are output-aligned by construction, while DPG methods expose structural evidence and graph diagnostics. This distinction is important when discussing fidelity.

```text
               method                              main_role                                                 score_field                               size_field                                                              unsupported_or_limited
                  dpg             structure-aware diagnostic explanation_confidence, support_margin, competitor_exposure                 active graph nodes/edges                                             not optimized for output fidelity alone
  dpg_execution_trace                trace-faithful ablation explanation_confidence, support_margin, competitor_exposure                 active trace nodes/edges                                         larger/readability cost than aggregated DPG
                 shap     output-aligned feature attribution         score_margin_pred_vs_competitor, contribution norms               non-zero attribution count                                                        no path or transition object
                 lime         output-aligned local surrogate         score_margin_pred_vs_competitor, contribution norms               non-zero coefficient count                                                          no executed-path semantics
              anchors                rule-surrogate baseline                           anchor_precision, anchor_coverage                     rule predicate count                                      rule is perturbation-induced, not forest trace
                 lore rule/counterfactual surrogate baseline         lore_fidelity_neighborhood, counterfactual distance               local rule predicate count                       lightweight LORE-style implementation, not full LOREM package
            tree_path           simple path-feature baseline                             score_margin_pred_vs_competitor              non-zero path feature count                                          feature-level only; loses transition graph
       raw_path_union           simple path-control baseline                             score_margin_pred_vs_competitor          unique executed predicate count no DPG canonicalization, class support graph, competitor exposure, or critical node
             path_bag           simple path-control baseline                             score_margin_pred_vs_competitor executed predicate count with duplicates              keeps path volume but loses graph transitions and semantic diagnostics
random_same_size_path         negative path-control baseline                             score_margin_pred_vs_competitor                  sampled predicate count                                     size control only; no local execution semantics
                  ice       diagnostic response-profile only                                   ice_prob_range, ice_slope                       one varied feature                             not a local route explainer; excluded from main ranking
```

## Main Method-Level Results

The following table reports the bootstrap mean and 95% confidence interval generated from the full 10-seed final test output. Local fidelity is `local_matches_model_rate`, i.e. the rate at which the local explanation prediction agrees with the black-box model. Local accuracy is agreement with the true class labels. Explanation confidence is available for DPG-based methods because it is a structural diagnostic score, not a common output of all baselines.

```text
               method                  group  datasets       model_accuracy       local_fidelity       local_accuracy           confidence              runtime_ms
                 lime feature/output-aligned        15 0.858 [0.840, 0.875] 0.415 [0.381, 0.446] 0.413 [0.385, 0.440]                  n/a   131.2 [80.410, 192.3]
                 shap feature/output-aligned        15 0.858 [0.842, 0.874] 1.000 [1.000, 1.000] 0.858 [0.841, 0.874]                  n/a    3.470 [2.927, 4.070]
            tree_path feature/output-aligned        15 0.858 [0.841, 0.874] 1.000 [1.000, 1.000] 0.858 [0.840, 0.874]                  n/a    2.202 [2.164, 2.240]
             path_bag           path-control        15 0.905 [0.890, 0.920] 1.000 [1.000, 1.000] 0.905 [0.891, 0.920]                  n/a    4.626 [4.165, 5.112]
random_same_size_path           path-control        15 0.905 [0.890, 0.920] 1.000 [1.000, 1.000] 0.905 [0.890, 0.919]                  n/a 55.151 [42.950, 68.410]
       raw_path_union           path-control        15 0.905 [0.890, 0.919] 1.000 [1.000, 1.000] 0.905 [0.889, 0.920]                  n/a    4.412 [3.985, 4.858]
              anchors         rule-surrogate        15 0.858 [0.841, 0.875] 1.000 [1.000, 1.000] 0.858 [0.842, 0.874]                  n/a    2.455 [2.425, 2.485]
                 lore         rule-surrogate        15 0.858 [0.840, 0.874] 0.971 [0.967, 0.975] 0.855 [0.836, 0.871]                  n/a 37.894 [26.684, 50.804]
                  dpg        structure-aware        15 0.858 [0.840, 0.874] 0.836 [0.807, 0.864] 0.762 [0.732, 0.790] 0.491 [0.467, 0.515] 34.103 [24.823, 46.932]
  dpg_execution_trace        structure-aware        15 0.858 [0.841, 0.874] 0.865 [0.838, 0.890] 0.786 [0.759, 0.813] 0.661 [0.644, 0.678]   9.851 [9.569, 10.127]
```

Main observations for the paper:

- `dpg_execution_trace` improves over aggregated `dpg` on local fidelity, local accuracy, confidence, support margin, path purity, and runtime. This supports keeping the execution-trace variant as the main journal version of the local DPG explanation.
- `shap`, `tree_path`, `anchors`, and the path controls obtain fidelity equal to 1.000 because their local prediction is output-aligned with the model or constructed from the model prediction. This should not be presented as structural superiority over DPG.
- `lore` is a suitable rule-surrogate comparator. It has high local fidelity, but its native explanation object is a compact surrogate rule/counterfactual rather than a graph over decision predicates.
- `lime` is weak in this setting, with mean local fidelity 0.415 and mean local accuracy 0.413. It should remain as a standard local surrogate baseline, but not as the main comparator for the structural claim.
- Path controls reach perfect output fidelity but do not provide class-contrastive DPG quantities such as confidence, support margin, competitor exposure, recombination, trace coverage, or critical-node diagnostics. They are best used as controls demonstrating that output fidelity alone is an incomplete criterion.

## Corrected Paired Tests

The paired Wilcoxon tests use the 15 datasets as paired units, with Holm correction across comparisons within each metric family. The reference method is `dpg_execution_trace`. Positive mean difference means that `dpg_execution_trace` is larger than the comparison method.

```text
                    metric    reference_method     comparison_method  n_datasets reference_mean comparison_mean mean_difference_ref_minus_comparison p_value p_holm rank_biserial_ref_minus_comparison
  local_matches_model_rate dpg_execution_trace               anchors          15         0.8651          1.0000                              -0.1349  0.0001 0.0005                            -1.0000
  local_matches_model_rate dpg_execution_trace                   dpg          15         0.8651          0.8361                               0.0290  0.0004 0.0005                             0.9333
  local_matches_model_rate dpg_execution_trace                  lime          15         0.8651          0.4146                               0.4505  0.0001 0.0005                             1.0000
  local_matches_model_rate dpg_execution_trace                  lore          15         0.8651          0.9711                              -0.1060  0.0001 0.0005                            -1.0000
  local_matches_model_rate dpg_execution_trace              path_bag          15         0.8651          1.0000                              -0.1349  0.0001 0.0005                            -1.0000
  local_matches_model_rate dpg_execution_trace random_same_size_path          15         0.8651          1.0000                              -0.1349  0.0001 0.0005                            -1.0000
  local_matches_model_rate dpg_execution_trace        raw_path_union          15         0.8651          1.0000                              -0.1349  0.0001 0.0005                            -1.0000
  local_matches_model_rate dpg_execution_trace                  shap          15         0.8651          1.0000                              -0.1349  0.0001 0.0005                            -1.0000
  local_matches_model_rate dpg_execution_trace             tree_path          15         0.8651          1.0000                              -0.1349  0.0001 0.0005                            -1.0000
            local_accuracy dpg_execution_trace               anchors          15         0.7859          0.8576                              -0.0717  0.0003 0.0024                            -0.9500
            local_accuracy dpg_execution_trace                   dpg          15         0.7859          0.7623                               0.0236  0.0020 0.0024                             0.8500
            local_accuracy dpg_execution_trace                  lime          15         0.7859          0.4133                               0.3726  0.0001 0.0005                             1.0000
            local_accuracy dpg_execution_trace                  lore          15         0.7859          0.8546                              -0.0687  0.0012 0.0024                            -0.8833
            local_accuracy dpg_execution_trace              path_bag          15         0.7859          0.9051                              -0.1193  0.0003 0.0024                            -0.9500
            local_accuracy dpg_execution_trace random_same_size_path          15         0.7859          0.9053                              -0.1194  0.0003 0.0024                            -0.9500
            local_accuracy dpg_execution_trace        raw_path_union          15         0.7859          0.9051                              -0.1193  0.0003 0.0024                            -0.9500
            local_accuracy dpg_execution_trace                  shap          15         0.7859          0.8576                              -0.0717  0.0003 0.0024                            -0.9500
            local_accuracy dpg_execution_trace             tree_path          15         0.7859          0.8576                              -0.0717  0.0003 0.0024                            -0.9500
avg_explanation_confidence dpg_execution_trace                   dpg          15         0.6615          0.4911                               0.1704  0.0001 0.0001                             1.0000
        avg_support_margin dpg_execution_trace                   dpg          15         0.6957          0.5975                               0.0982  0.0006 0.0006                             0.9167
   avg_competitor_exposure dpg_execution_trace                   dpg          15         0.1798          0.2423                              -0.0626  0.0006 0.0006                            -0.9167
           avg_path_purity dpg_execution_trace                   dpg          15         0.8202          0.7577                               0.0626  0.0006 0.0006                             0.9167
    avg_recombination_rate dpg_execution_trace                   dpg          15         0.0000          0.1405                              -0.1405  0.0001 0.0001                            -1.0000
```

Statistical interpretation:

- `dpg_execution_trace` is significantly better than aggregated `dpg` for local fidelity, local accuracy, explanation confidence, support margin, and path purity after Holm correction.
- `dpg_execution_trace` is significantly worse than output-aligned or control methods on raw local fidelity and local accuracy. This should be acknowledged directly and explained as a consequence of comparing structural explanations with output-preserving predictors or controls.
- The significant gap against LORE in local fidelity and local accuracy means that the paper should not claim predictive superiority over LORE. The correct claim is that DPG provides additional structural diagnostics not produced by LORE.
- The recombination rate is significantly lower for `dpg_execution_trace` than for aggregated `dpg`, with mean 0.000 versus 0.141. This is expected because the execution-trace variant restricts the explanation to actually executed local routes. A zero recombination rate means no cross-route recombination is introduced in the trace explanation; it is not a failure.

## DPG-Specific Structural Diagnostics

```text
             method avg_explanation_confidence_mean avg_support_margin_mean avg_competitor_exposure_mean avg_path_purity_mean avg_recombination_rate_mean avg_runtime_ms_mean
                dpg                          0.4911                  0.5975                       0.2423               0.7577                      0.1405             34.1025
dpg_execution_trace                          0.6615                  0.6957                       0.1798               0.8202                      0.0000              9.8511
```

The execution-trace DPG variant has higher confidence, larger class support margin, lower competitor exposure, higher path purity, and zero recombination. These results support a journal framing centered on local structural diagnostics rather than on treating DPG as another feature-ranking explainer.

## Diagnostic Value of DPG Scores

The diagnostic analysis evaluates whether DPG structural scores identify difficult or unreliable cases. Higher AUROC indicates better ranking of problematic instances. The strongest result is the ability of low DPG confidence and competitor exposure to identify local disagreement and model error.

```text
             method             target                         score  n_datasets mean_positive_rate auroc_mean auroc_ci95_low auroc_ci95_high auprc_mean auprc_ci95_low auprc_ci95_high
                dpg local_disagreement        critical_node_contrast          15             0.2424     0.5997         0.5248          0.6772     0.3404         0.2204          0.4733
                dpg local_disagreement dpg_competitor_exposure_score          15             0.1639     0.8596         0.7971          0.9149     0.4752         0.4139          0.5436
                dpg local_disagreement      dpg_low_confidence_score          15             0.1639     0.8777         0.8314          0.9218     0.5206         0.4415          0.5956
                dpg local_disagreement         dpg_uncertainty_score          15             0.1639     0.8660         0.8124          0.9161     0.4781         0.4136          0.5434
                dpg local_disagreement              num_active_nodes          15             0.1639     0.5454         0.4762          0.6018     0.1906         0.1206          0.2825
                dpg local_disagreement            recombination_rate          15             0.1639     0.6042         0.5432          0.6701     0.2661         0.1695          0.3840
dpg_execution_trace local_disagreement        critical_node_contrast           1             0.3659     0.6615            n/a             n/a     0.5291            n/a             n/a
dpg_execution_trace local_disagreement dpg_competitor_exposure_score          15             0.1349     0.8985         0.8501          0.9382     0.4927         0.4325          0.5553
dpg_execution_trace local_disagreement      dpg_low_confidence_score          15             0.1349     0.9149         0.8635          0.9539     0.5580         0.4794          0.6342
dpg_execution_trace local_disagreement         dpg_uncertainty_score          15             0.1349     0.8964         0.8444          0.9404     0.4784         0.4195          0.5420
dpg_execution_trace local_disagreement              num_active_nodes          15             0.1349     0.6076         0.5328          0.6744     0.1724         0.0946          0.2773
dpg_execution_trace local_disagreement            recombination_rate          15             0.1349     0.5000         0.5000          0.5000     0.1349         0.0667          0.2262
                dpg        model_error        critical_node_contrast          13             0.1902     0.5482         0.4902          0.6115     0.2368         0.1750          0.2953
                dpg        model_error dpg_competitor_exposure_score          15             0.1424     0.7316         0.6706          0.7949     0.2557         0.1941          0.3152
                dpg        model_error      dpg_low_confidence_score          15             0.1424     0.7167         0.6606          0.7777     0.2494         0.1910          0.3021
                dpg        model_error         dpg_uncertainty_score          15             0.1424     0.7281         0.6619          0.7928     0.2481         0.1878          0.3063
                dpg        model_error              num_active_nodes          15             0.1424     0.6006         0.5551          0.6515     0.1745         0.1243          0.2254
                dpg        model_error            recombination_rate          15             0.1424     0.5109         0.4576          0.5617     0.1588         0.1107          0.2062
dpg_execution_trace        model_error        critical_node_contrast           2             0.2764     0.4329         0.2200          0.6458     0.3111         0.2797          0.3425
dpg_execution_trace        model_error dpg_competitor_exposure_score          15             0.1424     0.7600         0.7019          0.8193     0.2688         0.2085          0.3320
dpg_execution_trace        model_error      dpg_low_confidence_score          15             0.1424     0.7597         0.7046          0.8160     0.2844         0.2235          0.3385
dpg_execution_trace        model_error         dpg_uncertainty_score          15             0.1424     0.7540         0.6891          0.8164     0.2620         0.2018          0.3202
dpg_execution_trace        model_error              num_active_nodes          15             0.1424     0.6269         0.5738          0.6824     0.1926         0.1406          0.2462
dpg_execution_trace        model_error            recombination_rate          15             0.1424     0.5000         0.5000          0.5000     0.1424         0.0923          0.1922
```

Diagnostic conclusions for the paper:

- For local disagreement, `dpg_execution_trace` low-confidence score reaches AUROC 0.915 [0.864, 0.954], and competitor exposure reaches AUROC 0.898 [0.850, 0.938]. These are strong results and should be emphasized.
- For model error, `dpg_execution_trace` low-confidence score reaches AUROC 0.760 [0.705, 0.816], and competitor exposure reaches AUROC 0.760 [0.702, 0.819]. This supports the claim that DPG can flag locally uncertain or error-prone regions of the model.
- Critical-node contrast is weaker and less consistently available than confidence and competitor exposure. It should be discussed as a conditional descriptive diagnostic, not as a central standalone contribution.
- Recombination rate does not act as a useful risk score for `dpg_execution_trace` because it is exactly zero by construction in the trace-restricted setting.

## Size and Runtime

Native size denotes the method-specific explanation size, such as active DPG nodes/edges, non-zero attributions, rule predicates, or path-control predicates. The numbers are not directly identical semantic units, but they document the cost/readability trade-off.

```text
               method           method_group  n_datasets native_size_mean native_size_ci95_low native_size_ci95_high runtime_ms_mean runtime_ms_ci95_low runtime_ms_ci95_high
                 lime feature/output-aligned          15           15.394               12.064                18.457           131.2              23.159                331.3
                 shap feature/output-aligned          15           35.610               17.835                58.500           3.470               2.222                5.403
            tree_path feature/output-aligned          15           22.367               14.127                31.898           2.202               2.073                2.312
             path_bag           path-control          15            501.8                303.6                 710.1           4.626               3.167                6.100
random_same_size_path           path-control          15            376.1                215.3                 561.6          55.151              20.056                101.7
       raw_path_union           path-control          15            376.1                213.2                 564.4           4.412               3.050                5.749
                 lore         rule-surrogate          15            3.055                2.787                 3.328          37.894               9.121               83.843
                  dpg        structure-aware          15           63.775               56.500                70.933          34.103              17.355               58.886
  dpg_execution_trace        structure-aware          15           63.765               56.031                70.377           9.851               8.930               10.740
```

Runtime interpretation:

- `dpg_execution_trace` is substantially faster than aggregated `dpg` in the final test run, with mean runtime 9.851 ms versus 34.103 ms.
- `tree_path`, `anchors`, `shap`, and the deterministic path controls are faster, but they do not expose the same graph diagnostics.
- `lime` and `lore` are slower on average, especially on larger datasets. This supports reporting runtime as a secondary operational consideration.

## Per-Dataset DPG Results

The following table reports mean values across the 10 seeds for the two DPG variants. These are the most useful numbers for dataset-specific discussion.

```text
                dataset              method model_accuracy local_matches_model_rate local_accuracy avg_explanation_confidence avg_support_margin avg_competitor_exposure avg_path_purity avg_recombination_rate avg_runtime_ms
banknote-authentication                 dpg         0.9738                   0.9589         0.9335                     0.5942             0.8297                  0.0851          0.9149                 0.0620        13.1913
banknote-authentication dpg_execution_trace         0.9738                   0.9705         0.9451                     0.7718             0.9020                  0.0490          0.9510                 0.0000         8.4819
          breast_cancer                 dpg         0.9465                   0.9833         0.9526                     0.7022             0.9370                  0.0315          0.9685                 0.0339        14.4494
          breast_cancer dpg_execution_trace         0.9465                   0.9860         0.9535                     0.7501             0.9447                  0.0277          0.9723                 0.0000        11.1098
               diabetes                 dpg         0.7299                   0.8532         0.6987                     0.3697             0.4737                  0.2632          0.7368                 0.3013        22.1275
               diabetes dpg_execution_trace         0.7299                   0.8851         0.7188                     0.6861             0.7123                  0.1439          0.8561                 0.0000        11.7313
                 digits                 dpg         0.8650                   0.6425         0.6031                     0.1585             0.1837                  0.6341          0.3659                 0.4736          171.6
                 digits dpg_execution_trace         0.8650                   0.7508         0.6906                     0.4933             0.3309                  0.4575          0.5425                 0.0000         9.5627
             ionosphere                 dpg         0.9380                   0.8394         0.8028                     0.6157             0.8225                  0.0888          0.9112                 0.0674        19.6236
             ionosphere dpg_execution_trace         0.9380                   0.8859         0.8549                     0.7528             0.8551                  0.0724          0.9276                 0.0000        10.9167
                   iris                 dpg         0.9433                   0.9100         0.8533                     0.4841             0.7827                  0.1101          0.8899                 0.2017        54.6944
                   iris dpg_execution_trace         0.9433                   0.9833         0.9467                     0.7383             0.9307                  0.0348          0.9652                 0.0000         7.2902
                 isolet                 dpg         0.7447                   0.3149         0.2751                     0.4033             0.1474                  0.6391          0.3609                 0.0100        12.7955
                 isolet dpg_execution_trace         0.7447                   0.3525         0.2973                     0.4127             0.1365                  0.6490          0.3510                 0.0000         8.5327
                madelon                 dpg         0.6562                   0.7144         0.5983                     0.5102             0.3946                  0.3027          0.6973                 0.0274        13.5565
                madelon dpg_execution_trace         0.6562                   0.7115         0.5981                     0.5436             0.4000                  0.3000          0.7000                 0.0000         9.5614
                phoneme                 dpg         0.8100                   0.9340         0.7808                     0.6335             0.7887                  0.1056          0.8944                 0.0295        13.4060
                phoneme dpg_execution_trace         0.8100                   0.9486         0.7863                     0.6918             0.7952                  0.1024          0.8976                 0.0000         8.8056
            qsar-biodeg                 dpg         0.8166                   0.8882         0.8109                     0.4010             0.4291                  0.2855          0.7145                 0.2848        29.9632
            qsar-biodeg dpg_execution_trace         0.8166                   0.9289         0.8024                     0.6781             0.7125                  0.1437          0.8563                 0.0000        11.0335
                segment                 dpg         0.8654                   0.8100         0.7299                     0.4314             0.5217                  0.3020          0.6980                 0.0549        14.9917
                segment dpg_execution_trace         0.8654                   0.8489         0.7606                     0.6324             0.6103                  0.2359          0.7641                 0.0000         6.7653
               spambase                 dpg         0.9053                   0.9066         0.8432                     0.5194             0.6270                  0.1865          0.8135                 0.1457        15.7392
               spambase dpg_execution_trace         0.9053                   0.9086         0.8408                     0.7273             0.7990                  0.1005          0.8995                 0.0000         8.5631
                vehicle                 dpg         0.7200                   0.8671         0.6765                     0.2736             0.3081                  0.4501          0.5499                 0.3176        81.8513
                vehicle dpg_execution_trace         0.7200                   0.8694         0.6888                     0.5581             0.5168                  0.2700          0.7300                 0.0000        10.8071
                   wdbc                 dpg         0.9518                   0.9526         0.9114                     0.6940             0.9299                  0.0351          0.9649                 0.0306        14.8603
                   wdbc dpg_execution_trace         0.9518                   0.9632         0.9237                     0.7516             0.9341                  0.0329          0.9671                 0.0000        11.4891
                   wine                 dpg         0.9972                   0.9667         0.9639                     0.5752             0.7874                  0.1156          0.8844                 0.0676        18.6744
                   wine dpg_execution_trace         0.9972                   0.9833         0.9806                     0.7344             0.8555                  0.0768          0.9232                 0.0000        13.1164
```

Dataset-level interpretation:

- `isolet` is the clearest stress case: it has 26 classes, 617 features, and 1560 evaluated test samples. DPG local fidelity and local accuracy are much lower there, while competitor exposure is high. This should be discussed as evidence that high-dimensional multiclass settings expose the limits of compact local graph explanations.
- `madelon` is another difficult high-dimensional case, but less severe than `isolet`. It supports the discussion that dimensionality and weak boundary separation affect the practical usefulness of local structural explanations.
- Low-dimensional binary or small multiclass datasets generally show stronger DPG behavior, especially for confidence and margin diagnostics.

## Per-Dataset Local Fidelity by Method

Values are means across the 10 seeds.

```text
                dataset   dpg dpg_execution_trace anchors  lime  lore  shap tree_path path_bag random_same_size_path raw_path_union
banknote-authentication 0.959               0.971   1.000 0.530 0.996 1.000     1.000    1.000                 1.000          1.000
          breast_cancer 0.983               0.986   1.000 0.369 0.988 1.000     1.000    1.000                 1.000          1.000
               diabetes 0.853               0.885   1.000 0.276 0.978 1.000     1.000    1.000                 1.000          1.000
                 digits 0.642               0.751   1.000 0.094 0.965 1.000     1.000    1.000                 1.000          1.000
             ionosphere 0.839               0.886   1.000 0.679 0.994 1.000     1.000    1.000                 1.000          1.000
                   iris 0.910               0.983   1.000 0.403 0.990 1.000     1.000    1.000                 1.000          1.000
                 isolet 0.315               0.352   1.000 0.199 0.916 1.000     1.000    1.000                 1.000          1.000
                madelon 0.714               0.712   1.000 0.498 0.938 1.000     1.000    1.000                 1.000          1.000
                phoneme 0.934               0.949   1.000 0.717 0.989 1.000     1.000    1.000                 1.000          1.000
            qsar-biodeg 0.888               0.929   1.000 0.739 0.952 1.000     1.000    1.000                 1.000          1.000
                segment 0.810               0.849   1.000 0.247 0.967 1.000     1.000    1.000                 1.000          1.000
               spambase 0.907               0.909   1.000 0.529 0.963 1.000     1.000    1.000                 1.000          1.000
                vehicle 0.867               0.869   1.000 0.272 0.941 1.000     1.000    1.000                 1.000          1.000
                   wdbc 0.953               0.963   1.000 0.329 0.990 1.000     1.000    1.000                 1.000          1.000
                   wine 0.967               0.983   1.000 0.336 1.000 1.000     1.000    1.000                 1.000          1.000
```

## Per-Dataset Local Accuracy by Method

Values are means across the 10 seeds.

```text
                dataset   dpg dpg_execution_trace anchors  lime  lore  shap tree_path path_bag random_same_size_path raw_path_union
banknote-authentication 0.933               0.945   0.974 0.556 0.970 0.974     0.974    0.996                 0.996          0.996
          breast_cancer 0.953               0.954   0.946 0.368 0.950 0.946     0.946    0.947                 0.949          0.947
               diabetes 0.699               0.719   0.730 0.359 0.726 0.730     0.730    0.742                 0.742          0.742
                 digits 0.603               0.691   0.865 0.101 0.858 0.865     0.865    0.969                 0.969          0.969
             ionosphere 0.803               0.855   0.938 0.648 0.941 0.938     0.938    0.954                 0.954          0.954
                   iris 0.853               0.947   0.943 0.393 0.940 0.943     0.943    0.943                 0.943          0.943
                 isolet 0.275               0.297   0.745 0.189 0.744 0.745     0.745    0.939                 0.939          0.939
                madelon 0.598               0.598   0.656 0.467 0.657 0.656     0.656    0.707                 0.707          0.707
                phoneme 0.781               0.786   0.810 0.710 0.806 0.810     0.810    0.898                 0.898          0.898
            qsar-biodeg 0.811               0.802   0.817 0.662 0.792 0.817     0.817    0.871                 0.871          0.871
                segment 0.730               0.761   0.865 0.236 0.872 0.865     0.865    0.970                 0.970          0.970
               spambase 0.843               0.841   0.905 0.522 0.898 0.905     0.905    0.945                 0.945          0.945
                vehicle 0.676               0.689   0.720 0.285 0.708 0.720     0.720    0.738                 0.738          0.738
                   wdbc 0.911               0.924   0.952 0.368 0.960 0.952     0.952    0.957                 0.957          0.957
                   wine 0.964               0.981   0.997 0.333 0.997 0.997     0.997    1.000                 1.000          1.000
```

## Recommended Manuscript Updates

Suggested Results section points:

- State explicitly that the final test evaluation uses 15 datasets, 10 seeds, all test samples, and 10 methods or controls, yielding 1500 dataset-method-seed runs.
- Report DPG execution trace as the main local DPG variant and aggregated DPG as an ablation. The execution-trace variant is more faithful to executed local routes and provides better confidence, support margin, runtime, and recombination behavior.
- Avoid claiming that DPG has the highest raw local fidelity. It does not. Instead, state that raw output fidelity is highest for output-aligned baselines and path controls, while DPG contributes graph-level evidence, confidence, competitor exposure, path purity, and recombination diagnostics.
- Use LORE as the main rule-surrogate comparator, and keep ICE outside the main ranking or in an appendix as previously decided.
- Present path controls as a validity check: they show that preserving path-level output information can be trivial or nearly trivial, but this does not provide the semantic graph diagnostics required by the proposed contribution.
- Use the diagnostic-value results as a central contribution: DPG confidence and competitor exposure identify local disagreement and model error with strong AUROC values.
- Discuss critical nodes cautiously. The evidence does not support making critical nodes a main contribution. They can be described as an interpretable local descriptor that is useful in selected cases, especially when a clear contrastive node exists.

Suggested Discussion section points:

- The paper should be framed around what decision predicate graphs enable: local route structure, class support, competitor exposure, uncertainty diagnostics, and graph-level path analysis.
- The term `DPG local` is narrower than the contribution. A stronger framing is that a Decision Predicate Graph supports local explanation as a structured diagnostic object, rather than only producing a local surrogate prediction.
- The reviewer concern about ICE is valid. ICE is a response-profile diagnostic and should not be treated as a direct local route explainer. Keeping ICE in the appendix is a defensible compromise.
- The journal version should clearly distinguish predictive agreement, structural faithfulness, and diagnostic usefulness. These are different criteria and the final results make that distinction necessary.

## Conservative Claim Set Supported by the Results

The following claims are supported by the completed 10-seed experiments:

1. Decision Predicate Graphs provide local explanations as structured graph objects over model predicates, not only as feature rankings or surrogate rules.
2. The execution-trace DPG variant improves over the aggregated local DPG variant in fidelity to the model output, local accuracy, confidence, margin, path purity, recombination behavior, and runtime.
3. DPG structural confidence and competitor exposure are useful diagnostic scores for identifying local disagreement and model error.
4. Output-aligned feature and path-control methods can dominate raw local fidelity, but they do not provide the same structural diagnostics. Therefore local fidelity alone is insufficient to evaluate the proposed contribution.
5. Critical nodes should be presented as a secondary descriptive diagnostic rather than as the principal contribution.

Claims that should be avoided:

- Do not claim that DPG is the most faithful local predictor across all baselines.
- Do not claim that critical nodes are consistently effective across all datasets.
- Do not claim that recombination rate is always beneficial. In the execution-trace variant, zero recombination is desirable because it means the explanation remains on executed routes.
- Do not compare ICE as if it were a direct local route or rule explainer in the main table.

## Short Text Candidate for the Paper

The final evaluation was conducted on the locked test split of 15 tabular classification datasets using 10 random seeds per dataset and method. For each dataset-method-seed combination, all test instances were explained, resulting in 1500 completed runs. The execution-trace DPG variant achieved a mean local fidelity of 0.865 and a mean local accuracy of 0.786, improving over the aggregated DPG variant, which obtained 0.836 and 0.762, respectively. This improvement was statistically significant under paired Wilcoxon tests across datasets with Holm correction. The execution-trace variant also increased the mean explanation confidence from 0.491 to 0.661, increased the support margin from 0.598 to 0.696, reduced competitor exposure from 0.242 to 0.180, and removed graph recombination by construction.

The strongest evidence for the proposed contribution is not raw output fidelity alone. Output-aligned baselines and path controls reached perfect or near-perfect agreement with the model prediction, but they do not produce the same structural diagnostics. In contrast, DPG confidence and competitor exposure provided strong diagnostic value: for local disagreement, the execution-trace DPG low-confidence score obtained AUROC 0.915 [0.864, 0.954], and competitor exposure obtained AUROC 0.898 [0.850, 0.938]. For model error, both scores reached approximately AUROC 0.760. These results support the use of Decision Predicate Graphs as local structural diagnostic objects that expose not only the predicted class, but also the supporting route structure and competing decision evidence.

## Next Analysis Steps Before Manuscript Integration

- Regenerate the heatmap figure from `summary_selected_test.csv` and remove ICE from the main visual comparison.
- Update the Results subsection with the 10-seed confidence intervals and corrected p-values from `paired_tests_holm.csv`.
- Add an explicit sentence in the experimental protocol stating that all test samples were evaluated for each dataset-method-seed pair.
- Move ICE to an appendix or remove it from the main comparative claims.
- Keep the critical-node discussion in the analysis/discussion section rather than the main contribution list.
