# Journal Final-Test Report

Generated from the validation-selected, locked final-test outputs.

## Main Method Summary

| method                | group                  | datasets | local_matches_model_rate | local_accuracy       | avg_explanation_confidence | avg_runtime_ms           |
| --------------------- | ---------------------- | -------- | ------------------------ | -------------------- | -------------------------- | ------------------------ |
| lime                  | feature/output-aligned | 15       | 0.425 [0.329, 0.511]     | 0.421 [0.338, 0.505] | n/a                        | 86.080 [26.298, 196.602] |
| shap                  | feature/output-aligned | 15       | 1.000 [1.000, 1.000]     | 0.858 [0.805, 0.904] | n/a                        | 4.917 [3.264, 7.746]     |
| tree_path             | feature/output-aligned | 15       | 1.000 [1.000, 1.000]     | 0.858 [0.808, 0.909] | n/a                        | 3.427 [3.033, 3.758]     |
| path_bag              | path-control           | 15       | 1.000 [1.000, 1.000]     | 0.902 [0.843, 0.947] | n/a                        | 7.442 [5.079, 9.703]     |
| random_same_size_path | path-control           | 15       | 1.000 [1.000, 1.000]     | 0.902 [0.852, 0.949] | n/a                        | 78.709 [34.733, 129.318] |
| raw_path_union        | path-control           | 15       | 1.000 [1.000, 1.000]     | 0.902 [0.847, 0.950] | n/a                        | 7.480 [5.113, 9.706]     |
| anchors               | rule-surrogate         | 15       | 1.000 [1.000, 1.000]     | 0.858 [0.802, 0.912] | n/a                        | 4.288 [3.950, 4.591]     |
| lore                  | rule-surrogate         | 15       | 0.971 [0.958, 0.983]     | 0.852 [0.801, 0.906] | n/a                        | 34.992 [11.840, 70.087]  |
| dpg                   | structure-aware        | 15       | 0.825 [0.725, 0.905]     | 0.752 [0.671, 0.831] | 0.477 [0.384, 0.559]       | n/a                      |
| dpg_execution_trace   | structure-aware        | 15       | 0.860 [0.767, 0.931]     | 0.783 [0.692, 0.865] | 0.654 [0.590, 0.705]       | n/a                      |

## Size And Runtime

| method                | group                  | native_size                | runtime_ms               |
| --------------------- | ---------------------- | -------------------------- | ------------------------ |
| lime                  | feature/output-aligned | 15.394 [12.062, 18.757]    | 86.080 [24.001, 192.750] |
| shap                  | feature/output-aligned | 35.343 [17.847, 57.313]    | 4.917 [3.225, 7.546]     |
| tree_path             | feature/output-aligned | 22.025 [14.136, 30.521]    | 3.427 [3.082, 3.772]     |
| path_bag              | path-control           | 501.156 [317.564, 708.744] | 7.442 [5.269, 9.685]     |
| random_same_size_path | path-control           | 375.264 [199.900, 553.855] | 78.709 [36.649, 127.088] |
| raw_path_union        | path-control           | 375.265 [205.591, 551.839] | 7.480 [4.952, 10.057]    |
| lore                  | rule-surrogate         | 3.014 [2.727, 3.297]       | 34.992 [11.169, 66.680]  |
| dpg                   | structure-aware        | 58.254 [48.881, 66.545]    | n/a                      |
| dpg_execution_trace   | structure-aware        | 63.962 [56.708, 70.568]    | n/a                      |

## Corrected Paired Tests

| metric                     | reference_method    | comparison_method     | n_datasets | mean_difference_ref_minus_comparison | p_value   | p_holm    | rank_biserial_ref_minus_comparison |
| -------------------------- | ------------------- | --------------------- | ---------- | ------------------------------------ | --------- | --------- | ---------------------------------- |
| local_matches_model_rate   | dpg_execution_trace | anchors               | 15         | -0.1402                              | 0.001469  | 0.01175   | -1                                 |
| local_matches_model_rate   | dpg_execution_trace | dpg                   | 15         | 0.03482                              | 0.06404   | 0.06404   | 0.5619                             |
| local_matches_model_rate   | dpg_execution_trace | lime                  | 15         | 0.4349                               | 6.104e-05 | 0.0005493 | 1                                  |
| local_matches_model_rate   | dpg_execution_trace | lore                  | 15         | -0.1109                              | 0.002218  | 0.01175   | -1                                 |
| local_matches_model_rate   | dpg_execution_trace | path_bag              | 15         | -0.1402                              | 0.001469  | 0.01175   | -1                                 |
| local_matches_model_rate   | dpg_execution_trace | random_same_size_path | 15         | -0.1402                              | 0.001469  | 0.01175   | -1                                 |
| local_matches_model_rate   | dpg_execution_trace | raw_path_union        | 15         | -0.1402                              | 0.001469  | 0.01175   | -1                                 |
| local_matches_model_rate   | dpg_execution_trace | shap                  | 15         | -0.1402                              | 0.001469  | 0.01175   | -1                                 |
| local_matches_model_rate   | dpg_execution_trace | tree_path             | 15         | -0.1402                              | 0.001469  | 0.01175   | -1                                 |
| local_accuracy             | dpg_execution_trace | anchors               | 15         | -0.07434                             | 0.003324  | 0.01662   | -0.9231                            |
| local_accuracy             | dpg_execution_trace | dpg                   | 15         | 0.03124                              | 0.1094    | 0.1094    | 0.4857                             |
| local_accuracy             | dpg_execution_trace | lime                  | 15         | 0.3627                               | 6.104e-05 | 0.0005493 | 1                                  |
| local_accuracy             | dpg_execution_trace | lore                  | 15         | -0.06885                             | 0.007649  | 0.01662   | -0.8718                            |
| local_accuracy             | dpg_execution_trace | path_bag              | 15         | -0.1186                              | 0.001871  | 0.01497   | -0.978                             |
| local_accuracy             | dpg_execution_trace | random_same_size_path | 15         | -0.1186                              | 0.001871  | 0.01497   | -0.978                             |
| local_accuracy             | dpg_execution_trace | raw_path_union        | 15         | -0.1186                              | 0.001871  | 0.01497   | -0.978                             |
| local_accuracy             | dpg_execution_trace | shap                  | 15         | -0.07426                             | 0.004154  | 0.01662   | -0.9011                            |
| local_accuracy             | dpg_execution_trace | tree_path             | 15         | -0.07426                             | 0.004154  | 0.01662   | -0.9011                            |
| avg_explanation_confidence | dpg_execution_trace | dpg                   | 15         | 0.1766                               | 6.104e-05 | 6.104e-05 | 1                                  |
| avg_support_margin         | dpg_execution_trace | dpg                   | 15         | 0.07252                              | 0.04126   | 0.04126   | 0.6                                |
| avg_competitor_exposure    | dpg_execution_trace | dpg                   | 15         | -0.03725                             | 0.05536   | 0.05536   | -0.5667                            |
| avg_path_purity            | dpg_execution_trace | dpg                   | 15         | 0.03725                              | 0.05536   | 0.05536   | 0.5667                             |
| avg_recombination_rate     | dpg_execution_trace | dpg                   | 15         | -0.1339                              | 6.104e-05 | 6.104e-05 | -1                                 |

## Diagnostic Value

| method              | target             | score                         | n_datasets | mean_positive_rate | auroc_mean | auroc_ci95_low | auroc_ci95_high | auprc_mean | auprc_ci95_low | auprc_ci95_high |
| ------------------- | ------------------ | ----------------------------- | ---------- | ------------------ | ---------- | -------------- | --------------- | ---------- | -------------- | --------------- |
| dpg                 | local_disagreement | critical_node_contrast        | 5          | 0.3026             | 0.4302     | 0.1112         | 0.7391          | 0.4358     | 0.1946         | 0.6674          |
| dpg                 | local_disagreement | dpg_competitor_exposure_score | 15         | 0.175              | 0.8642     | 0.7851         | 0.9271          | 0.5779     | 0.4831         | 0.6805          |
| dpg                 | local_disagreement | dpg_low_confidence_score      | 15         | 0.175              | 0.9001     | 0.8528         | 0.9431          | 0.6706     | 0.5716         | 0.7704          |
| dpg                 | local_disagreement | dpg_uncertainty_score         | 15         | 0.175              | 0.8689     | 0.7838         | 0.9369          | 0.6138     | 0.5084         | 0.7311          |
| dpg                 | local_disagreement | num_active_nodes              | 15         | 0.175              | 0.4978     | 0.4087         | 0.5808          | 0.2147     | 0.1409         | 0.3182          |
| dpg                 | local_disagreement | recombination_rate            | 15         | 0.175              | 0.5676     | 0.4752         | 0.6482          | 0.2846     | 0.1766         | 0.4133          |
| dpg_execution_trace | local_disagreement | dpg_competitor_exposure_score | 13         | 0.1617             | 0.8832     | 0.8229         | 0.9361          | 0.5729     | 0.4623         | 0.7076          |
| dpg_execution_trace | local_disagreement | dpg_low_confidence_score      | 13         | 0.1617             | 0.9046     | 0.8416         | 0.9538          | 0.6509     | 0.5424         | 0.7562          |
| dpg_execution_trace | local_disagreement | dpg_uncertainty_score         | 13         | 0.1617             | 0.8834     | 0.8314         | 0.936           | 0.5667     | 0.4609         | 0.69            |
| dpg_execution_trace | local_disagreement | num_active_nodes              | 13         | 0.1617             | 0.5491     | 0.4709         | 0.6361          | 0.2138     | 0.1167         | 0.3403          |
| dpg_execution_trace | local_disagreement | recombination_rate            | 13         | 0.1617             | 0.5        | 0.5            | 0.5             | 0.1617     | 0.08313        | 0.2572          |
| dpg                 | model_error        | critical_node_contrast        | 5          | 0.2098             | 0.5675     | 0.4609         | 0.6483          | 0.3252     | 0.1845         | 0.4595          |
| dpg                 | model_error        | dpg_competitor_exposure_score | 14         | 0.1533             | 0.7243     | 0.6575         | 0.7915          | 0.3008     | 0.2215         | 0.3679          |
| dpg                 | model_error        | dpg_low_confidence_score      | 14         | 0.1533             | 0.6796     | 0.6019         | 0.7575          | 0.2938     | 0.2074         | 0.3923          |
| dpg                 | model_error        | dpg_uncertainty_score         | 14         | 0.1533             | 0.7097     | 0.6456         | 0.779           | 0.283      | 0.2151         | 0.3449          |
| dpg                 | model_error        | num_active_nodes              | 14         | 0.1533             | 0.6063     | 0.5283         | 0.6896          | 0.2434     | 0.172          | 0.3175          |
| dpg                 | model_error        | recombination_rate            | 14         | 0.1533             | 0.5386     | 0.4439         | 0.6257          | 0.2031     | 0.155          | 0.2548          |
| dpg_execution_trace | model_error        | dpg_competitor_exposure_score | 14         | 0.1546             | 0.7471     | 0.6907         | 0.8102          | 0.3255     | 0.2411         | 0.4122          |
| dpg_execution_trace | model_error        | dpg_low_confidence_score      | 14         | 0.1546             | 0.7523     | 0.7023         | 0.8024          | 0.3216     | 0.2476         | 0.4039          |
| dpg_execution_trace | model_error        | dpg_uncertainty_score         | 14         | 0.1546             | 0.7367     | 0.6784         | 0.7962          | 0.315      | 0.2371         | 0.3982          |
| dpg_execution_trace | model_error        | num_active_nodes              | 14         | 0.1546             | 0.6454     | 0.5727         | 0.7235          | 0.239      | 0.169          | 0.3011          |
| dpg_execution_trace | model_error        | recombination_rate            | 14         | 0.1546             | 0.5        | 0.5            | 0.5             | 0.1546     | 0.1057         | 0.212           |

## Isolet And Madelon Failure Focus

| dataset | method                | n_model_classes | n_features | model_accuracy | local_matches_model_rate | local_accuracy | avg_explanation_confidence | avg_support_margin | avg_competitor_exposure | avg_num_active_nodes | avg_num_paths | avg_runtime_ms |
| ------- | --------------------- | --------------- | ---------- | -------------- | ------------------------ | -------------- | -------------------------- | ------------------ | ----------------------- | -------------------- | ------------- | -------------- |
| isolet  | anchors               |                 |            | 0.7455         | 1                        | 0.7455         |                            |                    |                         |                      |               | 3.933          |
| isolet  | dpg                   | 26              | 617        | 0.7288         | 0.3051                   | 0.2821         | 0.3768                     | 0.09292            | 0.6768                  | 83.72                | 20.09         |                |
| isolet  | dpg_execution_trace   | 26              | 617        | 0.7288         | 0.2776                   | 0.2603         | 0.3921                     | 0.08573            | 0.686                   | 86.73                | 20            |                |
| isolet  | lime                  |                 |            | 0.7455         | 0.2058                   | 0.1987         |                            |                    |                         |                      |               | 812.5          |
| isolet  | lore                  |                 |            | 0.7455         | 0.9167                   | 0.7417         |                            |                    |                         | 3.606                | 1             | 209.8          |
| isolet  | path_bag              |                 |            | 0.9372         | 1                        | 0.9372         |                            |                    |                         | 1132                 | 100           | 10.15          |
| isolet  | random_same_size_path |                 |            | 0.9372         | 1                        | 0.9372         |                            |                    |                         | 1078                 | 100           | 361.4          |
| isolet  | raw_path_union        |                 |            | 0.9372         | 1                        | 0.9372         |                            |                    |                         | 1078                 | 100           | 15.49          |
| isolet  | shap                  |                 |            | 0.7455         | 1                        | 0.7455         |                            |                    |                         |                      |               | 20.72          |
| isolet  | tree_path             |                 |            | 0.7455         | 1                        | 0.7455         |                            |                    |                         | 3.998                |               | 3.929          |
| madelon | anchors               |                 |            | 0.6519         | 1                        | 0.6519         |                            |                    |                         |                      |               | 4.508          |
| madelon | dpg                   | 2               | 500        | 0.6423         | 0.7712                   | 0.6442         | 0.4622                     | 0.2962             | 0.3519                  | 79.9                 | 21.08         |                |
| madelon | dpg_execution_trace   | 2               | 500        | 0.6423         | 0.7212                   | 0.625          | 0.4931                     | 0.323              | 0.3385                  | 81.04                | 20            |                |
| madelon | lime                  |                 |            | 0.6519         | 0.5                      | 0.4788         |                            |                    |                         |                      |               | 91.25          |
| madelon | lore                  |                 |            | 0.6519         | 0.9596                   | 0.6577         |                            |                    |                         | 3.602                | 1             | 147.4          |
| madelon | path_bag              |                 |            | 0.6923         | 1                        | 0.6923         |                            |                    |                         | 1191                 | 100           | 16.23          |
| madelon | random_same_size_path |                 |            | 0.6923         | 1                        | 0.6923         |                            |                    |                         | 1133                 | 100           | 149.6          |
| madelon | raw_path_union        |                 |            | 0.6923         | 1                        | 0.6923         |                            |                    |                         | 1133                 | 100           | 10.87          |
| madelon | shap                  |                 |            | 0.6519         | 1                        | 0.6519         |                            |                    |                         |                      |               | 3.647          |
| madelon | tree_path             |                 |            | 0.6519         | 1                        | 0.6519         |                            |                    |                         | 3.996                |               | 4.037          |

## Reviewer-Facing Notes

- ICE is documented as a response-profile diagnostic, not a main route/path explainer.
- Structural trace recovery should be presented as a construction check; diagnostic AUROC/AUPRC rows are the stronger value evidence.
- Path-control baselines directly address the basic path-union critique; interpret their output-fidelity rows as controls, not as semantic diagnostics.
- Critical-node intervention with random controls remains the next reviewer-critical experiment.
