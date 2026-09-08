# Journal Final-Test Report

Generated from the validation-selected, locked final-test outputs.

## Main Method Summary

| method              | group                  | datasets | local_matches_model_rate | local_accuracy       | avg_explanation_confidence | avg_runtime_ms           |
| ------------------- | ---------------------- | -------- | ------------------------ | -------------------- | -------------------------- | ------------------------ |
| lime                | feature/output-aligned | 15       | 0.425 [0.329, 0.511]     | 0.421 [0.338, 0.505] | n/a                        | 86.080 [26.298, 196.602] |
| shap                | feature/output-aligned | 15       | 1.000 [1.000, 1.000]     | 0.858 [0.798, 0.906] | n/a                        | 4.917 [3.232, 7.871]     |
| tree_path           | feature/output-aligned | 15       | 1.000 [1.000, 1.000]     | 0.858 [0.807, 0.908] | n/a                        | 3.427 [3.077, 3.746]     |
| anchors             | rule-surrogate         | 15       | 1.000 [1.000, 1.000]     | 0.858 [0.802, 0.912] | n/a                        | 4.288 [3.950, 4.591]     |
| lore                | rule-surrogate         | 15       | 0.971 [0.958, 0.983]     | 0.852 [0.801, 0.906] | n/a                        | 34.992 [11.840, 70.087]  |
| dpg                 | structure-aware        | 15       | 0.825 [0.725, 0.905]     | 0.752 [0.671, 0.831] | 0.477 [0.384, 0.559]       | n/a                      |
| dpg_execution_trace | structure-aware        | 15       | 0.860 [0.767, 0.931]     | 0.783 [0.692, 0.865] | 0.654 [0.590, 0.705]       | n/a                      |

## Size And Runtime

| method              | group                  | native_size             | runtime_ms               |
| ------------------- | ---------------------- | ----------------------- | ------------------------ |
| lime                | feature/output-aligned | 15.394 [11.889, 18.689] | 86.080 [25.385, 191.139] |
| shap                | feature/output-aligned | 35.343 [18.554, 57.431] | 4.917 [3.264, 7.746]     |
| tree_path           | feature/output-aligned | 22.025 [13.433, 30.381] | 3.427 [3.066, 3.790]     |
| lore                | rule-surrogate         | 3.014 [2.762, 3.280]    | 34.992 [11.587, 63.083]  |
| dpg                 | structure-aware        | 58.254 [47.863, 67.226] | n/a                      |
| dpg_execution_trace | structure-aware        | 63.962 [57.254, 70.224] | n/a                      |

## Corrected Paired Tests

| metric                     | reference_method    | comparison_method | n_datasets | mean_difference_ref_minus_comparison | p_value   | p_holm    | rank_biserial_ref_minus_comparison |
| -------------------------- | ------------------- | ----------------- | ---------- | ------------------------------------ | --------- | --------- | ---------------------------------- |
| local_matches_model_rate   | dpg_execution_trace | anchors           | 15         | -0.1402                              | 0.001469  | 0.007344  | -1                                 |
| local_matches_model_rate   | dpg_execution_trace | dpg               | 15         | 0.03482                              | 0.06404   | 0.06404   | 0.5619                             |
| local_matches_model_rate   | dpg_execution_trace | lime              | 15         | 0.4349                               | 6.104e-05 | 0.0003662 | 1                                  |
| local_matches_model_rate   | dpg_execution_trace | lore              | 15         | -0.1109                              | 0.002218  | 0.007344  | -1                                 |
| local_matches_model_rate   | dpg_execution_trace | shap              | 15         | -0.1402                              | 0.001469  | 0.007344  | -1                                 |
| local_matches_model_rate   | dpg_execution_trace | tree_path         | 15         | -0.1402                              | 0.001469  | 0.007344  | -1                                 |
| local_accuracy             | dpg_execution_trace | anchors           | 15         | -0.07434                             | 0.003324  | 0.01662   | -0.9231                            |
| local_accuracy             | dpg_execution_trace | dpg               | 15         | 0.03124                              | 0.1094    | 0.1094    | 0.4857                             |
| local_accuracy             | dpg_execution_trace | lime              | 15         | 0.3627                               | 6.104e-05 | 0.0003662 | 1                                  |
| local_accuracy             | dpg_execution_trace | lore              | 15         | -0.06885                             | 0.007649  | 0.01662   | -0.8718                            |
| local_accuracy             | dpg_execution_trace | shap              | 15         | -0.07426                             | 0.004154  | 0.01662   | -0.9011                            |
| local_accuracy             | dpg_execution_trace | tree_path         | 15         | -0.07426                             | 0.004154  | 0.01662   | -0.9011                            |
| avg_explanation_confidence | dpg_execution_trace | dpg               | 15         | 0.1766                               | 6.104e-05 | 6.104e-05 | 1                                  |
| avg_support_margin         | dpg_execution_trace | dpg               | 15         | 0.07252                              | 0.04126   | 0.04126   | 0.6                                |
| avg_competitor_exposure    | dpg_execution_trace | dpg               | 15         | -0.03725                             | 0.05536   | 0.05536   | -0.5667                            |
| avg_path_purity            | dpg_execution_trace | dpg               | 15         | 0.03725                              | 0.05536   | 0.05536   | 0.5667                             |
| avg_recombination_rate     | dpg_execution_trace | dpg               | 15         | -0.1339                              | 6.104e-05 | 6.104e-05 | -1                                 |

## Diagnostic Value

| method              | target             | score                         | n_datasets | mean_positive_rate | auroc_mean | auroc_ci95_low | auroc_ci95_high | auprc_mean | auprc_ci95_low | auprc_ci95_high |
| ------------------- | ------------------ | ----------------------------- | ---------- | ------------------ | ---------- | -------------- | --------------- | ---------- | -------------- | --------------- |
| dpg                 | local_disagreement | critical_node_contrast        | 5          | 0.3026             | 0.4302     | 0.1214         | 0.7323          | 0.4358     | 0.2148         | 0.6568          |
| dpg                 | local_disagreement | dpg_competitor_exposure_score | 15         | 0.175              | 0.8642     | 0.7762         | 0.9265          | 0.5779     | 0.4711         | 0.6823          |
| dpg                 | local_disagreement | dpg_low_confidence_score      | 15         | 0.175              | 0.9001     | 0.8508         | 0.9434          | 0.6706     | 0.5724         | 0.7638          |
| dpg                 | local_disagreement | dpg_uncertainty_score         | 15         | 0.175              | 0.8689     | 0.7876         | 0.9441          | 0.6138     | 0.4974         | 0.7077          |
| dpg                 | local_disagreement | num_active_nodes              | 15         | 0.175              | 0.4978     | 0.4046         | 0.5736          | 0.2147     | 0.1357         | 0.3119          |
| dpg                 | local_disagreement | recombination_rate            | 15         | 0.175              | 0.5676     | 0.4767         | 0.6511          | 0.2846     | 0.1663         | 0.4117          |
| dpg_execution_trace | local_disagreement | dpg_competitor_exposure_score | 13         | 0.1617             | 0.8832     | 0.8238         | 0.9334          | 0.5729     | 0.4578         | 0.7008          |
| dpg_execution_trace | local_disagreement | dpg_low_confidence_score      | 13         | 0.1617             | 0.9046     | 0.8355         | 0.9542          | 0.6509     | 0.5513         | 0.7677          |
| dpg_execution_trace | local_disagreement | dpg_uncertainty_score         | 13         | 0.1617             | 0.8834     | 0.8274         | 0.9353          | 0.5667     | 0.4574         | 0.7049          |
| dpg_execution_trace | local_disagreement | num_active_nodes              | 13         | 0.1617             | 0.5491     | 0.47           | 0.6291          | 0.2138     | 0.1071         | 0.3272          |
| dpg_execution_trace | local_disagreement | recombination_rate            | 13         | 0.1617             | 0.5        | 0.5            | 0.5             | 0.1617     | 0.08402        | 0.2788          |
| dpg                 | model_error        | critical_node_contrast        | 5          | 0.2098             | 0.5675     | 0.4724         | 0.6509          | 0.3252     | 0.1845         | 0.454           |
| dpg                 | model_error        | dpg_competitor_exposure_score | 14         | 0.1533             | 0.7243     | 0.6625         | 0.787           | 0.3008     | 0.2297         | 0.3682          |
| dpg                 | model_error        | dpg_low_confidence_score      | 14         | 0.1533             | 0.6796     | 0.5938         | 0.7647          | 0.2938     | 0.1991         | 0.3836          |
| dpg                 | model_error        | dpg_uncertainty_score         | 14         | 0.1533             | 0.7097     | 0.6562         | 0.7843          | 0.283      | 0.2171         | 0.3461          |
| dpg                 | model_error        | num_active_nodes              | 14         | 0.1533             | 0.6063     | 0.5281         | 0.6875          | 0.2434     | 0.1724         | 0.3168          |
| dpg                 | model_error        | recombination_rate            | 14         | 0.1533             | 0.5386     | 0.4535         | 0.6325          | 0.2031     | 0.1522         | 0.2517          |
| dpg_execution_trace | model_error        | dpg_competitor_exposure_score | 14         | 0.1546             | 0.7471     | 0.6842         | 0.8093          | 0.3255     | 0.2355         | 0.4264          |
| dpg_execution_trace | model_error        | dpg_low_confidence_score      | 14         | 0.1546             | 0.7523     | 0.7035         | 0.801           | 0.3216     | 0.2432         | 0.4015          |
| dpg_execution_trace | model_error        | dpg_uncertainty_score         | 14         | 0.1546             | 0.7367     | 0.677          | 0.8             | 0.315      | 0.2313         | 0.3987          |
| dpg_execution_trace | model_error        | num_active_nodes              | 14         | 0.1546             | 0.6454     | 0.5799         | 0.7144          | 0.239      | 0.1697         | 0.3076          |
| dpg_execution_trace | model_error        | recombination_rate            | 14         | 0.1546             | 0.5        | 0.5            | 0.5             | 0.1546     | 0.09879        | 0.212           |

## Isolet And Madelon Failure Focus

| dataset | method              | n_model_classes | n_features | model_accuracy | local_matches_model_rate | local_accuracy | avg_explanation_confidence | avg_support_margin | avg_competitor_exposure | avg_num_active_nodes | avg_num_paths | avg_runtime_ms |
| ------- | ------------------- | --------------- | ---------- | -------------- | ------------------------ | -------------- | -------------------------- | ------------------ | ----------------------- | -------------------- | ------------- | -------------- |
| isolet  | anchors             |                 |            | 0.7455         | 1                        | 0.7455         |                            |                    |                         |                      |               | 3.933          |
| isolet  | dpg                 | 26              | 617        | 0.7288         | 0.3051                   | 0.2821         | 0.3768                     | 0.09292            | 0.6768                  | 83.72                | 20.09         |                |
| isolet  | dpg_execution_trace | 26              | 617        | 0.7288         | 0.2776                   | 0.2603         | 0.3921                     | 0.08573            | 0.686                   | 86.73                | 20            |                |
| isolet  | lime                |                 |            | 0.7455         | 0.2058                   | 0.1987         |                            |                    |                         |                      |               | 812.5          |
| isolet  | lore                |                 |            | 0.7455         | 0.9167                   | 0.7417         |                            |                    |                         | 3.606                | 1             | 209.8          |
| isolet  | shap                |                 |            | 0.7455         | 1                        | 0.7455         |                            |                    |                         |                      |               | 20.72          |
| isolet  | tree_path           |                 |            | 0.7455         | 1                        | 0.7455         |                            |                    |                         | 3.998                |               | 3.929          |
| madelon | anchors             |                 |            | 0.6519         | 1                        | 0.6519         |                            |                    |                         |                      |               | 4.508          |
| madelon | dpg                 | 2               | 500        | 0.6423         | 0.7712                   | 0.6442         | 0.4622                     | 0.2962             | 0.3519                  | 79.9                 | 21.08         |                |
| madelon | dpg_execution_trace | 2               | 500        | 0.6423         | 0.7212                   | 0.625          | 0.4931                     | 0.323              | 0.3385                  | 81.04                | 20            |                |
| madelon | lime                |                 |            | 0.6519         | 0.5                      | 0.4788         |                            |                    |                         |                      |               | 91.25          |
| madelon | lore                |                 |            | 0.6519         | 0.9596                   | 0.6577         |                            |                    |                         | 3.602                | 1             | 147.4          |
| madelon | shap                |                 |            | 0.6519         | 1                        | 0.6519         |                            |                    |                         |                      |               | 3.647          |
| madelon | tree_path           |                 |            | 0.6519         | 1                        | 0.6519         |                            |                    |                         | 3.996                |               | 4.037          |

## Reviewer-Facing Notes

- ICE is documented as a response-profile diagnostic, not a main route/path explainer.
- Structural trace recovery should be presented as a construction check; diagnostic AUROC/AUPRC rows are the stronger value evidence.
- Path-control baselines are still recommended as the next implementation step.
- Critical-node intervention with random controls remains the next reviewer-critical experiment.
