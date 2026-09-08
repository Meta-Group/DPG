# Readability And Cost Analysis

Source: `experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/per_sample_selected_test.csv`

This analysis measures the size and runtime cost of explanation objects in the locked final-test output. The `display_size` proxy is the number of graph nodes for DPG, the number of unique predicates for path controls, nonzero feature contributions for attribution methods, and rule length for rule/surrogate methods when available.

## Summary By Method

| method                | method_group         | n_samples | mean_display_size | p90_display_size | mean_runtime_ms | p90_runtime_ms | mean_local_match | mean_local_accuracy | mean_confidence | mean_support_margin | mean_competitor_exposure | mean_signal_per_100_nodes |
| --------------------- | -------------------- | --------- | ----------------- | ---------------- | --------------- | -------------- | ---------------- | ------------------- | --------------- | ------------------- | ------------------------ | ------------------------- |
| lore                  | local-surrogate-rule | 6079      | 3.155             | 4                | 75.76           | 203.8          | 0.9557           | 0.8097              |                 |                     |                          |                           |
| tree_path             | path-feature         | 6079      | 3.901             | 4                | 3.836           | 4.14           | 1                | 0.8113              |                 |                     |                          |                           |
| lime                  | feature-attribution  | 6079      | 16.05             | 20               | 239.3           | 768.4          | 0.4366           | 0.4256              |                 |                     |                          |                           |
| shap                  | feature-attribution  | 6079      | 54.96             | 127              | 8.172           | 21.04          | 1                | 0.8113              |                 |                     |                          |                           |
| dpg                   | DPG                  | 6079      | 68.58             | 84               |                 |                | 0.7108           | 0.6358              | 0.4671          | 0.4921              | 0.3226                   | 1.184                     |
| dpg_execution_trace   | DPG                  | 6079      | 72.5              | 88               |                 |                | 0.7248           | 0.646               | 0.5854          | 0.5314              | 0.3057                   | 0.9182                    |
| random_same_size_path | path-control         | 6079      | 666.4             | 1194             | 176.7           | 300.5          | 1                | 0.9051              |                 |                     |                          |                           |
| path_bag              | path-control         | 6079      | 666.4             | 1194             | 9.833           | 15.79          | 1                | 0.9051              |                 |                     |                          |                           |
| raw_path_union        | path-control         | 6079      | 666.4             | 1194             | 11.05           | 16.76          | 1                | 0.9051              |                 |                     |                          |                           |
| anchors               | rule                 | 6079      |                   |                  | 4.351           | 5.312          | 1                | 0.8118              |                 |                     |                          |                           |

## Summary By Explanation Family

| method_group         | n_samples | mean_display_size | median_display_size | p90_display_size | mean_active_edges | mean_num_paths | mean_runtime_ms | p90_runtime_ms | mean_local_match | mean_local_accuracy | mean_confidence | mean_support_margin | mean_competitor_exposure | mean_signal_per_100_nodes |
| -------------------- | --------- | ----------------- | ------------------- | ---------------- | ----------------- | -------------- | --------------- | -------------- | ---------------- | ------------------- | --------------- | ------------------- | ------------------------ | ------------------------- |
| local-surrogate-rule | 6079      | 3.155             | 4                   | 4                |                   | 1              | 75.76           | 203.8          | 0.9557           | 0.8097              |                 |                     |                          |                           |
| path-feature         | 6079      | 3.901             | 4                   | 4                |                   |                | 3.836           | 4.14           | 1                | 0.8113              |                 |                     |                          |                           |
| feature-attribution  | 12158     | 35.51             | 20                  | 112              |                   |                | 123.8           | 746.9          | 0.7183           | 0.6184              |                 |                     |                          |                           |
| DPG                  | 12158     | 70.54             | 74                  | 86               | 165.1             | 31.32          |                 |                | 0.7178           | 0.6409              | 0.5263          | 0.5117              | 0.3142                   | 1.051                     |
| path-control         | 18237     | 666.4             | 620                 | 1194             |                   | 81.94          | 65.86           | 287.8          | 1                | 0.9051              |                 |                     |                          |                           |
| rule                 | 6079      |                   |                     |                  |                   |                | 4.351           | 5.312          | 1                | 0.8118              |                 |                     |                          |                           |

## DPG-Local Versus Comparators

| base_method         | comparator            | display_size_ratio_comparator_over_dpg | runtime_ratio_comparator_over_dpg | local_match_delta_dpg_minus_comparator | local_accuracy_delta_dpg_minus_comparator |
| ------------------- | --------------------- | -------------------------------------- | --------------------------------- | -------------------------------------- | ----------------------------------------- |
| dpg_execution_trace | raw_path_union        | 9.192                                  |                                   | -0.2752                                | -0.2591                                   |
| dpg_execution_trace | path_bag              | 9.192                                  |                                   | -0.2752                                | -0.2591                                   |
| dpg_execution_trace | random_same_size_path | 9.192                                  |                                   | -0.2752                                | -0.2591                                   |
| dpg_execution_trace | tree_path             | 0.05381                                |                                   | -0.2752                                | -0.1653                                   |
| dpg_execution_trace | shap                  | 0.7581                                 |                                   | -0.2752                                | -0.1653                                   |
| dpg_execution_trace | lime                  | 0.2213                                 |                                   | 0.2882                                 | 0.2204                                    |
| dpg_execution_trace | anchors               |                                        |                                   | -0.2752                                | -0.1658                                   |
| dpg_execution_trace | lore                  | 0.04352                                |                                   | -0.231                                 | -0.1637                                   |

## Highest-Cost DPG Datasets

| dataset                 | n_samples | mean_display_size | p90_display_size | mean_active_edges | mean_runtime_ms | p90_runtime_ms | mean_local_match | mean_confidence | mean_competitor_exposure |
| ----------------------- | --------- | ----------------- | ---------------- | ----------------- | --------------- | -------------- | ---------------- | --------------- | ------------------------ |
| isolet                  | 1560      | 86.73             | 90               | 79.94             |                 |                | 0.2776           | 0.3921          | 0.686                    |
| madelon                 | 520       | 81.04             | 82               | 79.88             |                 |                | 0.7212           | 0.4931          | 0.3385                   |
| phoneme                 | 1081      | 73.9              | 75               | 79.76             |                 |                | 0.9399           | 0.677           | 0.1047                   |
| breast_cancer           | 114       | 71.07             | 75               | 75                |                 |                | 0.9825           | 0.7266          | 0.03528                  |
| wdbc                    | 114       | 70.87             | 76.7             | 71.63             |                 |                | 0.9825           | 0.7575          | 0.03061                  |
| spambase                | 921       | 69.78             | 72               | 79.07             |                 |                | 0.9012           | 0.7142          | 0.09477                  |
| qsar-biodeg             | 211       | 66.09             | 70               | 78.04             |                 |                | 0.9005           | 0.6679          | 0.1447                   |
| diabetes                | 154       | 63.85             | 67               | 78.36             |                 |                | 0.9156           | 0.6936          | 0.1232                   |
| ionosphere              | 71        | 61.66             | 67               | 71.08             |                 |                | 0.8873           | 0.7755          | 0.06424                  |
| wine                    | 36        | 61.06             | 66.5             | 63.19             |                 |                | 1                | 0.7096          | 0.07798                  |
| banknote-authentication | 275       | 60.19             | 66               | 72.33             |                 |                | 0.9564           | 0.7882          | 0.05069                  |
| digits                  | 360       | 57.96             | 63               | 78.89             |                 |                | 0.7194           | 0.4954          | 0.468                    |
| vehicle                 | 170       | 57.1              | 63               | 76.31             |                 |                | 0.8412           | 0.5533          | 0.2602                   |
| segment                 | 462       | 53.5              | 66               | 65.75             |                 |                | 0.8723           | 0.6333          | 0.2589                   |
| iris                    | 30        | 24.63             | 32.2             | 41.27             |                 |                | 1                | 0.7276          | 0.03786                  |

## Writing Guidance

- Use this as a readability/cost supplement, not as a new fidelity benchmark.
- Path controls can be output-faithful but often require substantially larger predicate sets, supporting the claim that DPG-local adds a structured diagnostic view rather than merely dumping all paths.
- Tree-path summaries are compact, but they discard predicate-transition topology and predicted-versus-competitor branch structure.
- High-cost datasets should be discussed as readability/scalability boundaries and motivate top-k, support-threshold, or sampled graph views.
- Do not claim a universal readability advantage: DPG-local is a graph object and needs focused views for large or high-dimensional cases.
