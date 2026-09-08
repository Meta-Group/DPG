# Local Stability Analysis

Source: `/home/sbarbonjr/projects/DPG/experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/per_sample_selected_test.csv`

For each dataset and method, this analysis compares every final-test sample with its 5 nearest test neighbours in standardized input space.
Stability is measured by explained-label agreement, top-feature agreement when available, and absolute changes in DPG or shared diagnostic scores.

## Method Summary

| method_key | datasets | pairs | mean_input_distance | explained_label_agreement | model_label_agreement | top_feature_agreement | mean_abs_delta_explanation_confidence | median_abs_delta_explanation_confidence | mean_abs_delta_support_margin | median_abs_delta_support_margin | mean_abs_delta_competitor_exposure | median_abs_delta_competitor_exposure | mean_abs_delta_model_vote_agreement | median_abs_delta_model_vote_agreement | mean_abs_delta_path_purity | median_abs_delta_path_purity | mean_abs_delta_score_margin_pred_vs_competitor | median_abs_delta_score_margin_pred_vs_competitor | mean_abs_delta_num_active_nodes | median_abs_delta_num_active_nodes | mean_abs_delta_runtime_ms | median_abs_delta_runtime_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchors | 15 | 30395 | 9.316 | 0.7912 | 0.7912 |  |  |  |  |  |  |  |  |  |  |  | 0 | 0 |  |  | 0.4297 | 0.1674 |
| dpg:aggregated_transitions | 15 | 30395 | 9.316 | 0.7855 | 0.7939 |  | 0.04929 | 0.02239 | 0.1233 | 0.04515 | 0.07106 | 0.02851 | 0.1028 | 0.05 | 0.07106 | 0.02851 |  |  | 3.058 | 1 |  |  |
| dpg_execution_trace:execution_trace | 15 | 30395 | 9.316 | 0.7853 | 0.7948 |  | 0.05627 | 0.02413 | 0.1273 | 0.04357 | 0.07239 | 0.02939 | 0.1039 | 0.05 | 0.07239 | 0.02939 |  |  | 1.382 | 1 |  |  |
| lime | 15 | 30395 | 9.316 | 0.7293 | 0.7893 | 0.7578 |  |  |  |  |  |  |  |  |  |  | 0.0324 | 0.01863 |  |  | 32.07 | 4.51 |
| lore | 15 | 30395 | 9.316 | 0.7858 | 0.79 | 0.4211 |  |  |  |  |  |  |  |  |  |  | 0.1096 | 0.0131 | 0.5278 | 0 | 18.13 | 1.875 |
| path_bag | 15 | 30395 | 9.316 | 0.796 | 0.796 | 0.6418 |  |  |  |  |  |  |  |  |  |  | 0.1843 | 0.1176 | 62.23 | 34 | 1.451 | 0.4348 |
| random_same_size_path | 15 | 30395 | 9.316 | 0.796 | 0.796 | 0.312 |  |  |  |  |  |  |  |  |  |  | 0.1843 | 0.1177 | 51 | 28 | 47.08 | 4.793 |
| raw_path_union | 15 | 30395 | 9.316 | 0.796 | 0.796 | 0.5638 |  |  |  |  |  |  |  |  |  |  | 0.1843 | 0.1176 | 51 | 28 | 1.094 | 0.4466 |
| shap | 15 | 30395 | 9.316 | 0.7905 | 0.7905 | 0.6069 |  |  |  |  |  |  |  |  |  |  | 0.1047 | 0.04854 |  |  | 0.6124 | 0.2092 |
| tree_path | 15 | 30395 | 9.316 | 0.7905 | 0.7905 | 0.6022 |  |  |  |  |  |  |  |  |  |  | 0.1047 | 0.04854 | 0.02122 | 0 | 0.1429 | 0.08643 |

## Input-Distance Correlations

Positive correlations mean that explanations change more as inputs become farther apart.

| method_key | metric | pairs | spearman_input_distance_vs_abs_delta |
| --- | --- | --- | --- |
| anchors | explanation_confidence | 0 |  |
| anchors | support_margin | 0 |  |
| anchors | competitor_exposure | 0 |  |
| anchors | model_vote_agreement | 0 |  |
| anchors | path_purity | 0 |  |
| anchors | score_margin_pred_vs_competitor | 30395 |  |
| anchors | num_active_nodes | 0 |  |
| anchors | runtime_ms | 30395 | 0.1056 |
| dpg:aggregated_transitions | explanation_confidence | 30395 | 0.2284 |
| dpg:aggregated_transitions | support_margin | 30395 | 0.292 |
| dpg:aggregated_transitions | competitor_exposure | 30395 | 0.3347 |
| dpg:aggregated_transitions | model_vote_agreement | 30395 | 0.2962 |
| dpg:aggregated_transitions | path_purity | 30395 | 0.3347 |
| dpg:aggregated_transitions | score_margin_pred_vs_competitor | 0 |  |
| dpg:aggregated_transitions | num_active_nodes | 30395 | 0.2727 |
| dpg:aggregated_transitions | runtime_ms | 0 |  |
| dpg_execution_trace:execution_trace | explanation_confidence | 30395 | 0.234 |
| dpg_execution_trace:execution_trace | support_margin | 30395 | 0.2906 |
| dpg_execution_trace:execution_trace | competitor_exposure | 30395 | 0.3258 |
| dpg_execution_trace:execution_trace | model_vote_agreement | 30395 | 0.2493 |
| dpg_execution_trace:execution_trace | path_purity | 30395 | 0.3258 |
| dpg_execution_trace:execution_trace | score_margin_pred_vs_competitor | 0 |  |
| dpg_execution_trace:execution_trace | num_active_nodes | 30395 | 0.2559 |
| dpg_execution_trace:execution_trace | runtime_ms | 0 |  |
| lime | explanation_confidence | 0 |  |
| lime | support_margin | 0 |  |
| lime | competitor_exposure | 0 |  |
| lime | model_vote_agreement | 0 |  |
| lime | path_purity | 0 |  |
| lime | score_margin_pred_vs_competitor | 30395 | -0.2789 |
| lime | num_active_nodes | 0 |  |
| lime | runtime_ms | 30395 | 0.3338 |
| lore | explanation_confidence | 0 |  |
| lore | support_margin | 0 |  |
| lore | competitor_exposure | 0 |  |
| lore | model_vote_agreement | 0 |  |
| lore | path_purity | 0 |  |
| lore | score_margin_pred_vs_competitor | 28887 | 0.3891 |
| lore | num_active_nodes | 30395 | -0.003256 |
| lore | runtime_ms | 30395 | 0.6741 |
| path_bag | explanation_confidence | 0 |  |
| path_bag | support_margin | 0 |  |
| path_bag | competitor_exposure | 0 |  |
| path_bag | model_vote_agreement | 0 |  |
| path_bag | path_purity | 0 |  |
| path_bag | score_margin_pred_vs_competitor | 30395 | 0.2598 |
| path_bag | num_active_nodes | 30395 | 0.2734 |
| path_bag | runtime_ms | 30395 | 0.02739 |
| random_same_size_path | explanation_confidence | 0 |  |
| random_same_size_path | support_margin | 0 |  |
| random_same_size_path | competitor_exposure | 0 |  |
| random_same_size_path | model_vote_agreement | 0 |  |
| random_same_size_path | path_purity | 0 |  |
| random_same_size_path | score_margin_pred_vs_competitor | 30395 | 0.2602 |
| random_same_size_path | num_active_nodes | 30395 | 0.3209 |
| random_same_size_path | runtime_ms | 30395 | 0.2347 |
| raw_path_union | explanation_confidence | 0 |  |
| raw_path_union | support_margin | 0 |  |
| raw_path_union | competitor_exposure | 0 |  |
| raw_path_union | model_vote_agreement | 0 |  |
| raw_path_union | path_purity | 0 |  |
| raw_path_union | score_margin_pred_vs_competitor | 30395 | 0.2598 |
| raw_path_union | num_active_nodes | 30395 | 0.3209 |
| raw_path_union | runtime_ms | 30395 | 0.2686 |
| shap | explanation_confidence | 0 |  |
| shap | support_margin | 0 |  |
| shap | competitor_exposure | 0 |  |
| shap | model_vote_agreement | 0 |  |
| shap | path_purity | 0 |  |
| shap | score_margin_pred_vs_competitor | 30395 | 0.159 |
| shap | num_active_nodes | 0 |  |
| shap | runtime_ms | 30395 | 0.3114 |
| tree_path | explanation_confidence | 0 |  |
| tree_path | support_margin | 0 |  |
| tree_path | competitor_exposure | 0 |  |
| tree_path | model_vote_agreement | 0 |  |
| tree_path | path_purity | 0 |  |
| tree_path | score_margin_pred_vs_competitor | 30395 | 0.159 |
| tree_path | num_active_nodes | 30395 | -0.05853 |
| tree_path | runtime_ms | 30395 | 0.1101 |

## Dataset Summary

| dataset | method_key | pairs | mean_input_distance | explained_label_agreement | model_label_agreement | top_feature_agreement | mean_abs_delta_explanation_confidence | mean_abs_delta_support_margin | mean_abs_delta_competitor_exposure | mean_abs_delta_model_vote_agreement | mean_abs_delta_path_purity | mean_abs_delta_score_margin_pred_vs_competitor | mean_abs_delta_num_active_nodes | mean_abs_delta_runtime_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| banknote-authentication | anchors | 1375 | 0.4284 | 0.9804 | 0.9804 |  |  |  |  |  |  | 0 |  | 0.566 |
| banknote-authentication | dpg:aggregated_transitions | 1375 | 0.4284 | 0.9556 | 0.9804 |  | 0.04068 | 0.0831 | 0.04155 | 0.05115 | 0.04155 |  | 2.34 |  |
| banknote-authentication | dpg_execution_trace:execution_trace | 1375 | 0.4284 | 0.9615 | 0.9804 |  | 0.03405 | 0.07026 | 0.03513 | 0.05062 | 0.03513 |  | 1.073 |  |
| banknote-authentication | lime | 1375 | 0.4284 | 1 | 0.9745 | 0.9782 |  |  |  |  |  | 0.04611 |  | 10.14 |
| banknote-authentication | lore | 1375 | 0.4284 | 0.9687 | 0.9804 | 0.8305 |  |  |  |  |  | 0.01124 | 0.5716 | 0.4811 |
| banknote-authentication | path_bag | 1375 | 0.4284 | 0.9804 | 0.9804 | 0.9789 |  |  |  |  |  | 0.0552 | 4.056 | 0.1013 |
| banknote-authentication | random_same_size_path | 1375 | 0.4284 | 0.9804 | 0.9804 | 0.376 |  |  |  |  |  | 0.0552 | 3.677 | 0.1818 |
| banknote-authentication | raw_path_union | 1375 | 0.4284 | 0.9804 | 0.9804 | 0.8691 |  |  |  |  |  | 0.0552 | 3.677 | 0.122 |
| banknote-authentication | shap | 1375 | 0.4284 | 0.9804 | 0.9804 | 0.8916 |  |  |  |  |  | 0.07002 |  | 0.1482 |
| banknote-authentication | tree_path | 1375 | 0.4284 | 0.9804 | 0.9804 | 0.8953 |  |  |  |  |  | 0.07002 | 0.05218 | 0.129 |
| breast_cancer | anchors | 570 | 3.545 | 0.9123 | 0.9123 |  |  |  |  |  |  | 0 |  | 0.1649 |
| breast_cancer | dpg:aggregated_transitions | 570 | 3.545 | 0.9228 | 0.9123 |  | 0.03912 | 0.09559 | 0.0478 | 0.08278 | 0.0478 |  | 1.642 |  |
| breast_cancer | dpg_execution_trace:execution_trace | 570 | 3.545 | 0.9351 | 0.9123 |  | 0.03502 | 0.08678 | 0.04339 | 0.07982 | 0.04339 |  | 1.486 |  |
| breast_cancer | lime | 570 | 3.545 | 1 | 0.9123 | 0.9596 |  |  |  |  |  | 0.06385 |  | 1.326 |
| breast_cancer | lore | 570 | 3.545 | 0.907 | 0.9123 | 0.2842 |  |  |  |  |  | 0.02944 | 1.002 | 4.523 |
| breast_cancer | path_bag | 570 | 3.545 | 0.9123 | 0.9123 | 0.8649 |  |  |  |  |  | 0.1207 | 4.277 | 0.1001 |
| breast_cancer | random_same_size_path | 570 | 3.545 | 0.9123 | 0.9123 | 0.1421 |  |  |  |  |  | 0.1231 | 4.139 | 0.5575 |
| breast_cancer | raw_path_union | 570 | 3.545 | 0.9123 | 0.9123 | 0.8719 |  |  |  |  |  | 0.1207 | 4.139 | 0.08341 |
| breast_cancer | shap | 570 | 3.545 | 0.9123 | 0.9123 | 0.6018 |  |  |  |  |  | 0.1438 |  | 0.1736 |
| breast_cancer | tree_path | 570 | 3.545 | 0.9123 | 0.9123 | 0.5105 |  |  |  |  |  | 0.1438 | 0.07763 | 0.0697 |
| diabetes | anchors | 770 | 1.653 | 0.8195 | 0.8195 |  |  |  |  |  |  | 0 |  | 0.4803 |
| diabetes | dpg:aggregated_transitions | 770 | 1.653 | 0.8468 | 0.8195 |  | 0.05585 | 0.1599 | 0.07993 | 0.09395 | 0.07993 |  | 6.594 |  |
| diabetes | dpg_execution_trace:execution_trace | 770 | 1.653 | 0.8468 | 0.8195 |  | 0.07917 | 0.1743 | 0.08715 | 0.111 | 0.08715 |  | 2.088 |  |
| diabetes | lime | 770 | 1.653 | 0.9896 | 0.8221 | 1 |  |  |  |  |  | 0.03775 |  | 0.9678 |
| diabetes | lore | 770 | 1.653 | 0.826 | 0.8221 | 0.739 |  |  |  |  |  | 0.0446 | 1.051 | 1.439 |
| diabetes | path_bag | 770 | 1.653 | 0.8013 | 0.8013 | 0.961 |  |  |  |  |  | 0.2063 | 43.01 | 0.4457 |
| diabetes | random_same_size_path | 770 | 1.653 | 0.8013 | 0.8013 | 1 |  |  |  |  |  | 0.2063 | 24.91 | 1.181 |
| diabetes | raw_path_union | 770 | 1.653 | 0.8013 | 0.8013 | 0.687 |  |  |  |  |  | 0.2063 | 24.91 | 1.671 |
| diabetes | shap | 770 | 1.653 | 0.8195 | 0.8195 | 0.5649 |  |  |  |  |  | 0.1718 |  | 0.1551 |
| diabetes | tree_path | 770 | 1.653 | 0.8195 | 0.8195 | 0.5494 |  |  |  |  |  | 0.1718 | 0.0387 | 0.1417 |
| digits | anchors | 1800 | 5.412 | 0.8056 | 0.8056 |  |  |  |  |  |  | 0 |  | 0.06712 |
| digits | dpg:aggregated_transitions | 1800 | 5.412 | 0.5922 | 0.8056 |  | 0.03564 | 0.2285 | 0.1765 | 0.1882 | 0.1765 |  | 13.41 |  |
| digits | dpg_execution_trace:execution_trace | 1800 | 5.412 | 0.68 | 0.8056 |  | 0.07522 | 0.1824 | 0.1337 | 0.1773 | 0.1337 |  | 2.779 |  |
| digits | lime | 1800 | 5.412 | 0.6933 | 0.7889 | 0.7833 |  |  |  |  |  | 0.01685 |  | 10.92 |
| digits | lore | 1800 | 5.412 | 0.775 | 0.7889 | 0.3678 |  |  |  |  |  | 0.09142 | 0.81 | 3.05 |
| digits | path_bag | 1800 | 5.412 | 0.8578 | 0.8578 | 0.655 |  |  |  |  |  | 0.2042 | 42.85 | 2.699 |
| digits | random_same_size_path | 1800 | 5.412 | 0.8578 | 0.8578 | 0.01889 |  |  |  |  |  | 0.2042 | 21.83 | 8.261 |
| digits | raw_path_union | 1800 | 5.412 | 0.8578 | 0.8578 | 0.4411 |  |  |  |  |  | 0.2042 | 21.83 | 0.4579 |
| digits | shap | 1800 | 5.412 | 0.8056 | 0.8056 | 0.5661 |  |  |  |  |  | 0.1248 |  | 0.474 |
| digits | tree_path | 1800 | 5.412 | 0.8056 | 0.8056 | 0.4367 |  |  |  |  |  | 0.1248 | 0.006167 | 0.1208 |
| ionosphere | anchors | 355 | 4.013 | 0.7577 | 0.7577 |  |  |  |  |  |  | 0 |  | 0.09881 |
| ionosphere | dpg:aggregated_transitions | 355 | 4.013 | 0.9268 | 0.7577 |  | 0.08148 | 0.1965 | 0.09826 | 0.1802 | 0.09826 |  | 5.454 |  |
| ionosphere | dpg_execution_trace:execution_trace | 355 | 4.013 | 0.8169 | 0.7577 |  | 0.08018 | 0.1446 | 0.07232 | 0.1499 | 0.07232 |  | 4.099 |  |
| ionosphere | lime | 355 | 4.013 | 1 | 0.7577 | 0.5268 |  |  |  |  |  | 0.08946 |  | 0.571 |
| ionosphere | lore | 355 | 4.013 | 0.7465 | 0.7465 | 0.2648 |  |  |  |  |  | 0.01877 | 1.141 | 2.131 |
| ionosphere | path_bag | 355 | 4.013 | 0.7437 | 0.7437 | 0.7944 |  |  |  |  |  | 0.1828 | 21.5 | 0.1541 |
| ionosphere | random_same_size_path | 355 | 4.013 | 0.7437 | 0.7437 | 0.1155 |  |  |  |  |  | 0.1828 | 18.2 | 0.6719 |
| ionosphere | raw_path_union | 355 | 4.013 | 0.7437 | 0.7437 | 0.5577 |  |  |  |  |  | 0.1828 | 18.2 | 0.3265 |
| ionosphere | shap | 355 | 4.013 | 0.7577 | 0.7577 | 0.5887 |  |  |  |  |  | 0.1665 |  | 0.1773 |
| ionosphere | tree_path | 355 | 4.013 | 0.7577 | 0.7577 | 0.6451 |  |  |  |  |  | 0.1665 | 0.3259 | 0.1331 |
| iris | anchors | 150 | 0.8473 | 0.8533 | 0.8533 |  |  |  |  |  |  | 0 |  | 0.1389 |
| iris | dpg:aggregated_transitions | 150 | 0.8473 | 0.7933 | 0.8533 |  | 0.08979 | 0.2223 | 0.1121 | 0.1115 | 0.1121 |  | 3.56 |  |
| iris | dpg_execution_trace:execution_trace | 150 | 0.8473 | 0.8533 | 0.8533 |  | 0.0501 | 0.1083 | 0.05413 | 0.091 | 0.05413 |  | 1.993 |  |
| iris | lime | 150 | 0.8473 | 0.9533 | 0.8533 | 1 |  |  |  |  |  | 0.07412 |  | 5.349 |
| iris | lore | 150 | 0.8473 | 0.8533 | 0.8533 | 0.7533 |  |  |  |  |  | 0.001369 | 0.7333 | 0.7218 |
| iris | path_bag | 150 | 0.8473 | 0.8533 | 0.8533 | 0.72 |  |  |  |  |  | 0.1154 | 3.54 | 0.117 |
| iris | random_same_size_path | 150 | 0.8473 | 0.8533 | 0.8533 | 0.5067 |  |  |  |  |  | 0.1154 | 1.78 | 0.1183 |
| iris | raw_path_union | 150 | 0.8473 | 0.8533 | 0.8533 | 0.8333 |  |  |  |  |  | 0.1154 | 1.78 | 0.124 |
| iris | shap | 150 | 0.8473 | 0.8533 | 0.8533 | 0.9 |  |  |  |  |  | 0.1154 |  | 0.112 |
| iris | tree_path | 150 | 0.8473 | 0.8533 | 0.8533 | 0.7867 |  |  |  |  |  | 0.1154 | 0.177 | 0.2456 |
| isolet | anchors | 7800 | 20.48 | 0.6417 | 0.6417 |  |  |  |  |  |  | 0 |  | 0.888 |
| isolet | dpg:aggregated_transitions | 7800 | 20.48 | 0.6358 | 0.6633 |  | 0.0264 | 0.06095 | 0.04842 | 0.09779 | 0.04842 |  | 1.634 |  |
| isolet | dpg_execution_trace:execution_trace | 7800 | 20.48 | 0.6127 | 0.6633 |  | 0.02739 | 0.05587 | 0.04569 | 0.09494 | 0.04569 |  | 1.242 |  |
| isolet | lime | 7800 | 20.48 | 0.3917 | 0.6417 | 0.6122 |  |  |  |  |  | 0.01037 |  | 101.6 |
| isolet | lore | 7800 | 20.48 | 0.6468 | 0.6417 | 0.1653 |  |  |  |  |  | 0.1668 | 0.4651 | 53.47 |
| isolet | path_bag | 7800 | 20.48 | 0.7465 | 0.7465 | 0.6731 |  |  |  |  |  | 0.2193 | 63.54 | 1.495 |
| isolet | random_same_size_path | 7800 | 20.48 | 0.7465 | 0.7465 | 0.007692 |  |  |  |  |  | 0.2193 | 60.68 | 102.3 |
| isolet | raw_path_union | 7800 | 20.48 | 0.7465 | 0.7465 | 0.4756 |  |  |  |  |  | 0.2193 | 60.68 | 1.697 |
| isolet | shap | 7800 | 20.48 | 0.6417 | 0.6417 | 0.626 |  |  |  |  |  | 0.05732 |  | 1.629 |
| isolet | tree_path | 7800 | 20.48 | 0.6417 | 0.6417 | 0.5788 |  |  |  |  |  | 0.05732 | 0.001436 | 0.1504 |
| madelon | anchors | 2600 | 29.24 | 0.6019 | 0.6019 |  |  |  |  |  |  | 0 |  | 0.1626 |
| madelon | dpg:aggregated_transitions | 2600 | 29.24 | 0.6208 | 0.5819 |  | 0.08381 | 0.2128 | 0.1064 | 0.118 | 0.1064 |  | 2.268 |  |
| madelon | dpg_execution_trace:execution_trace | 2600 | 29.24 | 0.6496 | 0.5819 |  | 0.09106 | 0.2263 | 0.1132 | 0.1262 | 0.1132 |  | 0.7973 |  |
| madelon | lime | 2600 | 29.24 | 0.5115 | 0.6019 | 0.6385 |  |  |  |  |  | 0.02295 |  | 21.07 |
| madelon | lore | 2600 | 29.24 | 0.6088 | 0.6019 | 0.09769 |  |  |  |  |  | 0.1626 | 0.6846 | 36.22 |
| madelon | path_bag | 2600 | 29.24 | 0.5854 | 0.5854 | 0.2273 |  |  |  |  |  | 0.1426 | 82.74 | 0.9257 |
| madelon | random_same_size_path | 2600 | 29.24 | 0.5854 | 0.5854 | 0.04231 |  |  |  |  |  | 0.1426 | 76.4 | 42.82 |
| madelon | raw_path_union | 2600 | 29.24 | 0.5854 | 0.5854 | 0.2135 |  |  |  |  |  | 0.1426 | 76.4 | 2.262 |
| madelon | shap | 2600 | 29.24 | 0.6019 | 0.6019 | 0.2938 |  |  |  |  |  | 0.09273 |  | 0.2233 |
| madelon | tree_path | 2600 | 29.24 | 0.6019 | 0.6019 | 0.2235 |  |  |  |  |  | 0.09273 | 0.006096 | 0.1473 |
| phoneme | anchors | 5405 | 0.4707 | 0.9317 | 0.9317 |  |  |  |  |  |  | 0 |  | 0.1683 |
| phoneme | dpg:aggregated_transitions | 5405 | 0.4707 | 0.9436 | 0.9317 |  | 0.0465 | 0.09066 | 0.04533 | 0.05827 | 0.04533 |  | 0.8792 |  |
| phoneme | dpg_execution_trace:execution_trace | 5405 | 0.4707 | 0.9297 | 0.9317 |  | 0.05035 | 0.1101 | 0.05504 | 0.06489 | 0.05504 |  | 0.447 |  |
| phoneme | lime | 5405 | 0.4707 | 1 | 0.9214 | 0.8087 |  |  |  |  |  | 0.0339 |  | 12.94 |
| phoneme | lore | 5405 | 0.4707 | 0.9341 | 0.9317 | 0.6575 |  |  |  |  |  | 0.04112 | 0.5441 | 1.624 |
| phoneme | path_bag | 5405 | 0.4707 | 0.8507 | 0.8507 | 0.7243 |  |  |  |  |  | 0.1764 | 68.56 | 3.257 |
| phoneme | random_same_size_path | 5405 | 0.4707 | 0.8507 | 0.8507 | 0.2747 |  |  |  |  |  | 0.1764 | 55.48 | 67.38 |
| phoneme | raw_path_union | 5405 | 0.4707 | 0.8507 | 0.8507 | 0.647 |  |  |  |  |  | 0.1764 | 55.48 | 0.7095 |
| phoneme | shap | 5405 | 0.4707 | 0.9214 | 0.9214 | 0.7334 |  |  |  |  |  | 0.07974 |  | 0.3721 |
| phoneme | tree_path | 5405 | 0.4707 | 0.9214 | 0.9214 | 0.8453 |  |  |  |  |  | 0.07974 | 0.004246 | 0.118 |
| qsar-biodeg | anchors | 1055 | 3.788 | 0.8152 | 0.8152 |  |  |  |  |  |  | 0 |  | 0.1591 |
| qsar-biodeg | dpg:aggregated_transitions | 1055 | 3.788 | 0.7896 | 0.8152 |  | 0.06995 | 0.1514 | 0.07568 | 0.09045 | 0.07568 |  | 3.837 |  |
| qsar-biodeg | dpg_execution_trace:execution_trace | 1055 | 3.788 | 0.8142 | 0.8152 |  | 0.09714 | 0.2158 | 0.1079 | 0.1434 | 0.1079 |  | 2.861 |  |
| qsar-biodeg | lime | 1055 | 3.788 | 1 | 0.8152 | 0.5308 |  |  |  |  |  | 0.06445 |  | 0.5305 |
| qsar-biodeg | lore | 1055 | 3.788 | 0.8209 | 0.818 | 0.4891 |  |  |  |  |  | 0.09315 | 0.8919 | 3.727 |
| qsar-biodeg | path_bag | 1055 | 3.788 | 0.8038 | 0.8038 | 0.6095 |  |  |  |  |  | 0.2651 | 71.18 | 0.6145 |
| qsar-biodeg | random_same_size_path | 1055 | 3.788 | 0.8038 | 0.8038 | 0.2294 |  |  |  |  |  | 0.2651 | 44.37 | 15.87 |
| qsar-biodeg | raw_path_union | 1055 | 3.788 | 0.8038 | 0.8038 | 0.654 |  |  |  |  |  | 0.2651 | 44.37 | 0.6175 |
| qsar-biodeg | shap | 1055 | 3.788 | 0.8152 | 0.8152 | 0.4682 |  |  |  |  |  | 0.1872 |  | 0.1044 |
| qsar-biodeg | tree_path | 1055 | 3.788 | 0.8152 | 0.8152 | 0.4152 |  |  |  |  |  | 0.1872 | 0.03047 | 0.4088 |
| segment | anchors | 2310 | 1.159 | 0.8749 | 0.8749 |  |  |  |  |  |  | 0 |  | 0.2349 |
| segment | dpg:aggregated_transitions | 2310 | 1.159 | 0.8784 | 0.8632 |  | 0.04098 | 0.1065 | 0.05965 | 0.07049 | 0.05965 |  | 1.419 |  |
| segment | dpg_execution_trace:execution_trace | 2310 | 1.159 | 0.8896 | 0.8749 |  | 0.0407 | 0.08758 | 0.05885 | 0.07305 | 0.05885 |  | 0.9156 |  |
| segment | lime | 2310 | 1.159 | 0.5675 | 0.8749 | 0.8212 |  |  |  |  |  | 0.0169 |  | 4.814 |
| segment | lore | 2310 | 1.159 | 0.8372 | 0.8749 | 0.5909 |  |  |  |  |  | 0.09339 | 0.1407 | 0.8583 |
| segment | path_bag | 2310 | 1.159 | 0.8528 | 0.8528 | 0.9013 |  |  |  |  |  | 0.105 | 39.35 | 0.53 |
| segment | random_same_size_path | 2310 | 1.159 | 0.8528 | 0.8528 | 0.5991 |  |  |  |  |  | 0.105 | 27.58 | 11.64 |
| segment | raw_path_union | 2310 | 1.159 | 0.8528 | 0.8528 | 0.8264 |  |  |  |  |  | 0.105 | 27.58 | 0.2062 |
| segment | shap | 2310 | 1.159 | 0.8749 | 0.8749 | 0.8229 |  |  |  |  |  | 0.0638 |  | 0.3517 |
| segment | tree_path | 2310 | 1.159 | 0.8749 | 0.8749 | 0.8013 |  |  |  |  |  | 0.0638 | 0.02366 | 0.1151 |
| spambase | anchors | 4605 | 4.079 | 0.8447 | 0.8447 |  |  |  |  |  |  | 0 |  | 0.486 |
| spambase | dpg:aggregated_transitions | 4605 | 4.079 | 0.8732 | 0.8517 |  | 0.07723 | 0.1823 | 0.09116 | 0.1501 | 0.09116 |  | 4.244 |  |
| spambase | dpg_execution_trace:execution_trace | 4605 | 4.079 | 0.8623 | 0.8517 |  | 0.08309 | 0.1936 | 0.09678 | 0.1393 | 0.09678 |  | 1.641 |  |
| spambase | lime | 4605 | 4.079 | 0.957 | 0.8517 | 0.9168 |  |  |  |  |  | 0.06327 |  | 1.498 |
| spambase | lore | 4605 | 4.079 | 0.8321 | 0.8517 | 0.5286 |  |  |  |  |  | 0.1304 | 0.2949 | 2.187 |
| spambase | path_bag | 4605 | 4.079 | 0.833 | 0.833 | 0.4169 |  |  |  |  |  | 0.2055 | 107.5 | 0.8789 |
| spambase | random_same_size_path | 4605 | 4.079 | 0.833 | 0.833 | 0.9474 |  |  |  |  |  | 0.2055 | 82 | 20.72 |
| spambase | raw_path_union | 4605 | 4.079 | 0.833 | 0.833 | 0.5535 |  |  |  |  |  | 0.2055 | 82 | 1.323 |
| spambase | shap | 4605 | 4.079 | 0.8517 | 0.8517 | 0.4771 |  |  |  |  |  | 0.1884 |  | 0.1642 |
| spambase | tree_path | 4605 | 4.079 | 0.8517 | 0.8517 | 0.5023 |  |  |  |  |  | 0.1884 | 0.0194 | 0.1284 |

## Writing Guidance

- Use this as a robustness/stability analysis, not as another raw-fidelity comparison.
- The central DPG claim should be that local diagnostic quantities vary smoothly for nearby samples when the model decision is locally stable, and expose instability when nearby samples disagree.
- If DPG has lower top-feature agreement than feature-ranking baselines, that is not necessarily a failure: DPG is a graph diagnostic, and its strongest stability evidence should come from confidence, margin, competitor exposure, and vote agreement.