# DPG 2.0 Remaining Analysis Results

## Consolidated Comparison

comparison_family           comparison_method  datasets  mean_fidelity  mean_local_accuracy  mean_margin  mean_num_paths  mean_num_active_nodes  mean_edge_precision  mean_edge_recall  mean_recombination_rate  mean_explanation_confidence
         baseline                        shap        15         1.0000               0.9087       0.7197             NaN                    NaN                  NaN               NaN                      NaN                          NaN
         baseline                         ice        15         0.9620               0.8647       0.4588             NaN                    NaN                  NaN               NaN                      NaN                          NaN
       legacy_dpg             legacy_dpg_best        15         0.9074               0.7977       0.6324        130.6535                52.7629                  NaN               NaN                      NaN                          NaN
  dpg2_next_phase        dpg2_execution_trace        15         0.8956               0.8025       0.7253         17.3333                57.4688               1.0000            0.9999                   0.0000                       0.6872
          pruning                pruning_best        15         0.8801               0.7910       0.6392         63.5309                51.3563                  NaN               NaN                      NaN                          NaN
  dpg2_next_phase dpg2_aggregated_transitions        15         0.8797               0.7914       0.6321         67.8794                55.9562               0.8732            0.8613                   0.1268                       0.5047
         baseline                        lime        15         0.4793               0.4361       0.3685             NaN                    NaN                  NaN               NaN                      NaN                          NaN

## Key Cohort Effect Sizes

graph_construction_mode        comparison                 metric  group_a group_b  mean_a  mean_b  delta_mean  cohen_d  cliffs_delta  n_a  n_b
 aggregated_transitions DISAGREE_vs_AGREE    competitor_exposure DISAGREE   AGREE  0.6143  0.2323      0.3820   1.7981        0.7931 1394 4685
 aggregated_transitions DISAGREE_vs_AGREE         edge_precision DISAGREE   AGREE  0.9481  0.9069      0.0413   0.3369        0.4613 1394 4685
 aggregated_transitions DISAGREE_vs_AGREE explanation_confidence DISAGREE   AGREE  0.3722  0.5313     -0.1591  -1.0418       -0.5814 1394 4685
 aggregated_transitions DISAGREE_vs_AGREE            path_purity DISAGREE   AGREE  0.3857  0.7677     -0.3820  -1.7981       -0.7931 1394 4685
 aggregated_transitions DISAGREE_vs_AGREE     recombination_rate DISAGREE   AGREE  0.0519  0.0931     -0.0413  -0.3369       -0.4121 1394 4685
 aggregated_transitions    MW-EM_vs_MC-EC    competitor_exposure    MW-EM   MC-EC  0.3475  0.2099      0.1376   0.6217        0.3742  764 3921
 aggregated_transitions    MW-EM_vs_MC-EC         edge_precision    MW-EM   MC-EC  0.9266  0.9030      0.0236   0.1952        0.1932  764 3921
 aggregated_transitions    MW-EM_vs_MC-EC explanation_confidence    MW-EM   MC-EC  0.4739  0.5425     -0.0686  -0.4194       -0.2424  764 3921
 aggregated_transitions    MW-EM_vs_MC-EC            path_purity    MW-EM   MC-EC  0.6525  0.7901     -0.1376  -0.6217       -0.3742  764 3921
 aggregated_transitions    MW-EM_vs_MC-EC     recombination_rate    MW-EM   MC-EC  0.0734  0.0970     -0.0236  -0.1952       -0.1956  764 3921
        execution_trace DISAGREE_vs_AGREE    competitor_exposure DISAGREE   AGREE  0.6136  0.1754      0.4382   2.1752        0.8584 1357 4722
        execution_trace DISAGREE_vs_AGREE         edge_precision DISAGREE   AGREE  1.0000  1.0000      0.0000   0.2635        0.2598 1357 4722
        execution_trace DISAGREE_vs_AGREE explanation_confidence DISAGREE   AGREE  0.4164  0.6878     -0.2714  -1.9692       -0.8564 1357 4722
        execution_trace DISAGREE_vs_AGREE            path_purity DISAGREE   AGREE  0.3864  0.8246     -0.4382  -2.1752       -0.8584 1357 4722
        execution_trace DISAGREE_vs_AGREE     recombination_rate DISAGREE   AGREE  0.0000  0.0000      0.0000      NaN        0.0000 1357 4722
        execution_trace    MW-EM_vs_MC-EC    competitor_exposure    MW-EM   MC-EC  0.2889  0.1534      0.1355   0.6535        0.4265  767 3955
        execution_trace    MW-EM_vs_MC-EC         edge_precision    MW-EM   MC-EC  1.0000  1.0000     -0.0000  -0.0228       -0.0011  767 3955
        execution_trace    MW-EM_vs_MC-EC explanation_confidence    MW-EM   MC-EC  0.5932  0.7061     -0.1129  -0.7711       -0.4471  767 3955
        execution_trace    MW-EM_vs_MC-EC            path_purity    MW-EM   MC-EC  0.7111  0.8466     -0.1355  -0.6535       -0.4265  767 3955
        execution_trace    MW-EM_vs_MC-EC     recombination_rate    MW-EM   MC-EC  0.0000  0.0000      0.0000      NaN        0.0000  767 3955

## Case Studies

                    case_label dataset  sample_idx graph_construction_mode true_label model_pred local_pred  explanation_confidence  support_margin  edge_precision  recombination_rate critical_node_label  critical_split_depth                                                                                                                                                                                                                                                                            aggregate_png
explanation_model_disagreement madelon          93  aggregated_transitions          1          0          1                  0.7918          0.9066          1.0000              0.0000                None                     0 /home/sbarbonjr/projects/DPG/experiments_local_explanation/experiment_dpg2_next_phase/_analysis/comparison_story_local_dpg_madelon/case_studies/explanation_model_disagreement/aggregated_transitions/explanation_model_disagreement_madelon_aggregated_transitions_aggregate_sid_93.png
explanation_model_disagreement madelon          93         execution_trace          1          0          1                  0.7918          0.9066          1.0000              0.0000                None                     0               /home/sbarbonjr/projects/DPG/experiments_local_explanation/experiment_dpg2_next_phase/_analysis/comparison_story_local_dpg_madelon/case_studies/explanation_model_disagreement/execution_trace/explanation_model_disagreement_madelon_execution_trace_aggregate_sid_93.png
