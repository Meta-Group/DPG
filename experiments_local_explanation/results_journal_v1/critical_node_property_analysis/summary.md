# Critical Nodes And Dataset Properties

This report asks whether global dataset properties explain when critical nodes become more useful.

Important caution: this is a dataset-level exploratory analysis over 15 datasets, so correlations are hypothesis-generating rather than confirmatory.

## Strongest Associations

### critical_node_present_rate

| predictor                  | n_datasets | spearman_rho |
| -------------------------- | ---------- | ------------ |
| avg_recombination_rate     | 15         | 0.8179       |
| avg_num_paths              | 15         | 0.6393       |
| avg_num_active_nodes       | 15         | -0.4558      |
| test_normalized_entropy    | 15         | -0.4504      |
| features_per_train_sample  | 15         | -0.4433      |
| avg_explanation_confidence | 15         | -0.4214      |

### critical_eligible_rate

| predictor                 | n_datasets | spearman_rho |
| ------------------------- | ---------- | ------------ |
| features_per_train_sample | 15         | -0.6188      |
| avg_num_active_nodes      | 15         | -0.4732      |
| n_features                | 15         | -0.4299      |
| log_n_features            | 15         | -0.4299      |
| avg_recombination_rate    | 15         | 0.2728       |
| avg_support_margin        | 15         | 0.2219       |

### critical_changed_to_competitor_rate

| predictor                 | n_datasets | spearman_rho |
| ------------------------- | ---------- | ------------ |
| features_per_train_sample | 10         | -0.5462      |
| avg_recombination_rate    | 10         | 0.4643       |
| test_normalized_entropy   | 10         | -0.4302      |
| n_features                | 10         | -0.3733      |
| log_n_features            | 10         | -0.3733      |
| test_imbalance_ratio      | 10         | 0.3456       |

### critical_changed_advantage_vs_random

| predictor                 | n_datasets | spearman_rho |
| ------------------------- | ---------- | ------------ |
| model_accuracy            | 10         | 0.653        |
| local_accuracy            | 10         | 0.4202       |
| features_per_train_sample | 10         | 0.4073       |
| n_train_core              | 10         | -0.3168      |
| n_test                    | 10         | -0.3168      |
| log_n_train_core          | 10         | -0.3168      |

### critical_competitor_delta_advantage_vs_random

| predictor                  | n_datasets | spearman_rho |
| -------------------------- | ---------- | ------------ |
| features_per_train_sample  | 10         | 0.5515       |
| n_features                 | 10         | 0.5106       |
| log_n_features             | 10         | 0.5106       |
| avg_explanation_confidence | 10         | -0.4667      |
| avg_num_paths              | 10         | -0.3818      |
| train_core_imbalance_ratio | 10         | 0.3455       |

## Most Favorable Contexts

| dataset                 | n_train_core | n_features | n_classes_train | train_core_imbalance_ratio | train_core_minority_fraction | model_accuracy | avg_support_margin | avg_competitor_exposure | critical_node_present_rate | critical_eligible_rate | critical_changed_to_competitor_rate | control_random_path_changed_to_competitor_rate | critical_changed_advantage_vs_random | critical_competitor_delta_advantage_vs_random |
| ----------------------- | ------------ | ---------- | --------------- | -------------------------- | ---------------------------- | -------------- | ------------------ | ----------------------- | -------------------------- | ---------------------- | ----------------------------------- | ---------------------------------------------- | ------------------------------------ | --------------------------------------------- |
| banknote-authentication | 877          | 4          | 2               | 1.249                      | 0.4447                       | 0.9745         | 0.782              | 0.109                   | 0.2364                     | 0.2364                 | 0.1846                              | 0.1231                                         | 0.06154                              | -0.0348                                       |
| digits                  | 1149         | 64         | 10              | 1.054                      | 0.09661                      | 0.8806         | 0.4696             | 0.3489                  | 0.6833                     | 0.1667                 | 0.01667                             | 0.01667                                        | 0                                    | 0.004294                                      |
| iris                    | 96           | 4          | 3               | 1                          | 0.3333                       | 0.9333         | 0.8097             | 0.09602                 | 0.06667                    | 0.06667                | 0                                   | 0                                              | 0                                    | -0.1                                          |
| wdbc                    | 364          | 30         | 2               | 1.676                      | 0.3736                       | 0.9474         | 0.9305             | 0.03476                 | 0.03509                    | 0.03509                | 0                                   | 0                                              | 0                                    | -0.006483                                     |
| wine                    | 113          | 13         | 3               | 1.5                        | 0.2655                       | 1              | 0.8926             | 0.05952                 | 0.02778                    | 0.02778                | 0                                   | 0                                              | 0                                    | 0.05                                          |
| isolet                  | 4989         | 617        | 26              | 1.011                      | 0.03808                      | 0.7288         | 0.09292            | 0.6768                  | 0.000641                   | 0.000641               | 0                                   | 0                                              | 0                                    | -0.0168                                       |
| spambase                | 2944         | 57         | 2               | 1.538                      | 0.394                        | 0.9121         | 0.7028             | 0.1486                  | 0.08035                    | 0.06949                | 0                                   | 0.01562                                        | -0.01562                             | -0.01889                                      |
| phoneme                 | 3458         | 5          | 2               | 2.407                      | 0.2935                       | 0.8122         | 0.8147             | 0.09267                 | 0.06938                    | 0.0592                 | 0.01562                             | 0.09375                                        | -0.07812                             | -0.03168                                      |
| diabetes                | 491          | 8          | 2               | 1.871                      | 0.3483                       | 0.7143         | 0.5686             | 0.2157                  | 0.7857                     | 0.2208                 | 0.02941                             | 0.1176                                         | -0.08824                             | 0.005882                                      |
| segment                 | 1478         | 19         | 7               | 1.005                      | 0.1428                       | 0.8377         | 0.5333             | 0.2814                  | 0.01948                    | 0.01948                | 0                                   | 0.1111                                         | -0.1111                              | -0.01894                                      |

## Target-Class Minority/Majority Cohorts

| target_is_minority | target_is_majority | samples | datasets | mean_target_train_fraction | critical_present_rate | critical_eligible_rate | critical_changed_to_competitor_rate | control_random_changed_to_competitor_rate | critical_advantage_vs_random |
| ------------------ | ------------------ | ------- | -------- | -------------------------- | --------------------- | ---------------------- | ----------------------------------- | ----------------------------------------- | ---------------------------- |
| True               | True               | 550     | 2        | 0.4909                     | 0.02182               | 0.003636               | 0                                   | 0.005455                                  | -0.005455                    |
| True               | False              | 1492    | 12       | 0.2613                     | 0.128                 | 0.08713                | 0.004692                            | 0.06367                                   | -0.05898                     |
| False              | True               | 3638    | 13       | 0.3879                     | 0.09016               | 0.03079                | 0.002749                            | 0.01429                                   | -0.01154                     |
| False              | False              | 399     | 3        | 0.131                      | 0.5539                | 0.1504                 | 0.002506                            | 0.01003                                   | -0.007519                    |

## Most Favorable Target Classes

| dataset                 | target_label | target_train_fraction | target_imbalance_vs_majority | target_is_minority | target_is_majority | samples | critical_present_rate | critical_eligible_rate | critical_changed_to_competitor_rate | control_random_changed_to_competitor_rate | control_same_depth_changed_to_competitor_rate | critical_advantage_vs_random | critical_advantage_vs_same_depth |
| ----------------------- | ------------ | --------------------- | ---------------------------- | ------------------ | ------------------ | ------- | --------------------- | ---------------------- | ----------------------------------- | ----------------------------------------- | --------------------------------------------- | ---------------------------- | -------------------------------- |
| diabetes                | 1            | 0.3483                | 1.871                        | True               | False              | 33      | 0.7273                | 0.5152                 | 0.1212                              | 0.1212                                    | 0.06061                                       | 0                            | 0.06061                          |
| banknote-authentication | 0            | 0.5553                | 1                            | False              | True               | 156     | 0.3718                | 0.3718                 | 0.05769                             | 0.05769                                   | 0.01923                                       | 0                            | 0.03846                          |
| digits                  | 7            | 0.09922               | 1.026                        | False              | False              | 26      | 1                     | 0.07692                | 0.03846                             | 0.03846                                   | 0                                             | 0                            | 0.03846                          |
| breast_cancer           | 0            | 0.3736                | 1.676                        | True               | False              | 40      | 0                     | 0                      | 0                                   | 0                                         | 0                                             | 0                            | 0                                |
| digits                  | 0            | 0.09922               | 1.026                        | False              | False              | 33      | 1                     | 0.06061                | 0                                   | 0                                         | 0                                             | 0                            | 0                                |
| digits                  | 1            | 0.1018                | 1                            | False              | True               | 25      | 1                     | 0                      | 0                                   | 0                                         | 0                                             | 0                            | 0                                |
| digits                  | 2            | 0.09835               | 1.035                        | False              | False              | 29      | 1                     | 0                      | 0                                   | 0                                         | 0                                             | 0                            | 0                                |
| digits                  | 3            | 0.1018                | 1                            | False              | True               | 8       | 1                     | 0                      | 0                                   | 0                                         | 0                                             | 0                            | 0                                |
| digits                  | 4            | 0.101                 | 1.009                        | False              | False              | 42      | 0.07143               | 0                      | 0                                   | 0                                         | 0                                             | 0                            | 0                                |
| digits                  | 5            | 0.101                 | 1.009                        | False              | False              | 153     | 0.549                 | 0.3464                 | 0                                   | 0                                         | 0                                             | 0                            | 0                                |
| digits                  | 6            | 0.101                 | 1.009                        | False              | False              | 35      | 0.8286                | 0.08571                | 0                                   | 0                                         | 0                                             | 0                            | 0                                |
| digits                  | 9            | 0.1001                | 1.017                        | False              | False              | 9       | 1                     | 0                      | 0                                   | 0                                         | 0                                             | 0                            | 0                                |

## Working Interpretation

- Critical-node usefulness should be treated as conditional and dataset-dependent.
- Imbalance-related columns are included explicitly; if they are not among the strongest associations, the paper should avoid claiming imbalance is the main driver.
- The class-level table tests whether minority target classes behave differently from majority target classes.
- Favorable contexts are better identified by the actual advantage over controls, not just by critical-node occurrence.
- Because n=15 datasets, any property-level conclusion should be framed as exploratory and moved to discussion or appendix unless the signal is very strong.
