# Confidence Score Sensitivity

Source: `experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/per_sample_selected_test.csv`

This analysis tests whether DPG diagnostic performance depends strongly on the original equal-weight confidence formula and multiplicative trace-coverage gate.

Scores are evaluated as risk scores, so higher values indicate higher expected disagreement or error. For confidence-like variants, risk is `1 - confidence_variant`.

## Local Disagreement

| method              | score                        | n_datasets | mean_positive_rate | auroc_mean | auroc_ci95_low | auroc_ci95_high | auprc_mean | auprc_ci95_low | auprc_ci95_high |
| ------------------- | ---------------------------- | ---------- | ------------------ | ---------- | -------------- | --------------- | ---------- | -------------- | --------------- |
| dpg                 | risk_low_vote_agreement_only | 15         | 0.175              | 0.9237     | 0.8479         | 0.9725          | 0.7539     | 0.6503         | 0.8567          |
| dpg                 | risk_low_multiplicative_gate | 15         | 0.175              | 0.9001     | 0.8501         | 0.9438          | 0.6706     | 0.57           | 0.766           |
| dpg                 | risk_low_reported            | 15         | 0.175              | 0.9001     | 0.8514         | 0.9442          | 0.6706     | 0.5739         | 0.7704          |
| dpg                 | risk_low_additive_coverage   | 15         | 0.175              | 0.88       | 0.799          | 0.9439          | 0.6516     | 0.5545         | 0.7481          |
| dpg                 | risk_low_no_gate             | 15         | 0.175              | 0.8742     | 0.7921         | 0.9408          | 0.6163     | 0.5354         | 0.7104          |
| dpg                 | risk_low_margin_only         | 15         | 0.175              | 0.8689     | 0.7901         | 0.9361          | 0.6138     | 0.5039         | 0.723           |
| dpg                 | risk_competitor_exposure     | 15         | 0.175              | 0.8642     | 0.7819         | 0.9295          | 0.5779     | 0.482          | 0.6811          |
| dpg                 | risk_low_path_purity         | 15         | 0.175              | 0.8642     | 0.778          | 0.9289          | 0.5779     | 0.4758         | 0.6762          |
| dpg                 | risk_low_concentration_only  | 15         | 0.175              | 0.2596     | 0.1499         | 0.3767          | 0.141      | 0.08089        | 0.2194          |
| dpg_execution_trace | risk_low_vote_agreement_only | 15         | 0.1402             | 0.9808     | 0.9682         | 0.9929          | 0.9104     | 0.8497         | 0.9651          |
| dpg_execution_trace | risk_low_multiplicative_gate | 15         | 0.1402             | 0.9046     | 0.8404         | 0.9528          | 0.6509     | 0.5522         | 0.7599          |
| dpg_execution_trace | risk_low_reported            | 15         | 0.1402             | 0.9046     | 0.841          | 0.9544          | 0.6509     | 0.5471         | 0.7643          |
| dpg_execution_trace | risk_low_additive_coverage   | 15         | 0.1402             | 0.9045     | 0.8465         | 0.953           | 0.6509     | 0.5465         | 0.7614          |
| dpg_execution_trace | risk_low_no_gate             | 15         | 0.1402             | 0.9043     | 0.8454         | 0.9549          | 0.6507     | 0.5459         | 0.7616          |
| dpg_execution_trace | risk_low_margin_only         | 15         | 0.1402             | 0.8834     | 0.8226         | 0.9349          | 0.5667     | 0.457          | 0.6935          |
| dpg_execution_trace | risk_competitor_exposure     | 15         | 0.1402             | 0.8832     | 0.8281         | 0.9317          | 0.5729     | 0.4603         | 0.7003          |
| dpg_execution_trace | risk_low_path_purity         | 15         | 0.1402             | 0.8832     | 0.8291         | 0.9337          | 0.5729     | 0.4653         | 0.6967          |
| dpg_execution_trace | risk_low_concentration_only  | 15         | 0.1402             | 0.0876     | 0.04462        | 0.1375          | 0.1045     | 0.04955        | 0.1834          |

## Model Error

| method              | score                        | n_datasets | mean_positive_rate | auroc_mean | auroc_ci95_low | auroc_ci95_high | auprc_mean | auprc_ci95_low | auprc_ci95_high |
| ------------------- | ---------------------------- | ---------- | ------------------ | ---------- | -------------- | --------------- | ---------- | -------------- | --------------- |
| dpg                 | risk_low_vote_agreement_only | 15         | 0.1431             | 0.7408     | 0.671          | 0.8162          | 0.3612     | 0.2638         | 0.4658          |
| dpg                 | risk_low_no_gate             | 15         | 0.1431             | 0.7264     | 0.6674         | 0.7953          | 0.3156     | 0.2351         | 0.4084          |
| dpg                 | risk_low_path_purity         | 15         | 0.1431             | 0.7245     | 0.6631         | 0.7892          | 0.3008     | 0.2299         | 0.3649          |
| dpg                 | risk_competitor_exposure     | 15         | 0.1431             | 0.7243     | 0.6606         | 0.7869          | 0.3008     | 0.2282         | 0.3631          |
| dpg                 | risk_low_additive_coverage   | 15         | 0.1431             | 0.7208     | 0.6608         | 0.7885          | 0.3075     | 0.2208         | 0.4016          |
| dpg                 | risk_low_margin_only         | 15         | 0.1431             | 0.7097     | 0.647          | 0.7816          | 0.283      | 0.2161         | 0.3415          |
| dpg                 | risk_low_multiplicative_gate | 15         | 0.1431             | 0.6796     | 0.5994         | 0.7617          | 0.2938     | 0.2044         | 0.397           |
| dpg                 | risk_low_reported            | 15         | 0.1431             | 0.6796     | 0.5946         | 0.7609          | 0.2938     | 0.2029         | 0.3951          |
| dpg                 | risk_low_concentration_only  | 15         | 0.1431             | 0.4221     | 0.3323         | 0.4978          | 0.1558     | 0.1047         | 0.2075          |
| dpg_execution_trace | risk_low_vote_agreement_only | 15         | 0.1443             | 0.7638     | 0.6983         | 0.8249          | 0.3417     | 0.259          | 0.4185          |
| dpg_execution_trace | risk_low_no_gate             | 15         | 0.1443             | 0.7523     | 0.7011         | 0.8051          | 0.3216     | 0.2391         | 0.4043          |
| dpg_execution_trace | risk_low_additive_coverage   | 15         | 0.1443             | 0.7523     | 0.7016         | 0.8032          | 0.3216     | 0.2401         | 0.4053          |
| dpg_execution_trace | risk_low_multiplicative_gate | 15         | 0.1443             | 0.7523     | 0.7017         | 0.8038          | 0.3216     | 0.2383         | 0.4044          |
| dpg_execution_trace | risk_low_reported            | 15         | 0.1443             | 0.7523     | 0.7011         | 0.8034          | 0.3216     | 0.2419         | 0.4033          |
| dpg_execution_trace | risk_competitor_exposure     | 15         | 0.1443             | 0.7471     | 0.6888         | 0.8055          | 0.3255     | 0.2406         | 0.4098          |
| dpg_execution_trace | risk_low_path_purity         | 15         | 0.1443             | 0.7471     | 0.6908         | 0.8094          | 0.3255     | 0.2421         | 0.4141          |
| dpg_execution_trace | risk_low_margin_only         | 15         | 0.1443             | 0.7367     | 0.6739         | 0.8009          | 0.315      | 0.235          | 0.4039          |
| dpg_execution_trace | risk_low_concentration_only  | 15         | 0.1443             | 0.3265     | 0.2373         | 0.4172          | 0.13       | 0.07913        | 0.1883          |

## Ranking Stability Against Reported Low Confidence

| method              | reference_score   | score                        | n_samples | spearman_rho |
| ------------------- | ----------------- | ---------------------------- | --------- | ------------ |
| dpg                 | risk_low_reported | risk_low_reported            | 6079      | 1            |
| dpg                 | risk_low_reported | risk_low_multiplicative_gate | 6079      | 1            |
| dpg                 | risk_low_reported | risk_low_additive_coverage   | 6079      | 0.8912       |
| dpg                 | risk_low_reported | risk_low_margin_only         | 6079      | 0.7549       |
| dpg                 | risk_low_reported | risk_low_path_purity         | 6079      | 0.7424       |
| dpg                 | risk_low_reported | risk_competitor_exposure     | 6079      | 0.7424       |
| dpg                 | risk_low_reported | risk_low_no_gate             | 6079      | 0.7365       |
| dpg                 | risk_low_reported | risk_low_vote_agreement_only | 6079      | 0.7132       |
| dpg                 | risk_low_reported | risk_low_concentration_only  | 6079      | -0.4913      |
| dpg_execution_trace | risk_low_reported | risk_low_reported            | 6079      | 1            |
| dpg_execution_trace | risk_low_reported | risk_low_multiplicative_gate | 6079      | 1            |
| dpg_execution_trace | risk_low_reported | risk_low_additive_coverage   | 6079      | 1            |
| dpg_execution_trace | risk_low_reported | risk_low_no_gate             | 6079      | 1            |
| dpg_execution_trace | risk_low_reported | risk_low_margin_only         | 6079      | 0.9542       |
| dpg_execution_trace | risk_low_reported | risk_competitor_exposure     | 6079      | 0.9453       |
| dpg_execution_trace | risk_low_reported | risk_low_path_purity         | 6079      | 0.9453       |
| dpg_execution_trace | risk_low_reported | risk_low_vote_agreement_only | 6079      | 0.9192       |
| dpg_execution_trace | risk_low_reported | risk_low_concentration_only  | 6079      | -0.6715      |

## Writing Guidance

- Treat confidence as a diagnostic ranking index, not a calibrated probability.
- If several variants perform similarly, the paper can keep the simple reported formula and state that conclusions are robust to reasonable alternatives.
- If margin-only or competitor-exposure scores dominate, emphasize the class-contrastive DPG diagnostics rather than the aggregate confidence formula.
- Use local-disagreement results as the primary diagnostic-value evidence; model-error results are secondary because model error is affected by dataset difficulty and classifier quality.
