# KBS local-DPG experiments

Experiments for the Knowledge-Based Systems version of the DPG-local paper (resubmission
after the AAAI-27 rejection). The method is `dpg.local_dpg` (trace-indexed local DPG,
exact class support, local context order, critical predicate). The revision rationale
and pilot evidence are in `JOURNAL_paper/plans/KBS_REVISION_PLAN_2026-10-09.md` on the
`local_explanation` branch.

## Protocol

- **One black box per (dataset, family, seed).** It is chosen on validation accuracy
  only (`select_blackbox.py`), refitted on train + validation, and explained by
  every method and control. The AAAI-27 runs instead selected a forest per
  explainer, so methods explained different models.
- **DPG-local has no tuned hyper-parameters.** It uses `context_order="auto"` and
  `decimal_threshold=6`, which affects label formatting only.
- **Datasets (16).** banknote-authentication, breast_cancer, diabetes, digits,
  ionosphere, iris, isolet, madelon, phoneme, qsar-biodeg, satimage, segment,
  spambase, vehicle, wine, wine_quality. `wdbc` duplicates `breast_cancer` and
  `optdigits` contains `digits`; both are excluded. Splits come from
  `splits/split_registry.json`, and the test sets are the untouched prepared test
  arrays.
- **Families and validation grids (100 estimators).**

  | Family | Grid |
  |---|---|
  | RandomForest, ExtraTrees, Bagging(tree) | depth {4, 8, None} |
  | AdaBoost (SAMME) | base depth {1, 2, 3} |
  | GradientBoosting | depth {2, 3, 5}, lr 0.1 |

- **Seeds.** 27, 42, 100–107 for the main study. Baselines use 3 seeds × 50 test
  samples by default (cost).

## Research questions → outputs

| RQ | Question | Stage | Report table |
|---|---|---|---|
| RQ1 | How often does merging traces invent routes? What does route faithfulness (k*) cost? | `dpg`, `scaling` | `rq1_route_faithfulness.csv`, `scaling.csv` |
| RQ2 | Does the critical predicate localise the decision better than controls? (whole-model intervention; separation from differently- vs same-predicted training neighbours) | `dpg`, `baselines` | `rq2_localisation_summary.csv`, `rq2_localisation_tests.csv` |
| RQ3 | Do structural signals detect model errors beyond model-native uncertainty? (expected: no; reported honestly) | `dpg` | `rq3_error_detection_auc.csv`, `rq3_incremental_value.csv` |
| RQ4 | Runtime and explanation size vs TreeSHAP, LIME, Anchors, LORE | `dpg`, `baselines` | `rq4_cost_size.csv` |

**RQ2 controls.** Executed predicates of the same sample chosen at random, by
smallest slack, by most trees, or on the top feature of TreeSHAP / LIME / Anchors /
LORE. Each is crossed at its threshold exactly like the critical predicate.

## Running on the server

```bash
git fetch && git checkout feature/local-explanation-v2
cd experiments/local_dpg_kbs
PYTHON=python3.13 bash setup_server.sh       # venv in <repo>/.venv-kbs, runs tests/test_local_dpg.py
bash launch_server.sh smoke                   # a few minutes; check results_smoke/report/summary.md
nohup bash launch_server.sh all > kbs_all.log 2>&1 &
```

### Stages

- **Order.** `all` runs select → dpg → scaling → analyze → baselines → analyze.
  Stages can also be run one at a time (`bash launch_server.sh dpg`).
- **Resuming.** Jobs are resumable: existing outputs are skipped. Failed jobs leave
  `results/<stage>/<job>.error.txt`; re-running the stage retries them.
- **Logs.** `results/logs/<stage>.log`.

### Cost

These are rough extrapolations from small local smoke runs. Verify against the
smoke stage before planning around them.

- **DPG-local.** About 0.1–0.35 s per test sample at 100 trees, including pivots
  and five interventions.
- **Multiclass gradient boosting is the long tail.** It grows one tree per class
  per stage: isolet has 26 × 100 = 2,600 trees.
  - Expect the isolet/GBM fits and explanations to take hours per seed.
  - TreeSHAP does not support multiclass `GradientBoostingClassifier`, so the
    SHAP-guided control is absent there.
- **Anchors and LORE dominate baseline cost.** Anchors took about 10 s per sample
  on vehicle in the smoke run. Knobs:
  - `BASE_SAMPLES` and `BASE_SEEDS` set how many samples and seeds the baselines use.
  - `LORE_GEN=random` swaps the genetic neighbourhood for the faster random one.
  - `--families` / `--datasets` can be passed by running a script directly.
- **Runtimes under load.** Measured runtimes include parallel load. `scaling`
  defaults to 8 workers so its timings are less contended.

### Bringing results back

`results/report/` holds the paper tables (`summary.md` plus CSVs). Please also
archive the raw per-sample CSVs, which the figures and any re-analysis need:

```bash
tar czf kbs_results_$(date +%F).tgz results/
```
