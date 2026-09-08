# Semantic Faithfulness Provenance

Date mirrored into journal result root: 2026-06-19

This directory contains the semantic-faithfulness artifacts used by `JOURNAL_paper/main_journal.tex`.

## Origin

The files were copied without modification from:

`experiments_local_explanation/experiment_dpg2_next_phase/_analysis/semantic_faithfulness/`

They are mirrored here so all journal-facing numerical artifacts are available under:

`experiments_local_explanation/results_journal_v1/`

## Files

- `semantic_faithfulness_summary.csv`
- `semantic_faithfulness_per_sample.csv`
- `critical_branch_flip_summary.csv`
- `critical_case_reports.csv`

## Scope

The focused semantic-faithfulness analysis contains 2,975 evaluated instances across 15 datasets. It uses a matched feature-budget perturbation protocol for sufficiency and comprehensiveness.

## Manuscript Values

Rounded values reported in the journal manuscript:

| Method | Sufficiency probability | Comprehensiveness drop |
| --- | --- | --- |
| DPG-local execution trace | 0.54 | 0.13 |
| LIME | 0.55 | 0.17 |
| ICE | 0.48 | 0.07 |

## Copy Verification

The mirrored files were checked with SHA-256 and match the source files byte-for-byte:

- `semantic_faithfulness_summary.csv`: `3f40d05f5528676e53fc66ea78b345553253a541391835f98ef27b3aa8ad64b0`
- `semantic_faithfulness_per_sample.csv`: `87bae7c6a226e716614409fec998ca4fc68395399f29f638c716b1b96789e644`
- `critical_branch_flip_summary.csv`: `f0d56287e4f43e46d6de7c8b1ec68995b9fa649c973199bbb0dd50c33e21f3a4`
- `critical_case_reports.csv`: `d7d03e9092e472f241e1c7dbe5deeb42a10cd64eff629beb5a7bd1984cb1b86c`

## Residual Note

For a final public reproducibility package, the analysis can be rerun with `experiments_local_explanation/analyze_semantic_faithfulness.py` using an output directory under `results_journal_v1`. The present copy is sufficient for manuscript provenance because it preserves the exact audited files.
