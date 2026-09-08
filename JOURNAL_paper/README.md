# JOURNAL_paper Revision Workspace

This directory is the working space for the journal version of the ECML-submitted DPG-local paper.

## Provenance

- `ECML26_DPG.tex`: submitted ECML source copied from `paper_ecml_rejected`.
- `ECML26_DPG_LocalExplanation.zip`: original submitted archive kept for provenance.
- `journal_paper.tex`: editable journal-version source, initialized from `ECML26_DPG.tex`.
- `reviews/ecml_rejection_comments.txt`: ECML decision and reviewer comments.
- `plans/`: implementation and writing plans for the journal revision.

## Revision Principle

Keep the submitted version immutable as the reference point. Develop the journal paper in `journal_paper.tex`, and treat all new experiments as registered artifacts with explicit scripts, configs, seeds, and outputs.

## Suggested Order

1. Read `plans/reviewer_issue_matrix.md` to see what each reviewer criticism requires.
2. Follow `plans/implementation_plan.md` to run the new experiments.
3. Use `plans/writing_plan.md` to rewrite the manuscript around the new evidence.
