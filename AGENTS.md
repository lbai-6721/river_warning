# Repository workflow

- This directory is the Git repository for the river monitoring project.
- Preserve original datasets, labels, weights and historical scripts. Do not move large datasets just to reorganize code.
- New experiment code belongs in `riverlab/`, experiment configurations in `configs/`, tests in `tests/`, and documentation in `docs/`.
- Keep generated manifests, predictions, checkpoints and figures under ignored `artifacts/`, `runs/` or `paper_output/` directories.
- Never upload this private dataset or push the repository without explicit permission.
- Default to CPU verification on this computer. Full training is for a subsequently selected remote GPU workspace.
- Split by original image, shared frames, date and verified event before generating patches or temporal windows. Never tune on the test set.
- Record configurations, seeds, dataset fingerprints and Git revisions with runs. Preserve source licenses and identify legacy implementations accurately.
- Before finishing code changes, run meaningful CPU tests and `git diff --check`, review and stage explicit task-owned paths, and create a focused commit using the commit-task-changes skill.
- Report commit SHA and verification. Do not amend, rebase, tag or push unless explicitly requested.
