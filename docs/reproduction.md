# Reproduction from a fresh clone

## Install and test

Use Linux and Python 3.13. Install a CUDA-compatible PyTorch build if using an NVIDIA GPU, then install the project:

```sh
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Small unit-test fixtures are included. Checks of full experimental artifacts are skipped when those local artifacts are absent. Core training/resume tests use self-contained fixtures.

Reference resources: about 27 CPU cores, NVIDIA A2 (15 GB VRAM), 180 GiB available RAM. These are observed resources, not a validated minimum. Reduce workers if memory or GPU capacity is limited. Data, caches, replay buffers and intermediate checkpoints need much more disk space than the source code. Measure free space before a full run.

## 1. Recreate the earlier baseline inputs

The final training pipeline depends on calibration, hyperparameter selection and full-state TD3/SAC checkpoints from the baseline study. None are shipped on GitHub. Generate them in order:

```sh
python scripts/robust_study.py prepare --workers 12
python scripts/robust_study.py download --workers 12
python scripts/robust_study.py calibrate --workers 12
python scripts/robust_study.py tune --workers 12
python scripts/robust_study.py validate --workers 12
python scripts/robust_study.py select --workers 12
python scripts/robust_study.py train-final --workers 12
```

This uses `configs/robust_study.yaml`, including the declared methods, representations, tuning seeds and final seeds. It is a large historical experiment, not a short smoke test. Download availability and provider access conditions apply. No old publication report or baseline test is required for the final training stages below.

## 2. Synthetic checkpoints and historical fine-tuning

```sh
python scripts/train_checkpoint_study.py --workers 18
python scripts/finetune_checkpoint_study.py train --workers 6
python scripts/finetune_grid_study.py train --config configs/finetune_grid.json --workers 12
python scripts/finetune_grid_study.py train --config configs/finetune_grid_seed53.json --workers 12
python scripts/final_selection.py classical
python scripts/final_selection.py grid
python scripts/finetune_confirm.py train --workers 6 --threads 4
python scripts/final_selection.py select
```

Run these sequentially. Both grid seeds must finish before selecting hyperparameters. Checkpoints are scored on validation only; the primary rule is one tick. All newly generated candidates include zero-, one- and two-tick scores. Missing seeds depend on the selected configurations; the script determines the confirmation jobs.

## 3. Historical test and figures

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python scripts/final_test.py --workers 22
python scripts/figures_finales.py
```

Use CPU affinity (`taskset` on Linux) if a strict core limit is needed. Each test process handles one session. The test script refuses to overwrite an existing final results CSV. To run another independent experiment, use a separate checkout and output directories, retaining the old outputs separately.

Figures use local final results and selection files. The training-validation figure also requires local validation records and retained checkpoints; generate it with `python scripts/audit_training_budget.py`.

## Paths and provenance

Generated experiments live in `experiences/`; presentation exports live in `resultats/`. Optional environment variables: `CJR_TEST_RAW` (raw-data directories separated by the operating-system path separator), `CJR_VALIDATION_CACHE`, `CJR_TEST_CACHE` and `CJR_RESUME_ROOT`. Default raw data: `data/raw/dot/`; causal caches: `data/cache/`.

Local historical snapshots retain their original source text and paths. The directory rename updated metadata paths and dependent hashes; model bytes were verified unchanged.

Generated experiment files are excluded from Git. Three presentation snapshots are included in `docs/figures/` for the repository README; they can be refreshed from `resultats/figures/` after regenerating the plots. The repository history has been consolidated into one commit.
