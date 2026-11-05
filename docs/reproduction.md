# Reproduction guide

This guide describes how run `c30863ba7974171b` was produced and how to repeat it.

## Environment of the original run

| Item | Value |
|---|---|
| Operating system | Ubuntu 24.04.5 LTS |
| CPU | Intel Core i9-14900K, 24 cores / 32 threads |
| Memory | 31 GiB RAM |
| GPU | none (all models on CPU) |
| Python | 3.12.3, packages in `requirements-lock.txt` |
| R | 4.3.3, `survey` 4.5, packages in `renv.lock` |
| TabPFN | package 9.2.0, weights `tabpfn-v3.5-20260909.safetensors` |

The full hardware record is in `docs/hardware.txt`.

## Order of the steps

1. **Data.** Put `ESS11e04_2.sav` in `data/` and check its SHA-256 (see `data/README.md`).
2. **Stage-1 recoding.**
   ```
   python scripts/run_preprocess.py --config config/audit.yml
   ```
   Writes `local/analytic_stage1.parquet` (respondent level, not committed) and the aggregate evidence in `results/preprocessing/`.
3. **Phase 00 audit.**
   ```
   python scripts/run_audit.py --config config/audit.yml
   quarto render 00_data_audit.qmd
   ```
   The status becomes `APPROVED` only with all automatic checks passing and the five human decisions in `config/audit_signoff.yaml`. The R comparison of the bootstrap needs `Rscript` with `survey` and `jsonlite`.
4. **TabPFN feasibility pilot (decision D16).** Already done for this run; its results are in `results/benchmarking/runtime/`. The pilot script only runs while `config/tabpfn.yml` has `status: draft`.
5. **Benchmark.**
   ```
   python scripts/run_benchmark.py --model xgboost --resume
   ```
   Run one model at a time or use `--all`. Each outer fold is written as a checkpoint; `--resume` skips folds whose checkpoint is complete and intact. `--max-folds 1` is useful for a test run.
6. **Collect metrics.**
   ```
   python scripts/run_benchmark.py --collect c30863ba7974171b
   ```
7. **Inference.**
   ```
   python scripts/run_inference.py --run-id c30863ba7974171b \
       --models logistic elastic_net random_forest xgboost lightgbm catboost mlp tabpfn --reps 1000
   ```
   Without `--models` only seven models (no TabPFN) are analysed.
8. **Paper numbers.**
   ```
   python scripts/make_paper_numbers.py --run-id c30863ba7974171b
   ```

## Run times of the original run

Total wall-clock time over 15 outer folds, tuning included (`results/paper/runtime_summary.csv`):

| Model | Hours |
|---|---|
| TabPFN | 26.8 |
| Elastic net | 5.3 |
| CatBoost | 1.1 |
| LightGBM | 1.0 |
| MLP | 0.7 |
| XGBoost | 0.65 |
| Random forest | 0.6 |
| Logistic regression | < 0.01 |

Thread settings differed between libraries (16 threads for the tree libraries, 24 Torch threads for TabPFN, one core for the `saga` solver of elastic net), so the times are indicative. The MLP was run with the numerical libraries limited to eight threads:

```
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 BLIS_NUM_THREADS=8 \
python scripts/run_benchmark.py --model mlp --resume
```

## Things to know

- **Do not change `config/benchmarking.yml`.** Its content is part of the run identity; any change gives a new identity and new folds.
- **Do not change `config/tabpfn.yml`, not even its line endings.** Its SHA-256 is stored in every TabPFN checkpoint. If it differs, `--resume` treats the TabPFN folds as not done and runs them again (about 27 hours).
- **Line endings.** All recorded hashes were computed on files with LF line endings. Keep `.gitattributes`; on Windows run `git config core.autocrlf input` before cloning.
- **Floating-point differences.** On other hardware or library versions, results may differ in the last digits. Calibration coefficients are found by numerical optimisation and can differ around 1e-8.
- **Private files.** `local/`, prediction files and fold assignments contain respondent-level data. They are git-ignored and must not be shared.
