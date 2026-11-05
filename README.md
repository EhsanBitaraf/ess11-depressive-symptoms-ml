# ess11-depressive-symptoms-ml

Reproducible benchmark of eight machine learning algorithms, including TabPFN, for **elevated depressive symptoms** (CES-D8 ≥ 9) in the **European Social Survey Round 11** (ESS11, 48,242 respondents, 30 countries), with survey-design-aware evaluation.

This repository accompanies the MIE 2027 paper:

> Bitaraf E, Jafarpour M, Katsch F, Jouned A, Stamm T, Duftschmid G. *Predicting Elevated Depressive Symptoms Across 30 European Countries: A Comparison of Eight Machine Learning Approaches Using ESS Round 11.* Medical Informatics Europe (MIE) 2027.

Release `v1.0.0-mie2027` contains the code, configuration and aggregate results of benchmark run **`c30863ba7974171b`**. Every number in the paper can be traced to a file here; see [docs/paper_claims.md](docs/paper_claims.md).

The outcome is a survey-defined symptom indicator, **not a clinical diagnosis**. The models are research tools, not screening instruments.

## What is in the repository

| Folder | Content |
|---|---|
| `config/` | Audit and benchmark settings, the TabPFN configuration (decision D16), the variable map, derived variables and the signed audit decisions |
| `src/python/` | `audit.py` (Phase 00 audit), `ess_preprocess/` (stage-1 recoding), `ess_benchmark/` (folds, models, nested CV, metrics, Rao-Wu bootstrap) |
| `scripts/` | Command-line entry points (`run_*.py`) and `make_paper_numbers.py` |
| `tests/` | Unit tests on synthetic data only |
| `results/` | Aggregate outputs only: audit, preprocessing, benchmark metrics, inference, checkpoint manifests and `results/paper/` |
| `docs/` | Claim-to-file map, reproduction guide, hardware record |
| `00_data_audit.qmd` | Quarto report of the Phase 00 audit |
| `data/` | Empty; instructions for obtaining the ESS file |

No respondent-level data are included. Prediction files, fold assignments and the analytic data set stay on the analyst's computer (see `.gitignore`).

## Methods in brief

- **Data:** ESS11 integrated file, edition 4.2; 50,116 respondents, 48,242 eligible (aged ≥ 18, valid sex, CES-D8 and weights).
- **Predictors:** set S-3, 91 questionnaire variables (144 encoded columns).
- **Algorithms:** unpenalised and elastic-net logistic regression, random forest, XGBoost, LightGBM, CatBoost, multilayer perceptron (scikit-learn) and TabPFN (package 9.2.0, TabPFN-3.5 weights).
- **Validation:** five-fold outer cross-validation repeated three times, stratified by country × outcome; three inner folds and 50 Optuna TPE trials (log-loss) for tuned models; all learned preprocessing fitted on training folds only.
- **TabPFN:** four stratified subsets of 10,000 training respondents per outer fold (≈70% of each training fold), probabilities averaged.
- **Estimand:** equal-country mean of within-country `pspwght`-weighted AUROC, averaged over repeats.
- **Inference:** 1,000 paired Rao-Wu bootstrap replicates on the full source design, restricted to the eligible domain; Holm adjustment for seven comparisons with elastic net. Intervals are conditional on the stored predictions.

## Installation

Python 3.12, CPU only (no GPU was used).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q                # unit tests, synthetic data only
```

`requirements.txt` pins the direct dependencies to the versions of the original run; `requirements-lock.txt` is the full `pip freeze` of that environment.

The R part (an independent check of the Rao-Wu bootstrap with the `survey` package, run from `audit.py`) uses R 4.3.3 and `renv`:

```bash
Rscript -e 'install.packages("renv"); renv::restore()'
```

### TabPFN weights

TabPFN-3.5 weights are **not** included. They are licensed by Prior Labs GmbH under a non-commercial license and are downloaded on first use after you accept the license (see the TabPFN documentation). TabPFN outputs may be used for non-commercial research only. The run used checkpoint `tabpfn-v3.5-20260909.safetensors` (SHA-256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`).

## Reproducing the results

Obtain the ESS file first ([data/README.md](data/README.md)). Then, from the repository root:

```bash
python scripts/run_preprocess.py --config config/audit.yml      # stage-1 recoding
python scripts/run_audit.py      --config config/audit.yml      # Phase 00 audit
python scripts/run_benchmark.py  --all --resume                 # nested CV, all models
python scripts/run_benchmark.py  --collect c30863ba7974171b     # metrics
python scripts/run_inference.py  --run-id c30863ba7974171b \
    --models logistic elastic_net random_forest xgboost lightgbm catboost mlp tabpfn --reps 1000
python scripts/make_paper_numbers.py --run-id c30863ba7974171b  # numbers of the paper
```

`make_paper_numbers.py` runs on the published aggregate files alone, without ESS data, except for `tabpfn_coverage.csv`. Details, run times and pitfalls are in [docs/reproduction.md](docs/reproduction.md).

The run identity `c30863ba7974171b` is a hash of the benchmark configuration and of the analytic data, feature specification and audit summary. A run on the same inputs gives the same identity and the same outer folds.

## Licenses

- **Code** (`src/`, `scripts/`, `tests/`, `config/`): Apache License 2.0, see [LICENSE](LICENSE).
- **Documentation and aggregate results** (`docs/`, `results/`, `00_data_audit.qmd`): Creative Commons Attribution 4.0 International, see [LICENSE-docs](LICENSE-docs).
- **ESS11 data:** not included; ESS ERIC conditions of use apply.
- **TabPFN weights:** not included; Prior Labs non-commercial license applies.

See [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff) for copyright and citation.

## Note on file paths and hashes

`results/audit/audit_summary.json` contains a few absolute paths of the machine on which the audit was run. They are metadata only. The file is kept unchanged because its SHA-256 is part of the run identity.

All hashes recorded in the results were computed on files with LF line endings. `.gitattributes` keeps LF on every operating system; on Windows also run `git config core.autocrlf input`.
