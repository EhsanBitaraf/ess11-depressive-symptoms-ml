# Paper claims and their sources

Each statement or number in the MIE 2027 paper is listed with the file that supports it. Paths are relative to the repository root; `<id>` is the run identity `c30863ba7974171b`. Files in `results/paper/` are written by `scripts/make_paper_numbers.py`; `results/paper/key_numbers.csv` lists every single number quoted in the text.

## Data, population and predictors (Methods 2.1)

| Claim in the paper | Value | Source |
|---|---|---|
| Source respondents | 50,116 | `results/preprocessing/flow.csv` |
| Excluded: missing age / aged < 18 / ≥ 2 missing CES-D8 items | 393 / 1,064 / 417 | `results/preprocessing/flow.csv` |
| Eligible respondents | 48,242 | `results/preprocessing/flow.csv`, `results/preprocessing/preprocess_summary.json` |
| Outcome CES-D8 ≥ 9 (unweighted share) | 11,186 (23.2%) | `results/audit/audit_summary.json` (`outcome`) |
| Prorated scores among eligible respondents | 836 | `results/audit/audit_summary.json` (`outcome.eligible_prorated`) |
| Country sample sizes | 659 to 2,734 | `results/paper/country_summary.csv` |
| Audit of identifiers, design fields and skip patterns | all checks PASS, status APPROVED | `results/audit/checks.csv`, `results/audit/audit_summary.json`, `config/audit_signoff.yaml` |
| Routing rules R1–R3 | decision register | `results/audit/routing_decision_register.csv`, `results/preprocessing/recode_log.csv` |
| Predictor set S-3: 91 variables (41 / 48 / 2) | 91 | `results/preprocessing/feature_spec.csv` |
| 144 encoded columns | 144 | `results/benchmarking/checkpoints/<id>/tabpfn/*/*/runtime.json` (`encoded_features`) |
| Variable dictionary and coding rules | | `config/variable_map.csv`, `config/derived_variables.csv`, `src/python/ess_preprocess/recode.py` |

## Algorithms, preprocessing and tuning (Methods 2.2)

| Claim in the paper | Value | Source |
|---|---|---|
| Training-fold preprocessing (99.9th percentile cap, missing indicators > 5% and income, imputation, one-hot, scaling) | | `src/python/ess_benchmark/models.py` (`ClipAlcohol`, `MissingFlags`, `MatrixMaker`) |
| Optuna TPE, 50 trials, three inner folds, log-loss | | `src/python/ess_benchmark/runner.py` (`train_fold`), `config/benchmarking.yml` |
| Search spaces | | `src/python/ess_benchmark/models.py` (`search_params`) |
| Selected hyperparameters per fold | | `results/paper/tuning_best_params.csv`, `results/benchmarking/checkpoints/<id>/*/*/*/tuning.json` |
| 4,500 trials, none failed | 4,500 / 0 | `results/paper/tuning_summary.csv` |
| 120 outer-fold fits completed | 120 | `results/benchmarking/checkpoints/<id>/*/*/*/manifest.json` (`state: COMPLETE`) |
| Software versions | Python 3.12.3, TabPFN 9.2.0, … | `requirements.txt`, `requirements-lock.txt`, `renv.lock` |
| Seeds | | `config/benchmarking.yml` (`seed`), `config/tabpfn.yml` (`seed_rule`) |
| TabPFN feasibility pilot on training data only | strategy B | `results/benchmarking/runtime/tabpfn_D16_<id>_*.json`, `config/tabpfn.yml` |
| Four stratified subsets of 10,000; seed rule | | `config/tabpfn.yml`, `src/python/ess_benchmark/runner.py` (`train_tabpfn_b`) |
| Unique training respondents per fold | 26,874–27,067 (≈70%) | `results/paper/tabpfn_coverage.csv` |

## Validation, measures and inference (Methods 2.3–2.4)

| Claim in the paper | Value | Source |
|---|---|---|
| Five outer folds × three repeats, stratified by country × outcome | | `src/python/ess_benchmark/core.py` (`make_folds`), `results/benchmarking/folds/<id>.json` |
| Unweighted training, weighted evaluation | | `config/benchmarking.yml` (`weighted_training: false`), `src/python/ess_benchmark/metrics.py` |
| Primary estimand θ_m | | `src/python/ess_benchmark/metrics.py` (`aggregate`) |
| Recalibration intercept and slope, fitted jointly | | `src/python/ess_benchmark/metrics.py` (`score`) |
| Hardware | i9-14900K, 31 GiB RAM, no GPU | `docs/hardware.txt` |
| Rao-Wu bootstrap on the full source design, 1,000 replicates, Holm (7 comparisons) | | `src/python/ess_benchmark/inference.py`, `results/benchmarking/inference/<id>_auroc.csv` |
| Python Rao-Wu agrees with R `survey` | | `results/audit/audit_summary.json` (`gate_evidence`) |

## Results

| Claim in the paper | Value | Source |
|---|---|---|
| Table 1 (AUROC with CI, AUPRC, Brier, ΔAUROC vs elastic net with CI, time) | | `results/paper/table1.csv` |
| Highest AUROC: TabPFN | 0.8213 (0.8155–0.8272) | `results/paper/table1.csv` |
| Lowest Brier: XGBoost | 0.1254 | `results/paper/table1.csv` |
| Time TabPFN vs XGBoost | 26.8 h vs 0.65 h (≈41×) | `results/paper/runtime_summary.csv`, `results/paper/key_numbers.csv` |
| H1 supported (lowest lower CI 0.7981, MLP) | | `results/paper/hypotheses.csv` |
| H2 supported for XGBoost, CatBoost, LightGBM | ΔAUROC 0.010–0.012, CIs within (0, 0.03) | `results/paper/hypotheses.csv` |
| Exploratory TabPFN − XGBoost | +0.0008 (−0.0008 to 0.0022) | `results/paper/top_model_contrasts.csv` |
| TabPFN higher than XGBoost in 14 of 30 countries | 14 | `results/paper/top_model_contrasts.csv` |
| Ranking unchanged under unweighted evaluation | True | `results/paper/key_numbers.csv`, `results/benchmarking/combined/<id>_benchmark.csv` |
| XGBoost above elastic net in 29 of 30 countries (not Sweden) | 29 | `results/paper/country_summary.csv`, `results/paper/key_numbers.csv` |
| XGBoost recalibration intercepts −0.73 (Finland) to +0.72 (Sweden) | | `results/paper/country_summary.csv` |
| Country-level metrics for every model and repeat | | `results/benchmarking/combined/<id>_country_metrics.csv` |
