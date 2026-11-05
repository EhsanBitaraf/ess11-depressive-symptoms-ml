# Changelog

## [1.0.0-mie2027] - 2026-11-05

First public release, accompanying the MIE 2027 paper
"Predicting Elevated Depressive Symptoms Across 30 European Countries: A Comparison of Eight Machine Learning Approaches Using ESS Round 11".

- Phase 00 data audit with signed-off decisions (`config/audit_signoff.yaml`).
- Stage-1 recoding of ESS11 edition 4.2 (`src/python/ess_preprocess`).
- Repeated nested cross-validation of eight algorithms, run `c30863ba7974171b` (`src/python/ess_benchmark`).
- Paired Rao-Wu source-design bootstrap (1,000 replicates).
- Aggregate results, checkpoint manifests and the paper numbers (`results/`, `scripts/make_paper_numbers.py`).
