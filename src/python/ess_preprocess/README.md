# ess_preprocess

Stage 1 preprocessing of the ESS11 integrated file (edition 4.2) for the elevated depressive symptoms study.

`recode.py` contains fixed rules that do not learn from data. They are applied to the whole file before the cross-validation folds are made.

- Special codes to missing, by variable, from `config/variable_map.csv`.
- CES-D8 score (0-24) with proration of one missing item; outcomes at cut-offs 8, 9 and 10.
- Eligibility flags and participant flow, with missing age and age under 18 shown separately.
- Routing rules of Appendix C:
  - R1: a substantive filter answer governs; the dependent question takes the structural value for its type.
  - R2: the filter is nonresponse but the dependent has a substantive answer; the dependent is kept.
  - R3: a multi-response group conflict; the whole group is missing for that respondent.
- Derived variables listed in `config/derived_variables.csv`.
- Fixed plausibility limits for height, weight, BMI and days of physical activity.

The raw data frame is never modified; a hash check raises an error if it changes.

Stage 2 (steps that learn from data and are fitted on training folds only: alcohol truncation at the training 99.9th percentile, missing indicators, imputation, one-hot encoding and scaling) is part of the benchmark, in `src/python/ess_benchmark/models.py`.

## Usage

```python
from ess_preprocess import recode, features_for_set

res = recode(raw, variable_map, derived_variables)
data = res.data[res.data.eligible]
features = features_for_set(res.feature_spec, "S-3")
```

To run on the real file and write the evidence used by the audit:

```
python scripts/run_preprocess.py --config config/audit.yml
```

Individual-level output goes to `local/analytic_stage1.parquet` (never committed). `results/preprocessing/` contains only aggregates: the recode log, the participant flow, before/after cross-tables for each rule, and missingness by feature and by country.
