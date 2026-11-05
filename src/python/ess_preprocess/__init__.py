"""ESS11 preprocessing: stage 1 (deterministic recoding) and stage 2 (fold-fitted).

Modules:
    recode: stage 1, fixed rules applied to the whole file (outcome,
        eligibility, special codes, routing rules R1-R3, derived variables).

Stage 2 (steps fitted on training folds only) is part of the benchmark, in
:mod:`ess_benchmark.models`.
"""
from .recode import RecodeResult, recode, score_cesd8, eligibility, flow_table, feature_spec, features_for_set

__all__ = ["RecodeResult", "recode", "score_cesd8", "eligibility", "flow_table", "feature_spec",
           "features_for_set"]
