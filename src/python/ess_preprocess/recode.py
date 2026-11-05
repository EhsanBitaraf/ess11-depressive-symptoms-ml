"""Stage 1 preprocessing: deterministic recoding of the ESS11 integrated file.

Everything in this module is a fixed rule that does not learn from the data
(Study Design v0.5, "Data Preprocessing"; Appendix C; rules R1-R3 approved by
the PI on 2026-10-14, pending the team decisions log). It can therefore be
applied to the whole file before folds are created. Steps that learn from the
data (99.9th percentile truncation, imputation, missing indicators, encoding,
scaling, weight rescaling) are fitted on training folds only, in
:mod:`ess_benchmark.models`.

Routing rules
-------------
R1  The filter has a substantive answer that means the dependent question
    should not have been asked: the filter governs and the dependent takes
    the structural value of Appendix C for its variable type.
R2  The filter is nonresponse (refusal, don't know, no answer, system
    missing) but the dependent has a substantive answer: the dependent is
    kept and the filter stays missing.
R3  Multi-response group conflict (a nonresponse flag or "none of these"
    together with a marked item, or "none" together with a nonresponse
    flag): all items of the group are missing for that respondent.

The raw input is never modified; the function works on a copy and checks
this with a hash at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import re
import warnings

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants fixed in the protocol
# --------------------------------------------------------------------------
CESD_ITEMS = ["fltdpr", "flteeff", "slprl", "wrhpp", "fltlnl", "enjlf", "fltsd", "cldgng"]
CESD_REVERSED = {"wrhpp", "enjlf"}
CUTOFFS = (8, 9, 10)

HEIGHT_LIMITS = (120.0, 220.0)   # cm
WEIGHT_LIMITS = (30.0, 250.0)    # kg
BMI_LIMITS = (12.0, 70.0)
DOSPRT_LIMITS = (0, 7)

YES_NO = {1: 1.0, 2: 0.0}        # 1 = yes, 2 = no
NONORDINAL_EDU = {0, 55}         # not harmonisable / other

# Multi-response groups: items, "none of these" item, nonresponse flags,
# not-applicable flag. Items are listed explicitly; a shared prefix is never
# used to find them (hltprca is a separate question).
MULTI_GROUPS: dict[str, dict] = {
    "hltpr": dict(items=["hltprhc", "hltprhb", "hltprbp", "hltpral", "hltprbn", "hltprpa",
                         "hltprpf", "hltprsd", "hltprsc", "hltprsh", "hltprdi"],
                  none="hltprnt", flags=["hltprref", "hltprdk", "hltprna"], nap=None),
    "pacc": dict(items=["paccmoro", "paccdwlr", "pacclift", "paccnbsh", "paccocrw",
                        "paccxhoc", "paccnois", "paccinro"],
                 none="paccnt", flags=["paccref", "paccdk", "paccna"], nap=None),
    "jbexp": dict(items=["jbexpvi", "jbexpti", "jbexpml", "jbexpmc"],
                  none="jbexpnt", flags=["jbexpref", "jbexpdk", "jbexpna"], nap="jbexpnap"),
    "jbexe": dict(items=["jbexevl", "jbexevh", "jbexevc", "jbexera", "jbexecp", "jbexebs"],
                  none="jbexent", flags=["jbexeref", "jbexedk", "jbexena"], nap="jbexenap"),
    "trhlt": dict(items=["trhltacu", "trhltacp", "trhltcm", "trhltch", "trhltos", "trhltho",
                         "trhltht", "trhlthy", "trhltmt", "trhltpt", "trhltre", "trhltsh"],
                  none="trhltnt", flags=["trhltref", "trhltdk", "trhltna"], nap=None),
    "dshlt": dict(items=["dshltgp", "dshltms"],
                  none="dshltnt", flags=["dshltref", "dshltdk", "dshltna"], nap=None),
    # Reasons for unmet need: asked only if medtrun = 1; no "none" option.
    "medtr": dict(items=["medtrnp", "medtrnt", "medtroc", "medtrnl", "medtrwl", "medtrnaa", "medtroth"],
                  none=None, flags=["medtrref", "medtrdk", "medtrna"], nap="medtrnap"),
}
# Household grid: positions present in the ESS11 integrated file (edition 4.2
# codebook lists rshipa2-rshipa13 and rshipa15; no rshipa14, gndr14 or yrbrn14).
# The derivation uses whatever rshipa<k> columns exist in the raw file and
# treats a position that is needed (k <= hhmmb) but absent as unobserved.
HOUSEHOLD_GRID = [f"rshipa{i}" for i in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15)]
GRID_PATTERN = r"^rshipa(\d+)$"

# Category labels of derived nominal variables (stored as strings).
COVID_LEVELS = ["not_infected", "infected_no_persistent", "persistent_past", "persistent_current"]
CANCER_LEVELS = ["never", "current", "previous_lt5y", "previous_5_10y", "previous_gt10y"]
HC_LEVELS = ["no_need", "needed_obtained", "unmet_cost", "unmet_other"]
EMPLREL_LEVELS = {1: "employee", 2: "self_employed", 3: "family_business", 6: "never_paid_job"}
BMI_CAT = [(-np.inf, 18.5, "underweight"), (18.5, 25.0, "normal"),
           (25.0, 30.0, "overweight"), (30.0, np.inf, "obese")]


# --------------------------------------------------------------------------
# Result container and logging
# --------------------------------------------------------------------------
@dataclass
class RecodeResult:
    """Output of :func:`recode`.

    Attributes:
        data (pandas.DataFrame): One row per input record with design fields,
            outcome, eligibility flags and recoded predictors.
        log (pandas.DataFrame): One row per rule application, with the number
            of affected records overall (``n_all``) and among eligible
            respondents (``n_eligible``).
        crosstabs (dict): Before/after cross-tables per routing rule, for
            manual review.
        feature_spec (pandas.DataFrame or None): Feature specification for
            stage 2, if ``derived_variables`` was given.
    """
    data: pd.DataFrame                 # one row per input record, analytic columns
    log: pd.DataFrame                  # one row per rule application
    crosstabs: dict = field(default_factory=dict)   # before/after tables per rule
    feature_spec: pd.DataFrame | None = None


class _Log:
    """Collects one record per applied rule for the recode log."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, rule: str, variable: str, mask, description: str, eligible=None):
        """Record how many rows a rule affected.

        Args:
            rule (str): Rule label, e.g. ``"R1"``, ``"R2"``, ``"R3"``,
                ``"structural"``, ``"implausible"``.
            variable (str): Variable the rule changes.
            mask (array-like of bool): Rows affected by the rule.
            description (str): Plain-language description of the rule.
            eligible (pandas.Series, optional): Eligibility flag; if given, the
                count among eligible respondents is recorded too.
        """
        mask = pd.Series(mask).fillna(False).astype(bool)
        row = dict(rule=rule, variable=variable, n_all=int(mask.sum()), description=description)
        if eligible is not None:
            row["n_eligible"] = int((mask & eligible.reindex(mask.index).fillna(False)).sum())
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        """Return the log as a data frame.

        Returns:
            pandas.DataFrame: Columns ``rule``, ``variable``, ``n_all``,
            ``n_eligible`` and ``description``.
        """
        return pd.DataFrame(self.rows, columns=["rule", "variable", "n_all", "n_eligible", "description"])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _codes(text) -> set[int]:
    """Parse a list of numeric codes such as ``"77 88, 99"``.

    Args:
        text: Codes separated by spaces or commas; empty or NaN gives no codes.

    Returns:
        set[int]: The codes.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)) or str(text).strip() == "":
        return set()
    return {int(float(t)) for t in str(text).replace(",", " ").split()}


class _Codes:
    """Nonresponse and not-applicable codes per variable from variable_map.csv.

    Attributes:
        nonresponse (dict): Variable -> set of nonresponse codes
            (refusal, don't know, no answer).
        not_applicable (dict): Variable -> set of not-applicable codes.
    """

    def __init__(self, variable_map: pd.DataFrame):
        """Read the codes.

        Args:
            variable_map (pandas.DataFrame): ``config/variable_map.csv``.
        """
        vm = variable_map.fillna("")
        self.nonresponse = {r.variable: _codes(r.nonresponse_codes) for r in vm.itertuples()}
        self.not_applicable = {r.variable: _codes(r.not_applicable_codes) for r in vm.itertuples()}

    def special(self, v: str) -> set[int]:
        """All special codes of a variable.

        Args:
            v (str): Variable name.

        Returns:
            set[int]: Nonresponse and not-applicable codes together.
        """
        return self.nonresponse.get(v, set()) | self.not_applicable.get(v, set())


def _num(df: pd.DataFrame, v: str) -> pd.Series:
    """Return a column as float; non-numeric values become NaN.

    Args:
        df (pandas.DataFrame): Raw data.
        v (str): Column name.

    Returns:
        pandas.Series: The column as float.

    Raises:
        KeyError: If the column is not in the file.
    """
    if v not in df.columns:
        raise KeyError(f"Required ESS variable absent: {v}")
    return pd.to_numeric(df[v], errors="coerce").astype(float)


def _isin(s: pd.Series, values: Iterable) -> pd.Series:
    """Element-wise membership test that accepts any iterable (e.g. a set).

    Args:
        s (pandas.Series): Values to test.
        values (Iterable): Allowed values.

    Returns:
        pandas.Series: Boolean mask.
    """
    return s.isin(list(values))


def _hash(df: pd.DataFrame) -> int:
    """Content hash of a data frame, used to prove the raw input was not changed.

    Args:
        df (pandas.DataFrame): Data to hash (index included).

    Returns:
        int: Sum of the row hashes.
    """
    return int(pd.util.hash_pandas_object(df, index=True).sum())


# --------------------------------------------------------------------------
# Outcome and eligibility
# --------------------------------------------------------------------------
def score_cesd8(df: pd.DataFrame, cutoffs: Iterable[int] = CUTOFFS) -> pd.DataFrame:
    """CES-D8 total 0-24 (Appendix B); one missing item is prorated.

    Each item is coded 1-4 in ESS. Items are rescored to 0-3; the two
    positively worded items (``wrhpp``, ``enjlf``) are reversed. With one item
    missing, the score is ``round(sum of 7 items * 8 / 7)``; with two or more
    missing, the score is missing.

    Args:
        df (pandas.DataFrame): Raw data with the eight CES-D8 items.
        cutoffs (Iterable[int]): Cut-offs for the binary outcomes
            ``d_eds<c>`` (score >= c).

    Returns:
        pandas.DataFrame: ``d_cesd8_score``, ``cesd8_answered``,
        ``cesd8_prorated`` and one ``d_eds<c>`` column per cut-off
        (1.0 / 0.0 / NaN).
    """
    items = {}
    for v in CESD_ITEMS:
        x = _num(df, v).where(lambda s: s.isin([1, 2, 3, 4]))
        items[v] = (4 - x) if v in CESD_REVERSED else (x - 1)
    items = pd.DataFrame(items, index=df.index)
    answered = items.notna().sum(axis=1)
    total = items.sum(axis=1, min_count=1)
    # 8*s/7 never has a fractional part of exactly 0.5 for integer s, so the
    # rounding convention cannot matter; half-up is used for transparency.
    prorated = np.floor(total * 8.0 / 7.0 + 0.5)
    score = total.where(answered == 8, prorated.where(answered == 7))
    out = pd.DataFrame({"d_cesd8_score": score, "cesd8_answered": answered,
                        "cesd8_prorated": answered == 7}, index=df.index)
    for c in cutoffs:
        out[f"d_eds{c}"] = (score >= c).astype(float).where(score.notna())
    return out


def eligibility(df: pd.DataFrame, cesd: pd.DataFrame, min_age: int = 18) -> pd.DataFrame:
    """Compute the eligibility flags.

    A respondent is eligible if aged ``min_age``-120, sex is 1 or 2, the
    CES-D8 score is defined and both weights (``pspwght``, ``anweight``) are
    positive.

    Args:
        df (pandas.DataFrame): Raw data.
        cesd (pandas.DataFrame): Output of :func:`score_cesd8`.
        min_age (int): Minimum age.

    Returns:
        pandas.DataFrame: One boolean column per criterion, ``eligible``, and
        ``eligible_15plus`` (same rules with minimum age 15, for sensitivity
        analysis SB13).
    """
    age = _num(df, "agea")
    sex = _num(df, "gndr")
    w_post = _num(df, "pspwght")
    w_ana = _num(df, "anweight")
    age_ok = age.between(min_age, 120)
    sex_ok = sex.isin([1, 2])
    cesd_ok = cesd["d_cesd8_score"].notna()
    w_ok = np.isfinite(w_post) & (w_post > 0) & np.isfinite(w_ana) & (w_ana > 0)
    return pd.DataFrame({
        "elig_age_missing": age.isna() | age.eq(999),
        "elig_age_ok": age_ok, "elig_sex_ok": sex_ok, "elig_cesd8_ok": cesd_ok, "elig_weights_ok": w_ok,
        "eligible": age_ok & sex_ok & cesd_ok & w_ok,
        "eligible_15plus": age.between(15, 120) & sex_ok & cesd_ok & w_ok,
    }, index=df.index)


def flow_table(elig: pd.DataFrame, min_age: int = 18) -> pd.DataFrame:
    """Participant flow with age missing and under 18 shown separately.

    Args:
        elig (pandas.DataFrame): Output of :func:`eligibility`.
        min_age (int): Minimum age, used in the stage label.

    Returns:
        pandas.DataFrame: Columns ``stage``, ``remaining`` and ``excluded``,
        one row per sequential step.
    """
    n = len(elig)
    miss = elig["elig_age_missing"]
    s1 = ~miss
    s2 = s1 & elig["elig_age_ok"]
    s3 = s2 & elig["elig_sex_ok"]
    s4 = s3 & elig["elig_cesd8_ok"]
    s5 = s4 & elig["elig_weights_ok"]
    rows = [("All records", n, 0), ("Age recorded", s1.sum(), n - s1.sum()),
            (f"Aged {min_age} or over", s2.sum(), s1.sum() - s2.sum()),
            ("Sex recorded", s3.sum(), s2.sum() - s3.sum()),
            ("Valid CES-D8 (at most one item missing)", s4.sum(), s3.sum() - s4.sum()),
            ("Valid weights (eligible)", s5.sum(), s4.sum() - s5.sum())]
    return pd.DataFrame(rows, columns=["stage", "remaining", "excluded"]).astype({"remaining": int, "excluded": int})


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def recode(df_raw: pd.DataFrame, variable_map: pd.DataFrame,
           derived_variables: pd.DataFrame | None = None, *,
           age_min: int = 18, cutoffs: Iterable[int] = CUTOFFS) -> RecodeResult:
    """Apply all deterministic rules. Returns analytic columns; raw is untouched.

    Args:
        df_raw (pandas.DataFrame): The ESS11 integrated file as read from the
            ``.sav`` file (numeric codes, no value labels).
        variable_map (pandas.DataFrame): ``config/variable_map.csv``.
        derived_variables (pandas.DataFrame, optional):
            ``config/derived_variables.csv``. If given, the feature
            specification is built and checked.
        age_min (int): Minimum age for eligibility.
        cutoffs (Iterable[int]): CES-D8 cut-offs.

    Returns:
        RecodeResult: Recoded data, rule log, cross-tables and feature
        specification.

    Raises:
        KeyError: If a required ESS variable is missing.
        RuntimeError: If the raw input changed, or a feature in the
            specification was not produced.
    """
    before = _hash(df_raw)
    df = df_raw.copy()
    codes = _Codes(variable_map)
    log = _Log()
    xt: dict[str, pd.DataFrame] = {}

    cesd = score_cesd8(df, cutoffs)
    elig = eligibility(df, cesd, min_age=age_min)
    E = elig["eligible"]
    out = pd.DataFrame(index=df.index)
    for c in ["cntry", "idno", "psu", "stratum", "pspwght", "anweight", "dweight", "mode"]:
        if c in df.columns:
            out[c] = df[c]
    out = pd.concat([out, cesd, elig], axis=1)
    log.add("outcome", "d_cesd8_score", cesd["cesd8_prorated"], "Score prorated (one item missing)", E)

    vm = variable_map.fillna("")
    measure = dict(zip(vm.variable, vm.get("measure", pd.Series("", index=vm.index))))

    # ---- simple predictors (no routing) -------------------------------------
    routed = {"alcfreq", "alcbnge", "alcwkdy", "alcwknd", "hlpfmly", "hlpfmhr", "emplrel"}
    multi_items = {i for g in MULTI_GROUPS.values() for i in g["items"]}
    simple = [r.variable for r in vm.itertuples()
              if r.role in ("predictor", "sensitivity") and r.variable not in routed | multi_items]
    simple_cols = {}
    # Variables without routing: special codes -> missing, plus a few
    # variable-specific plausibility rules.
    for v in simple:
        x = _num(df, v)
        special = codes.special(v)
        bad = _isin(x, special)
        if codes.not_applicable.get(v) and _isin(x, codes.not_applicable[v]).any():
            log.add("NA_without_rule", v, _isin(x, codes.not_applicable[v]),
                    "Not-applicable code in a variable without routing rule -> missing", E)
        x = x.where(~bad)
        if v in ("eisced", "eiscedf", "eiscedm"):
            m = _isin(x, NONORDINAL_EDU)
            log.add("nonordinal_code", v, m, "ES-ISCED 0 or 55 -> missing", E)
            x = x.where(~m)
        if v == "hhmmb":
            m = x.eq(0)
            log.add("implausible", v, m, "Household size 0 (respondent always counts) -> missing", E)
            x = x.where(~m)
        if v == "dosprt":
            m = x.notna() & ~x.between(*DOSPRT_LIMITS)
            log.add("implausible", v, m, "Days of physical activity outside 0-7 -> missing", E)
            x = x.where(~m)
        if measure.get(v) == "binary":
            # Yes/no items: 1 = yes -> 1.0, 2 = no -> 0.0. Sex: male -> 0, female -> 1.
            x = x.map(YES_NO) if v != "gndr" else x.map({1: 0.0, 2: 1.0})
        simple_cols[v] = x
    out = pd.concat([out, pd.DataFrame(simple_cols, index=df.index)], axis=1)

    # ---- employment relation: NA kept as category (Appendix C) --------------
    x = _num(df, "emplrel")
    out["emplrel"] = x.map(EMPLREL_LEVELS)

    with warnings.catch_warnings():
        # columns are added one by one for readability; defragmented below
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        _alcohol(df, out, codes, log, E, xt)
        _caregiving(df, out, codes, log, E, xt)
        groups = _multi_response(df, out, log, E, xt)
        _never_worked(df, out, groups, log, E, xt)
        _covid(df, out, log, E, xt)
        _cancer(df, out, log, E, xt)
        _health_care_access(df, out, groups, log, E, xt)
        _children(df, out, log, E, xt)
        _bmi(df, out, codes, log, E)
    out = out.copy()

    # Safety check: the raw input must be exactly as it was at the start.
    if _hash(df_raw) != before:
        raise RuntimeError("Raw input was modified during recoding")

    spec = feature_spec(variable_map, derived_variables) if derived_variables is not None else None
    if spec is not None:
        absent = [f for f in spec.feature if f not in out.columns]
        if absent:
            raise RuntimeError("Features in the specification not produced: " + ", ".join(absent))
    return RecodeResult(data=out, log=log.frame(), crosstabs=xt, feature_spec=spec)


# --------------------------------------------------------------------------
# Rule implementations
# --------------------------------------------------------------------------
def _crosstab(name, xt, before_a, before_b, after):
    """Store a before/after cross-table for manual review.

    Args:
        name (str): Key in ``xt``.
        xt (dict): Cross-table store; changed in place.
        before_a (pandas.Series): Filter question (raw).
        before_b (pandas.Series): Dependent question (raw).
        after (pandas.Series): Recoded value.
    """
    t = pd.crosstab([before_a.fillna(-1), before_b.fillna(-1)], after.fillna("<NA>").astype(str), dropna=False)
    xt[name] = t


def _alcohol(df, out, codes, log, E, xt):
    """Recode alcohol frequency, grams per drinking day and binge drinking.

    ``alcfreq = 7`` (never drinks) is the filter. Never-drinkers get 0 grams
    and binge frequency 5 (never), also if they reported a value (R1).
    Values reported after a nonresponse filter are kept (R2).

    Args:
        df (pandas.DataFrame): Raw data.
        out (pandas.DataFrame): Output frame; columns are added in place.
        codes (_Codes): Special codes.
        log (_Log): Rule log.
        E (pandas.Series): Eligibility flag.
        xt (dict): Cross-table store.
    """
    freq = _num(df, "alcfreq")
    freq_ok = freq.where(freq.between(1, 7))
    freq_nonresp = ~freq.between(1, 7)          # 77, 88, 99 or system missing
    never = freq.eq(7)
    out["alcfreq"] = freq_ok

    for v in ("alcwkdy", "alcwknd"):
        g = _num(df, v).round(2)               # removes float32 artefacts
        substantive = g.notna() & ~_isin(g, codes.special(v))
        res = g.where(substantive)
        r1 = never
        log.add("R1", v, never & substantive, "alcfreq = 7 with reported grams -> 0 grams", E)
        res = res.mask(r1, 0.0)
        log.add("R2", v, freq_nonresp & substantive, "alcfreq nonresponse, grams reported -> kept", E)
        log.add("structural", v, never & ~substantive, "alcfreq = 7, not asked -> 0 grams", E)
        log.add("missing", v, ~never & ~substantive, "Nonresponse or not applicable outside never-drinkers -> missing", E)
        out[v] = res
        _crosstab(f"alcfreq_x_{v}", xt, freq, g.where(~substantive, -2), res.where(res.isna(), "value"))

    b = _num(df, "alcbnge")
    substantive = b.between(1, 5)
    res = b.where(substantive)
    log.add("structural", "alcbnge", never & ~substantive, "alcfreq = 7, not asked -> 5 (never)", E)
    log.add("R1", "alcbnge", never & substantive & ~b.eq(5), "alcfreq = 7 with binge frequency reported -> 5 (never)", E)
    res = res.mask(never, 5.0)
    log.add("R2", "alcbnge", freq_nonresp & substantive, "alcfreq nonresponse, binge answered -> kept", E)
    out["alcbnge"] = res
    _crosstab("alcfreq_x_alcbnge", xt, freq, b, res)


def _caregiving(df, out, codes, log, E, xt):
    """Recode family caregiving (``hlpfmly``) and its hours (``hlpfmhr``).

    Hours become an ordinal scale 0-7 (0 = no caregiving). Respondents who do
    not care for anyone get 0 hours (R1 / structural); hours reported after a
    nonresponse filter are kept (R2).

    Args:
        df, out, codes, log, E, xt: As in :func:`_alcohol`.
    """
    f = _num(df, "hlpfmly")
    h = _num(df, "hlpfmhr")
    # Ordinal order: 0 none < 55 (<1 h) < 1 (1-10 h) < ... < 6 (>50 h)
    order = {55: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: 7.0}
    substantive = _isin(h, order)
    res = h.map(order)
    no = f.eq(2)
    log.add("R1", "hlpfmhr", no & substantive, "hlpfmly = 2 with hours reported -> 0 hours", E)
    log.add("structural", "hlpfmhr", no & ~substantive, "hlpfmly = 2, not asked -> 0 hours", E)
    res = res.mask(no, 0.0)
    log.add("R2", "hlpfmhr", ~f.isin([1, 2]) & substantive, "hlpfmly nonresponse, hours reported -> kept", E)
    out["hlpfmhr"] = res
    out["hlpfmly"] = f.map(YES_NO)
    _crosstab("hlpfmly_x_hlpfmhr", xt, f, h, res)


def _multi_response(df, out, log, E, xt) -> dict:
    """Protocol rule plus R3. Returns per-group status masks for later rules.

    For each multi-response group, items are 0/1. The whole group is missing
    for a respondent with a nonresponse flag, a not-applicable flag, or a
    conflict (R3), for example "none of these" together with a marked item.
    Count variables (``d_n_housing``, ``d_n_chronic``, ``d_n_occ_exp``) and
    ``d_any_comp_treat`` are derived here.

    Args:
        df (pandas.DataFrame): Raw data.
        out (pandas.DataFrame): Output frame; columns are added in place.
        log (_Log): Rule log.
        E (pandas.Series): Eligibility flag.
        xt (dict): Cross-table store.

    Returns:
        dict: Per group, the masks ``items``, ``marked``, ``flag``, ``none``,
        ``nap``, ``conflict`` and ``missing``.
    """
    status = {}
    for g, spec in MULTI_GROUPS.items():
        items = pd.DataFrame({v: _num(df, v) for v in spec["items"]}, index=df.index)
        marked = items.eq(1).any(axis=1)
        flag = pd.DataFrame({v: _num(df, v).eq(1) for v in spec["flags"]}, index=df.index).any(axis=1)
        none = _num(df, spec["none"]).eq(1) if spec["none"] else pd.Series(False, index=df.index)
        nap = _num(df, spec["nap"]).eq(1) if spec["nap"] else pd.Series(False, index=df.index)
        conflict = (flag & marked) | (none & marked) | (none & flag) | (nap & marked)
        missing = flag | conflict | nap
        log.add("R3", g, conflict, "Group conflict (flag/none/not-applicable with marked item) -> group missing", E)
        log.add("flag", g, flag & ~conflict, "Nonresponse flag -> group missing", E)
        if spec["nap"]:
            log.add("not_applicable", g, nap & ~conflict, "Not-applicable flag -> items missing", E)
        if g != "medtr":                       # medtr reasons only feed d_hc_access
            for v in spec["items"]:
                out[v] = items[v].where(items[v].isin([0, 1])).where(~missing)
        status[g] = dict(items=items, marked=marked, flag=flag, none=none, nap=nap,
                         conflict=conflict, missing=missing)
        xt[f"{g}_flags"] = pd.crosstab([flag, none, nap], marked, rownames=["flag", "none", "nap"], colnames=["any_marked"])

    def count(groups):
        """Sum the items of one or more groups for each respondent."""
        # skipna=False: a count is missing if any of its items is missing
        return pd.concat([out[MULTI_GROUPS[g]["items"]] for g in groups], axis=1).sum(axis=1, skipna=False)

    out["d_n_housing"] = count(["pacc"])
    out["d_n_chronic"] = count(["hltpr"])
    out["d_n_occ_exp"] = count(["jbexp", "jbexe"])
    n_treat = count(["trhlt"])
    out["d_any_comp_treat"] = (n_treat > 0).astype(float).where(n_treat.notna())
    return status


def _never_worked(df, out, groups, log, E, xt):
    """Derive ``d_never_worked`` (never had a paid job).

    The not-applicable flags of the occupational-exposure groups mark people
    who never worked. If those flags conflict or are nonresponse, the
    employment relation (``emplrel = 6``) is used instead.

    Args:
        df, out, log, E, xt: As in :func:`_alcohol`.
        groups (dict): Output of :func:`_multi_response`.
    """
    emp = _num(df, "emplrel")
    nap = groups["jbexp"]["nap"] | groups["jbexe"]["nap"]
    conflict = groups["jbexp"]["conflict"] | groups["jbexe"]["conflict"]
    flagged = (groups["jbexp"]["flag"] | groups["jbexe"]["flag"]) & ~nap
    res = pd.Series(np.where(nap, 1.0, 0.0), index=df.index)
    # If the occupational flags conflict, or both are nonresponse, use the
    # employment relation (emplrel 6 = never had a paid job).
    from_emp = emp.map({1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0})
    undetermined = conflict | flagged
    res = res.mask(undetermined, from_emp)
    log.add("R3", "d_never_worked", conflict, "Occupational-exposure conflict -> derived from emplrel", E)
    log.add("flag", "d_never_worked", flagged & ~conflict, "Exposure nonresponse -> derived from emplrel", E)
    log.add("check", "d_never_worked", (res.eq(1) & emp.isin([1, 2, 3])) | (res.eq(0) & emp.eq(6)),
            "Disagreement with emplrel (reported, not changed)", E)
    out["d_never_worked"] = res
    xt["never_worked_x_emplrel"] = pd.crosstab(res.fillna(-1), emp.fillna(-1))


def _covid(df, out, log, E, xt):
    """Derive ``d_covid_status`` from infection and persistent-symptom questions.

    Levels: not infected, infected without persistent symptoms, persistent
    symptoms in the past, persistent symptoms now.

    Args:
        df, out, log, E, xt: As in :func:`_alcohol`.
    """
    a = _num(df, "respc19a")
    s = _num(df, "symtc19")
    c = _num(df, "symtnc19")
    res = pd.Series(np.nan, index=df.index, dtype=object)
    infected = a.isin([1, 2])
    a_nonresp = ~a.isin([1, 2, 3])
    # persistent-symptom chain, used for infected and (R2) for nonresponse
    chain = pd.Series(np.nan, index=df.index, dtype=object)
    chain[s.eq(2)] = "infected_no_persistent"
    chain[s.eq(1) & c.eq(1)] = "persistent_current"
    chain[s.eq(1) & c.eq(2)] = "persistent_past"
    res[infected] = chain[infected]
    log.add("R1", "d_covid_status", s.eq(2) & c.isin([1, 2]),
            "symtc19 = 2 with symtnc19 answered -> infected, no persistent symptoms", E)
    log.add("R1", "d_covid_status", a.eq(3) & s.isin([1, 2]), "respc19a = 3 with symtc19 answered -> not infected", E)
    res[a.eq(3)] = "not_infected"
    r2 = a_nonresp & chain.notna()
    log.add("R2", "d_covid_status", r2, "respc19a nonresponse, symptom chain answered -> derived from chain", E)
    res[r2] = chain[r2]
    out["d_covid_status"] = res
    _crosstab("covid", xt, a, s, res)


def _cancer(df, out, log, E, xt):
    """Derive ``d_cancer_status`` from cancer history and time since cancer.

    Previous cancer without a known duration is set to missing (decision D38).

    Args:
        df, out, log, E, xt: As in :func:`_alcohol`.
    """
    h = _num(df, "hltprca")
    f = _num(df, "cancfre")
    dur = f.map({1: "previous_lt5y", 2: "previous_5_10y", 3: "previous_gt10y"})
    res = pd.Series(np.nan, index=df.index, dtype=object)
    res[h.eq(3)] = "never"
    res[h.eq(1)] = "current"
    res[h.eq(2)] = dur[h.eq(2)]
    log.add("R1", "d_cancer_status", h.isin([1, 3]) & dur.notna(), "hltprca 1 or 3 with duration reported -> filter governs", E)
    r2 = ~h.isin([1, 2, 3]) & dur.notna()
    log.add("R2", "d_cancer_status", r2, "hltprca nonresponse, duration reported -> previous cancer", E)
    res[r2] = dur[r2]
    log.add("missing", "d_cancer_status", h.eq(2) & dur.isna(), "Previous cancer, duration unknown -> missing", E)
    out["d_cancer_status"] = res
    _crosstab("cancer", xt, h, f, res)


def _health_care_access(df, out, groups, log, E, xt):
    """Derive ``d_hc_access`` (unmet need for medical care and its reason).

    Levels: no need, need met, unmet because of cost, unmet for another
    reason. Cost takes precedence when several reasons are marked.

    Args:
        df, out, log, E, xt: As in :func:`_alcohol`.
        groups (dict): Output of :func:`_multi_response` (``medtr`` reasons).
    """
    u = _num(df, "medtrun")
    nu = _num(df, "medtrnu")
    g = groups["medtr"]
    items = g["items"]
    cost = items["medtrnp"].eq(1)
    other = items.drop(columns="medtrnp").eq(1).any(axis=1)
    from_reasons = pd.Series(np.nan, index=df.index, dtype=object)
    from_reasons[other] = "unmet_other"
    from_reasons[cost] = "unmet_cost"              # cost takes precedence
    from_reasons[g["flag"] | g["conflict"]] = np.nan
    from_nu = nu.map({1: "needed_obtained", 2: "no_need"})

    res = pd.Series(np.nan, index=df.index, dtype=object)
    unable = u.eq(1)
    res[unable] = from_reasons[unable]
    log.add("R1", "d_hc_access", unable & from_nu.notna(), "medtrun = 1 with medtrnu answered -> derived from reasons", E)
    log.add("missing", "d_hc_access", unable & from_reasons.isna(), "Unmet need without valid reason -> missing", E)
    asked_nu = u.isin([2, 8])
    res[asked_nu] = from_nu[asked_nu]
    log.add("R1", "d_hc_access", u.eq(2) & (cost | other), "medtrun = 2 with reasons marked -> derived from medtrnu", E)
    nonresp = ~u.isin([1, 2])
    r2 = nonresp & res.isna() & from_reasons.notna()
    log.add("R2", "d_hc_access", r2 | (u.eq(8) & from_nu.notna()), "medtrun nonresponse, dependent answered -> kept", E)
    res[r2] = from_reasons[r2]
    out["d_hc_access"] = res
    _crosstab("hc_access", xt, u, nu, res)


def _children(df, out, log, E, xt):
    """Derive ``d_lives_with_children`` and ``d_ever_children``.

    A child in the household is a grid member with relationship code 2
    (son or daughter). The value is missing when no child is listed but the
    grid is incomplete or the household size is unknown. ESS11 has no grid
    position 14; a needed position that is absent counts as unobserved.

    Args:
        df, out, log, E, xt: As in :func:`_alcohol`.
    """
    present = sorted((int(m.group(1)), c) for c in df.columns
                     if (m := re.match(GRID_PATTERN, str(c))))
    positions = [k for k, _ in present]
    grid = pd.DataFrame({c: _num(df, c) for _, c in present}, index=df.index)
    child = grid.eq(2).any(axis=1)
    unknown = grid.isin([77, 88, 99]).any(axis=1)
    hh = _num(df, "hhmmb")
    # Needed grid positions are 2..hhmmb; any of them absent from the file is unobserved.
    absent = [k for k in range(2, int(np.nanmax(hh.where(hh < 77)) if hh.notna().any() else 2) + 1)
              if k not in positions]
    unobserved = pd.Series(False, index=df.index)
    for k in absent:
        unobserved |= hh.ge(k) & hh.lt(77)
    # A listed member with system-missing relationship is also unobserved.
    for k, c in present:
        unobserved |= hh.ge(k) & hh.lt(77) & grid[c].isna()
    hh_unknown = ~hh.between(1, 76)          # nonresponse (77/88/99), system missing or implausible 0
    lives = pd.Series(np.where(child, 1.0, 0.0), index=df.index)
    lives = lives.mask(~child & (unknown | unobserved | hh_unknown), np.nan)
    lives = lives.mask(hh.eq(1), 0.0)
    log.add("check", "d_lives_with_children", ~child & hh_unknown,
            "Household size unknown or 0 and no child listed -> missing", E)
    log.add("check", "d_lives_with_children", unobserved,
            f"Needed grid position absent or empty (positions in file: {positions}; absent: {absent})", E)
    log.add("missing", "d_lives_with_children", lives.isna(), "Household grid incomplete, no child listed -> missing", E)
    out["d_lives_with_children"] = lives
    hh_label = hh.where(hh.between(0, 76)).fillna(-1)          # -1 = nonresponse or system missing
    xt["household_grid_coverage"] = pd.crosstab([hh_label, child], unobserved,
                                                 rownames=["hhmmb (-1 = nonresponse)", "child_listed"],
                                                 colnames=["unobserved"])

    ch = _num(df, "chldhhe")
    ever = ch.map({1: 1.0, 2: 0.0})
    ever = ever.mask(lives.eq(1), 1.0)
    inconsistent = lives.eq(0) & ch.eq(6)
    log.add("check", "d_ever_children", inconsistent, "Grid lists no child but chldhhe not applicable -> missing", E)
    ever = ever.mask(inconsistent, np.nan)
    out["d_ever_children"] = ever
    xt["children"] = pd.crosstab(lives.fillna(-1), ch.fillna(-1))


def _bmi(df, out, codes, log, E):
    """Derive BMI and BMI category from height and weight.

    Implausible height (outside 120-220 cm), weight (outside 30-250 kg) and
    BMI (outside 12-70) are set to missing.

    Args:
        df, out, codes, log, E: As in :func:`_alcohol`.
    """
    h = _num(df, "height")
    w = _num(df, "weighta")
    h = h.where(~_isin(h, codes.special("height")))
    w = w.where(~_isin(w, codes.special("weighta")))
    bad_h = h.notna() & ~h.between(*HEIGHT_LIMITS)
    bad_w = w.notna() & ~w.between(*WEIGHT_LIMITS)
    log.add("implausible", "height", bad_h, "Height outside 120-220 cm -> missing", E)
    log.add("implausible", "weighta", bad_w, "Weight outside 30-250 kg -> missing", E)
    h, w = h.where(~bad_h), w.where(~bad_w)
    bmi = w / (h / 100.0) ** 2
    bad_b = bmi.notna() & ~bmi.between(*BMI_LIMITS)
    log.add("implausible", "d_bmi", bad_b, "BMI outside 12-70 -> missing", E)
    bmi = bmi.where(~bad_b)
    out["d_bmi"] = bmi
    cat = pd.Series(np.nan, index=df.index, dtype=object)
    for lo, hi, lab in BMI_CAT:
        cat[(bmi >= lo) & (bmi < hi)] = lab
    out["d_bmi_cat"] = cat


# --------------------------------------------------------------------------
# Feature specification for stage 2
# --------------------------------------------------------------------------
_SET_ORDER = {"S-1": 1, "S-2": 2, "S-3": 3, "S-4": 4}


def feature_spec(variable_map: pd.DataFrame, derived_variables: pd.DataFrame) -> pd.DataFrame:
    """Build the feature specification used by stage 2.

    Args:
        variable_map (pandas.DataFrame): ``config/variable_map.csv``.
        derived_variables (pandas.DataFrame): ``config/derived_variables.csv``.

    Returns:
        pandas.DataFrame: Columns ``feature``, ``set``, ``measure``
        (continuous, count, ordinal, binary or nominal), ``source`` (raw or
        derived) and ``set_rank``, sorted by set and name.
    """
    vm = variable_map.fillna("")
    dv = derived_variables.fillna("")
    raw = vm[vm.role.isin(["predictor", "sensitivity"]) & vm.set.ne("")]
    rows = [dict(feature=r.variable, set=r.set, measure=r.measure, source="raw") for r in raw.itertuples()]
    type_map = {"binary": "binary", "count": "count", "continuous": "continuous", "categorical": "nominal"}
    for r in dv[dv.role.eq("predictor")].itertuples():
        rows.append(dict(feature=r.variable, set=r.set, measure=type_map.get(r.type, r.type), source="derived"))
    spec = pd.DataFrame(rows)
    spec.loc[spec.feature.eq("emplrel"), "measure"] = "nominal"
    spec["set_rank"] = spec.set.map(_SET_ORDER)
    return spec.sort_values(["set_rank", "feature"]).reset_index(drop=True)


def features_for_set(spec: pd.DataFrame, predictor_set: str) -> list[str]:
    """Hierarchical sets: S-2 includes S-1, S-3 includes S-2, S-4 includes S-3.

    Args:
        spec (pandas.DataFrame): Output of :func:`feature_spec`.
        predictor_set (str): ``"S-1"`` to ``"S-4"``.

    Returns:
        list[str]: Feature names of the set.
    """
    rank = _SET_ORDER[predictor_set]
    return spec.loc[spec.set_rank <= rank, "feature"].tolist()
