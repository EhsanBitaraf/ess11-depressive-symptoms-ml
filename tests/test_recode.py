"""Tests of stage-1 recoding (ess_preprocess.recode) on synthetic respondents built in conftest.py."""
import numpy as np
import pandas as pd
import pytest

from ess_preprocess import recode, flow_table, features_for_set


def run(make_raw, variable_map, derived_variables, *rows):
    """Build raw records and run stage-1 recoding on them."""
    raw = make_raw(*rows)
    return raw, recode(raw, variable_map, derived_variables)


# ---- integrity ---------------------------------------------------------------
def test_raw_unchanged_and_all_features_produced(make_raw, variable_map, derived_variables):
    """The raw input is unchanged and every specified feature is produced."""
    raw = make_raw({}, {"alcfreq": 7, "alcwkdy": 50.0})
    snapshot = raw.copy(deep=True)
    res = recode(raw, variable_map, derived_variables)
    pd.testing.assert_frame_equal(raw, snapshot)
    assert len(res.data) == len(raw)
    assert set(res.feature_spec.feature) <= set(res.data.columns)


def test_default_record_is_clean(make_raw, variable_map, derived_variables):
    """The default synthetic respondent triggers no rule."""
    _, res = run(make_raw, variable_map, derived_variables, {})
    feats = res.feature_spec.feature
    row = res.data.loc[0, feats]
    assert row.notna().all(), row[row.isna()].index.tolist()
    assert res.data.loc[0, "eligible"]


def test_predictor_sets_are_nested(make_raw, variable_map, derived_variables):
    """S-1 is part of S-2, S-2 of S-3 and S-3 of S-4."""
    _, res = run(make_raw, variable_map, derived_variables, {})
    s = [set(features_for_set(res.feature_spec, k)) for k in ("S-1", "S-2", "S-3", "S-4")]
    assert s[0] < s[1] < s[2] < s[3]
    assert {"stflife", "ctrlife"} == s[3] - s[2]
    assert {"health", "hlthhmp"} == s[2] - s[1]
    assert "d_alcohol" not in s[3] and "happy" not in s[3]
    assert not ({"fltdpr", "wrhpp", "medtrun", "hltprca", "respc19a"} & s[3])


# ---- outcome and eligibility --------------------------------------------------
def test_cesd8_scoring_reverse_and_prorating(make_raw, variable_map, derived_variables):
    """CES-D8 reverse-coding and one-item proration are correct."""
    all4 = {v: 4 for v in ["fltdpr", "flteeff", "slprl", "fltlnl", "fltsd", "cldgng"]}
    rows = [
        {},                                                   # 6 x 0 + reversed 1 -> 3 each = 6
        {**all4, "wrhpp": 1, "enjlf": 1},                     # maximum 24
        {**all4, "wrhpp": 1, "enjlf": 8},                     # 7 items sum 21 -> 24
        {"fltdpr": 7, "slprl": 9},                            # two missing -> no score
    ]
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    d = res.data
    assert d.loc[0, "d_cesd8_score"] == 6 and d.loc[1, "d_cesd8_score"] == 24
    assert d.loc[2, "d_cesd8_score"] == 24 and d.loc[2, "cesd8_prorated"]
    assert np.isnan(d.loc[3, "d_cesd8_score"]) and not d.loc[3, "eligible"]
    assert d.loc[1, "d_eds9"] == 1 and d.loc[0, "d_eds9"] == 0


def test_eligibility_and_flow(make_raw, variable_map, derived_variables):
    """Eligibility rules and the participant flow counts are correct."""
    rows = [{}, {"agea": 999}, {"agea": 16}, {"gndr": 9}, {"pspwght": 0.0}]
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    assert res.data.eligible.tolist() == [True, False, False, False, False]
    assert res.data.eligible_15plus.tolist() == [True, False, True, False, False]
    f = flow_table(res.data)
    assert f.excluded.tolist() == [0, 1, 1, 1, 0, 1]


# ---- alcohol (R1, R2) ---------------------------------------------------------
def test_alcohol_rules(make_raw, variable_map, derived_variables):
    """Alcohol rules R1, R2 and the structural zero for never-drinkers."""
    rows = [
        {"alcfreq": 7, "alcwkdy": 6666, "alcwknd": 35.0, "alcbnge": 6},   # R1: never drinker
        {"alcfreq": 77, "alcwkdy": 12.5, "alcwknd": 6666, "alcbnge": 3},  # R2: refused frequency
        {"alcfreq": np.nan, "alcwkdy": 6666, "alcwknd": 6666, "alcbnge": 2},
        {"alcfreq": 2, "alcwkdy": 9.600000381469727, "alcwknd": 8888, "alcbnge": 9},
    ]
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    d = res.data
    assert d.loc[0, ["alcwkdy", "alcwknd", "alcbnge"]].tolist() == [0.0, 0.0, 5.0]
    assert d.loc[1, "alcwkdy"] == 12.5 and np.isnan(d.loc[1, "alcwknd"]) and d.loc[1, "alcbnge"] == 3
    assert np.isnan(d.loc[1, "alcfreq"]) and np.isnan(d.loc[2, "alcfreq"])
    assert d.loc[2, "alcbnge"] == 2 and np.isnan(d.loc[2, "alcwkdy"])
    assert d.loc[3, "alcwkdy"] == 9.6 and np.isnan(d.loc[3, "alcwknd"]) and np.isnan(d.loc[3, "alcbnge"])
    log = res.log
    assert log.loc[log.rule.eq("R1") & log.variable.eq("alcwknd"), "n_all"].sum() == 1


# ---- caregiving -----------------------------------------------------------------
def test_caregiving_order_and_rules(make_raw, variable_map, derived_variables):
    """Caregiving hours are ordered correctly and follow R1 and R2."""
    rows = [{"hlpfmly": 2, "hlpfmhr": 3},     # R1 -> 0 hours
            {"hlpfmly": 1, "hlpfmhr": 55},    # <1 h -> 1
            {"hlpfmly": 1, "hlpfmhr": 1},     # 1-10 h -> 2
            {"hlpfmly": 8, "hlpfmhr": 6},     # R2 -> kept (7)
            {"hlpfmly": 7, "hlpfmhr": 66}]
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    h = res.data.hlpfmhr.tolist()
    assert h[:4] == [0.0, 1.0, 2.0, 7.0] and np.isnan(h[4])
    assert res.data.hlpfmly.tolist()[:3] == [0.0, 1.0, 1.0]


# ---- multi-response (protocol rule and R3) -------------------------------------
def test_multi_response_groups(make_raw, variable_map, derived_variables):
    """R3 conflicts and nonresponse flags make the whole group missing."""
    rows = [
        {"hltprnt": 0, "hltprhc": 1, "hltprdi": 1},                 # clean, two conditions
        {"hltprnt": 0, "hltprref": 1},                              # flag -> group missing
        {"hltprnt": 0, "hltprdk": 1, "hltprhb": 1},                 # R3 flag + item
        {"hltprnt": 1, "hltprhb": 1},                               # R3 none + item
        {"trhltnt": 1, "trhltna": 1},                               # R3 none + flag
        {"hltprca": 1},                                             # cancer is NOT in hltpr
    ]
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    d = res.data
    assert d.loc[0, "d_n_chronic"] == 2 and d.loc[0, "hltprhc"] == 1 and d.loc[0, "hltprhb"] == 0
    for i in (1, 2, 3):
        assert d.loc[i, ["hltprhc", "hltprhb", "d_n_chronic"]].isna().all()
    assert d.loc[4, ["trhltacu", "d_any_comp_treat"]].isna().all()
    assert d.loc[5, "d_n_chronic"] == 0 and d.loc[5, "d_cancer_status"] == "current"


def test_occupational_exposure_and_never_worked(make_raw, variable_map, derived_variables):
    """d_never_worked uses exposure flags, with emplrel as fallback."""
    rows = [
        {"jbexpnt": 0, "jbexpvi": 1},                                     # worked, one exposure
        {"jbexpnt": 0, "jbexpnap": 1, "jbexent": 0, "jbexenap": 1, "emplrel": 6},
        {"jbexpnt": 0, "jbexpnap": 1, "jbexpti": 1, "emplrel": 1},       # R3 conflict -> emplrel
        {"jbexpnt": 0, "jbexpref": 1, "jbexent": 0, "jbexeref": 1, "emplrel": 6},
    ]
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    d = res.data
    assert d.loc[0, "d_n_occ_exp"] == 1 and d.loc[0, "d_never_worked"] == 0
    assert d.loc[1, "d_never_worked"] == 1 and d.loc[1, ["jbexpvi", "jbexevl", "d_n_occ_exp"]].isna().all()
    assert d.loc[1, "emplrel"] == "never_paid_job"
    assert d.loc[2, "d_never_worked"] == 0 and np.isnan(d.loc[2, "jbexpti"])
    assert d.loc[3, "d_never_worked"] == 1


# ---- derived status variables ---------------------------------------------------
def test_covid_status(make_raw, variable_map, derived_variables):
    """COVID status levels, including R1 and R2."""
    rows = [{"respc19a": 3, "symtc19": 2, "symtnc19": 6},     # R1 -> not infected
            {"respc19a": 1, "symtc19": 2, "symtnc19": 2},     # R1 -> no persistent
            {"respc19a": 2, "symtc19": 1, "symtnc19": 1},
            {"respc19a": 1, "symtc19": 1, "symtnc19": 2},
            {"respc19a": 8, "symtc19": 2, "symtnc19": 6},     # R2 -> from chain
            {"respc19a": 9, "symtc19": 6, "symtnc19": 6},
            {"respc19a": 1, "symtc19": 8, "symtnc19": 6}]
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    s = res.data.d_covid_status.tolist()
    assert s[:5] == ["not_infected", "infected_no_persistent", "persistent_current",
                     "persistent_past", "infected_no_persistent"]
    assert pd.isna(s[5]) and pd.isna(s[6])


def test_cancer_status(make_raw, variable_map, derived_variables):
    """Cancer status levels; previous cancer without duration is missing."""
    rows = [{"hltprca": 3, "cancfre": 3}, {"hltprca": 2, "cancfre": 1}, {"hltprca": 2, "cancfre": 8},
            {"hltprca": 7, "cancfre": 6}]
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    s = res.data.d_cancer_status.tolist()
    assert s[0] == "never" and s[1] == "previous_lt5y" and pd.isna(s[2]) and pd.isna(s[3])


def test_health_care_access(make_raw, variable_map, derived_variables):
    """Unmet-need levels; cost takes precedence over other reasons."""
    unable = {"medtrun": 1, "medtrnu": 6, "medtrnap": 0}
    rows = [{"medtrun": 2, "medtrnu": 2}, {"medtrun": 2, "medtrnu": 1},
            {**unable, "medtrnp": 1, "medtrwl": 1},                  # cost has precedence
            {**unable, "medtrwl": 1},
            {**unable, "medtrnu": 1, "medtrnl": 1},                  # R1 -> from reasons
            {**unable, "medtrref": 1},                               # flag -> missing
            {"medtrun": 8, "medtrnu": 1}]
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    s = res.data.d_hc_access.tolist()
    assert s[:5] == ["no_need", "needed_obtained", "unmet_cost", "unmet_other", "unmet_other"]
    assert pd.isna(s[5]) and s[6] == "needed_obtained"


def test_children(make_raw, variable_map, derived_variables):
    """Children in the household from the household grid."""
    rows = [{},                                                   # child in grid
            {"rshipa2": 1, "chldhhe": 1},                         # partner only, had children
            {"rshipa2": 1, "chldhhe": 2},
            {"hhmmb": 1, "rshipa2": 66, "chldhhe": 2},
            {"rshipa2": 99, "chldhhe": 6}]                        # incomplete grid
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    d = res.data
    assert d.d_lives_with_children.tolist()[:4] == [1.0, 0.0, 0.0, 0.0]
    assert d.d_ever_children.tolist()[:4] == [1.0, 1.0, 0.0, 0.0]
    assert np.isnan(d.loc[4, "d_lives_with_children"]) and np.isnan(d.loc[4, "d_ever_children"])


def test_bmi_and_plausibility(make_raw, variable_map, derived_variables):
    """BMI is computed and implausible values are set to missing."""
    rows = [{}, {"height": 110}, {"weighta": 999}, {"height": 200, "weighta": 60}, {"dosprt": 8},
            {"height": 200, "weighta": 31}]                       # BMI 7.75 < 12 -> missing
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    b = res.data.d_bmi
    assert b[0] == pytest.approx(70 / 1.7 ** 2) and res.data.loc[0, "d_bmi_cat"] == "normal"
    assert b[1:3].isna().all()
    assert b[3] == pytest.approx(15.0) and res.data.loc[3, "d_bmi_cat"] == "underweight"
    assert np.isnan(b[5]) and pd.isna(res.data.loc[5, "d_bmi_cat"])
    assert np.isnan(res.data.loc[4, "dosprt"])


def test_special_codes_and_binary(make_raw, variable_map, derived_variables):
    """Special codes become missing and yes/no items become 1/0."""
    rows = [{"gndr": 2, "brncntr": 2, "eisced": 55, "hinctnta": 77, "maritalb": 99, "mnactic": 66},
            {"eiscedf": 0, "ppltrst": 0, "stflife": 88}]
    _, res = run(make_raw, variable_map, derived_variables, *rows)
    d = res.data
    assert d.loc[0, "gndr"] == 1 and d.loc[0, "brncntr"] == 0
    assert d.loc[0, ["eisced", "hinctnta", "maritalb", "mnactic"]].isna().all()
    assert np.isnan(d.loc[1, "eiscedf"]) and d.loc[1, "ppltrst"] == 0 and np.isnan(d.loc[1, "stflife"])
