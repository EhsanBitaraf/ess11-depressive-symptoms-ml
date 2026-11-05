"""Synthetic ESS11-like records for preprocessing tests. No real ESS data."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))


@pytest.fixture(scope="session")
def variable_map():
    """Session fixture: the real config/variable_map.csv (no respondent data)."""
    return pd.read_csv(ROOT / "config" / "variable_map.csv", keep_default_na=False)


@pytest.fixture(scope="session")
def derived_variables():
    """Session fixture: the real config/derived_variables.csv."""
    return pd.read_csv(ROOT / "config" / "derived_variables.csv", keep_default_na=False)


def _defaults(variable_map: pd.DataFrame) -> dict:
    """One fully valid, routing-consistent respondent."""
    d = {v: 1 for v in variable_map.variable}           # valid code for most scales
    d.update(cntry="AT", idno=1, psu=1, stratum=1, pspwght=1.0, anweight=1.0, dweight=1.0,
             pweight=1.0, prob=0.001, domain=1, mode=1,
             agea=40, gndr=1, hhmmb=2, dosprt=3, eisced=4, eiscedf=3, eiscedm=3,
             alcfreq=3, alcwkdy=10.0, alcwknd=20.0, alcbnge=4,
             hlpfmly=2, hlpfmhr=66, emplrel=1, chldhhe=6,
             medtrun=2, medtrnu=2, respc19a=3, symtc19=6, symtnc19=6,
             hltprca=3, cancfre=6, height=170, weighta=70)
    for v in variable_map.variable:
        if v.startswith("rshipa"):
            d[v] = 66
    d["rshipa2"] = 2                                     # one child in the household
    from ess_preprocess.recode import MULTI_GROUPS
    for g, s in MULTI_GROUPS.items():
        for i in s["items"]:
            d[i] = 0
        for f in s["flags"]:
            d[f] = 0
        if s["none"]:
            d[s["none"]] = 1
        if s["nap"]:
            d[s["nap"]] = 0
    d["medtrnap"] = 1                                    # reasons not asked (medtrun = 2)
    return d


@pytest.fixture
def make_raw(variable_map):
    """Fixture: factory that builds synthetic raw records.

    Each argument of the returned function is a dict of overrides for one
    respondent; every respondent gets its own idno and PSU.
    """
    base = _defaults(variable_map)

    def _make(*overrides: dict) -> pd.DataFrame:
        """Build one raw record per override dict."""
        rows = []
        for k, o in enumerate(overrides or ({},)):
            r = dict(base)
            r["idno"] = k + 1
            r["psu"] = k + 1
            r.update(o)
            rows.append(r)
        return pd.DataFrame(rows)
    return _make
