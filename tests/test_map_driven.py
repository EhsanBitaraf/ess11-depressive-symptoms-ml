"""Tests that the audit reads routing rules and multi-response groups from the variable map correctly."""
import importlib.util
from pathlib import Path
import pandas as pd
spec=importlib.util.spec_from_file_location('audit_map',Path(__file__).parents[1]/'src/python/audit.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
vm=m._normalise_variable_map(pd.DataFrame([
 {'variable':'hltprhc','role':'predictor'}, {'variable':'hltprca','role':'derivation_input'},
 {'variable':'hltprref','role':'routing_flag'}, {'variable':'hltprdk','role':'routing_flag'},
 {'variable':'hltprna','role':'routing_flag'}, {'variable':'hltprnt','role':'routing_flag'},
 {'variable':'dshltgp','role':'predictor'}, {'variable':'dshltref','role':'routing_flag'},
 {'variable':'dshltnt','role':'routing_flag'},
 {'variable':'medtrun','role':'derivation_input'},
 {'variable':'medtrnp','role':'derivation_input','routing':'medtrun = 1'},
 {'variable':'cancfre','role':'derivation_input','not_applicable_codes':'6.0'},
 ]))

def test_real_map_and_derived_source_separate():
    """Raw variables and derived variables come from separate files."""
    assert len(vm)==12
    assert m._parse_codes(vm.loc[vm.variable.eq('cancfre'),'not_applicable_codes'].iloc[0])==[6]

def test_hltprca_excluded_dynamic():
    """hltprca is excluded from the hltpr group when groups are read from the map."""
    names=vm.loc[vm.variable.str.startswith('hltpr'),'variable'].tolist()
    df=pd.DataFrame({n:[0,0] for n in names})
    df['hltprca']=[1,1]
    df['hltprref']=[1,0]
    df['hltprhc']=[0,1]
    df['hltprnt']=[0,1]
    r=next(r for r in m.appendix_c_diagnostics(df,vm) if r['rule']=='hltpr multiresponse flags')
    assert r['violations']==1
    assert r['none_with_substantive_n']==1

def test_missing_conditional_fields_are_review_not_pass():
    """A rule whose columns are missing gives REVIEW, not PASS."""
    results=m.appendix_c_diagnostics(pd.DataFrame(),vm)
    assert any(r['rule']=='medtrun -> medtrnp' for r in results)
    assert any(r['rule']=='dshlt multiresponse flags' for r in results)


def test_unasked_binary_zero_not_substantive():
    """A 0 in an unasked binary item is not counted as a substantive answer."""
    vm2=m._normalise_variable_map(pd.DataFrame([
        {'variable':'medtrnp','role':'derivation_input','routing':'medtrun = 1'}]))
    df=pd.DataFrame({'medtrun':[2,2,1,2],'medtrnp':[0,1,0,0]})
    r=next(r for r in m.appendix_c_diagnostics(df,vm2) if r['rule']=='medtrun -> medtrnp')
    assert r['unasked_substantive_n']==1
    assert r['violations']==1

def test_medtrnt_is_an_item_not_none_of_above():
    """medtrnt is a reason item, not a "none of these" option."""
    vm2=m._normalise_variable_map(pd.DataFrame([
        {'variable':'medtrun','role':'derivation_input'},
        {'variable':'medtrnp','role':'derivation_input','routing':'medtrun = 1'},
        {'variable':'medtrnt','role':'derivation_input','routing':'medtrun = 1'},
        {'variable':'medtrref','role':'routing_flag'}]))
    df=pd.DataFrame({'medtrun':[1,1], 'medtrnp':[0,1], 'medtrnt':[1,0], 'medtrref':[0,0]})
    r=next(r for r in m.appendix_c_diagnostics(df,vm2) if r['rule']=='medtr multiresponse flags')
    assert r['violations']==0
    assert r['none_with_substantive_n']==0
    assert r['flag_with_substantive_n']==0
    df.loc[0,'medtrref']=1
    r=next(r for r in m.appendix_c_diagnostics(df,vm2) if r['rule']=='medtr multiresponse flags')
    assert r['flag_with_substantive_n']==1
