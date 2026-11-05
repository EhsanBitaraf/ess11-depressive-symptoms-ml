"""Tests of the Phase 00 audit functions in src/python/audit.py, on small synthetic data."""
import numpy as np
import pandas as pd
from src.python.audit import cesd8,kish,audit_dataframe,ITEMS

def test_cesd8_reverse_and_extremes():
    """Lowest and highest answers give scores 6 and 18 after reverse-coding."""
    df=pd.DataFrame([{k:1 for k in ITEMS},{k:4 for k in ITEMS}])
    result,invalid,ambiguous=cesd8(df)
    assert result.cesd8_score.tolist()==[6.,18.]
    assert sum(invalid.values())==0
    assert ambiguous==0

def test_prorated_one_vs_two_missing():
    """One missing item is prorated; two missing items give no score."""
    df=pd.DataFrame([{k:2 for k in ITEMS},{k:2 for k in ITEMS}])
    df.loc[0,'fltdpr']=7
    df.loc[1,['fltdpr','flteeff']]=[7,8]
    result,_,_=cesd8(df)
    assert pd.notna(result.loc[0,'cesd8_score'])
    assert pd.isna(result.loc[1,'cesd8_score'])

def test_invalid_ces_response_not_interpreted_as_symptom():
    """Codes outside the valid range are counted as invalid, not scored."""
    df=pd.DataFrame([{k:2 for k in ITEMS}]);df.loc[0,'fltdpr']=66
    scores,invalid,_=cesd8(df)
    assert invalid['fltdpr']==1
    assert bool(scores.loc[0,'prorated'])

def test_kish_equal_weights():
    """With equal weights the Kish effective sample size equals n."""
    assert kish(pd.Series([3.,3.,3.,3.]))==4.0

def test_audit_no_silent_readiness():
    """The audit must not report readiness without a recorded Appendix C sign-off."""
    config={'columns':dict(country='cntry',respondent='idno',age='agea',sex='gndr',weight_post='pspwght',weight_analysis='anweight',psu='psu',stratum='stratum'),
            'country_min_eligible':500,'country_min_cases':100,'country_min_controls':100,
            # Point to a sign-off file that does not exist, so the test does not
            # read the project's real (approved) config/audit_signoff.yaml.
            'signoff_path':'config/__no_signoff_for_tests__.yaml'}
    records=[]
    for i in range(4):
        row={k:2 for k in ITEMS}
        row.update(dict(cntry='AT',idno=i+1,agea=30,gndr=1,pspwght=1.,anweight=1.,psu=i//2+1,stratum=1))
        records.append(row)
    vm=pd.DataFrame([{'name':'fltdpr','role':'outcome','set':'','review_status':'verified_protocol'}])
    result=audit_dataframe(pd.DataFrame(records),config,vm)
    assert result['countries'][0]['eligible_n']==4
    assert any('Appendix C' in x for x in result['issues'])

def test_appendix_c_flags_unasked_substantive_answer():
    """An answer to a question that should not have been asked is flagged."""
    from src.python.audit import appendix_c_diagnostics
    frame = pd.DataFrame({'alcfreq':[7,2,7], 'alcwkdy':[6666,13,25]})
    diagnostic = next(x for x in appendix_c_diagnostics(frame) if x['rule']=='alcfreq -> alcwkdy')
    assert diagnostic['violations']==1
    assert diagnostic['unasked_substantive_n']==1

def test_eligibility_does_not_include_sex_nonresponse():
    """Respondents with sex nonresponse are not eligible."""
    config={'columns':dict(country='cntry',respondent='idno',age='agea',sex='gndr',weight_post='pspwght',weight_analysis='anweight',psu='psu',stratum='stratum'),
            'country_min_eligible':1,'country_min_cases':1,'country_min_controls':1}
    rows=[]
    for i,sex in enumerate([1,9]):
        r={k:2 for k in ITEMS}
        r.update(cntry='AT',idno=i+1,agea=30,gndr=sex,pspwght=1.,anweight=1.,psu=i+1,stratum=1)
        rows.append(r)
    result=audit_dataframe(pd.DataFrame(rows),config,pd.DataFrame([{'variable':'fltdpr','role':'target'}]))
    assert result['countries'][0]['eligible_n']==1


def test_covid_followup_uses_symtc19_not_infection_alone():
    """The COVID follow-up check uses symtc19 as the filter."""
    from src.python.audit import appendix_c_diagnostics
    frame = pd.DataFrame({
        'respc19a': [1, 1, 1, 3],
        'symtc19': [1, 2, 1, 6],
        'symtnc19': [1, 6, 2, 6],
    })
    rule = next(x for x in appendix_c_diagnostics(frame)
                if x['rule'] == 'symtc19 -> symtnc19')
    assert rule['violations'] == 0
    frame.loc[1, 'symtnc19'] = 1
    rule = next(x for x in appendix_c_diagnostics(frame)
                if x['rule'] == 'symtc19 -> symtnc19')
    assert rule['violations'] == 1


def test_singletons_report_source_vs_eligible():
    """Single-PSU strata are labelled as source-created or eligibility-created."""
    from src.python.audit import strata_diagnostics
    frame = pd.DataFrame({
        'cntry': ['AT'] * 4, 'stratum': [1] * 4,
        'psu': [10, 10, 20, 20]
    })
    eligible = pd.Series([True, True, False, False])
    rows = strata_diagnostics(frame, eligible,
        {'country': 'cntry', 'stratum': 'stratum', 'psu': 'psu'})
    assert len(rows) == 1
    assert rows[0]['source_psu'] == 2
    assert rows[0]['eligible_psu'] == 1
    assert rows[0]['origin'] == 'created_by_eligibility'
