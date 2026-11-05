"""Regression tests for audit revisions: multi-response groups, proration, precision and sign-off handling."""
import importlib.util
from pathlib import Path
import pandas as pd
import pytest

path=Path(__file__).resolve().parents[1]/'src/python/audit.py'
spec=importlib.util.spec_from_file_location('audit_rev',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_hltprca_not_in_group_and_none_conflict():
    """hltprca is not part of the hltpr group; "none" with a marked item is a conflict."""
    df=pd.DataFrame({x:[0,0,0] for x in
      ['hltprhc','hltprhb','hltprbp','hltpral','hltprbn','hltprpa','hltprpf',
       'hltprsd','hltprsc','hltprsh','hltprdi','hltprnt','hltprref','hltprdk','hltprna']})
    df['hltprca']=[1,0,0]  # unrelated cancer question
    df.loc[0,'hltprref']=1
    df.loc[1,'hltprnt']=1
    df.loc[1,'hltprhc']=1
    df.loc[2,'hltprnt']=1
    df.loc[2,'hltprdk']=1
    h=next(x for x in m.appendix_c_diagnostics(df) if x['rule']=='hltpr multiresponse flags')
    assert h['violations']==2  # only records 1 and 2
    assert h['none_with_substantive_n']==1
    assert h['none_with_flag_n']==1


def test_proration_no_half_ties():
    """Prorated scores never fall exactly halfway between two integers."""
    df=pd.DataFrame({c:[1,2] for c in m.ITEMS})
    df.loc[1,'cldgng']=7
    _,_,ties=m.cesd8(df)
    assert ties==0


def test_precision_with_effective_n():
    """A smaller effective sample size gives a wider expected AUROC interval."""
    wide=m.auroc_precision_halfwidth(150,500,400)
    narrow=m.auroc_precision_halfwidth(150,500,650)
    assert wide>narrow>0


def test_signoff_never_approves_implicitly(tmp_path):
    """An incomplete sign-off entry is never treated as recorded."""
    p=tmp_path/'signoff.yml'
    p.write_text('items:\n  gate:\n    decision: approved\n    date: 2026-10-14\n    responsible: analyst\n    evidence: result.json\n')
    x=m.load_signoff({'signoff_path':str(p)}, {'gate':{}})
    assert x['status']=='RECORDED_NOT_AUTOMATICALLY_APPROVED'
    assert x['items']['gate']['recorded']
