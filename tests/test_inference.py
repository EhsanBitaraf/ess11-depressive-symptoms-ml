"""Tests of the paired Rao-Wu bootstrap in ess_benchmark.inference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'python'))
import numpy as np
import pandas as pd
from ess_benchmark.inference import bootstrap_comparison,replicate_psu_weights,_source_structure

def test_rwu_count_and_paired_ci():
    """Rao-Wu draws n-1 PSUs per stratum and the paired comparison output is complete."""
    source=pd.DataFrame([(c,1,p) for c in ('AT','FI') for p in range(1,5)],columns=['cntry','stratum','psu'])
    rng=np.random.default_rng(3)
    samples=[]
    for model in ('elastic_net','logistic'):
        for repeat in (0,1):
            for country in ('AT','FI'):
                for idno in range(60):
                    psu=idno%4+1;y=int(idno%3==0)
                    pr=(.72 if y else .28)+(0.02 if model=='logistic' else 0.)
                    samples.append((model,repeat,country,idno,1,psu,1.,y,pr))
    pred=pd.DataFrame(samples,columns=['model','repeat','cntry','idno','stratum','psu','pspwght','y_true','y_prob'])
    result,rep=bootstrap_comparison(pred,source,reps=30)
    assert len(result)==2 and len(rep)==30
    assert result.set_index('model').loc['elastic_net','difference_vs_reference']==0
    w=replicate_psu_weights(_source_structure(source),np.random.default_rng(2))
    for country in ('AT','FI'):
        assert abs(sum(w[(country,'1',str(p))] for p in range(1,5))-4)<1e-7
