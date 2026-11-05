"""Tests of the Phase 01 benchmark code: folds, training-only preprocessing, checkpoints and metrics."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'python'))
from ess_benchmark.core import make_folds,checkpoint_valid,atomic_json,atomic_parquet,digest
from ess_benchmark.models import preprocessing,estimator
from ess_benchmark.metrics import metrics_table,aggregate
from ess_benchmark.runner import train_fold

def data(n=120):
    """Synthetic eligible respondents for the benchmark tests."""
    rng=np.random.default_rng(5)
    df=pd.DataFrame({'cntry':np.repeat(['AT','FI'],n//2),'idno':list(range(n//2))*2,
      'agea':rng.normal(50,13,n),'maritalb':rng.choice(['a','b','c'],n),'alcwkdy':rng.normal(2,1,n),
      'hinctnta':rng.normal(5,2,n), 'd_eds9':np.tile([0,1],n//2),
      'pspwght':np.ones(n),'anweight':np.ones(n),'psu':np.arange(n),'stratum':np.arange(n)//10})
    df.loc[::8,'hinctnta']=np.nan
    return df

def spec():
    """Feature specification for the synthetic data."""
    return pd.DataFrame({'feature':['agea','maritalb','alcwkdy','hinctnta'],'set':['S-1']*4,'measure':['continuous','nominal','continuous','ordinal']})

def test_split_repeat_strata():
    """Each row is in one outer fold per repeat, and folds are stratified."""
    d=data();f=make_folds(d,repeats=2,outer=5)
    assert len(f)==len(d)*2
    assert f.groupby(['repeat','row_index']).size().eq(1).all()
    for (r,k),g in f.groupby(['repeat','outer_fold']):
        assert len(d.iloc[g.row_index].groupby(['cntry','d_eds9']))==4

def test_train_only_median_and_quantile():
    """Medians and alcohol caps are learned from training rows only."""
    d=data();s=spec();cols=s.feature.tolist()
    p=preprocessing(s,'logistic'); p.fit(d.iloc[:60][cols]);x=p.transform(d.iloc[60:][cols])
    assert np.isfinite(x.to_numpy()).all()
    assert 'hinctnta__missing' in x
    assert p.named_steps['matrix'].medians_['agea']==pytest.approx(d.iloc[:60].agea.median())
    assert p.named_steps['clip'].caps_['alcwkdy']==pytest.approx(d.iloc[:60].alcwkdy.quantile(.999))

def test_models_smoke():
    """Every estimator fits and predicts probabilities between 0 and 1."""
    d=data();s=spec();cols=s.feature.tolist()
    for model in ['logistic','elastic_net','random_forest']:
        p=preprocessing(s,model);a=p.fit_transform(d.iloc[:100][cols]); b=p.transform(d.iloc[100:][cols]);
        est=estimator(model,{},42);est.fit(a,d.iloc[:100].d_eds9)
        probs=est.predict_proba(b)[:,1]
        assert len(probs)==20 and np.isfinite(probs).all()

def test_checkpoint_tamper(tmp_path):
    """A checkpoint whose predictions were changed is not valid."""
    pytest.importorskip("pyarrow")
    df=data(40).head(10)
    folder=tmp_path/'one';folder.mkdir()
    pd.DataFrame({'cntry':df.cntry,'idno':df.idno,'y_prob':np.ones(len(df))*.5}).pipe(lambda p:atomic_parquet(folder/'predictions.parquet',p))
    atomic_json(folder/'manifest.json',{'identity':'abc','state':'COMPLETE','prediction_sha256':digest(folder/'predictions.parquet')})
    ids=set(zip(df.cntry.astype(str),df.idno.astype(str)))
    assert checkpoint_valid(folder,'abc',ids)
    atomic_json(folder/'manifest.json',{'identity':'abc','state':'COMPLETE','prediction_sha256':'invalid'})
    assert not checkpoint_valid(folder,'abc',ids)

def test_nested_fold_smoke():
    """One full nested fold runs end to end."""
    d=data();s=spec();pred,p,t,detail=train_fold('logistic',d,s,s.feature.tolist(),np.arange(96),np.arange(96,120),{'inner_folds':3,'n_trials':1},seed=12)
    assert len(pred)==24 and not t and detail['actual_train_rows']==96

def test_weighted_metric_aggregation():
    """Metrics are averaged over countries, then over repeats."""
    d=data(40);d['model']='logistic';d['repeat']=0;d['y_true']=d.d_eds9;d['y_prob']=np.where(d.d_eds9,.8,.2)
    m=metrics_table(d);a=aggregate(m)
    assert len(m)==4 and len(a)==2
    assert a.auroc.eq(1).all()
