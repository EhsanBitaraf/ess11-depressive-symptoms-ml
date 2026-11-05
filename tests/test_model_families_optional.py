"""Smoke tests of the optional model libraries; skipped when a library is not installed."""
import importlib.util,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'python'))
import numpy as np
import pandas as pd
import pytest
from ess_benchmark.models import estimator,preprocessing

@pytest.mark.parametrize('model,dependency',[('xgboost','xgboost'),('lightgbm','lightgbm'),('catboost','catboost'),('mlp','sklearn')])
def test_optional_model_smoke(model,dependency):
    """Each optional model fits and predicts valid probabilities on small data."""
    if importlib.util.find_spec(dependency) is None:pytest.skip(dependency+' unavailable')
    rng=np.random.default_rng(1)
    d=pd.DataFrame({'agea':rng.normal(size=80),'maritalb':rng.choice(['a','b'],80),'hinctnta':rng.normal(size=80)})
    d.loc[::9,'hinctnta']=np.nan
    spec=pd.DataFrame({'feature':d.columns,'set':['S-3']*3,'measure':['continuous','nominal','ordinal']})
    y=np.tile([0,1],40)
    prep=preprocessing(spec,model)
    a=prep.fit_transform(d.iloc[:64]);b=prep.transform(d.iloc[64:])
    est=estimator(model,{'n_estimators':15},seed=4)
    est.fit(a,y[:64]);p=est.predict_proba(b)[:,1]
    assert len(p)==16 and np.isfinite(p).all()
