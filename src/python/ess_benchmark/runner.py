"""Fold-by-fold independent nested CV with integrity-checked resumability.

Workflow of :func:`run`:

1. Check that the Phase 00 audit is approved and that the eligible sample
   has the expected size.
2. Select the predictors of the configured predictor set and refuse any
   outcome or survey-design field.
3. Compute the run identity and load (or create once) the shared outer folds.
4. For every model, repeat and outer fold: skip it if a valid checkpoint
   exists; otherwise tune on inner folds, fit on the outer training rows,
   predict the outer test rows and write the checkpoint atomically.

:func:`collect` then joins all checkpoints and writes the country-level and
aggregated metrics.

Checkpoint folder layout::

    results/benchmarking/checkpoints/<run_id>/<model>/repeat_RR/fold_FF/
        predictions.parquet  tuning.json  runtime.json  manifest.json

``manifest.json`` is written last; a fold without a ``COMPLETE`` manifest is
treated as not done.
"""
from __future__ import annotations
import json, os, time, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from .__init__ import MODELS
from .core import atomic_json,atomic_parquet,make_folds,inner_splits,run_id,digest,checkpoint_valid
from .models import estimator,preprocessing,search_params
from .metrics import metrics_table,aggregate


def read_spec(path):
    """Read and check the feature specification.

    Args:
        path (str or pathlib.Path): ``feature_spec.csv`` written by the
            preprocessing step.

    Returns:
        pandas.DataFrame: The specification, with at least the columns
        ``feature``, ``set`` and ``measure``.

    Raises:
        ValueError: If a required column is missing or a feature is listed twice.
    """
    spec=pd.read_csv(path)
    needed={'feature','set','measure'}
    if not needed.issubset(spec):raise ValueError('feature_spec lacks '+str(needed-set(spec)))
    if spec.feature.duplicated().any():raise ValueError('Duplicate feature names')
    return spec


def features_for_set(spec,name):
    """List the features of a cumulative predictor set.

    Predictor sets are nested: S-3 contains every feature of S-1, S-2 and S-3.

    Args:
        spec (pandas.DataFrame): Feature specification.
        name (str): ``"S-1"``, ``"S-2"``, ``"S-3"`` or ``"S-4"``.

    Returns:
        list[str]: Feature names, in the order of the specification.

    Raises:
        ValueError: If ``name`` is not a known set.
    """
    order={'S-1':1,'S-2':2,'S-3':3,'S-4':4}
    if name not in order:raise ValueError(name)
    return spec.loc[spec['set'].map(order).le(order[name]),'feature'].tolist()


def _stratified_positions(frame, n, seed):
    """D16: largest-remainder proportional sampling within country x outcome.

    Draws ``n`` rows without replacement so that each country x outcome cell
    gets its proportional share. Rounding uses the largest-remainder method:
    every cell first gets the floor of its exact share, and the remaining rows
    go to the cells with the largest fractional parts (ties broken by cell
    name, so the result is deterministic).

    Args:
        frame (pandas.DataFrame): Training rows with ``cntry`` and ``d_eds9``.
        n (int): Number of rows to draw.
        seed (int): Random seed.

    Returns:
        numpy.ndarray: Sorted row positions in ``frame``.
    """
    groups=frame.groupby(['cntry','d_eds9'],dropna=False,sort=True).indices
    keys=sorted(groups,key=str)
    sizes=np.array([len(groups[k]) for k in keys]); exact=n*sizes/len(frame)
    counts=np.floor(exact).astype(int)
    remainder=n-int(counts.sum())
    order=sorted(range(len(keys)),key=lambda i:(-(exact[i]-counts[i]),str(keys[i])))
    for i in order[:remainder]:counts[i]+=1
    rng=np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(groups[k],int(counts[i]),replace=False)
                         for i,k in enumerate(keys) if counts[i]]))


def _d16_config(root):
    """Load the approved TabPFN configuration (decision D16) and check it.

    The run stops unless the configuration is approved, the installed TabPFN
    and torch versions and the TabPFN pretraining limits equal the recorded
    values, and the sampling and hardware rules match the approved strategy.

    Args:
        root (str or pathlib.Path): Repository root.

    Returns:
        tuple: ``(config, sha256)`` where ``config`` is the parsed
        ``config/tabpfn.yml`` and ``sha256`` is the hash of its bytes. The hash
        is stored in every TabPFN checkpoint.

    Raises:
        RuntimeError: If any check fails.
    """
    import yaml,importlib.metadata
    path=Path(root)/'config'/'tabpfn.yml'
    cfg=yaml.safe_load(path.read_text())
    if cfg.get('status')!='approved' or cfg.get('selected_strategy')!='B_subsample_ensemble' or str(cfg.get('protocol_version'))!='0.6':
        raise RuntimeError('D16 must be approved for B_subsample_ensemble under protocol 0.6')
    if importlib.metadata.version('tabpfn')!=str(cfg['software']['tabpfn_version']):
        raise RuntimeError('TabPFN version not equal to approved D16')
    import torch
    if torch.__version__!=str(cfg['software']['torch_version']):
        raise RuntimeError('Torch version not equal to approved D16')
    # The pretraining limits (10,000 samples, 500 features) are read from the
    # installed library, not only from the configuration file.
    from tabpfn.inference_config import InferenceConfig
    if (int(InferenceConfig.MAX_NUMBER_OF_SAMPLES)!=int(cfg['documented_limits']['max_training_samples'])
        or int(InferenceConfig.MAX_NUMBER_OF_FEATURES)!=int(cfg['documented_limits']['max_features'])):
        raise RuntimeError('TabPFN pretraining limits differ from approved D16')
    st=cfg['strategies']['B_subsample_ensemble']
    if int(st['n_subsets'])!=4 or int(st['n_per_subset'])!=10000:
        raise RuntimeError('D16 must specify exactly four subsets of 10000')
    if cfg['subsampling']['stratify_by']!=['cntry','d_eds9'] or cfg['subsampling']['without_replacement'] is not True:
        raise RuntimeError('D16 stratification rules unexpected')
    if (str(cfg['subsampling']['seed_rule']).replace(' ','')!=
        'base_seed+1000*repeat+100*outer_fold+subset'):
        raise RuntimeError('D16 seed rule unexpected')
    if cfg['hardware']['device']!='cpu' or bool(cfg['hardware']['gpu_used']):
        raise RuntimeError('D16 hardware settings unexpected')
    return cfg,digest(path)


def train_tabpfn_b(df,spec,features,train,test,d16,repeat,outer):
    """D16 strategy B: 4 seeded, stratified training-only subsets; mean probabilities.

    TabPFN accepts at most 10,000 training rows per model. For each of four
    subsets, 10,000 rows are drawn from the outer training fold (stratified by
    country x outcome, without replacement within a subset). Subsets may
    overlap. Each subset gets its own preprocessing and TabPFN model, and the
    test predictions of the four models are averaged.

    Seed of subset ``s``: ``base_seed + 1000 * repeat + 100 * outer + s``.

    Args:
        df (pandas.DataFrame): All eligible respondents.
        spec (pandas.DataFrame): Feature specification.
        features (list[str]): Predictor names.
        train (numpy.ndarray): Row positions of the outer training fold.
        test (numpy.ndarray): Row positions of the outer test fold.
        d16 (dict): Approved configuration from :func:`_d16_config`.
        repeat (int): Cross-validation repeat (0-based).
        outer (int): Outer fold (0-based).

    Returns:
        tuple: ``(probabilities, best_params, trials, details)`` in the same
        form as :func:`train_fold`. ``best_params`` and ``trials`` are empty
        because TabPFN is not tuned; ``details`` records timing, rows per
        subset, encoded feature count and seeds.

    Raises:
        ValueError: If the training fold is smaller than a subset, the batch
            size is invalid, the encoded matrix has too many columns, or a
            prediction is not finite.
    """
    import torch
    from tabpfn import TabPFNClassifier
    torch.set_num_threads(int(d16['hardware']['torch_threads']))
    X=df[features]; y=df.d_eds9.astype(int).to_numpy()
    training_frame=df.iloc[train].reset_index(drop=True)
    n=int(d16['strategies']['B_subsample_ensemble']['n_per_subset'])
    if len(train)<n:raise ValueError('Outer training fold smaller than D16 subset')
    batch=int(d16['prediction']['batch_size'])
    if batch<=0:raise ValueError('D16 batch_size must be positive')
    acc=np.zeros(len(test),dtype=np.float64)
    start=time.perf_counter(); enc=[];seed_list=[]
    for subset in range(4):
        seed=int(d16['subsampling']['base_seed'])+1000*repeat+100*outer+subset
        # Positions inside the training fold -> positions in df.
        positions=_stratified_positions(training_frame,n,seed)
        ids=train[positions]
        # Fit preprocessing only on the selected training subset (not on the test fold).
        prep=preprocessing(spec,'tabpfn')
        x_train=prep.fit_transform(X.iloc[ids]); count=x_train.shape[1]
        if count>int(d16['documented_limits']['max_features']):
            raise ValueError(f'TabPFN features {count} exceed pretrained limit')
        # ignore_pretraining_limits=False keeps the library's own size checks on.
        clf=TabPFNClassifier(device='cpu',model_path='auto',ignore_pretraining_limits=False,random_state=seed)
        clf.fit(x_train,y[ids]);enc.append(count);seed_list.append(seed)
        # Predict in batches to limit memory; add 1/4 of each model's probability.
        for pos in range(0,len(test),batch):
            idx=test[pos:pos+batch]
            prob=clf.predict_proba(prep.transform(X.iloc[idx]))[:,1]
            if not np.isfinite(prob).all():raise ValueError('TabPFN nonfinite prediction')
            acc[pos:pos+len(idx)]+=prob/4.0
        del clf,x_train,prep
    return acc,{},[],{'final_fit_predict_seconds':time.perf_counter()-start,
        'actual_train_rows':n,'aggregate_fit_rows':4*n,'encoded_features':max(enc),
        'subset_seeds':seed_list,'ensemble_members':4,'strategy':'B_subsample_ensemble'}


def train_fold(model,df,spec,features,train,test,config,seed):
    """Tune, fit and predict one model on one outer fold.

    Tuned models run an Optuna TPE search (``config['n_trials']`` trials) that
    minimises the mean log-loss over the inner folds of the training rows.
    The best parameters are then refitted on all outer training rows and used
    to predict the outer test rows. Unpenalised logistic regression is not
    tuned.

    Args:
        model (str): Model name (not ``"tabpfn"``; see :func:`train_tabpfn_b`).
        df (pandas.DataFrame): All eligible respondents.
        spec (pandas.DataFrame): Feature specification.
        features (list[str]): Predictor names.
        train (numpy.ndarray): Row positions of the outer training fold.
        test (numpy.ndarray): Row positions of the outer test fold.
        config (dict): Benchmark configuration.
        seed (int): Seed for inner folds, the sampler and the estimator.

    Returns:
        tuple: ``(probabilities, best_params, trials, details)``:

        * ``probabilities`` (numpy.ndarray): predicted risk for the test rows;
        * ``best_params`` (dict): selected hyperparameters;
        * ``trials`` (list[dict]): every Optuna trial with its value and state;
        * ``details`` (dict): final fit-and-predict time, number of training
          rows and number of encoded columns.
    """
    X=df[features];y=df.d_eds9.astype(int).to_numpy()
    inner=inner_splits(df,train,folds=config['inner_folds'],seed=seed)
    budget=config['n_trials'];tuning=[]
    def build_fit(params,ids):
        """Fit preprocessing and the estimator on the given rows.

        Args:
            params (dict): Hyperparameters.
            ids (numpy.ndarray): Row positions to fit on.

        Returns:
            tuple: ``(fitted_preprocessing, fitted_estimator, n_rows_used)``.
        """
        prep=preprocessing(spec,model)
        x=prep.fit_transform(X.iloc[ids]); est=estimator(model,params,seed)
        # Legacy TabPFN path, kept for completeness. The benchmark trains
        # TabPFN only through train_tabpfn_b (see run()).
        if model in ('tabpfn',) and config.get('tabpfn_max_train_rows'):
            limit=int(config['tabpfn_max_train_rows'])
            if len(ids)>limit:
                # D16: train-only fixed subsample; record this differing training set.
                rng=np.random.default_rng(seed);chosen=np.sort(rng.choice(len(ids),limit,replace=False))
                x=x.iloc[chosen];target=y[ids][chosen]
            else:target=y[ids]
        else:target=y[ids]
        if model=='tabpfn' and x.shape[1]>int(config.get('tabpfn_max_features',10000)):
            raise ValueError('TabPFN encoded features exceed configured limit')
        est.fit(x,target)
        return prep,est,len(target)
    if model not in ('logistic','tabpfn'):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        def objective(trial):
            """Mean inner-fold log-loss of one hyperparameter sample (lower is better)."""
            params=search_params(trial,model)
            vals=[]
            for a,b in inner:
                prep,est,_=build_fit(params,a)
                val=est.predict_proba(prep.transform(X.iloc[b]))[:,1]
                vals.append(log_loss(y[b],np.clip(val,1e-7,1-1e-7),labels=[0,1]))
            return float(np.mean(vals))
        # A seeded TPE sampler makes the search reproducible. catch=() means
        # any failing trial stops the fold instead of being skipped silently.
        study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective,n_trials=budget,catch=())
        params=study.best_params
        tuning=[dict(number=t.number,value=t.value,params=t.params,state=str(t.state)) for t in study.trials]
    else:params={}
    # Final model: refit on all outer training rows with the chosen parameters.
    start=time.perf_counter()
    prep,est,n_train=build_fit(params,train)
    pred=est.predict_proba(prep.transform(X.iloc[test]))[:,1]
    duration=time.perf_counter()-start
    return pred,params,tuning,{'final_fit_predict_seconds':duration,'actual_train_rows':n_train,'encoded_features':len(prep.transform(X.iloc[test[:1]]).columns)}


def run(config, *, root, model, resume=True, max_jobs=None):
    """Run the nested cross-validation benchmark.

    Args:
        config (dict): Parsed ``config/benchmarking.yml``.
        root (str or pathlib.Path): Repository root; all configured paths are
            relative to it.
        model (str): A model name, or ``"all"`` for every model.
        resume (bool): Skip folds that already have a valid checkpoint.
        max_jobs (int, optional): Stop after this many newly completed folds
            (useful for a test run).

    Returns:
        str: The run identity.

    Raises:
        RuntimeError: If the audit is not approved, the D16 checks fail, or
            folds failed.
        ValueError: If the eligible sample size, the predictor list or the
            frozen fold file is not as expected.
    """
    root=Path(root); settings=dict(config)
    data_path=root/config['analytic_data'];spec_path=root/config['feature_spec'];audit_path=root/config['audit_summary']
    # Gate: modelling starts only after the approved Phase 00 audit.
    audit=json.loads(audit_path.read_text())
    if audit.get('status')!='APPROVED':raise RuntimeError('Phase 00 audit not APPROVED')
    if audit.get('metadata',{}).get('protocol') not in (config['protocol_version'], None):
        raise RuntimeError('Audit protocol version mismatch; explicitly record version provenance')
    df=pd.read_parquet(data_path)
    df=df.loc[df.eligible.astype(bool)].reset_index(drop=True)
    if len(df)!=int(config.get('expected_eligible',48242)) and not config.get('synthetic_mode',False):
        raise ValueError('Eligible sample count differs from frozen audit')
    spec=read_spec(spec_path);features=features_for_set(spec,config['predictor_set'])
    if not features or set(features)-set(df):raise ValueError('Feature list empty or missing from data')
    # Leakage guard: the outcome, its score, identifiers and design fields may
    # never be predictors.
    if set(features)&{'d_eds9','d_cesd8_score','cntry','psu','stratum','idno'}:
        raise ValueError('Leakage or survey-design field in predictors')
    fold_config={k:config[k] for k in ('outer_folds','inner_folds','repeats','seed')}
    out=root/config['results_dir'];out.mkdir(parents=True,exist_ok=True)
    frozen_paths={'data':data_path,'feature_spec':spec_path,'audit':audit_path}
    identity,payload=run_id({k:v for k,v in config.items() if k not in ('synthetic_mode',)},frozen_paths)
    # Folds are created once per run identity and then only read; their hash
    # is checked every time.
    fold_path=out/'folds'/f'{identity}.parquet';fold_manifest=out/'folds'/f'{identity}.json'
    if fold_path.exists():
        if not fold_manifest.exists() or json.loads(fold_manifest.read_text()).get('sha256')!=digest(fold_path):
            raise ValueError('Frozen fold artifact corrupted')
        assignments=pd.read_parquet(fold_path)
    else:
        assignments=make_folds(df,repeats=config['repeats'],outer=config['outer_folds'],seed=config['seed'])
        atomic_parquet(fold_path,assignments)
        atomic_json(fold_manifest,{'sha256':digest(fold_path),'identity':identity,'config':fold_config})
    atomic_json(out/'runs'/f'{identity}.json',dict(identity=identity,inputs=payload,models=list(MODELS),status='READY'))
    models=list(MODELS) if model=='all' else [model]
    if any(m not in MODELS for m in models):raise ValueError(model)
    d16=None; d16_hash=None
    if 'tabpfn' in models:
        d16,d16_hash=_d16_config(root)
        if config.get('weighted_training',False):
            raise ValueError('TabPFN weighted fitting not supported')
        # Lifts only TabPFN's CPU speed guard; the pretraining limits stay active.
        os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET']='1'
    done=0;errors=[]
    for m in models:
        if m=='tabpfn' and config.get('weighted_training',False):
            raise ValueError('TabPFN weighted fitting not supported; SA2 excludes TabPFN')
        for repeat in range(config['repeats']):
            for outer in range(config['outer_folds']):
                test=assignments.loc[assignments.repeat.eq(repeat)&assignments.outer_fold.eq(outer),'row_index'].to_numpy(dtype=int)
                train=np.setdiff1d(np.arange(len(df)),test,assume_unique=True)
                loc=out/'checkpoints'/identity/m/f'repeat_{repeat:02d}'/f'fold_{outer:02d}'
                expected=set(zip(df.iloc[test].cntry.astype(str),df.iloc[test].idno.astype(str)))
                # TabPFN checkpoints additionally bind the approved D16 file hash.
                d16_checkpoint_ok=True
                if m=='tabpfn':
                    try:
                        d16_checkpoint_ok=(json.loads((loc/'manifest.json').read_text()).get('d16_sha256')==d16_hash)
                    except (OSError,ValueError):d16_checkpoint_ok=False
                if resume and d16_checkpoint_ok and checkpoint_valid(loc,identity,expected):
                    print(f'SKIP verified {m} r{repeat} f{outer}',flush=True);continue
                # Warning: from here on the fold is re-run. If tabpfn.yml changed
                # (even only its line endings), a finished TabPFN fold is re-run.
                if (loc/'manifest.json').exists():
                    (loc/'manifest.json').unlink() # invalidate previous checkpoint before re-run
                atomic_json(loc/'running.json',{'started_utc':pd.Timestamp.utcnow().isoformat(),'identity':identity})
                started=time.perf_counter()
                try:
                    if m=='tabpfn':
                        pred,params,trials,details=train_tabpfn_b(df,spec,features,train,test,d16,repeat,outer)
                    else:
                        # Fold-specific seed: base seed + 100 * repeat + fold.
                        pred,params,trials,details=train_fold(m,df,spec,features,train,test,config,config['seed']+100*repeat+outer)
                    g=df.iloc[test]
                    # Respondent-level predictions with design fields; these
                    # files stay local and are not published.
                    p=pd.DataFrame({'cntry':g.cntry.to_numpy(),'idno':g.idno.to_numpy(),
                       'psu':g.psu.to_numpy(),'stratum':g.stratum.to_numpy(),
                       'pspwght':g.pspwght.to_numpy(),'anweight':g.anweight.to_numpy(),
                       'y_true':g.d_eds9.astype(int).to_numpy(),'y_prob':pred,
                       'model':m,'predictor_set':config['predictor_set'],'repeat':repeat,'outer_fold':outer})
                    if not np.isfinite(pred).all() or not np.all((pred>=0)&(pred<=1)):raise ValueError('Nonfinite/out-of-range probabilities')
                    # Write order matters: the manifest is written last, so a
                    # crash before it leaves the fold marked as not complete.
                    atomic_parquet(loc/'predictions.parquet',p)
                    atomic_json(loc/'tuning.json',{'best_params':params,'trials':trials})
                    atomic_json(loc/'runtime.json',dict(wall_seconds=time.perf_counter()-started,**details))
                    atomic_json(loc/'manifest.json',{'identity':identity,'model':m,'repeat':repeat,'outer_fold':outer,
                        'state':'COMPLETE','n_test':len(test),'prediction_sha256':digest(loc/'predictions.parquet'),
                        'best_params':params,'trained_rows':len(train),'actual_train_rows':details['actual_train_rows'],
                         **({'d16_sha256':d16_hash,'tabpfn_strategy':'B_subsample_ensemble'} if m=='tabpfn' else {})})
                    (loc/'running.json').unlink(missing_ok=True)
                    done+=1;print(f'COMPLETE {m} r{repeat} f{outer} n={len(test)}',flush=True)
                except Exception as exc:
                    atomic_json(loc/'error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
                    print(f'FAILED {m} r{repeat} f{outer}: {exc}',flush=True)
                    errors.append((m,repeat,outer,str(exc)))
                    if config.get('stop_on_error',True):raise
                if max_jobs is not None and done>=max_jobs:return identity
    if errors:raise RuntimeError(f'{len(errors)} folds failed: {errors[:5]}')
    return identity


def collect(config,root,identity,models=None):
    """Join finished checkpoints and write predictions and metrics.

    Writes, under ``results/benchmarking/combined/``:

    * ``<id>_predictions.parquet``: all out-of-fold predictions (respondent
      level; not for publication);
    * ``<id>_country_metrics.csv``: metrics per model, repeat, country and
      weighting;
    * ``<id>_benchmark.csv``: equal-country summary per model and weighting.

    Args:
        config (dict): Benchmark configuration.
        root (str or pathlib.Path): Repository root.
        identity (str): Run identity.
        models (list[str], optional): Models to collect; default all models.

    Returns:
        pandas.DataFrame: All collected predictions.

    Raises:
        RuntimeError: If a fold has no manifest.
        ValueError: If a checkpoint belongs to another run, its predictions
            hash does not match, or predictions are duplicated.
    """
    root=Path(root);out=root/config['results_dir'];models=MODELS if models is None else models
    frames=[]
    for model in models:
        for r in range(config['repeats']):
            for f in range(config['outer_folds']):
                loc=out/'checkpoints'/identity/model/f'repeat_{r:02d}'/f'fold_{f:02d}'
                if not (loc/'manifest.json').exists():raise RuntimeError(f'Incomplete: {model} r{r} f{f}')
                manifest=json.loads((loc/'manifest.json').read_text());p=loc/'predictions.parquet'
                if manifest.get('identity')!=identity or manifest.get('prediction_sha256')!=digest(p):
                    raise ValueError(f'Checkpoint corrupt: {loc}')
                frames.append(pd.read_parquet(p))
    frame=pd.concat(frames,ignore_index=True)
    if frame.duplicated(['model','repeat','cntry','idno']).any():raise ValueError('Duplicate OOF predictions')
    atomic_parquet(out/'combined'/f'{identity}_predictions.parquet',frame)
    table=metrics_table(frame);table.to_csv(out/'combined'/f'{identity}_country_metrics.csv',index=False)
    aggregate(table).to_csv(out/'combined'/f'{identity}_benchmark.csv',index=False)
    return frame
