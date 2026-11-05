#!/usr/bin/env python3
"""D16 computational feasibility pilot using EXACT frozen benchmarking outer folds.

No outer-test performance or predictions are used. No existing benchmark files are edited.
Run from repository root. This is a pilot ONLY, never the final 15-fold benchmark.

Usage::

    python scripts/run_tabpfn_pilot.py --repeat 0 --outer-fold 0

For one outer fold, each TabPFN strategy of ``config/tabpfn.yml`` is tried in
the configured order, each in a separate worker process. The supervisor
watches wall time, resident memory and swap, and stops a worker that exceeds
a threshold. With ``--strategy auto`` it stops at the first strategy that
passes. Timing uses a "proxy" set drawn from the training fold, of the same
size as the test fold; the outer test rows are never read.

Outputs in ``results/benchmarking/runtime/``:
    * ``tabpfn_D16_<run_id>_r<R>_f<F>_<strategy>.json``: one per strategy tried;
    * ``tabpfn_D16_<run_id>_pilot_decision.json``: summary of this invocation
      (overwritten by each new invocation).

The pilot only runs while ``config/tabpfn.yml`` has ``status: draft``; the
strategy is approved by the PI afterwards (decision D16).
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, pathlib, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'/'python'))

def sha(p):
    """Compute the SHA-256 of a file's bytes.

    Args:
        p (pathlib.Path): File to hash.

    Returns:
        str: Hexadecimal digest.
    """
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def prepare(benchmark, d16, repeat, fold):
    """Load data, predictors and the frozen folds for one outer fold.

    Args:
        benchmark (dict): Benchmark configuration.
        d16 (dict): TabPFN configuration (unused here; kept for symmetry).
        repeat (int): Repeat (0-based).
        fold (int): Outer fold (0-based).

    Returns:
        tuple: ``(run_id, train_frame, spec, features, n_test, input_hashes,
        folds_sha256)``. Only the training rows are returned.

    Raises:
        RuntimeError: If the audit is not approved, the frozen folds are
            missing or corrupted, the eligible count differs, or the predictors
            are invalid. Folds are never regenerated here.
    """
    import numpy as np, pandas as pd
    from ess_benchmark.runner import read_spec,features_for_set
    from ess_benchmark.core import run_id, digest
    audit_path=ROOT/benchmark['audit_summary']; audit=json.loads(audit_path.read_text())
    if audit.get('status')!='APPROVED':raise RuntimeError('Audit is not APPROVED')
    paths={'data':ROOT/benchmark['analytic_data'],'feature_spec':ROOT/benchmark['feature_spec'],'audit':audit_path}
    identity,_=run_id({k:v for k,v in benchmark.items() if k!='synthetic_mode'},paths)
    foldfile=ROOT/benchmark['results_dir']/'folds'/f'{identity}.parquet'
    manifest=foldfile.with_suffix('.json')
    if not foldfile.is_file() or not manifest.is_file() or json.loads(manifest.read_text())['sha256']!=digest(foldfile):
        raise RuntimeError(f'Frozen fold artifact absent or corrupted for run {identity}; DO NOT regenerate folds')
    assignments=pd.read_parquet(foldfile)
    df=pd.read_parquet(paths['data']);df=df.loc[df.eligible.astype(bool)].reset_index(drop=True)
    if len(df)!=benchmark['expected_eligible']:raise RuntimeError('Eligible count mismatch')
    spec=read_spec(paths['feature_spec']); features=features_for_set(spec,benchmark['predictor_set'])
    if not features or not set(features).issubset(df.columns):raise RuntimeError('Feature list invalid')
    forbidden={'d_eds9','d_cesd8_score','cntry','psu','stratum','idno','pspwght','anweight'}
    if forbidden.intersection(features):raise RuntimeError('Leakage feature present')
    holdout=assignments.loc[assignments.repeat.eq(repeat)&assignments.outer_fold.eq(fold),'row_index'].to_numpy(dtype=int)
    train=np.setdiff1d(np.arange(len(df),dtype=int),holdout,assume_unique=True)
    if not len(train) or not len(holdout) or len(np.intersect1d(train,holdout)):
        raise RuntimeError('Invalid fold partition')
    return identity,df.iloc[train].copy().reset_index(drop=True),spec,features,len(holdout), {k:sha(v) for k,v in paths.items()},sha(foldfile)

def stratified_positions(frame,n,seed):
    """Proportional stratified sample by country x outcome (largest remainder).

    Same algorithm as ``ess_benchmark.runner._stratified_positions``.

    Args:
        frame (pandas.DataFrame): Rows with ``cntry`` and ``d_eds9``.
        n (int): Sample size.
        seed (int): Random seed.

    Returns:
        numpy.ndarray: Sorted row positions.

    Raises:
        ValueError: If ``n`` exceeds the number of rows.
    """
    import numpy as np
    if n>len(frame):raise ValueError('Sample larger than pool')
    groups=frame.groupby(['cntry','d_eds9'],dropna=False,sort=True).indices
    keys=sorted(groups,key=str)
    sizes=np.array([len(groups[k]) for k in keys]); quota=n*sizes/len(frame)
    counts=np.floor(quota).astype(int); left=n-int(counts.sum())
    order=sorted(range(len(keys)),key=lambda i:(-(quota[i]-counts[i]),str(keys[i])))
    for i in order[:left]:counts[i]+=1
    rng=np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(groups[k],size=int(counts[j]),replace=False) for j,k in enumerate(keys) if counts[j]]))

def worker(args):
    """Time one TabPFN strategy in this process and print a JSON result line.

    Fits the strategy's subsets on a training pool (training fold minus the
    proxy rows) and predicts the proxy rows. ``A_full`` is rejected without
    fitting when the training fold exceeds the 10,000-sample limit.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        RuntimeError: If versions, limits, the CPU-guard variable or the
            encoded features are not as required.
    """
    import numpy as np, torch
    from tabpfn import TabPFNClassifier
    from tabpfn.inference_config import InferenceConfig
    from ess_benchmark.models import preprocessing
    import pandas as pd
    import yaml
    benchmark=yaml.safe_load((ROOT/args.benchmark_config).read_text())
    cfg=yaml.safe_load((ROOT/args.d16_config).read_text())
    if str(cfg['protocol_version'])!='0.6' or cfg['decision']!='D16':raise RuntimeError('D16/protocol mismatch')
    if importlib.metadata.version('tabpfn')!=str(cfg['software']['tabpfn_version']):raise RuntimeError('TabPFN version mismatch')
    if torch.__version__!=str(cfg['software']['torch_version']):raise RuntimeError('Torch version mismatch')
    if os.getenv('TABPFN_ALLOW_CPU_LARGE_DATASET')!='1':raise RuntimeError('CPU guard override not enabled')
    # TabPFN 9.2.0 defines InferenceConfig as a validated dataclass with
    # required PREPROCESS_TRANSFORMS. Read its class-level default values
    # rather than constructing it with missing required fields.
    max_n=int(InferenceConfig.MAX_NUMBER_OF_SAMPLES)
    max_f=int(InferenceConfig.MAX_NUMBER_OF_FEATURES)
    if max_n!=int(cfg['documented_limits']['max_training_samples']) or max_f!=int(cfg['documented_limits']['max_features']):raise RuntimeError('Library limits changed')
    torch.set_num_threads(int(cfg['hardware']['torch_threads']))
    identity,train,spec,features,n_holdout,hashes,fold_hash=prepare(benchmark,cfg,args.repeat,args.outer_fold)
    # Disjoint training-only proxy, approx the held-out-fold size.
    # The proxy stands in for the test fold when timing predict_proba. Exclusion avoids training/predict overlap.
    proxy_idx=stratified_positions(train,min(n_holdout,len(train)//4),int(cfg['subsampling']['base_seed'])+99101)
    proxy=train.iloc[proxy_idx].copy().reset_index(drop=True)
    pool=train.drop(index=proxy_idx).reset_index(drop=True)
    st=cfg['strategies'][args.strategy]
    n=len(pool) if args.strategy=='A_full' else int(st['n_per_subset'])
    n_sub=1 if args.strategy=='A_full' else int(st['n_subsets'])
    if args.strategy=='A_full':
        # A should use full outer train: infeasible when > pretrained max, known without fitting.
        n=len(train)
    if n>max_n:
        print(json.dumps({'strategy':args.strategy,'passed':False,'reason':f'pretraining_limit_{n}_exceeds_{max_n}', 'run_id':identity}));return
    if n>len(pool):raise RuntimeError('Insufficient training pool after creating disjoint proxy')
    fit_seconds=0.; pred_seconds=0.; enc_count=[]
    started=time.monotonic()
    for sub in range(n_sub):
        seed=int(cfg['subsampling']['base_seed'])+1000*args.repeat+100*args.outer_fold+sub
        ids=stratified_positions(pool,n,seed)
        sample=pool.iloc[ids]
        pre=preprocessing(spec,'tabpfn')
        x=pre.fit_transform(sample[features]); xp=pre.transform(proxy[features])
        if x.shape[1]!=xp.shape[1]:raise RuntimeError('Feature mismatch')
        if x.shape[1]>max_f:
            print(json.dumps({'strategy':args.strategy,'passed':False,'reason':f'encoded_features_{x.shape[1]}_exceeds_{max_f}','run_id':identity}));return
        if x.isna().all(axis=0).any():
            raise RuntimeError('All-NaN feature(s); resolve encoding before pilot')
        enc_count.append(x.shape[1]);clf=TabPFNClassifier(device='cpu',model_path='auto',ignore_pretraining_limits=False,random_state=seed)
        t=time.monotonic();clf.fit(x,sample.d_eds9.astype(int).to_numpy());fit_seconds+=time.monotonic()-t
        t=time.monotonic();batch=int(cfg['prediction']['batch_size'])
        for pos in range(0,len(xp),batch):
            _=clf.predict_proba(xp.iloc[pos:pos+batch])
        pred_seconds+=time.monotonic()-t
        del clf,x,xp,pre
    print(json.dumps({'strategy':args.strategy,'passed':True,'run_id':identity,'repeat':args.repeat,'outer_fold':args.outer_fold,
      'n_outer_train':len(train),'n_outer_test_not_read':n_holdout,'proxy_rows':len(proxy),'train_size_per_subset':n,
      'n_subsets':n_sub,'encoded_features_max':max(enc_count),'fit_seconds':fit_seconds,'predict_seconds':pred_seconds,
      'wall_seconds':time.monotonic()-started,'torch_threads':torch.get_num_threads(),
      'data_hashes':hashes,'frozen_folds_sha256':fold_hash,'note':'Feasibility only; no outer-test metric or prediction'}))

def supervise(args):
    """Start one worker per strategy and enforce the resource thresholds.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        RuntimeError: If the TabPFN configuration is not a draft for
            protocol 0.6.
        SystemExit: With code 2 if no strategy passed.
    """
    import yaml
    cfgpath=ROOT/args.d16_config; cfg=yaml.safe_load(cfgpath.read_text()); benchpath=ROOT/args.benchmark_config
    bench=yaml.safe_load(benchpath.read_text())
    if cfg['status']!='draft':raise RuntimeError('Pilot expects D16 draft, approval occurs after pilot')
    if str(cfg['protocol_version'])!='0.6':raise RuntimeError('D16 must reference v0.6')
    # Resolve identity and verify frozen folds BEFORE launching expensive CPU task.
    identity,train,_,_,n_test,_,_=prepare(bench,cfg,args.repeat,args.outer_fold)
    out=ROOT/bench['results_dir']/'runtime';out.mkdir(parents=True,exist_ok=True)
    import psutil
    results=[]
    for strategy in ([args.strategy] if args.strategy!='auto' else cfg['strategy_order']):
        cmd=[sys.executable,str(pathlib.Path(__file__).resolve()),'--_worker','--strategy',strategy,'--benchmark-config',args.benchmark_config,
             '--d16-config',args.d16_config,'--repeat',str(args.repeat),'--outer-fold',str(args.outer_fold)]
        env=os.environ.copy();env['TABPFN_ALLOW_CPU_LARGE_DATASET']='1'
        t=time.monotonic(); proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        peak_rss=0;peak_swap=0; timeout=False;oom=False
        maxseconds=float(cfg['thresholds']['max_wall_hours_per_outer_fold'])*3600
        maxrss=float(cfg['thresholds']['max_peak_rss_gb'])*1024**3
        while proc.poll() is None:
            try:
                process=psutil.Process(proc.pid)
                rss=process.memory_info().rss; swap=getattr(process.memory_full_info(),'swap',0)
                peak_rss=max(peak_rss,rss);peak_swap=max(peak_swap,swap)
                if rss>maxrss or (not cfg['thresholds']['swap_allowed'] and swap>0):oom=True;proc.kill();break
            except psutil.Error:pass
            if time.monotonic()-t>maxseconds:timeout=True;proc.kill();break
            time.sleep(.25)
        stdout,stderr=proc.communicate(); elapsed=time.monotonic()-t
        try: item=json.loads(stdout.strip().splitlines()[-1])
        except (ValueError,IndexError):item={'strategy':strategy,'passed':False,'reason':'worker_failed','stderr_tail':stderr[-1500:]}
        item.update({'exit_code':proc.returncode,'wall_seconds_observed':elapsed,'peak_rss_gb':round(peak_rss/1024**3,4),
                     'peak_swap_gb':round(peak_swap/1024**3,4),'timed_out':timeout,'resource_limit_exceeded':oom,
                     'pilot_only':True,'d16_config_sha256':sha(cfgpath),'benchmark_config_sha256':sha(benchpath)})
        item['passed']=bool(item.get('passed') and proc.returncode==0 and not timeout and not oom and elapsed<=maxseconds and peak_rss<=maxrss and (cfg['thresholds']['swap_allowed'] or peak_swap==0))
        results.append(item)
        target=out/f'tabpfn_D16_{identity}_r{args.repeat}_f{args.outer_fold}_{strategy}.json'
        target.write_text(json.dumps(item,indent=2,sort_keys=True)+'\n')
        print(f'{strategy}: '+('PASS' if item['passed'] else 'NOT PASS')+f' | {target}',flush=True)
        if args.strategy=='auto' and item['passed']:break
    report={'decision':'D16','protocol_version':'0.6','status':'PILOT_ONLY_REQUIRES_PI_APPROVAL',
            'benchmark_run_id':identity,'pilot_repeat':args.repeat,'pilot_outer_fold':args.outer_fold,
            'outer_test_rows_never_evaluated':n_test,'recommended_strategy':next((r['strategy'] for r in results if r['passed']),None),
            'results':results}
    reportpath=out/f'tabpfn_D16_{identity}_pilot_decision.json'
    reportpath.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('PILOT REPORT:',reportpath)
    print('Candidate strategy (NOT approved):',report['recommended_strategy'])
    if not report['recommended_strategy']:sys.exit(2)

def main():
    """Parse the command line and run the supervisor or a worker."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-config',default='config/benchmarking.yml')
    p.add_argument('--d16-config',default='config/tabpfn.yml')
    p.add_argument('--repeat',type=int,default=0);p.add_argument('--outer-fold',type=int,default=0)
    p.add_argument('--strategy',default='auto',choices=['auto','A_full','B_subsample_ensemble','C_single_subsample','D_small_subsample'])
    p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    (worker if a._worker else supervise)(a)
if __name__=='__main__':main()
