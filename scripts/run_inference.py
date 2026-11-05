#!/usr/bin/env python3
"""Run protocol Rao-Wu cluster AUROC inference for selected completed models.

Usage (from the repository root), as used for the paper (all eight models)::

    python scripts/run_inference.py --run-id c30863ba7974171b \
        --models logistic elastic_net random_forest xgboost lightgbm catboost mlp tabpfn \
        --reps 1000

Note: without ``--models`` only the seven non-TabPFN models are analysed.

The script first calls :func:`ess_benchmark.runner.collect` (which rewrites the
files in ``results/benchmarking/combined/``), then reads ``cntry``, ``stratum``
and ``psu`` of all respondents from the ESS source file and runs
:func:`ess_benchmark.inference.bootstrap_comparison` with elastic net as the
reference.

Outputs in ``results/benchmarking/inference/``:
    * ``<run_id>_auroc.csv``: estimates, CIs, differences, Wald and Holm p-values;
    * ``<run_id>_auroc_replicates.parquet``: replicate AUROC per model
      (no respondent-level data).
"""
import argparse,sys,yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'/'python'))
from ess_benchmark import MODELS
from ess_benchmark.runner import collect
from ess_benchmark.inference import bootstrap_comparison
from ess_benchmark.core import atomic_parquet

DEFAULT_MODELS = ['logistic','elastic_net','random_forest','xgboost','lightgbm','catboost','mlp']

def main():
    """Parse the command line, run the bootstrap and write the results."""
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='config/benchmarking.yml')
    p.add_argument('--run-id',required=True)
    p.add_argument('--reps',type=int,default=1000)
    p.add_argument('--models',nargs='+',choices=MODELS,default=DEFAULT_MODELS)
    a=p.parse_args()
    if a.reps < 1: p.error('--reps must be >= 1')
    if len(set(a.models)) != len(a.models):p.error('--models contains duplicates')
    cfg=yaml.safe_load((ROOT/a.config).read_text())
    predictions=collect(cfg,ROOT,a.run_id,models=a.models)
    import pyreadstat
    raw=ROOT/cfg.get('source_file','data/ESS11e04_2.sav')
    source,_=pyreadstat.read_sav(str(raw),usecols=['cntry','stratum','psu'],apply_value_formats=False)
    output=ROOT/cfg['results_dir']/'inference';output.mkdir(parents=True,exist_ok=True)
    comparison,replicates=bootstrap_comparison(predictions,source,reps=a.reps,metric='auroc')
    comparison.to_csv(output/f'{a.run_id}_auroc.csv',index=False)
    atomic_parquet(output/f'{a.run_id}_auroc_replicates.parquet',replicates)
    print('Models:', ', '.join(a.models))
    print('Wrote design-based AUROC inference:',output)
if __name__=='__main__':main()
