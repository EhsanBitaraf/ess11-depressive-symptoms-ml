#!/usr/bin/env python3
"""ESS11 model-by-model, outer-fold checkpointed nested CV.

Usage (from the repository root)::

    # run one model (all 15 outer folds), resuming finished folds
    python scripts/run_benchmark.py --model xgboost --resume

    # run every model
    python scripts/run_benchmark.py --all --resume

    # test run: stop after one newly completed fold
    python scripts/run_benchmark.py --model logistic --max-folds 1

    # join finished checkpoints and write metrics
    python scripts/run_benchmark.py --collect c30863ba7974171b

The run identity is printed at the end of a run. See
:mod:`ess_benchmark.runner` for the checkpoint layout.

Warning:
    Without ``--resume`` every fold is re-run. Re-running TabPFN takes about
    27 hours on the reference CPU.
"""
import argparse,sys,yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'/'python'))
from ess_benchmark import MODELS
from ess_benchmark.runner import run,collect

def main():
    """Parse the command line and run or collect the benchmark."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/benchmarking.yml')
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-folds',type=int)
    parser.add_argument('--collect',metavar='RUN_ID',help='Assemble completed model predictions; requires all requested folds')
    args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/args.config).read_text())
    if args.collect:
        models = [args.model] if args.model else [
            "logistic", "elastic_net", "random_forest",
            "xgboost", "lightgbm", "catboost", "mlp", "tabpfn"
        ]
        result=collect(cfg,ROOT,args.collect,models=models)
        print(f'Collected {len(result)} OOF predictions');return
    if not(args.model or args.all):parser.error('Supply --model NAME or --all')
    identity=run(cfg,root=ROOT,model='all' if args.all else args.model,resume=args.resume,max_jobs=args.max_folds)
    print('RUN_ID:',identity)
if __name__=='__main__':main()
