#!/usr/bin/env python3
"""Phase 00 data audit command line tool. Never uploads or redistributes ESS microdata.

Usage (from the repository root)::

    python scripts/run_audit.py --config config/audit.yml

Reads the ESS11 file named in the configuration, runs
:func:`src.python.audit.audit_dataframe` and writes ``audit_summary.json`` and
CSV tables to the configured output folder (``results/audit``).

The status is ``APPROVED`` only if every automatic check passes and all five
required human sign-offs in ``config/audit_signoff.yaml`` are recorded as
``approved``; otherwise it is ``REQUIRES_REVIEW``. Without the data file the
status is ``BLOCKED`` and no results are invented.

Exit codes: 0 approved, 1 review needed or error, 2 data file missing.
"""
import argparse,sys,datetime,os,subprocess
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.python.audit import read_config,checksum,load_ess,audit_dataframe,write_results,package_versions

def main():
    """Run the audit and write the results.

    Returns:
        int: Exit code (0 approved, 1 review or failure, 2 data missing).
    """
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='config/audit.yml')
    args=ap.parse_args()
    cfg=read_config(ROOT/args.config)
    target=ROOT/cfg['source_file']; folder=ROOT/cfg['output_dir']
    meta={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'source_edition_claimed':cfg['data_edition'],'protocol':cfg['protocol_version'],
          'environment':package_versions(),'source_exists':target.exists(),'data_sha256':checksum(target) if target.exists() else None}
    # Record the code version, if the folder is a git checkout.
    try: meta['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
    except Exception:meta['git_commit']='not-a-git-checkout'
    if not target.exists():
        out={'status':'BLOCKED','summary':{'records':None,'countries_in_file':None},'checks':[],
             'countries':[],'variable_audit':[],'flow':[],'outcome':{},'routing':[],
             'issues':[f'ESS source missing: {cfg["source_file"]}. Download ESS11 edition 4.2 from ESS Data Portal and store locally.'],
             'metadata':meta}
        print('BLOCKED: ESS dataset not present; no fake audit results were generated.',file=sys.stderr)
        write_results(out,folder);return 2
    try:
        df,sav_meta=load_ess(target)
        vm=pd.read_csv(ROOT/cfg['variable_map'],dtype=str,keep_default_na=False)
        out=audit_dataframe(df,cfg,vm)
        meta['source_rows']=len(df)
        meta['spss_file_label']=str(getattr(sav_meta,'file_label',''))
        meta['spss_mr_metadata_present']=bool(getattr(sav_meta,'missing_ranges',None))
        # PASS only when every technical check passes and all required
        # human approvals are explicitly recorded.
        all_checks_pass = (
            bool(out.get("checks"))
            and all(c["status"] == "PASS" for c in out["checks"])
        )

        required_signoffs = (
            "Domain-design inference validated on eligibility-created singletons",
            "Independent R Rao-Wu variance reference",
            "Synthetic hardware and API feasibility verified",
            "Appendix C routing validated",
            "Routed predictor recoding independently verified",
        )

        signoff_items = out.get("gate_evidence", {}).get("signoff", {}).get("items", {})

        all_signoffs_approved = all(
            signoff_items.get(name, {}).get("recorded") is True
            and str(signoff_items[name].get("decision", "")).strip().lower() == "approved"
            for name in required_signoffs
        )

        all_clear = all_checks_pass and all_signoffs_approved and not out.get("issues")

        out["status"] = "APPROVED" if all_clear else "REQUIRES_REVIEW"
        out["metadata"] = meta

        write_results(out, folder)

        print(f"Audit completed: {out['status']}")
        print("Report:", folder / "audit_summary.json")

        return 0 if all_clear else 1
    except Exception as e:
        out={'status':'BLOCKED','summary':{},'checks':[], 'countries':[],'variable_audit':[],
             'flow':[],'outcome':{},'routing':[], 'issues':[f'Audit failed: {type(e).__name__}: {e}'],'metadata':meta}
        write_results(out,folder)
        print(out['issues'][0],file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
