"""Run stage-1 preprocessing on the real ESS11 file and write audit evidence.

Usage (from the repository root):
    python scripts/run_preprocess.py --config config/audit.yml

Settings are read from the same config file as the data audit:
    source_file, variable_map, age_min, primary_cutoff, sensitivity_cutoffs, columns
Optional keys (defaults in brackets):
    derived_variables         [derived_variables.csv next to variable_map]
    preprocess_output_dir     [results/preprocessing]
    analytic_data             [local/analytic_stage1.parquet]

Individual-level output goes to analytic_data (keep it in local/, never commit).
Only aggregate tables go to preprocess_output_dir; they are the evidence for the
audit check "Routed predictor recoding independently verified".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))
from ess_preprocess import recode, flow_table  # noqa: E402

# Column names the recoding rules are written for (config "columns" must match).
EXPECTED_COLUMNS = dict(country="cntry", respondent="idno", age="agea", sex="gndr",
                        weight_post="pspwght", weight_analysis="anweight", psu="psu", stratum="stratum")


def sha256(path: Path) -> str:
    """Compute the SHA-256 of a file's bytes.

    Args:
        path (pathlib.Path): File to hash.

    Returns:
        str: Hexadecimal digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str | Path) -> Path:
    """Relative paths in the config are relative to the repository root.

    Args:
        path (str or pathlib.Path): Configured path.

    Returns:
        pathlib.Path: Absolute path.
    """
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: Path) -> dict:
    """Read and check the configuration and fill in default paths.

    Args:
        path (pathlib.Path): Configuration file (``config/audit.yml``).

    Returns:
        dict: Configuration with defaults for ``derived_variables``,
        ``preprocess_output_dir`` and ``analytic_data``.

    Raises:
        KeyError: If a required key is missing.
        ValueError: If the configured column names differ from the names the
            recoding rules use.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in ("source_file", "variable_map", "age_min", "primary_cutoff",
                           "sensitivity_cutoffs", "columns") if k not in cfg]
    if missing:
        raise KeyError(f"{path}: missing config keys: {', '.join(missing)}")
    cols = cfg["columns"] or {}
    wrong = {k: (cols.get(k), v) for k, v in EXPECTED_COLUMNS.items() if cols.get(k) != v}
    if wrong:
        raise ValueError("config 'columns' differ from the names the recoding rules use: "
                         + "; ".join(f"{k}: config={c!r}, expected={e!r}" for k, (c, e) in wrong.items()))
    vm = resolve(cfg["variable_map"])
    cfg.setdefault("derived_variables", str(vm.parent / "derived_variables.csv"))
    cfg.setdefault("preprocess_output_dir", "results/preprocessing")
    cfg.setdefault("analytic_data", "local/analytic_stage1.parquet")
    return cfg


def main() -> None:
    """Run stage 1 and write the analytic data and the aggregate evidence.

    Outputs:
        * ``analytic_data`` (respondent level, keep local),
        * ``recode_log.csv``, ``flow.csv``, ``feature_spec.csv``,
          ``crosstabs/*.csv``, ``feature_missingness*.csv`` and
          ``preprocess_summary.json`` in ``preprocess_output_dir``.

    Raises:
        FileNotFoundError: If an input file is missing.
        RuntimeError: If the raw file changes on disk while it is processed.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="e.g. config/audit.yml")
    a = ap.parse_args()
    cfg = load_config(resolve(a.config))

    data = resolve(cfg["source_file"])
    vm_path = resolve(cfg["variable_map"])
    dv_path = resolve(cfg["derived_variables"])
    out = resolve(cfg["preprocess_output_dir"])
    out_data = resolve(cfg["analytic_data"])
    for p in (data, vm_path, dv_path):
        if not p.exists():
            raise FileNotFoundError(p)
    cutoffs = tuple(sorted({int(cfg["primary_cutoff"]), *map(int, cfg["sensitivity_cutoffs"])}))
    age_min = int(cfg["age_min"])

    import pyreadstat
    raw_hash_before = sha256(data)
    df, _ = pyreadstat.read_sav(str(data), apply_value_formats=False, user_missing=True)
    vm = pd.read_csv(vm_path, keep_default_na=False)
    dv = pd.read_csv(dv_path, keep_default_na=False)

    # Stage 1: fixed rules only; the raw file is hashed before and after.
    res = recode(df, vm, dv, age_min=age_min, cutoffs=cutoffs)
    if sha256(data) != raw_hash_before:
        raise RuntimeError("Raw data file changed on disk during preprocessing")

    (out / "crosstabs").mkdir(parents=True, exist_ok=True)
    res.log.to_csv(out / "recode_log.csv", index=False)
    flow_table(res.data, min_age=age_min).to_csv(out / "flow.csv", index=False)
    res.feature_spec.to_csv(out / "feature_spec.csv", index=False)
    for name, tab in res.crosstabs.items():
        tab.to_csv(out / "crosstabs" / f"{name}.csv")

    elig = res.data[res.data.eligible]
    feats = res.feature_spec.feature.tolist()
    miss = (elig[feats].isna().mean().rename("missing_share_eligible").to_frame()
            .join(res.feature_spec.set_index("feature")[["set", "measure"]]))
    miss.to_csv(out / "feature_missingness.csv")
    elig.groupby("cntry")[feats].apply(lambda g: g.isna().mean()).to_csv(out / "feature_missingness_by_country.csv")

    out_data.parent.mkdir(parents=True, exist_ok=True)
    res.data.to_parquet(out_data, index=False)

    # The hashes recorded here are checked again by the audit cross-check.
    summary = dict(config=str(a.config), protocol_version=cfg.get("protocol_version"),
                   data_edition=cfg.get("data_edition"), raw_file=str(cfg["source_file"]),
                   raw_sha256=raw_hash_before, variable_map_sha256=sha256(vm_path),
                   derived_variables_sha256=sha256(dv_path), output_sha256=sha256(out_data),
                   age_min=age_min, cutoffs=list(cutoffs), primary_cutoff=int(cfg["primary_cutoff"]),
                   records=len(res.data), eligible=int(res.data.eligible.sum()), features=len(feats),
                   rule_totals=res.log.groupby("rule")[["n_all", "n_eligible"]].sum().to_dict("index"))
    (out / "preprocess_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("records", "eligible", "features", "rule_totals")}, indent=2, default=str))


if __name__ == "__main__":
    main()
