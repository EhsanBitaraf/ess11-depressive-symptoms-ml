#!/usr/bin/env python3
"""Recompute every number reported in the MIE 2027 paper from the stored results.

Usage (from the repository root)::

    python scripts/make_paper_numbers.py --run-id c30863ba7974171b

Most outputs need only the aggregate files that are published in this
repository:

* ``results/benchmarking/combined/<run_id>_benchmark.csv``
* ``results/benchmarking/combined/<run_id>_country_metrics.csv``
* ``results/benchmarking/inference/<run_id>_auroc.csv``
* ``results/benchmarking/inference/<run_id>_auroc_replicates.parquet``
* ``results/benchmarking/checkpoints/<run_id>/**/runtime.json`` and ``tuning.json``
* ``results/benchmarking/runtime/tabpfn_D16_<run_id>_*.json``

``tabpfn_coverage.csv`` (unique training respondents used by TabPFN) also needs
the respondent-level files ``local/analytic_stage1.parquet`` and
``results/benchmarking/folds/<run_id>.parquet``, which are not published. If
they are missing, the existing ``tabpfn_coverage.csv`` is kept.

Outputs are written to ``results/paper/``:

* ``table1.csv``: Table 1 of the paper.
* ``hypotheses.csv``: H1 and H2 with their CI criteria.
* ``top_model_contrasts.csv``: exploratory paired contrasts between the four
  leading models (not part of the Holm family).
* ``country_summary.csv``: per country sample size, events, AUROC of every
  model and recalibration intercepts.
* ``runtime_summary.csv``: computation time per model.
* ``tuning_summary.csv`` and ``tuning_best_params.csv``: Optuna trials and the
  selected hyperparameters of every outer fold.
* ``tabpfn_coverage.csv``: unique training respondents per fold for TabPFN.
* ``key_numbers.csv``: every single number quoted in the text, with its source.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

MODEL_LABELS = {
    "tabpfn": "TabPFN", "xgboost": "XGBoost", "catboost": "CatBoost", "lightgbm": "LightGBM",
    "random_forest": "Random forest", "logistic": "Logistic regression",
    "elastic_net": "Elastic net", "mlp": "MLP",
}
GBT = ["xgboost", "catboost", "lightgbm"]          # models covered by hypothesis H2
TOP = ["tabpfn", "xgboost", "catboost", "lightgbm"]


def load_json(path: Path) -> dict:
    """Read a JSON file.

    Args:
        path (pathlib.Path): File to read.

    Returns:
        dict: Parsed content.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_table(bench: Path, run_id: str) -> pd.DataFrame:
    """Summarise the recorded wall-clock time of every model.

    Args:
        bench (pathlib.Path): ``results/benchmarking``.
        run_id (str): Run identity.

    Returns:
        pandas.DataFrame: Per model, the total over the 15 outer folds (hours,
        tuning included), the median and the maximum per fold.
    """
    rows = []
    for f in sorted((bench / "checkpoints" / run_id).glob("*/repeat_*/fold_*/runtime.json")):
        rows.append(dict(model=f.parts[-4], seconds=float(load_json(f)["wall_seconds"])))
    t = pd.DataFrame(rows).groupby("model").seconds.agg(["count", "sum", "median", "max"]).reset_index()
    return pd.DataFrame({
        "model": t["model"], "n_folds": t["count"],
        "total_hours": t["sum"] / 3600, "median_minutes_per_fold": t["median"] / 60,
        "max_hours_per_fold": t["max"] / 3600,
    })


def tuning_tables(bench: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count Optuna trials and list the selected hyperparameters per fold.

    Args:
        bench (pathlib.Path): ``results/benchmarking``.
        run_id (str): Run identity.

    Returns:
        tuple: ``(summary, best_params)`` data frames.
    """
    summary, best = [], []
    for f in sorted((bench / "checkpoints" / run_id).glob("*/repeat_*/fold_*/tuning.json")):
        model, rep, fold = f.parts[-4], f.parts[-3], f.parts[-2]
        t = load_json(f)
        states = [tr["state"] for tr in t.get("trials", [])]
        summary.append(dict(model=model, trials=len(states),
                            complete=sum(s.endswith("COMPLETE") for s in states)))
        for name, value in (t.get("best_params") or {}).items():
            best.append(dict(model=model, repeat=int(rep.split("_")[1]), outer_fold=int(fold.split("_")[1]),
                             parameter=name, value=value))
    s = pd.DataFrame(summary).groupby("model")[["trials", "complete"]].sum().reset_index()
    s["failed"] = s["trials"] - s["complete"]
    return s, pd.DataFrame(best)


def tabpfn_coverage(bench: Path, run_id: str, analytic: Path) -> pd.DataFrame | None:
    """Reconstruct how many distinct training respondents TabPFN used per fold.

    Uses the same sampler and seeds as ``runner.train_tabpfn_b``.

    Args:
        bench (pathlib.Path): ``results/benchmarking``.
        run_id (str): Run identity.
        analytic (pathlib.Path): ``local/analytic_stage1.parquet``.

    Returns:
        pandas.DataFrame or None: One row per repeat and fold, or ``None`` if
        the respondent-level inputs are not available.
    """
    folds_path = bench / "folds" / f"{run_id}.parquet"
    if not (folds_path.is_file() and analytic.is_file()):
        return None
    import yaml
    from ess_benchmark.runner import _stratified_positions
    d16 = yaml.safe_load((ROOT / "config" / "tabpfn.yml").read_text())
    base = int(d16["subsampling"]["base_seed"])
    n = int(d16["strategies"]["B_subsample_ensemble"]["n_per_subset"])
    k = int(d16["strategies"]["B_subsample_ensemble"]["n_subsets"])
    # Same row order as runner.run(): eligible rows, index reset.
    df = pd.read_parquet(analytic)
    df = df.loc[df.eligible.astype(bool)].reset_index(drop=True)
    folds = pd.read_parquet(folds_path)
    rows = []
    for (r, f), g in folds.groupby(["repeat", "outer_fold"], sort=True):
        test = g.row_index.to_numpy(dtype=int)
        train = np.setdiff1d(np.arange(len(df)), test, assume_unique=True)
        frame = df.iloc[train].reset_index(drop=True)
        used = [train[_stratified_positions(frame, n, base + 1000 * int(r) + 100 * int(f) + s)] for s in range(k)]
        unique = len(np.unique(np.concatenate(used)))
        rows.append(dict(repeat=int(r), outer_fold=int(f), n_train=len(train),
                         rows_per_subset=n, subsets=k, unique_train_rows=unique, share=unique / len(train)))
    return pd.DataFrame(rows)


def main() -> None:
    """Compute all paper numbers and write them to ``results/paper``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="c30863ba7974171b")
    ap.add_argument("--analytic-data", default="local/analytic_stage1.parquet",
                    help="respondent-level stage-1 data; only needed for tabpfn_coverage.csv")
    a = ap.parse_args()
    bench = ROOT / "results" / "benchmarking"
    out = ROOT / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    rid = a.run_id

    bm = pd.read_csv(bench / "combined" / f"{rid}_benchmark.csv")
    cm = pd.read_csv(bench / "combined" / f"{rid}_country_metrics.csv")
    inf = pd.read_csv(bench / "inference" / f"{rid}_auroc.csv").set_index("model")
    rep = pd.read_parquet(bench / "inference" / f"{rid}_auroc_replicates.parquet")
    rt = runtime_table(bench, rid).set_index("model")
    w = bm[bm.weighting.eq("pspwght")].set_index("model")
    uw = bm[bm.weighting.eq("unweighted")].set_index("model")

    # ---- Table 1 --------------------------------------------------------------
    t1 = pd.DataFrame({
        "algorithm": [MODEL_LABELS[m] for m in w.index],
        "model": w.index,
        "auroc": w.auroc, "auroc_ci_low": inf.ci_low.reindex(w.index), "auroc_ci_high": inf.ci_high.reindex(w.index),
        "auprc": w.auprc, "brier": w.brier,
        "diff_vs_elastic_net": inf.difference_vs_reference.reindex(w.index),
        "diff_ci_low": inf.difference_ci_low.reindex(w.index),
        "diff_ci_high": inf.difference_ci_high.reindex(w.index),
        "p_holm": inf.p_holm.reindex(w.index),
        "time_hours": rt.total_hours.reindex(w.index),
    }).sort_values("auroc", ascending=False)
    t1.to_csv(out / "table1.csv", index=False)

    # ---- Hypotheses ------------------------------------------------------------
    lowest = inf.ci_low.idxmin()
    hyp = [dict(hypothesis="H1", model=lowest, criterion="lowest AUROC lower 95% limit >= 0.75",
                value=inf.ci_low[lowest], lower=inf.ci_low[lowest], upper=np.nan,
                supported=bool((inf.ci_low >= 0.75).all()))]
    for m in GBT:
        lo, hi = inf.difference_ci_low[m], inf.difference_ci_high[m]
        hyp.append(dict(hypothesis="H2", model=m, criterion="paired difference vs elastic net within (0, 0.03)",
                        value=inf.difference_vs_reference[m], lower=lo, upper=hi, supported=bool(lo > 0 and hi < 0.03)))
    pd.DataFrame(hyp).to_csv(out / "hypotheses.csv", index=False)

    # ---- Country table ----------------------------------------------------------
    cw = cm[cm.weighting.eq("pspwght")]
    auc = cw.groupby(["cntry", "model"]).auroc.mean().unstack()
    intercept = cw.groupby(["cntry", "model"]).calibration_intercept.mean().unstack()
    size = cw[(cw.model == "elastic_net") & (cw.repeat == 0)].set_index("cntry")[["n", "events"]]
    country = size.join(auc.add_prefix("auroc_")).join(intercept[["xgboost", "tabpfn"]].add_prefix("recalibration_intercept_"))
    country.reset_index().to_csv(out / "country_summary.csv", index=False)

    # ---- Exploratory contrasts between the leading models -------------------------
    contrasts = []
    for i, m1 in enumerate(TOP):
        for m2 in TOP[i + 1:]:
            d = rep[m1] - rep[m2]
            eff = inf.point[m1] - inf.point[m2]
            se = float(d.std(ddof=0))
            lo, hi = np.quantile(d, [0.025, 0.975])
            contrasts.append(dict(model_1=m1, model_2=m2, difference=eff, se=se, ci_low=lo, ci_high=hi,
                                  p_unadjusted=float(2 * norm.sf(abs(eff / se))),
                                  countries_model_1_higher=int((auc[m1] > auc[m2]).sum()),
                                  note="exploratory; not in the prespecified Holm family"))
    pd.DataFrame(contrasts).to_csv(out / "top_model_contrasts.csv", index=False)

    # ---- Runtime, tuning, TabPFN coverage ------------------------------------------
    pilot = load_json(bench / "runtime" / f"tabpfn_D16_{rid}_r0_f0_B_subsample_ensemble.json")
    rts = rt.reset_index()
    rts["peak_rss_gb_pilot"] = np.where(rts.model.eq("tabpfn"), pilot.get("peak_rss_gb"), np.nan)
    rts.to_csv(out / "runtime_summary.csv", index=False)
    tsum, tbest = tuning_tables(bench, rid)
    tsum.to_csv(out / "tuning_summary.csv", index=False)
    tbest.to_csv(out / "tuning_best_params.csv", index=False)
    cov = tabpfn_coverage(bench, rid, ROOT / a.analytic_data)
    if cov is not None:
        cov.to_csv(out / "tabpfn_coverage.csv", index=False)
    elif (out / "tabpfn_coverage.csv").is_file():
        print("Respondent-level inputs absent: kept existing tabpfn_coverage.csv")
        cov = pd.read_csv(out / "tabpfn_coverage.csv")

    # ---- Key numbers quoted in the text ---------------------------------------------
    by_repeat = cw.groupby(["model", "repeat"]).auroc.mean().unstack()
    unweighted_rank = list(uw.auroc.sort_values(ascending=False).index)
    weighted_rank = list(w.auroc.sort_values(ascending=False).index)
    tt = contrasts[0]  # tabpfn vs xgboost
    k = [
        ("eligible_respondents", int(cw[(cw.model == "elastic_net") & (cw.repeat == 0)].n.sum()), "country_metrics.csv"),
        ("outcome_events", int(cw[(cw.model == "elastic_net") & (cw.repeat == 0)].events.sum()), "country_metrics.csv"),
        ("country_n_min", int(size.n.min()), "country_summary.csv"),
        ("country_n_max", int(size.n.max()), "country_summary.csv"),
        ("tabpfn_vs_xgboost_difference", tt["difference"], "top_model_contrasts.csv"),
        ("tabpfn_vs_xgboost_ci_low", tt["ci_low"], "top_model_contrasts.csv"),
        ("tabpfn_vs_xgboost_ci_high", tt["ci_high"], "top_model_contrasts.csv"),
        ("tabpfn_higher_than_xgboost_countries", tt["countries_model_1_higher"], "top_model_contrasts.csv"),
        ("xgboost_higher_than_elastic_net_countries", int((auc.xgboost > auc.elastic_net).sum()), "country_summary.csv"),
        ("xgboost_not_higher_country", ",".join(auc.index[auc.xgboost <= auc.elastic_net]), "country_summary.csv"),
        ("xgboost_intercept_min", float(intercept.xgboost.min()), "country_summary.csv"),
        ("xgboost_intercept_min_country", intercept.xgboost.idxmin(), "country_summary.csv"),
        ("xgboost_intercept_max", float(intercept.xgboost.max()), "country_summary.csv"),
        ("xgboost_intercept_max_country", intercept.xgboost.idxmax(), "country_summary.csv"),
        ("ranking_unchanged_unweighted", unweighted_rank == weighted_rank, "benchmark.csv"),
        ("max_auroc_range_across_repeats", float((by_repeat.max(axis=1) - by_repeat.min(axis=1)).max()), "country_metrics.csv"),
        ("tabpfn_to_xgboost_time_ratio", float(rt.total_hours["tabpfn"] / rt.total_hours["xgboost"]), "runtime_summary.csv"),
        ("tuning_trials_total", int(tsum.trials.sum()), "tuning_summary.csv"),
        ("tuning_trials_failed", int(tsum.failed.sum()), "tuning_summary.csv"),
        ("outer_fold_fits", int(rt.n_folds.sum()), "runtime_summary.csv"),
    ]
    if cov is not None:
        k += [("tabpfn_unique_train_min", int(cov.unique_train_rows.min()), "tabpfn_coverage.csv"),
              ("tabpfn_unique_train_max", int(cov.unique_train_rows.max()), "tabpfn_coverage.csv"),
              ("tabpfn_unique_share_mean", float(cov.share.mean()), "tabpfn_coverage.csv")]
    pd.DataFrame(k, columns=["key", "value", "source"]).to_csv(out / "key_numbers.csv", index=False)
    print(f"Wrote {len(list(out.glob('*.csv')))} files to {out}")


if __name__ == "__main__":
    main()
