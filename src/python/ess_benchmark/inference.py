"""Paired Rao-Wu bootstrap inference on stored out-of-fold predictions.

This module estimates the uncertainty of the primary estimand (the
equal-country mean of within-country weighted AUROC) and of the paired
differences between each model and a reference model.

Design choices:

* The bootstrap follows the full source survey design. PSU multipliers are
  drawn for every stratum of the original ESS file (all 50,116 respondents),
  and only afterwards restricted to the eligible analysis domain. This avoids
  problems with strata that have a single PSU only after eligibility
  filtering.
* All models and all three cross-validation repeats share the same multipliers
  in a replicate, so comparisons are paired. A respondent's three predictions
  (one per repeat) get the same multiplier and are not treated as independent.
* Models are not retrained in a replicate. The intervals are therefore
  conditional on the stored predictions.

Performance note: the multipliers of one replicate are drawn into one array,
in the same random order as :func:`replicate_psu_weights`, and mapped to rows
with a precomputed integer index instead of a per-row dictionary lookup.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from .metrics import score


def _key(value):
    """Turn a design value into a stable string key.

    Survey design fields can be read as int, float or str depending on the
    source file. ``3``, ``3.0`` and ``"3"`` must map to the same key, so whole
    numbers are written without a decimal part.

    Args:
        value: A country, stratum or PSU identifier.

    Returns:
        str: The normalised key.
    """
    try:
        v=float(value)
        if np.isfinite(v) and v.is_integer():return str(int(v))
    except (TypeError, ValueError):pass
    return str(value)


def _source_structure(source):
    """List the PSUs of every (country, stratum) in the source design.

    Args:
        source (pandas.DataFrame): The full source file with columns
            ``cntry``, ``stratum`` and ``psu`` (all respondents, before
            eligibility filtering).

    Returns:
        dict: Maps ``(country, stratum)`` to a sorted list of PSU keys.

    Raises:
        ValueError: If a stratum of the source design has fewer than two PSUs.
            Rao-Wu resampling needs at least two PSUs per stratum.
    """
    groups={}
    for (country,stratum),g in source.groupby(['cntry','stratum'],sort=True):
        psus=np.sort(g.psu.unique())
        if len(psus)<2:raise ValueError(f'Original single-PSU stratum {country}/{stratum}; partner needed')
        groups[(_key(country),_key(stratum))]=[_key(x) for x in psus]
    return groups


def replicate_psu_weights(structure,rng):
    """Draw one Rao-Wu replicate and return the multipliers as a dictionary.

    In every stratum with ``n`` PSUs, ``n - 1`` PSUs are drawn with
    replacement. A PSU drawn ``k`` times gets the multiplier ``k * n / (n - 1)``.

    This dictionary version is kept for tests and backward compatibility.
    :func:`replicate_psu_vector` makes exactly the same draws faster.

    Args:
        structure (dict): Output of :func:`_source_structure`.
        rng (numpy.random.Generator): Random generator; its state advances.

    Returns:
        dict: Maps ``(country, stratum, psu)`` to the multiplier (float).
    """
    weights={}
    for (country,stratum),psus in structure.items():
        n=len(psus)
        draw=rng.integers(0,n,size=n-1)
        cnt=np.bincount(draw,minlength=n)*n/(n-1)
        for psu,weight in zip(psus,cnt):weights[(country,stratum,psu)]=float(weight)
    return weights


def _psu_positions(structure):
    """Give every PSU a fixed integer position.

    The order matches the order in which :func:`replicate_psu_vector` fills
    its output array.

    Args:
        structure (dict): Output of :func:`_source_structure`.

    Returns:
        tuple: ``(positions, sizes)`` where ``positions`` maps
        ``(country, stratum, psu)`` to an index and ``sizes`` lists the number
        of PSUs per stratum, in the same order.
    """
    keys=[(c,s,p) for (c,s),psus in structure.items() for p in psus]
    return {k:i for i,k in enumerate(keys)}, [len(p) for p in structure.values()]


def replicate_psu_vector(sizes,rng):
    """Draw one Rao-Wu replicate and return the multipliers as one array.

    Uses the same random draws, in the same order, as
    :func:`replicate_psu_weights`, so both give identical results for the same
    seed.

    Args:
        sizes (list[int]): Number of PSUs per stratum, from :func:`_psu_positions`.
        rng (numpy.random.Generator): Random generator; its state advances.

    Returns:
        numpy.ndarray: One multiplier per PSU, indexed by PSU position.
    """
    out=np.empty(sum(sizes));i=0
    for n in sizes:
        draw=rng.integers(0,n,size=n-1)
        out[i:i+n]=np.bincount(draw,minlength=n)*n/(n-1);i+=n
    return out


class _Prepared:
    """Arrays of one model's predictions, prepared once for fast replicates.

    Attributes:
        y (numpy.ndarray): Observed outcome (0/1).
        p (numpy.ndarray): Predicted probability.
        w (numpy.ndarray): Post-stratification weight ``pspwght``.
        pos (numpy.ndarray): PSU position of each row (see :func:`_psu_positions`).
        groups (list[numpy.ndarray]): Row indices of each (repeat, country)
            group, in a fixed sorted order.
        frame (pandas.DataFrame): The prediction rows, re-indexed from 0.
    """
    def __init__(self,frame,positions):
        """Build the arrays.

        Args:
            frame (pandas.DataFrame): Predictions of one model, with columns
                ``y_true``, ``y_prob``, ``pspwght``, ``cntry``, ``stratum``,
                ``psu`` and ``repeat``.
            positions (dict): PSU positions from :func:`_psu_positions`.
        """
        frame=frame.reset_index(drop=True)
        self.y=frame.y_true.to_numpy(dtype=int);self.p=frame.y_prob.to_numpy(dtype=float)
        self.w=frame.pspwght.to_numpy(dtype=float)
        self.pos=np.fromiter((positions[(_key(c),_key(s),_key(q))] for c,s,q in zip(frame.cntry,frame.stratum,frame.psu)),
                             dtype=np.int64,count=len(frame))
        self.groups=[idx for _,idx in sorted(frame.groupby(['repeat','cntry'],sort=True).indices.items())]
        self.frame=frame

    def value(self,metric,mult=None):
        """Compute the equal-country, repeat-averaged metric.

        The metric is computed in every (repeat, country) group and the group
        values are averaged. Because every repeat has the same 30 countries,
        this equals the mean over countries within each repeat, averaged over
        repeats.

        Args:
            metric (str): ``"auroc"`` or any key returned by
                :func:`ess_benchmark.metrics.score`.
            mult (numpy.ndarray, optional): Replicate PSU multipliers. ``None``
                gives the point estimate with the original weights.

        Returns:
            float: The aggregated metric, or ``nan`` if any group has zero
            total weight or only one outcome class after resampling.
        """
        vals=[]
        for idx in self.groups:
            # Replicate weight = survey weight x PSU multiplier of the row's PSU.
            w=self.w[idx] if mult is None else self.w[idx]*mult[self.pos[idx]]
            y=self.y[idx]
            # A group whose PSUs were all left out (or that lost one class) makes
            # the whole replicate undefined.
            if w.sum()<=0 or len(np.unique(y[w>0]))<2:return np.nan
            # AUROC is computed directly: it is fast and is the primary metric.
            if metric=='auroc':vals.append(float(roc_auc_score(y,self.p[idx],sample_weight=w)))
            else:vals.append(score(y,self.p[idx],w)[metric])
        return float(np.mean(vals))


def bootstrap_comparison(predictions,source,*,reference='elastic_net',reps=1000,seed=20261014,metric='auroc'):
    """Run the paired source-design bootstrap for all models.

    Args:
        predictions (pandas.DataFrame): Out-of-fold predictions of all models
            and repeats (output of :func:`ess_benchmark.runner.collect`).
        source (pandas.DataFrame): The full source file with ``cntry``,
            ``stratum`` and ``psu`` for all respondents.
        reference (str): Model used as the comparison reference.
        reps (int): Number of bootstrap replicates.
        seed (int): Seed of the random generator.
        metric (str): Metric to analyse; ``"auroc"`` for the primary analysis.

    Returns:
        tuple: ``(summary, replicates)``.

        * ``summary`` (pandas.DataFrame): one row per model with the point
          estimate, percentile 95% CI, difference from the reference with its
          bootstrap SE and percentile CI, two-sided Wald p-value, and the
          Holm-adjusted p-value over all non-reference models.
        * ``replicates`` (pandas.DataFrame): one column per model, one row per
          replicate. Contains no respondent-level data.

    Raises:
        ValueError: If ``reps`` < 1, the reference model is missing, models do
            not cover the same respondents and repeats, predictions are
            duplicated, or a predicted PSU is not in the source design.
        RuntimeError: If more than 1% of replicates are undefined for a model.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    models=sorted(predictions.model.unique())
    if reference not in models:raise ValueError('Reference model incomplete')
    # Pairing check: every model must have predictions for the same
    # (country, respondent, repeat) combinations.
    ids=[set(zip(g.cntry.astype(str),g.idno.astype(str),g.repeat)) for _,g in predictions.groupby('model')]
    if any(s!=ids[0] for s in ids[1:]):raise ValueError('Unpaired/missing predictions')
    if predictions[['model','repeat','cntry','idno']].duplicated().any():raise ValueError('Duplicate predictions')
    structure=_source_structure(source)
    positions,sizes=_psu_positions(structure)
    missing={(_key(c),_key(s),_key(p)) for c,s,p in zip(predictions.cntry,predictions.stratum,predictions.psu)}-set(positions)
    if missing:raise ValueError('Predicted PSU not present in original design: '+str(list(missing)[:4]))
    prepared={m:_Prepared(predictions.loc[predictions.model.eq(m)],positions) for m in models}
    point={m:prepared[m].value(metric) for m in models}
    rng=np.random.default_rng(seed);rep={m:np.full(reps,np.nan) for m in models}
    for i in range(reps):
        mult=replicate_psu_vector(sizes,rng)          # one draw per replicate, shared by all models
        for m in models:rep[m][i]=prepared[m].value(metric,mult)
    rows=[];ref=rep[reference]
    for m in models:
        a=rep[m];good=a[np.isfinite(a)]
        if len(good)<.99*reps:raise RuntimeError(f'Too many empty-domain replicates for {m}: {len(good)}/{reps}')
        # Percentile 95% interval of the model's own metric.
        interval=np.quantile(good,[.025,.975])
        # Paired differences: same replicate, model minus reference.
        difference=a-ref;difference=difference[np.isfinite(difference)]
        effect=point[m]-point[reference]
        se=float(difference.std(ddof=0))
        # Two-sided Wald test of "no difference", using the bootstrap SE.
        p=float(2*norm.sf(abs(effect/se))) if se>0 else (0. if effect else 1.)
        dlo,dhi=(np.quantile(difference,[.025,.975]) if m!=reference else (0.,0.))
        rows.append({'model':m,'point':point[m],'ci_low':interval[0],'ci_high':interval[1],
          'difference_vs_reference':effect,'difference_se':se,
          'difference_ci_low':float(dlo),'difference_ci_high':float(dhi),'p_wald':p,
          'valid_replicates':len(good),'metric':metric,'reference':reference,'reps':reps,'seed':seed})
    out=pd.DataFrame(rows)
    # Holm step-down adjustment over the non-reference models only. "running"
    # keeps the adjusted p-values monotone in the order of the raw p-values.
    others=out.model.ne(reference)
    pv=out.loc[others,'p_wald'].to_numpy()
    order=np.argsort(pv);adj=np.empty(len(pv));running=0.
    for rank,j in enumerate(order):
        running=max(running,(len(pv)-rank)*pv[j]);adj[j]=min(1,running)
    out.loc[others,'p_holm']=adj
    out['n_comparisons_holm']=int(others.sum())
    return out,pd.DataFrame(rep)
