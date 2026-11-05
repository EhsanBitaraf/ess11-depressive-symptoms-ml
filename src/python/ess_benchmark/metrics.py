"""Out-of-sample performance metrics and their aggregation.

Metrics are computed within each (model, repeat, country) cell on the pooled
out-of-fold predictions of that repeat, both unweighted and weighted with the
post-stratification weight ``pspwght``. The primary summary is the mean over
countries within a repeat, averaged over repeats, so every country has equal
weight.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from scipy.special import logit
from scipy.optimize import minimize


def score(y,p,w=None):
    """Compute discrimination, accuracy and calibration metrics.

    Args:
        y (array-like): Observed outcome, 0 or 1.
        p (array-like): Predicted probability of outcome 1.
        w (array-like, optional): Observation weights. ``None`` means
            unweighted.

    Returns:
        dict: With the keys

        * ``n``, ``events``: number of rows and of cases (unweighted counts);
        * ``auroc``: area under the ROC curve (``nan`` with one class only);
        * ``auprc``: average precision (``nan`` without cases);
        * ``brier``: mean squared error of the probabilities;
        * ``log_loss``: mean negative log-likelihood;
        * ``calibration_intercept``, ``calibration_slope``: coefficients of a
          logistic recalibration model ``logit P(y=1) = a + b * logit(p)``,
          with ``a`` and ``b`` fitted jointly. This intercept is not
          calibration-in-the-large, which would fix ``b = 1``;
        * ``ici_binned_diagnostic``: mean absolute difference between observed
          and predicted risk over deciles of predicted risk. This is a binned
          approximation, not the loess-based integrated calibration index.
    """
    # Clip so that logit(p) and log-loss stay finite for p = 0 or p = 1.
    y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    w=None if w is None else np.asarray(w,dtype=float)
    result={'n':len(y),'events':int(y.sum()),'auroc':float(roc_auc_score(y,p,sample_weight=w)) if len(set(y))==2 else np.nan,
            'auprc':float(average_precision_score(y,p,sample_weight=w)) if y.sum() else np.nan,
            'brier':float(brier_score_loss(y,p,sample_weight=w)),
            'log_loss':float(log_loss(y,p,labels=[0,1],sample_weight=w))}
    if len(set(y))<2:
        result.update(calibration_intercept=np.nan,calibration_slope=np.nan,ici_binned_diagnostic=np.nan);return result
    x=logit(p)
    def loss(beta):
        """Weighted logistic negative log-likelihood of the recalibration model."""
        # Clip the linear predictor to avoid overflow in exp().
        z=np.clip(beta[0]+beta[1]*x,-30,30)
        return float(np.average(np.logaddexp(0,z)-y*z,weights=w))
    # Start at (0, 1), i.e. "already perfectly calibrated".
    fit=minimize(loss,[0,1],method='BFGS')
    result['calibration_intercept']=float(fit.x[0]);result['calibration_slope']=float(fit.x[1])
    # ICI approximated by prediction-bin calibration, separate from smooth ICI.
    # Ranking first makes the ten bins equal-sized even when probabilities tie.
    bins=pd.qcut(pd.Series(p).rank(method='first'),q=min(10,len(p)),labels=False)
    obs=np.array([np.average(y[bins==b],weights=None if w is None else w[bins==b]) for b in np.unique(bins)])
    pred=np.array([np.average(p[bins==b],weights=None if w is None else w[bins==b]) for b in np.unique(bins)])
    bw=np.array([len(y[bins==b]) if w is None else sum(w[bins==b]) for b in np.unique(bins)])
    result['ici_binned_diagnostic']=float(np.average(abs(obs-pred),weights=bw))
    return result


def metrics_table(predictions):
    """Compute metrics for every model, repeat and country.

    Args:
        predictions (pandas.DataFrame): Out-of-fold predictions with columns
            ``model``, ``repeat``, ``cntry``, ``y_true``, ``y_prob`` and
            ``pspwght``.

    Returns:
        pandas.DataFrame: One row per (model, repeat, country, weighting),
        where ``weighting`` is ``"unweighted"`` or ``"pspwght"``, with the
        metrics of :func:`score`.
    """
    rows=[]
    for (model,repeat,country),g in predictions.groupby(['model','repeat','cntry'],sort=True):
        for wt,col in [('unweighted',None),('pspwght','pspwght')]:
            rows.append(dict(model=model,repeat=int(repeat),cntry=country,weighting=wt,
                **score(g.y_true,g.y_prob,None if col is None else g[col])))
    return pd.DataFrame(rows)


def aggregate(metrics):
    """Aggregate country-level metrics to the primary equal-country summary.

    Args:
        metrics (pandas.DataFrame): Output of :func:`metrics_table`.

    Returns:
        pandas.DataFrame: One row per (model, weighting) with each metric
        averaged first over countries within a repeat and then over repeats.
    """
    # Mean within-country, then across repeats; no pooling OOF repeats.
    cols=['auroc','auprc','brier','log_loss','calibration_intercept','calibration_slope','ici_binned_diagnostic']
    per_repeat=metrics.groupby(['model','weighting','repeat'])[cols].mean().reset_index()
    return per_repeat.groupby(['model','weighting'])[cols].mean().reset_index()
