"""Regression test: preserve old MatrixMaker encoding and eliminate fragmentation."""
import warnings
import numpy as np
import pandas as pd
from ess_benchmark.models import MatrixMaker


def test_matrix_equivalence_and_no_fragmentation():
    """MatrixMaker gives the same matrix as before and raises no fragmentation warning."""
    rng = np.random.default_rng(20261014)
    n = 140
    x = pd.DataFrame({'numeric': rng.normal(size=n), 'ordinal': rng.integers(0, 5, n).astype(float)})
    x.loc[2, 'numeric'] = np.nan
    for k in range(12):
        x[f'cat{k}'] = pd.Series(rng.choice([f'level{j}' for j in range(12)] + [None], n), dtype='string')
    spec = pd.DataFrame({'feature': list(x.columns), 'measure': ['continuous', 'ordinal'] + ['nominal'] * 12})
    for family in ('linear', 'forest', 'gbm', 'tabpfn'):
        fitted = MatrixMaker(spec, family).fit(x)
        with warnings.catch_warnings():
            warnings.simplefilter('error', pd.errors.PerformanceWarning)
            got = fitted.transform(x)
        expected = pd.DataFrame(index=x.index)
        # Reference construction intentionally mirrors the prior implementation.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
            for col in fitted.numeric_:
                a = pd.to_numeric(x[col], errors='coerce')
                if family not in ('gbm', 'tabpfn'):
                    a = a.fillna(fitted.medians_[col])
                if family == 'linear':
                    a = (a - fitted.means_[col]) / fitted.stds_[col]
                expected[col] = a.astype(float)
            for col in fitted.nominal_:
                a = x[col].astype('string')
                a = a.fillna('__missing__' if family in ('tabpfn', 'gbm') else fitted.modes_[col])
                for level in fitted.cat_levels_[col] + (['__missing__'] if family in ('tabpfn', 'gbm') else []):
                    expected[col + '__' + level] = (a == level).astype(float)
        pd.testing.assert_frame_equal(got, expected, check_exact=True)
