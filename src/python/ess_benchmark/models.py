"""Estimators, fold-fitted preprocessing and tuning search spaces.

All preprocessing steps learn their statistics (percentiles, medians, modes,
category levels, means and standard deviations) from the training rows only
and then apply them unchanged to validation or test rows. Nothing is fitted on
a test fold.

Preprocessing depends on the model family (see :func:`family_for`):

=========  ========================  ==================  ==============
family     numeric missing values    nominal variables   scaling
=========  ========================  ==================  ==============
linear     median imputation         one-hot, mode       z-score
forest     median imputation         one-hot, mode       none
gbm        kept as NaN               one-hot + missing   none
tabpfn     kept as NaN               one-hot + missing   none
=========  ========================  ==================  ==============

"linear" covers unpenalised and elastic-net logistic regression and the MLP;
"gbm" covers XGBoost, LightGBM and CatBoost.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class ClipAlcohol(BaseEstimator,TransformerMixin):
    """Cap alcohol quantities at the 99.9th percentile of the training rows.

    Very large reported quantities would otherwise dominate linear models.

    Args:
        columns (tuple[str]): Alcohol quantity columns (weekday and weekend).

    Attributes:
        caps_ (dict): Learned cap per column. Columns that are absent or all
            missing are not capped.
    """
    def __init__(self, columns=('alcwkdy','alcwknd')): self.columns=columns
    def fit(self,X,y=None):
        """Learn the caps from the training rows.

        Args:
            X (pandas.DataFrame): Training predictors.
            y: Ignored.

        Returns:
            ClipAlcohol: The fitted transformer.
        """
        self.caps_={c:float(pd.to_numeric(X[c],errors='coerce').quantile(.999)) for c in self.columns if c in X and X[c].notna().any()}
        return self
    def transform(self,X):
        """Apply the learned caps.

        Args:
            X (pandas.DataFrame): Predictors.

        Returns:
            pandas.DataFrame: A copy with capped alcohol columns.
        """
        x=X.copy()
        for col,cap in self.caps_.items():x[col]=pd.to_numeric(x[col],errors='coerce').clip(upper=cap)
        return x

class MissingFlags(BaseEstimator,TransformerMixin):
    """Add 0/1 indicators of missing values.

    Indicators are added for every column with more than 5% missing values in
    the training rows, and always for household income (``hinctnta``), whose
    nonresponse may itself be informative.

    Args:
        threshold (float): Stored for reference. Note: the code uses a fixed
            cut-off of 0.05; this argument is not used.

    Attributes:
        cols_ (list[str]): Columns that receive an indicator ``<col>__missing``.
    """
    def __init__(self, threshold=.05):self.threshold=threshold
    def fit(self,X,y=None):
        """Choose the columns that get an indicator.

        Args:
            X (pandas.DataFrame): Training predictors.
            y: Ignored.

        Returns:
            MissingFlags: The fitted transformer.
        """
        self.cols_=list(X.columns[X.isna().mean()>.05]);
        if 'hinctnta' in X and 'hinctnta' not in self.cols_:self.cols_.append('hinctnta')
        return self
    def transform(self,X):
        """Add the indicator columns.

        Args:
            X (pandas.DataFrame): Predictors.

        Returns:
            pandas.DataFrame: A copy with one ``<col>__missing`` column (0/1)
            per selected column. The original columns are kept.
        """
        x=X.copy()
        for col in self.cols_:x[col+'__missing']=x[col].isna().astype(float)
        return x

class MatrixMaker(BaseEstimator,TransformerMixin):
    """Fit encodings and imputers on training only; numeric matrix for estimators.

    Produces a fully numeric matrix: numeric columns first, then one 0/1
    column per category of each nominal variable.

    Args:
        spec (pandas.DataFrame): Feature specification with columns
            ``feature`` and ``measure``; ``measure == "nominal"`` marks
            nominal variables.
        family (str): Model family, see :func:`family_for`.

    Attributes:
        nominal_ (list[str]): Nominal columns.
        numeric_ (list[str]): All other columns, treated as numeric.
        cat_levels_ (dict): Category levels seen in training, per nominal column.
        medians_ (dict): Training median per numeric column.
        modes_ (dict): Training mode per nominal column.
        means_ (dict): Training mean per numeric column, after median imputation.
        stds_ (dict): Training standard deviation per numeric column (1 if 0).
    """
    def __init__(self,spec,family):self.spec=spec;self.family=family
    def fit(self,X,y=None):
        """Learn levels, medians, modes, means and standard deviations.

        Args:
            X (pandas.DataFrame): Training predictors.
            y: Ignored.

        Returns:
            MatrixMaker: The fitted transformer.
        """
        X=X.copy(); nominal=set(self.spec.loc[self.spec.measure.eq('nominal'),'feature'])&set(X.columns)
        self.nominal_=sorted(nominal);self.numeric_=sorted(set(X.columns)-nominal)
        self.cat_levels_={c:sorted(X[c].dropna().astype(str).unique()) for c in self.nominal_}
        # An all-missing column gets the median 0 / mode "__missing__" so that
        # transform() still works.
        self.medians_={c:(float(pd.to_numeric(X[c],errors='coerce').median()) if X[c].notna().any() else 0.) for c in self.numeric_}
        self.modes_={c:(X[c].dropna().astype(str).mode().iloc[0] if X[c].notna().any() else '__missing__') for c in self.nominal_}
        self.means_={};self.stds_={}
        for c in self.numeric_:
            # Scaling statistics are computed after median imputation; a
            # constant column gets std 1 to avoid division by zero.
            a=pd.to_numeric(X[c],errors='coerce').fillna(self.medians_[c]);self.means_[c]=float(a.mean());self.stds_[c]=float(a.std()) or 1.
        return self
    def transform(self,X):
        """Build the numeric matrix.

        Category levels not seen in training get 0 in every one-hot column.

        Args:
            X (pandas.DataFrame): Predictors.

        Returns:
            pandas.DataFrame: Numeric matrix with the same index as ``X``.
        """
        X=X.copy()
        # Collect the columns in insertion order, then construct one contiguous
        # DataFrame. This preserves names, values and the original column order
        # without repeatedly inserting columns (pandas PerformanceWarning).
        columns={}
        for c in self.numeric_:
            a=pd.to_numeric(X[c],errors='coerce')
            # Tree boosters and TabPFN handle NaN themselves; others need imputation.
            if self.family not in ('gbm','tabpfn'):a=a.fillna(self.medians_[c])
            if self.family=='linear':a=(a-self.means_[c])/self.stds_[c]
            columns[c]=a.astype(float)
        for c in self.nominal_:
            a=X[c].astype('string')
            # Boosters and TabPFN get an explicit "__missing__" category; the
            # other families impute the training mode.
            a=a.fillna('__missing__' if self.family in ('tabpfn','gbm') else self.modes_[c])
            for level in self.cat_levels_[c]+(['__missing__'] if self.family in ('tabpfn','gbm') else []):
                columns[c+'__'+level]=(a==level).astype(float)
        return pd.DataFrame(columns,index=X.index)



def family_for(model):
    """Map a model name to its preprocessing family.

    Args:
        model (str): One of :data:`ess_benchmark.MODELS`.

    Returns:
        str: ``"linear"``, ``"forest"``, ``"tabpfn"`` or ``"gbm"``.
    """
    return 'linear' if model in ('logistic','elastic_net','mlp') else ('forest' if model=='random_forest' else ('tabpfn' if model=='tabpfn' else 'gbm'))


def preprocessing(spec, model):
    """Build the preprocessing pipeline for a model.

    Steps, in order: :class:`ClipAlcohol`, :class:`MissingFlags`,
    :class:`MatrixMaker`.

    Args:
        spec (pandas.DataFrame): Feature specification.
        model (str): Model name.

    Returns:
        sklearn.pipeline.Pipeline: Unfitted pipeline.
    """
    return Pipeline([('clip',ClipAlcohol()),('flags',MissingFlags()),('matrix',MatrixMaker(spec,family_for(model)))])


def estimator(model, params,seed=0):
    """Create an unfitted estimator.

    Args:
        model (str): Model name.
        params (dict): Hyperparameters from :func:`search_params`; missing
            keys use the defaults written below.
        seed (int): Random seed of the estimator.

    Returns:
        A scikit-learn compatible classifier.

    Raises:
        ValueError: If the model name is unknown.

    Note:
        TabPFN is not fitted through this function in the benchmark. The
        runner uses ``runner.train_tabpfn_b``, which follows decision D16.
    """
    if model == 'logistic':
        # C=inf means no penalty (unpenalised maximum likelihood).
        return LogisticRegression(
            C=np.inf,
            max_iter=1500,
            random_state=seed
        )
    if model=='elastic_net':return LogisticRegression(penalty='elasticnet',solver='saga',C=params.get('C',1),l1_ratio=params.get('l1_ratio',.5),max_iter=1800,random_state=seed)
    if model=='random_forest':return RandomForestClassifier(n_estimators=params.get('n_estimators',300),max_depth=params.get('max_depth',None),min_samples_leaf=params.get('min_samples_leaf',2),n_jobs=16,random_state=seed)
    if model=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=params.get('n_estimators',400),max_depth=params.get('max_depth',4),learning_rate=params.get('learning_rate',.05),subsample=params.get('subsample',.9),tree_method='hist',eval_metric='logloss',n_jobs=16,random_state=seed)
    if model=='lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=params.get('n_estimators',400),num_leaves=params.get('num_leaves',31),learning_rate=params.get('learning_rate',.05),verbosity=-1,n_jobs=16,random_state=seed)
    if model=='catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=params.get('n_estimators',400),depth=params.get('depth',6),learning_rate=params.get('learning_rate',.05),verbose=False,thread_count=16,random_seed=seed,allow_writing_files=False)
    # Single hidden layer; early_stopping holds out 10% of the training rows.
    if model=='mlp':return MLPClassifier(hidden_layer_sizes=(params.get('width',64),),alpha=params.get('alpha',.001),learning_rate_init=params.get('lr',.001),max_iter=150,early_stopping=True,random_state=seed)
    if model=='tabpfn':
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device='cpu')
    raise ValueError(model)


def search_params(trial, model):
    """Sample hyperparameters for one Optuna trial.

    The same budget (50 trials) is used for every tuned model.

    Args:
        trial (optuna.trial.Trial): The current trial.
        model (str): Model name.

    Returns:
        dict: Sampled hyperparameters; empty for models that are not tuned
        (unpenalised logistic regression and TabPFN).
    """
    if model=='elastic_net':return {'C':trial.suggest_float('C',.001,100,log=True),'l1_ratio':trial.suggest_float('l1_ratio',0,1)}
    if model=='random_forest':return {'n_estimators':trial.suggest_int('n_estimators',150,500,step=50),'max_depth':trial.suggest_int('max_depth',4,24),'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,15)}
    if model=='xgboost':return {'n_estimators':trial.suggest_int('n_estimators',150,650,step=50),'max_depth':trial.suggest_int('max_depth',2,8),'learning_rate':trial.suggest_float('learning_rate',.015,.2,log=True),'subsample':trial.suggest_float('subsample',.65,1)}
    if model=='lightgbm':return {'n_estimators':trial.suggest_int('n_estimators',150,650,step=50),'num_leaves':trial.suggest_int('num_leaves',15,63),'learning_rate':trial.suggest_float('learning_rate',.015,.2,log=True)}
    if model=='catboost':return {'n_estimators':trial.suggest_int('n_estimators',150,650,step=50),'depth':trial.suggest_int('depth',4,8),'learning_rate':trial.suggest_float('learning_rate',.015,.2,log=True)}
    if model=='mlp':return {'width':trial.suggest_categorical('width',[32,64,128]),'alpha':trial.suggest_float('alpha',1e-5,.1,log=True),'lr':trial.suggest_float('lr',.0001,.01,log=True)}
    return {}
