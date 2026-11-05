"""ESS11 reproducible model benchmarking.

The package runs the Work Package A benchmark: repeated nested cross-validation
of eight algorithms, integrity-checked checkpoints, metrics and the paired
Rao-Wu bootstrap.

Modules:
    core: run identity, shared outer folds, atomic writes, checkpoint checks.
    models: fold-fitted preprocessing, estimators and tuning search spaces.
    runner: fold-by-fold nested cross-validation and result collection.
    metrics: weighted performance metrics and their aggregation.
    inference: paired source-design Rao-Wu bootstrap.

Attributes:
    MODELS (tuple[str]): Names of the eight algorithms, as used in file paths
        and command-line options.
"""
MODELS = ('logistic','elastic_net','random_forest','xgboost','lightgbm','catboost','mlp','tabpfn')
