"""Run identity, shared outer folds, atomic file writes and checkpoint checks.

These helpers make the benchmark reproducible and safe to resume:

* :func:`run_id` turns the configuration and the SHA-256 of the input files
  into a short run identity. Any change of an input gives a new identity.
* :func:`make_folds` creates the outer cross-validation folds once; all
  algorithms use the same folds.
* :func:`atomic_json` and :func:`atomic_parquet` write to a temporary file and
  rename it, so an interrupted run never leaves a half-written file.
* :func:`checkpoint_valid` decides whether a finished fold can be skipped on
  resume.
"""
from __future__ import annotations
import hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def digest(path):
    """Compute the SHA-256 of a file.

    The hash is taken over the raw bytes, so line endings matter: the same
    text with Windows (CRLF) and Linux (LF) line endings gives different
    hashes.

    Args:
        path (str or pathlib.Path): File to hash.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    h=hashlib.sha256()
    with open(path,'rb') as f:
        # Read in 1 MiB chunks so that large files do not fill memory.
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    """Write an object as JSON in one atomic step.

    The JSON is first written to ``<name>.tmp`` and then renamed. A rename is
    atomic on the same file system, so readers see either the old file or the
    complete new file.

    Args:
        path (str or pathlib.Path): Target file. Parent folders are created.
        obj: JSON-serialisable object. Values that JSON cannot represent are
            converted with ``str``.
    """
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str),encoding='utf8')
    os.replace(tmp,path)


def atomic_parquet(path, frame):
    """Write a data frame as Parquet in one atomic step.

    Args:
        path (str or pathlib.Path): Target file. Parent folders are created.
        frame (pandas.DataFrame): Data to write; the index is not stored.
    """
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    frame.to_parquet(tmp,index=False)
    os.replace(tmp,path)


def make_folds(frame, *, repeats=3, outer=5, seed=20261014):
    """Create the shared outer cross-validation folds.

    Folds are stratified by country x outcome, so every fold has about the
    same share of each country and of cases. Repeat ``r`` uses the seed
    ``seed + r``.

    Args:
        frame (pandas.DataFrame): Eligible respondents with ``cntry``,
            ``idno`` and ``d_eds9``.
        repeats (int): Number of repetitions of the K-fold split.
        outer (int): Number of outer folds.
        seed (int): Base random seed.

    Returns:
        pandas.DataFrame: Columns ``row_index`` (position in ``frame``),
        ``repeat`` and ``outer_fold``. Each row of ``frame`` appears exactly
        once per repeat.

    Raises:
        ValueError: If (country, idno) is not unique, the outcome has missing
            values, or a country x outcome cell has fewer rows than folds.
    """
    if frame[['cntry','idno']].duplicated().any(): raise ValueError('Nonunique country + id')
    if frame['d_eds9'].isna().any(): raise ValueError('Missing outcome in eligible frame')
    # Stratification label, e.g. "AT|1" for an Austrian case.
    labels=frame.cntry.astype(str)+'|'+frame.d_eds9.astype(int).astype(str)
    if labels.value_counts().min()<outer: raise ValueError('Country/outcome cells smaller than outer folds')
    rows=[]
    for repeat in range(repeats):
        skf=StratifiedKFold(n_splits=outer,shuffle=True,random_state=seed+repeat)
        # Only the test indices are stored; the training rows of a fold are
        # all other rows of the same repeat.
        for fold,(_,test) in enumerate(skf.split(np.zeros(len(frame)),labels)):
            rows.extend((int(i),repeat,fold) for i in test)
    result=pd.DataFrame(rows,columns=['row_index','repeat','outer_fold'])
    assert result.groupby(['repeat','row_index']).size().eq(1).all()
    return result


def inner_splits(frame, train_idx, *, folds=3, seed=0):
    """Create the inner folds used for hyperparameter tuning.

    The inner folds split only the training rows of one outer fold, so tuning
    never sees the outer test rows.

    Args:
        frame (pandas.DataFrame): All eligible respondents.
        train_idx (numpy.ndarray): Row positions of the outer training fold.
        folds (int): Number of inner folds.
        seed (int): Random seed of the inner split.

    Returns:
        list[tuple[numpy.ndarray, numpy.ndarray]]: ``(fit_rows, validation_rows)``
        pairs, given as positions in ``frame``.

    Raises:
        ValueError: If a country x outcome cell of the training fold has fewer
            rows than inner folds.
    """
    train=frame.iloc[train_idx]
    labels=train.cntry.astype(str)+'|'+train.d_eds9.astype(int).astype(str)
    if labels.value_counts().min()<folds:
        raise ValueError('Country/outcome cell smaller than inner fold requirement')
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    # cv.split returns positions inside "train"; map them back to positions in "frame".
    return [(train_idx[a],train_idx[b]) for a,b in cv.split(np.zeros(len(train)),labels)]


def run_id(config, paths):
    """Compute the run identity from the configuration and input files.

    Args:
        config (dict): Benchmark configuration (without run-time-only keys).
        paths (dict): Maps an input name (``data``, ``feature_spec``,
            ``audit``) to its file path.

    Returns:
        tuple: ``(identity, payload)``. ``identity`` is the first 16 hex
        characters of the SHA-256 of ``payload``; ``payload`` holds the
        configuration and the SHA-256 of each input file.
    """
    payload={'config':config,'inputs':{k:digest(v) for k,v in paths.items()}}
    # sort_keys makes the JSON, and therefore the identity, independent of key order.
    key=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()[:16]
    return key,payload


def checkpoint_valid(folder, identity, expected_ids):
    """Check whether a finished fold checkpoint can be reused.

    A checkpoint is valid only if all of these hold: the manifest says
    ``COMPLETE``; it belongs to the same run identity; the predictions file
    has the hash recorded in the manifest; it covers exactly the expected
    test respondents; and every probability is a number between 0 and 1.

    Args:
        folder (str or pathlib.Path): Checkpoint folder of one model, repeat
            and fold.
        identity (str): Current run identity.
        expected_ids (set[tuple[str, str]]): Expected ``(cntry, idno)`` pairs
            of the test fold.

    Returns:
        bool: ``True`` if the checkpoint is complete and intact.
    """
    folder=Path(folder); manifest=folder/'manifest.json'; predictions=folder/'predictions.parquet'
    if not(manifest.is_file() and predictions.is_file()):return False
    try:
        obj=json.loads(manifest.read_text()); p=pd.read_parquet(predictions)
        return (obj.get('state')=='COMPLETE' and obj.get('identity')==identity
                and obj.get('prediction_sha256')==digest(predictions)
                and len(p)==len(expected_ids)
                and set(zip(p.cntry.astype(str),p.idno.astype(str)))==expected_ids
                and p.y_prob.notna().all() and p.y_prob.between(0,1).all())
    # Any read or parse problem means "not valid"; the fold is then re-run.
    except (OSError,ValueError,KeyError,TypeError): return False
