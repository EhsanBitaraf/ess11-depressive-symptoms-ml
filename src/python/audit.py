"""Pre-analysis diagnostic utilities; NO prediction model training.

This module implements the Phase 00 data audit. It reads the ESS11 file,
checks identifiers, weights, design fields, the CES-D8 outcome, eligibility
and questionnaire routing, and collects the evidence needed for the audit
gate before any model is trained:

* consistency checks of questionnaire routing (Appendix C),
* an independent cross-check of the stage-1 recoding counts,
* a comparison of the Python Rao-Wu bootstrap with the R ``survey`` package,
* a technical check of the installed libraries (optionally a small synthetic
  TabPFN fit).

Human decisions are read from ``config/audit_signoff.yaml``; a gate is never
approved automatically. No outcome-based model metric is computed here.

All aggregates should be disclosure reviewed before publication.

Attributes:
    ITEMS (list[str]): The eight CES-D8 items.
    REVERSED (set[str]): Positively worded items that are reverse-scored.
    ESS_CES_MISSING (set[int]): ESS nonresponse codes of the CES-D8 items.
    ROUTES (list[tuple[str, str]]): (filter, dependent) pairs tabulated for review.
    CORE (list[str]): Core columns required by the audit.
"""
from __future__ import annotations
import hashlib, json, platform, sys, importlib.metadata, re, subprocess, shutil, tempfile, inspect, textwrap, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ITEMS = ['fltdpr','flteeff','slprl','wrhpp','fltlnl','enjlf','fltsd','cldgng']
REVERSED = {'wrhpp','enjlf'}
ESS_CES_MISSING = {7,8,9}

def _parse_codes(value):
    """Parse codes as written in the researcher's variable_map.csv, e.g. '6 7 8 9'.

    The audit only reports these codes; it never applies routed recoding.

    Args:
        value: Codes separated by spaces, commas, semicolons or ``|``.

    Returns:
        list[int]: Sorted unique codes; empty for a blank value.

    Raises:
        ValueError: If a code is not a whole number.
    """
    if pd.isna(value) or str(value).strip() == '':
        return []
    tokens = re.split(r'[\s,;|]+', str(value).strip())
    result = []
    for token in tokens:
        if not token:
            continue
        try:
            
            number = float(token)
            if not np.isfinite(number) or not number.is_integer():
                raise ValueError(token)
            result.append(int(number))
        except ValueError as exc:
            raise ValueError(f'Non-numeric missing code in variable_map.csv: {token!r}') from exc
    return sorted(set(result))

def _normalise_variable_map(variable_map):
    """Support the original six-column codebook-derived map without changing it.

    The earlier prototype schema (name/set/review_status) remains supported.

    Args:
        variable_map (pandas.DataFrame): Contents of ``variable_map.csv``.

    Returns:
        pandas.DataFrame: A copy with a ``variable`` column, all optional
        columns present (empty if absent) and no NaN values.

    Raises:
        ValueError: If variable names are missing, blank or duplicated.
    """
    vm = variable_map.copy()
    vm.columns = [str(c).lstrip('\ufeff').strip() for c in vm.columns]
    if 'variable' not in vm.columns and 'name' in vm.columns:
        vm = vm.rename(columns={'name':'variable'})
    if 'variable' not in vm.columns:
        raise ValueError('Variable map must contain a variable column (or legacy name).')
    vm['variable'] = vm['variable'].astype('string').str.strip()
    if vm['variable'].isna().any() or vm['variable'].eq('').any():
        raise ValueError('Variable map contains blank variable names.')
    if vm['variable'].duplicated().any():
        raise ValueError('Duplicate variable names: ' + ', '.join(vm.loc[vm.variable.duplicated(),'variable'].tolist()))
    for c in ['label','role','group','missing_codes','note','set','review_status','not_applicable_codes','nonresponse_codes','routing','derived_into','minimal_s1']:
        if c not in vm.columns: vm[c] = ''
    return vm.fillna('')

ROUTES = [('alcfreq','alcwkdy'),('alcfreq','alcwknd'),('alcfreq','alcbnge'),('respc19a','symtc19'),('symtc19','symtnc19'),('hlpfmly','hlpfmhr'),('medtrun','medtrnu'),('hltprca','cancfre')]
CORE = ['cntry','idno','agea','gndr','pspwght','anweight','psu','stratum'] + ITEMS

def read_config(path):
    """Read a YAML configuration file.

    Args:
        path (str or pathlib.Path): File to read.

    Returns:
        dict: Parsed configuration.
    """
    with open(path, encoding='utf-8') as f: return yaml.safe_load(f)

def checksum(path):
    """Compute the SHA-256 of a file's bytes.

    Args:
        path (str or pathlib.Path): File to hash.

    Returns:
        str: Hexadecimal digest.
    """
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for piece in iter(lambda:f.read(2**20), b''): h.update(piece)
    return h.hexdigest()

def load_ess(path):
    """Read the ESS ``.sav`` file with numeric codes.

    Value labels are not applied and user-defined missing codes are kept, so
    that every special code can be audited.

    Args:
        path (str or pathlib.Path): The ESS11 ``.sav`` file.

    Returns:
        tuple: ``(data, metadata)`` from ``pyreadstat.read_sav``.

    Raises:
        RuntimeError: If ``pyreadstat`` is not installed.
    """
    try: import pyreadstat
    except ImportError as e: raise RuntimeError('Install pyreadstat from requirements.txt to read ESS .sav') from e
    df,meta=pyreadstat.read_sav(str(path), apply_value_formats=False, user_missing=True)
    return df,meta

def cesd8(df):
    """Independent CES-D8 scoring used by the audit.

    This is a second implementation, separate from
    :func:`ess_preprocess.recode.score_cesd8`, so that the two can be compared.

    Args:
        df (pandas.DataFrame): Data with the eight CES-D8 items.

    Returns:
        tuple: ``(scores, invalid, ambiguous)``:

        * ``scores`` (pandas.DataFrame): ``cesd8_score``, ``answered_items``,
          ``prorated`` and the cut-off indicators ``eds_8``, ``eds_9``,
          ``eds_10``;
        * ``invalid`` (dict): Per item, the number of codes outside
          1-4 and 7-9;
        * ``ambiguous`` (int): Prorated scores exactly halfway between two
          integers (always 0, see the comment below).

    Raises:
        ValueError: If a CES-D8 item is missing.
    """
    missing=[x for x in ITEMS if x not in df]
    if missing: raise ValueError('Missing CES-D8 items: '+', '.join(missing))
    scores=pd.DataFrame(index=df.index)
    invalid={}
    for item in ITEMS:
        raw=pd.to_numeric(df[item],errors='coerce')
        invalid[item]=int((raw.notna() & ~raw.isin([1,2,3,4,7,8,9])).sum())
        raw=raw.where(raw.isin([1,2,3,4]))
        scores[item]=(4-raw if item in REVERSED else raw-1)
    count=scores.notna().sum(axis=1)
    summed=scores.sum(axis=1,min_count=7)
    # pandas/numpy round implements ties-to-even; protocol's rounding convention
    # is not specified, so half-integer ties are explicitly flagged for review.
    prorated=(summed*8/7).round()
    value=summed.where(count==8,prorated.where(count==7))
    ambiguous=((count==7) & ((summed*8/7)%1==0.5)).sum()
    return pd.DataFrame({'cesd8_score':value, 'answered_items':count, 'prorated':count==7,
                         'eds_8':value>=8,'eds_9':value>=9,'eds_10':value>=10},index=df.index), invalid, int(ambiguous)

def auroc_precision_halfwidth(n_pos, n_neg, kish_n, auc=0.75):
    """Expected normal 95% AUROC half-width; Hanley-McNeil using Kish n/n.

    Used only to flag countries where a country-level AUROC would be imprecise.
    The case and non-case counts are shrunk by the ratio of the Kish effective
    sample size to the actual sample size.

    Args:
        n_pos (int): Number of cases.
        n_neg (int): Number of non-cases.
        kish_n (float): Kish effective sample size.
        auc (float): Assumed AUROC.

    Returns:
        float: Expected half-width of the 95% CI; ``inf`` if not computable.
    """
    n=n_pos+n_neg
    if n_pos<2 or n_neg<2 or not kish_n or kish_n<=0: return float('inf')
    scale=min(1.0,kish_n/n)
    p=n_pos*scale; q=n_neg*scale
    q1=auc/(2-auc); q2=2*auc**2/(1+auc)
    var=(auc*(1-auc)+(p-1)*(q1-auc**2)+(q-1)*(q2-auc**2))/(p*q)
    return float(1.96*np.sqrt(max(0,var)))


def load_signoff(config, checks):
    """Human attestations are recorded, never converted into automatic approval.

    Args:
        config (dict): Audit configuration (key ``signoff_path``).
        checks (Iterable[str]): Names of the gates to look up.

    Returns:
        dict: ``status``, ``path`` and ``items``. For each gate, ``recorded``
        is ``True`` only if decision, date, responsible person and evidence
        are all filled in.

    Raises:
        ValueError: If ``items`` in the YAML file is not a mapping.
    """
    path=Path(config.get('signoff_path','config/audit_signoff.yaml'))
    if not path.exists():
        return {'status':'MISSING','path':str(path),'items':{}}
    content=yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    items=content.get('items',{})
    if not isinstance(items,dict): raise ValueError('signoff items must be mapping')
    accepted={}
    for name in checks:
        item=items.get(name,{}) or {}
        valid=all(str(item.get(k,'')).strip() for k in ['decision','date','responsible','evidence'])
        accepted[name]={'recorded':bool(valid),'decision':item.get('decision','')}
    return {'status':'RECORDED_NOT_AUTOMATICALLY_APPROVED','path':str(path),'items':accepted}

def _preprocessing_reference(config, routing_checks, df=None, eligible=None):
    """Compare independent stage-1 recoding with audit's per-rule diagnostics.

    Never approve solely because an output directory exists. Missing, stale or
    inconsistent evidence always gives REVIEW with explicit discrepancies.

    The check compares the preprocessing summary and log with the audit's
    own counts: raw-file and variable-map SHA-256, number of records and of
    eligible respondents, every R1 application, R3 per group, the alcohol R2
    counts, and the rule totals.

    Args:
        config (dict): Audit configuration with the paths of the
            preprocessing outputs.
        routing_checks (list[dict]): Output of :func:`appendix_c_diagnostics`.
        df (pandas.DataFrame, optional): Raw data, for record counts and
            checks that need raw values.
        eligible (pandas.Series, optional): Eligibility flag.

    Returns:
        dict: ``status`` (``PASS`` or ``REVIEW``), ``differences``,
        ``comparisons`` (audit vs preprocessing values), ``sources`` and
        ``detail``.
    """
    result = {'status': 'REVIEW', 'differences': [], 'comparisons': {}}
    differences = result['differences']
    config = config or {}

    def path_for(key, default):
        """Resolve a configured path relative to the current folder."""
        path = Path(config.get(key, default))
        return path if path.is_absolute() else Path.cwd() / path

    summary_path = path_for('preprocess_summary',
                            str(Path(config.get('preprocess_output_dir', 'results/preprocessing')) / 'preprocess_summary.json'))
    log_path = path_for('preprocess_log',
                        str(Path(config.get('preprocess_output_dir', 'results/preprocessing')) / 'recode_log.csv'))
    source_path = path_for('source_file', '')
    result['sources'] = {'summary': str(summary_path), 'log': str(log_path),
                         'raw': str(source_path)}
    for item in (summary_path, log_path, source_path):
        if not item.is_file():
            differences.append(f'Missing evidence file: {item}')
    if differences:
        result['detail'] = '; '.join(differences)
        return result
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        log = pd.read_csv(log_path, keep_default_na=False)
        required = {'rule', 'variable', 'n_all', 'description'}
        if not required.issubset(log.columns):
            raise ValueError('recode_log missing columns: ' + ', '.join(sorted(required - set(log.columns))))
        log['n_all'] = pd.to_numeric(log['n_all'], errors='raise').astype(int)
        if (log.n_all < 0).any():
            raise ValueError('Negative recode counts')
        sha = checksum(source_path)
        result['comparisons']['raw_sha256'] = {'audit': sha, 'preprocess': summary.get('raw_sha256')}
        if sha != summary.get('raw_sha256'):
            differences.append('Raw ESS SHA-256 mismatch')
        # Same variable map and same sample in both runs.
        vm_path = path_for('variable_map', 'config/variable_map.csv')
        vm_sha = checksum(vm_path) if vm_path.is_file() else None
        result['comparisons']['variable_map_sha256'] = {'audit': vm_sha, 'preprocess': summary.get('variable_map_sha256')}
        if vm_sha is None or vm_sha != summary.get('variable_map_sha256'):
            differences.append('Variable map SHA-256 mismatch')
        if summary.get('records') is None:
            differences.append('Preprocess summary missing records')
        elif df is not None:
            result['comparisons']['records'] = {'audit': len(df), 'preprocess': summary.get('records')}
            if int(summary['records']) != len(df):
                differences.append(f"Records: audit={len(df)}, preprocessing={summary['records']}")
        if eligible is not None:
            n_elig = int(pd.Series(eligible).sum())
            result['comparisons']['eligible'] = {'audit': n_elig, 'preprocess': summary.get('eligible')}
            if summary.get('eligible') is None or int(summary['eligible']) != n_elig:
                differences.append(f"Eligible: audit={n_elig}, preprocessing={summary.get('eligible')}")

        lookup = {r['rule']: r for r in routing_checks}

        def match(rule, variable, expected, label):
            """Compare one rule count from the log with the audit's own count."""
            rows = log.loc[log.rule.eq(rule) & log.variable.eq(variable)]
            if rows.empty:
                differences.append(f'Missing recode row: {label}')
                return
            found = int(rows.n_all.sum())
            result['comparisons'][label] = {'audit': expected, 'preprocess': found}
            if expected is None:
                differences.append(f'Missing independent audit diagnostic: {label}')
            elif int(expected) != found:
                differences.append(f'{label}: audit={expected}, preprocessing={found}')

        # Each R1 application is compared to its exact source diagnostic.
        # Descriptions distinguish two applications to the same derived variable.
        r1 = {
          'alcfreq = 7 with reported grams': 'alcfreq -> {variable}',
          'alcfreq = 7 with binge frequency reported': 'alcfreq -> alcbnge',
          'hlpfmly = 2 with hours reported': 'hlpfmly -> hlpfmhr',
          'symtc19 = 2 with symtnc19 answered': 'symtc19 -> symtnc19',
          'respc19a = 3 with symtc19 answered': 'respc19a -> symtc19',
          'hltprca 1 or 3 with duration reported': 'hltprca -> cancfre',
          'medtrun = 1 with medtrnu answered': 'medtrun -> medtrnu',
          'medtrun = 2 with reasons marked': None,
        }
        observed_r1 = log.loc[log.rule.eq('R1')]
        for _, row in observed_r1.iterrows():
            description = str(row.description)
            names = [key for key in r1 if description.startswith(key)]
            if len(names) != 1:
                differences.append(f'Unrecognised R1 application: {row.variable}: {description}')
                continue
            template = r1[names[0]]
            if template is None:
                # Audit reports medtrun -> medtr* individually; compare derived
                # reason conflicts against the union in a separate implementation.
                # This is zero in the reference run; nonzero requires review.
                expected = (int((pd.to_numeric(df['medtrun'], errors='coerce').eq(2) &
                     pd.DataFrame({c: pd.to_numeric(df[c], errors='coerce').eq(1) for c in
                       ('medtrnp','medtrnt','medtroc','medtrnl','medtrwl','medtrnaa','medtroth')}).any(axis=1)).sum())
                     if df is not None else None)
            elif names[0] == 'alcfreq = 7 with binge frequency reported':
                # The map's audit has no alcbnge specific diagnostic. Guard
                # against a nonzero unvalidated R1 instead of assuming equality.
                expected = (int((pd.to_numeric(df['alcfreq'], errors='coerce').eq(7) &
                           pd.to_numeric(df['alcbnge'], errors='coerce').isin([1,2,3,4])).sum())
                           if df is not None else None)
            else:
                route = template.format(variable=row.variable)
                diag = lookup.get(route)
                expected = diag.get('unasked_substantive_n') if diag else None
            name = f'R1 {row.variable} / {names[0]}'
            result['comparisons'][name] = {'audit': expected, 'preprocess': int(row.n_all)}
            if expected is None or int(expected) != int(row.n_all):
                differences.append(f'{name}: audit={expected}, preprocessing={row.n_all}')
        if len(observed_r1) != 9:
            differences.append(f'R1 row count changed: {len(observed_r1)}, expected 9')

        # Seven groups only; d_never_worked is a derived downstream decision.
        groups = ('hltpr', 'pacc', 'jbexp', 'jbexe', 'trhlt', 'dshlt', 'medtr')
        for group in groups:
            diag = lookup.get(f'{group} multiresponse flags')
            expected = diag.get('violations') if diag else None
            match('R3', group, expected, f'R3 {group}')
        unexpected_r3 = set(log.loc[log.rule.eq('R3'), 'variable']) - set(groups) - {'d_never_worked'}
        if unexpected_r3:
            differences.append('Unknown R3 applications: ' + ', '.join(sorted(unexpected_r3)))

        # R2 alcohol: independently tabulated audit filter nonresponse paths.
        for dependent in ('alcwkdy', 'alcwknd', 'alcbnge'):
            entries = [lookup.get(f'alcfreq filter_{kind} -> {dependent}')
                       for kind in ('nan', '77_88_99')]
            expected = (sum(int(e['substantive_dependent_n']) for e in entries)
                        if all(e is not None and e.get('substantive_dependent_n') is not None for e in entries)
                        else None)
            match('R2', dependent, expected, f'R2 alcohol {dependent}')

        # Corroborate reported rule totals against the underlying row-level log.
        for rule in ('R1', 'R2', 'R3'):
            count = int(log.loc[log.rule.eq(rule), 'n_all'].sum())
            declared = (summary.get('rule_totals', {}).get(rule) or {}).get('n_all')
            result['comparisons'][f'{rule} summary'] = {'log': count, 'summary': declared}
            if declared != count:
                differences.append(f'{rule} summary/log mismatch: {declared} vs {count}')
    except Exception as exc:
        differences.append(f'Evidence could not be verified: {type(exc).__name__}: {exc}')
    if not differences:
        result['status'] = 'PASS'
        result['detail'] = 'Independent preprocessing matches raw and variable-map SHA-256, records and eligible sample, per-rule R1/R3 and alcohol R2; summary totals verified'
    else:
        result['detail'] = '; '.join(differences[:12]) + (f'; +{len(differences)-12} more' if len(differences)>12 else '')
    return result


def kish(w):
    """Kish effective sample size, (sum w)^2 / sum w^2.

    Args:
        w (array-like): Weights; non-positive or non-finite values are dropped.

    Returns:
        float or None: Effective sample size, or ``None`` without valid weights.
    """
    a=pd.to_numeric(w,errors='coerce')
    a=a[np.isfinite(a) & (a>0)]
    if a.empty: return None
    return float(a.sum()**2/(a*a).sum())

def safe_scalar(x):
    """Convert a numpy scalar to a plain Python value for JSON.

    Args:
        x: Any scalar.

    Returns:
        The Python value, or ``None`` for missing values.
    """
    if pd.isna(x): return None
    if isinstance(x,(np.integer,np.floating)): return x.item()
    return x

def audit_dataframe(df, config, variable_map):
    """Run the full Phase 00 audit on an in-memory data frame.

    Args:
        df (pandas.DataFrame): Raw ESS11 data (numeric codes).
        config (dict): Audit configuration (``config/audit.yml``), including
            the column names and the country thresholds.
        variable_map (pandas.DataFrame): ``config/variable_map.csv``.

    Returns:
        dict: Audit results with ``summary``, ``checks`` (each ``PASS`` or
        ``REVIEW``), ``countries``, ``variable_audit``, ``routing``,
        ``routing_checks``, ``singleton_strata``, ``flow``,
        ``eligibility_age_diagnostics``, ``outcome``, ``gate_evidence`` and
        ``issues`` (all checks that did not pass). Only aggregates are
        returned.
    """
    variable_map = _normalise_variable_map(variable_map)
    checks=[]
    def check(name, ok, detail):
        """Append one check result (PASS if ``ok`` else REVIEW)."""
        checks.append({'check':name,'status':'PASS' if ok else 'REVIEW','detail':str(detail)})
    cols=config['columns']; required=list(cols.values())+ITEMS
    absent=sorted(set(required)-set(df.columns))
    check('Required columns present',not absent, 'Absent: '+', '.join(absent))
    overview={'records':len(df),'columns':len(df.columns),'countries_in_file':int(df[cols['country']].nunique(dropna=True)) if cols['country'] in df else None}
    if absent:
        return {'summary':overview,'checks':checks,'countries':[],'variable_audit':[],'routing':[], 'flow':[], 'outcome':{},'issues':['Required fields absent: '+str(absent)]}
    ids=df[[cols['country'],cols['respondent']]]
    duplicate=int(ids.duplicated(keep=False).sum())
    check('Country + respondent ID unique',duplicate==0,f'{duplicate} duplicate rows')
    check('Country code present',bool(df[cols['country']].notna().all()),'Country missing='+str(int(df[cols['country']].isna().sum())))
    score,invalid,ambiguous=cesd8(df)
    check('CES-D8 valid codes',sum(invalid.values())==0,invalid)
    # With integer seven-item sum s, 8*s/7 cannot have fractional part 0.5.
    # A rounding-tie check is therefore mathematically redundant.

    for label in ('weight_post','weight_analysis'):
        x=pd.to_numeric(df[cols[label]],errors='coerce')
        bad=int((~np.isfinite(x)|(x<=0)).sum())
        check(label+' positive finite',bad==0,str(bad)+' invalid or absent')
    for field in ('psu','stratum'):
        bad=int(df[cols[field]].isna().sum())
        check(field+' present',bad==0,str(bad)+' missing')
    # Unknown sex categories must be checked against source metadata/codebook.
    age=pd.to_numeric(df[cols['age']],errors='coerce')
    # Adult eligibility: age 18-120. Values outside this range (including the
    # ESS code 999 for missing age) are not eligible.
    age_ok=age.between(18,120)
    sex_numeric=pd.to_numeric(df[cols['sex']],errors='coerce')
    sex_present=sex_numeric.isin([1,2])
    post=pd.to_numeric(df[cols['weight_post']],errors='coerce')
    analysis=pd.to_numeric(df[cols['weight_analysis']],errors='coerce')
    weights_ok=np.isfinite(post)&(post>0)&np.isfinite(analysis)&(analysis>0)
    outcome_valid=score.cesd8_score.notna()
    eligible=age_ok&sex_present&weights_ok&outcome_valid
    check('Sex code valid for eligibility',bool((sex_numeric.isin([1,2,9]) | sex_numeric.isna()).all()),'Allowed substantive codes 1/2; 9 is nonresponse')
    age_missing=int((age.isna() | age.eq(999)).sum())
    age_minor=int(age.between(0,17).sum())
    stages=[('All records',len(df)),('Age recorded',len(df)-age_missing),('Adult age 18-120',int(age_ok.sum())),('Age and sex present',int((age_ok&sex_present).sum())),('Plus valid CES-D8',int((age_ok&sex_present&outcome_valid).sum())),('Plus valid weights (eligible)',int(eligible.sum()))]
    flow=[{'stage': name, 'remaining': n, 'excluded': (stages[i-1][1] - n if i else 0)} for i,(name,n) in enumerate(stages)]
    d=df.loc[eligible].copy(); d['__eds9__']=score.loc[eligible,'eds_9'].to_numpy()
    countries=[]
    for name,g in d.groupby(cols['country'],dropna=False,sort=True):
        y=g['__eds9__']; ncase=int(y.sum()); ncontrol=int((~y).sum()); n=len(g)
        # Country/stratum/PSU key must be unique at country + stratum + PSU granularity.
        units=g.groupby(cols['stratum'],dropna=False)[cols['psu']].nunique(dropna=True)
        one=int((units==1).sum()); zero=int((units==0).sum())
        countries.append({'country':str(name),'eligible_n':n,'events':ncase,'non_events':ncontrol,
             'unweighted_eds9':float(y.mean()),'kish_n_pspwght':kish(g[cols['weight_post']]),
             'weighted_eds9':float(np.average(y,weights=g[cols['weight_post']])),
             'strata':int(len(units)),'single_psu_strata':one,'empty_psu_strata':zero,
             'meets_country_n':bool(n>=config['country_min_eligible']),
             'eligible_for_loco_metrics':bool(ncase>=config['country_min_cases'] and ncontrol>=config['country_min_controls']),
             'precision_halfwidth_95':auroc_precision_halfwidth(ncase,ncontrol,kish(g[cols['weight_post']])),
             'precision_status':('FLAG: expected AUROC CI half-width >0.07' if auroc_precision_halfwidth(ncase,ncontrol,kish(g[cols['weight_post']]))>0.07 else 'PASS: expected AUROC CI half-width <=0.07')})
    check('Nonempty eligible sample',len(d)>0,f'{len(d)} eligible')
    singleton_detail=strata_diagnostics(df,eligible,cols)
    check('Source sampling strata have >=2 PSUs',
          all(r['origin']!='source_singleton' for r in singleton_detail),
          f"{sum(r['origin']=='source_singleton' for r in singleton_detail)} source singleton strata; {sum(r['origin']=='created_by_eligibility' for r in singleton_detail)} created only by eligibility; use domain estimation")
    # Report prevalence descriptive only; no predictive accuracy or outcome-guided tuning.
    outcome={'complete':int((score.answered_items==8).sum()),'one_missing_prorated':int(score.prorated.sum()),
             'two_or_more_missing':int((score.answered_items<7).sum()),
             'eligible_prorated':int((eligible & score.prorated).sum()), 'range_min':safe_scalar(score.cesd8_score.min()),
             'range_max':safe_scalar(score.cesd8_score.max()),'cutoffs':{str(c):{'n_positive':int(score.loc[eligible,'cesd8_score'].ge(c).sum()),'unweighted_share':float(score.loc[eligible,'cesd8_score'].ge(c).mean()) if eligible.any() else None} for c in [8,9,10]}}
    # The study team's original codebook-derived map is authoritative for variable
    # names, domain descriptions and raw missing-code declarations. No new column
    # (set/review_status/NA) is required. Missing codes are NOT automatically
    # transformed because the map combines routed Not Applicable and nonresponse.
    variable_audit=[]
    invalid_mapped_codes=[]
    for row in variable_map.to_dict('records'):
        name=str(row['variable']).strip()
        found=name in df.columns
        codes=_parse_codes(row.get('missing_codes',''))
        if found:
            series=df[name]
            numeric=pd.to_numeric(series, errors='coerce')
            code_counts={str(c):int(numeric.eq(c).sum()) for c in codes}
            mapped_n=int(numeric.isin(codes).sum())
            missing_raw=int(series.isna().sum())
        else:
            code_counts={str(c):None for c in codes}
            mapped_n=None
            missing_raw=None
        variable_audit.append({
            'variable':name,
            'label':row.get('label',''),
            'role':row.get('role',''),
            'group':row.get('group',''),
            'predictor_set':row.get('set',''),
            'present':bool(found),
            'missing_raw_n':missing_raw,
            'missing_codes':row.get('missing_codes',''),
            'mapped_special_code_n':mapped_n,
            'mapped_special_code_counts':json.dumps(code_counts,ensure_ascii=False),
            'note':row.get('note',''),
            'review_status':row.get('review_status',''),
            'not_applicable_codes':row.get('not_applicable_codes',''),
        })
    check('All mapped fields present', all(r['present'] for r in variable_audit),
          f"{sum(not r['present'] for r in variable_audit)} missing mapped fields")
    check('Codebook-derived variable map loaded', True,
          f'{len(variable_audit)} variables; schema: variable,label,role,group,missing_codes,note')
    missing_role=[r['variable'] for r in variable_audit if not r['role']]
    check('All mapped variables have roles',not missing_role,
          f'{len(missing_role)} variables have an empty role')
    # A source-codebook-derived map is not the same as a verified routing/recode.
    # In particular missing_codes may include NA codes and cannot be blindly
    # converted to NaN until the relevant filter questions have been checked.
    routes=[]
    for filter_name,dependent in ROUTES:
        if filter_name not in df or dependent not in df:
            routes.append({'filter':filter_name,'dependent':dependent,'status':'MISSING_FIELD','rows':None});continue
        tab=(df.groupby([filter_name,dependent],dropna=False).size().reset_index(name='n'))
        routes.append({'filter':filter_name,'dependent':dependent,'status':'REQUIRES_CODEBOOK_VALIDATION','rows':tab.to_dict('records')[:1000], 'truncated':len(tab)>1000})
    routing_checks=appendix_c_diagnostics(df, variable_map)
    n_routing_flags=sum((r.get('violations') or 0) for r in routing_checks)
    # Appendix C requires explicit human approval of routing policy (D33).
    # A populated YAML file by itself never constitutes such approval.
    appendix_signoff = load_signoff(config, ['Appendix C routing validated'])['items'].get('Appendix C routing validated', {})
    approved = appendix_signoff.get('recorded') and str(appendix_signoff.get('decision', '')).strip().lower() in ('approved', 'approve', 'pass')
    check('Appendix C routing validated', bool(approved),
          f'{len(routing_checks)} rule diagnostics; {n_routing_flags} potential contradictions; '
          + ('team signoff recorded' if approved else 'awaiting recorded team approval (D33)'))
    # Independent validation and technical feasibility are each reported only once below.

    # Evidence gates are computed in this module, so the original run_audit.py
    # needs no modification or additional imports. These checks do NOT fit models.
    evidence = _audit_acceptance_evidence(df, eligible, cols, routing_checks, config)
    for name, result in evidence['checks'].items():
        check(name, result['status'] == 'PASS', result['detail'])
    issues=[x['check']+': '+x['detail'] for x in checks if x['status']!='PASS']
    return {'summary':overview,'checks':checks,'countries':countries,'variable_audit':variable_audit,
            'routing':routes,'routing_checks':routing_checks,'singleton_strata':singleton_detail,'flow':flow,
            'eligibility_age_diagnostics':{'missing_or_999':age_missing,'under_18':age_minor},'outcome':outcome,'gate_evidence':evidence,'issues':issues}


# Appendix C diagnostics are checks, not automatic transformations of predictors.
# Zero violations means the *observed routing* is compatible with the stated rule;
# it does not attest to the semantic validity of a derived feature.
def appendix_c_diagnostics(df, variable_map=None):
    """Check observed questionnaire routing against the Appendix C rules.

    For each (filter, dependent) pair, counts respondents who were not asked
    the dependent question but still have a substantive answer, and
    respondents who were asked but have a not-applicable code. For each
    multi-response group, counts conflicts between nonresponse flags, "none
    of these" and marked items. Rows are diagnostics only; nothing is recoded.

    Args:
        df (pandas.DataFrame): Raw data.
        variable_map (pandas.DataFrame, optional): Variable map. If it has a
            ``routing`` column, machine-readable rules (e.g. ``medtrun = 2 or 8``)
            and ``routing_flag`` roles are taken from it; otherwise built-in
            specifications are used.

    Returns:
        list[dict]: One row per rule with ``rule``, ``status`` (``PASS`` or
        ``REVIEW``), counts and a ``reason``.
    """
    # A richer variable map is the authority when it contains routing metadata.
    # The legacy six-column map retains the original diagnostic specifications.
    map_driven = variable_map is not None and 'routing' in variable_map.columns and any(
        str(v).strip() for v in variable_map['routing'].fillna(''))
    specs = [
        ('alcfreq','alcwkdy',{1,2,3,4,5,6,88}, {6666}),
        ('alcfreq','alcwknd',{1,2,3,4,5,6,88}, {6666}),
        ('respc19a','symtc19',{1,2}, {6}),
        ('symtc19','symtnc19',{1}, {6}),
        ('hlpfmly','hlpfmhr',{1}, {66}),
        ('hltprca','cancfre',{2}, {6,66}),
        ('medtrun','medtrnu',{2,8}, {6}),
    ]
    if map_driven:
        specs = []
        for row in variable_map.to_dict('records'):
            rule = str(row.get('routing', '')).strip()
            # Only machine-readable conditions, e.g. 'medtrun = 2 or 8'.
            # Narrative rules (household grid, employment history) stay REVIEW.
            match = re.fullmatch(r'([a-z][a-z0-9]*)\s*=\s*(\d+)(?:\s+or\s+(\d+))*', rule, re.I)
            if not match:
                continue
            filt = match.group(1)
            asked = {int(n) for n in re.findall(r'\d+', rule)}
            specs.append((filt, str(row['variable']), asked,
                          set(_parse_codes(row.get('not_applicable_codes', '')))))
        # The alcohol entries are phrased narratively in the agreed map.
        for dep in ('alcwkdy','alcwknd'):
            row = variable_map.loc[variable_map.variable.eq(dep)]
            if not row.empty:
                specs.append(('alcfreq',dep,{1,2,3,4,5,6,88},
                              set(_parse_codes(row.iloc[0].get('not_applicable_codes','')))))
    out=[]
    for filt,dep,asked,na_codes in specs:
        if filt not in df or dep not in df:
            out.append(dict(rule=f'{filt} -> {dep}',status='REVIEW',tested=0,violations=None,
                            reason='Missing routing field; cannot assess rule'))
            continue
        x=pd.to_numeric(df[filt],errors='coerce')
        y=pd.to_numeric(df[dep],errors='coerce')
        asked_mask=x.isin(asked)
        na_mask=y.isin(na_codes)
        # If unasked, a substantive numeric response contradicts routing,
        # except missing/refusal/don't know/NA. Distinguish unknown from violation.
        map_miss={'alcbnge':{6,7,8,9},'cancfre':{6,7,8,9,66,77,88,99},
                  'hlpfmhr':{66,77,88,99,777,888,999},
                  'symtc19':{6,7,8,9},'symtnc19':{6,7,8,9},
                  'medtrnu':{6,7,8,9},'alcwkdy':{6666,7777,8888,9999},
                  'alcwknd':{6666,7777,8888,9999}}
        known_special=map_miss.get(dep,na_codes)
        known_negative=x.eq(7) if filt=='alcfreq' else x.notna() & ~asked_mask & ~x.isin([7,8,9,77,88,99])
        observed_codes=set(y.dropna().unique())
        binary=bool(observed_codes) and observed_codes.issubset({0,1})
        substantive=y.eq(1) if binary else (y.notna() & ~y.isin(known_special))
        bad_unasked=known_negative & substantive
        # 'asked' doesn't imply all respondents give substantive answers;
        # flag an NA-code found among asked as suspect rather than silently fixing.
        bad_asked=asked_mask&na_mask
        out.append(dict(rule=f'{filt} -> {dep}',status='REVIEW' if (bad_unasked|bad_asked).any() else 'PASS',
                        tested=int(len(df)),asked_n=int(asked_mask.sum()),unasked_n=int((~asked_mask).sum()),
                        violations=int((bad_unasked|bad_asked).sum()),
                        unasked_substantive_n=int(bad_unasked.sum()),asked_not_applicable_n=int(bad_asked.sum()),
                        reason='Codes and allowed routing sets must be confirmed against the source questionnaire; PASS is only an observed-data consistency check'))
    # Read actual multiresponse membership from the role="routing_flag"
    # entries. Never include similarly prefixed standalone variables (hltprca).
    if map_driven and 'role' in variable_map.columns:
        rows = variable_map.to_dict('records')
        flag_names = [str(r['variable']) for r in rows if r['role']=='routing_flag']
        prefixes = sorted({re.sub(r'(ref|dk|na|nt|nap)$', '', n)
                           for n in flag_names if re.search(r'(ref|dk|na|nt|nap)$', n)})
        groups = {}
        for prefix in prefixes:
            flags = [n for n in flag_names if n.startswith(prefix) and
                     re.fullmatch(re.escape(prefix)+r'(ref|dk|na|nap)',n)]
            role_by_name = {str(r['variable']): str(r['role']) for r in rows}
            none = prefix+'nt' if role_by_name.get(prefix+'nt') == 'routing_flag' else None
            items = []
            for row in rows:
                name = str(row['variable'])
                if not name.startswith(prefix) or name in flags or name == none:
                    continue
                if row['role'] not in ('predictor','derivation_input'):
                    continue
                # Separate single-response questions that merely share a prefix.
                if prefix=='medtr' and not re.fullmatch(r'medtrun\s*=\s*1',str(row.get('routing','')).strip()):
                    continue
                if prefix=='hltpr' and name=='hltprca':
                    continue
                items.append(name)
            groups[prefix] = (items, none, flags)
    else:
        groups={
          'hltpr': (['hltprhc','hltprhb','hltprbp','hltpral','hltprbn','hltprpa','hltprpf','hltprsd','hltprsc','hltprsh','hltprdi'], 'hltprnt', ['hltprref','hltprdk','hltprna']),
          'pacc': (['paccmoro','paccdwlr','pacclift','paccnbsh','paccocrw','paccxhoc','paccnois','paccinro'], 'paccnt', ['paccref','paccdk','paccna']),
          'jbexp': (['jbexpvi','jbexpti','jbexpml','jbexpmc'], 'jbexpnt', ['jbexpref','jbexpdk','jbexpna','jbexpnap']),
          'jbexe': (['jbexevl','jbexevh','jbexevc','jbexera','jbexecp','jbexebs'], 'jbexent', ['jbexeref','jbexedk','jbexena','jbexenap']),
          'trhlt': (['trhltacu','trhltacp','trhltcm','trhltch','trhltos','trhltho','trhltht','trhlthy','trhltmt','trhltpt','trhltre','trhltsh'], 'trhltnt', ['trhltref','trhltdk','trhltna']),
          'medtr': (['medtrnp','medtroc','medtrnl','medtrwl','medtrnaa','medtroth'], 'medtrnt', ['medtrnap','medtrref','medtrdk','medtrna']),
          'dshlt': (['dshltgp','dshltms'], 'dshltnt', ['dshltref','dshltdk','dshltna'])
        }
    for prefix,(items,none,flags) in groups.items():
        if not items or not flags:
            out.append(dict(rule=f'{prefix} multiresponse flags',status='REVIEW',violations=None,reason='Incomplete group specification in variable map'))
            continue
        required=items+([none] if none else [])+flags
        absent=[v for v in required if v not in df]
        if absent:
            out.append(dict(rule=f'{prefix} multiresponse flags',status='REVIEW',violations=None,
                            reason='Missing source columns: '+', '.join(absent)))
            continue
        marked=pd.DataFrame({v:pd.to_numeric(df[v],errors='coerce').eq(1) for v in items}).any(axis=1)
        none_marked=pd.to_numeric(df[none],errors='coerce').eq(1) if none else pd.Series(False,index=df.index)
        flag_marked=pd.DataFrame({v:pd.to_numeric(df[v],errors='coerce').eq(1) for v in flags}).any(axis=1)
        # Three disjoint causes: refusal/NA + substantive, none + substantive, none + flag.
        a=flag_marked & marked
        b=none_marked & marked
        c=none_marked & flag_marked
        conflict=a|b|c
        out.append(dict(rule=f'{prefix} multiresponse flags',status='REVIEW' if conflict.any() else 'PASS',
            tested=len(df),flag_n=int(flag_marked.sum()),violations=int(conflict.sum()),
            flag_with_substantive_n=int(a.sum()),none_with_substantive_n=int(b.sum()),
            none_with_flag_n=int(c.sum()),
            reason='Explicit item list; hltprca excluded; 0/1 flags and none-of-above checked independently'))
    # Document that other conditional categories need external grid/history metadata.
    for var,context in [('chldhhe','household grid F4'),('emplrel','paid employment history')]:
        if var in df:
            raw=pd.to_numeric(df[var],errors='coerce')
            out.append(dict(rule=f'{var} structural NA',status='REVIEW',tested=len(df),
                violations=None,na_code_6_n=int(raw.eq(6).sum()),
                reason=f'Code 6 needs {context} to derive analytic variable; count is NOT a contradiction'))
    if 'alcfreq' in df:
        freq=pd.to_numeric(df['alcfreq'],errors='coerce')
        for dep in ['alcwkdy','alcwknd','alcbnge']:
            if dep not in df: continue
            yy=pd.to_numeric(df[dep],errors='coerce')
            dep_special={'alcwkdy':{6666,7777,8888,9999},'alcwknd':{6666,7777,8888,9999},
                         'alcbnge':{6,7,8,9}}[dep]
            actual=yy.notna() & ~yy.isin(dep_special)
            for name,mask in [('filter_nan',freq.isna()),('filter_77_88_99',freq.isin([77,88,99]))]:
                out.append(dict(rule=f'alcfreq {name} -> {dep}',status='REVIEW',
                    tested=int(mask.sum()),violations=None,substantive_dependent_n=int((mask&actual).sum()),
                    by_country_mode='See audit_summary routing evidence',
                    reason='Unknown filter with substantive dependent is retained for recoding; not a confirmed routing violation'))
    return out

def strata_diagnostics(df, eligible, cols):
    """Compare original and eligible PSU counts; never infer source singletons.

    Args:
        df (pandas.DataFrame): Raw data (all respondents).
        eligible (pandas.Series): Eligibility flag.
        cols (dict): Column names from the audit configuration.

    Returns:
        list[dict]: One row per stratum that has at most one PSU among
        eligible respondents. ``origin`` tells whether the stratum already had
        one PSU in the source file (``source_singleton``) or only after
        eligibility filtering (``created_by_eligibility``).
    """
    keys = [cols['country'], cols['stratum']]
    source = df.groupby(keys, dropna=False)[cols['psu']].agg(['size','nunique'])
    kept = df.loc[eligible].groupby(keys, dropna=False)[cols['psu']].agg(['size','nunique'])
    rows = []
    for (nation, sid), eligible_stats in kept.iterrows():
        count = int(eligible_stats['nunique'])
        if count > 1:
            continue
        original = source.loc[(nation, sid)]
        original_count = int(original['nunique'])
        rows.append({'country':str(nation), 'stratum':str(sid),
                     'source_respondents':int(original['size']),
                     'source_psu':original_count,
                     'eligible_respondents':int(eligible_stats['size']),
                     'eligible_psu':count,
                     'origin':'created_by_eligibility' if original_count > 1 else 'source_singleton',
                     'action':'REVIEW DESIGN; never merge strata automatically'})
    return rows


def _singleton_policy(df, eligible, cols):
    """Protocol v0.5 fallback, recorded but NOT silently applied to survey analysis.

    The approved approach is domain estimation on the full source design
    (decision D34), so this proposal is only reported.

    Args:
        df (pandas.DataFrame): Raw data.
        eligible (pandas.Series): Eligibility flag.
        cols (dict): Column names.

    Returns:
        list[dict]: Per country with single-PSU strata, the proposed fallback
        and the status ``PROPOSED_NOT_APPLIED``.
    """
    out=[]
    country,stratum,psu=(cols[k] for k in ('country','stratum','psu'))
    for nation, country_data in df.loc[eligible].groupby(country, sort=True):
        singleton=[]
        for sid, sg in country_data.groupby(stratum):
            if sg[psu].nunique(dropna=True)==1:
                singleton.append(sid)
        if not singleton:
            continue
        affected=country_data.loc[country_data[stratum].isin(singleton),[stratum,psu]].drop_duplicates()
        pool_n=len(affected)  # distinct (stratum, PSU) pairs, not raw PSU codes
        action=('combine_singletons_into_pseudo_stratum' if pool_n>=2
                else 'country_unstratified_fallback')
        out.append({'country':str(nation),'original_singleton_strata':[str(v) for v in singleton],
                    'distinct_pooled_psus':pool_n,'proposed_fallback':action,
                    'status':'PROPOSED_NOT_APPLIED',
                    'reason':'ESS sampling-domain partner metadata not provided; no guessed stratum partner'})
    return out


def _rao_wu_replicates(data, outcome, *, reps=1000, seed=20261013):
    """Rao-Wu n_h-1 PSU rescaling with weighted ratio estimator.

    Here the input country must contain >=2 PSUs in each stratum. Never
    replace singletons implicitly; primary-model fallback is a separate gate.

    Args:
        data (pandas.DataFrame): Columns ``stratum``, ``psu``, ``weight`` and
            the outcome column.
        outcome (str): Column whose weighted mean is estimated.
        reps (int): Number of replicates.
        seed (int): Random seed.

    Returns:
        float: Bootstrap standard error of the weighted mean.

    Raises:
        ValueError: If values are not finite, weights are not positive, or a
            stratum has one PSU.
    """
    x=data.reset_index(drop=True)
    w=x['weight'].to_numpy(dtype=float)
    y=x[outcome].to_numpy(dtype=float)
    if not (np.isfinite(w).all() and np.isfinite(y).all() and (w>0).all()):
        raise ValueError('Rao-Wu validation requires finite values and positive weights')
    groups=[]
    for _, sg in x.groupby('stratum', sort=True):
        psus, codes=np.unique(sg['psu'].to_numpy(),return_inverse=True)
        if len(psus)<2:
            raise ValueError('Singleton stratum; resolve before Rao-Wu comparison')
        groups.append((sg.index.to_numpy(),codes,len(psus)))
    rng=np.random.default_rng(seed)
    values=np.empty(reps)
    for r in range(reps):
        multiplier=np.zeros(len(x))
        for indices,codes,npsu in groups:
            draw=rng.integers(0,npsu,size=npsu-1)
            multiplicity=np.bincount(draw,minlength=npsu)
            multiplier[indices]=multiplicity[codes]*npsu/(npsu-1)
        wr=w*multiplier
        values[r]=np.dot(wr,y)/wr.sum()
    return float(np.std(values,ddof=0))


def _rao_wu_reference(df, eligible, cols):
    """Run independent R survey subbootstrap and Taylor reference, locally.

    A per-respondent temporary input is always inside a private TemporaryDirectory
    and deleted afterwards. No additional repository R file is required.

    Compares the Python Rao-Wu standard errors (Austria; weighted mean age
    and share with CES-D8 >= 9) with the R ``survey`` subbootstrap and the
    Taylor-linearisation standard errors. PASS requires less than 10%
    relative difference for both.

    Args:
        df (pandas.DataFrame): Raw data.
        eligible (pandas.Series): Eligibility flag.
        cols (dict): Column names.

    Returns:
        dict: ``status``, ``reason`` and the Python and R standard errors.
    """
    result={'status':'REVIEW','country':'AT','replicates':1000,
            'reason':'Not attempted','python_se':{},'r_subbootstrap_se':{},'r_taylor_se':{}}
    rscript=shutil.which('Rscript')
    if not rscript:
        result['reason']='Rscript not installed; independent variance comparison not run'
        return result
    try:
        selected=df.loc[eligible & df[cols['country']].eq('AT')]
        if selected.empty:
            result['reason']='Austria missing from eligible sample'; return result
        if (selected.groupby(cols['stratum'])[cols['psu']].nunique()<2).any():
            result['reason']='Reference country contains singleton strata'; return result
        ces,_,_=cesd8(selected)
        frame=pd.DataFrame({'stratum':selected[cols['stratum']].to_numpy(),
                            'psu':selected[cols['psu']].to_numpy(),
                            'weight':selected[cols['weight_post']].to_numpy(),
                            'age':pd.to_numeric(selected[cols['age']],errors='coerce').to_numpy(),
                            'binary':ces['eds_9'].astype(float).to_numpy()})
        # One representative continuous variable and one binary proportion.
        py={k:_rao_wu_replicates(frame,k,seed=20261013+i) for i,k in enumerate(('age','binary'))}
        r_code="""
        suppressPackageStartupMessages(library(survey))
        suppressPackageStartupMessages(library(jsonlite))
        args <- commandArgs(trailingOnly=TRUE)
        x <- read.csv(args[1]); set.seed(20261014)
        d <- svydesign(ids=~psu, strata=~stratum, weights=~weight, data=x, nest=TRUE)
        b <- as.svrepdesign(d, type="subbootstrap", replicates=1000)
        rse <- SE(svymean(~age+binary, b))
        tse <- SE(svymean(~age+binary, d))
        write_json(list(bootstrap=list(age=unname(rse[1]),binary=unname(rse[2])),
                        taylor=list(age=unname(tse[1]),binary=unname(tse[2]))),
                   args[2], auto_unbox=TRUE)
        """
        with tempfile.TemporaryDirectory(prefix='ess_audit_rao_wu_') as td:
            inp=Path(td)/'validation.csv'; out=Path(td)/'reference.json'
            frame.to_csv(inp,index=False)
            run=subprocess.run([rscript,'-e',r_code,str(inp),str(out)],
                               capture_output=True,text=True,timeout=240)
            if run.returncode:
                result['reason']='R survey failed: '+run.stderr.strip()[-600:]
                return result
            ref=json.loads(out.read_text(encoding='utf-8'))
        rb=ref['bootstrap']; rt=ref['taylor']
        rel={k:abs(py[k]-rb[k])/rb[k] if rb[k]>0 else float('inf') for k in py}
        taylor_rel={k:abs(py[k]-rt[k])/rt[k] if rt[k]>0 else float('inf') for k in py}
        result.update(python_se=py,r_subbootstrap_se=rb,r_taylor_se=rt,
                      relative_disagreement=rel,taylor_relative_disagreement=taylor_rel)
        if all(np.isfinite(list(rel.values()))) and all(v<0.10 for v in rel.values()) and all(v<0.10 for v in taylor_rel.values()):
            result.update(status='PASS',reason='Python and independent R bootstrap SE differ <10%; Taylor proximity also <10%')
        else:
            result['reason']='Bootstrap/Taylor SE disagreement >=10%; independent review required'
    except Exception as exc:
        result['reason']=f'R/Python validation unavailable: {type(exc).__name__}: {exc}'
    return result


def _domain_bootstrap_reference(df, eligible, cols, country='FI', reps=500):
    """Build replicate weights on source PSUs, then estimate eligible-domain prevalence.

    Compare to R survey::subset() on the replicate design. Only the source
    design determines the PSU sampling; no eligibility-created strata merging.

    This validates decision D34: Finland has strata that have one PSU only
    after eligibility filtering. PASS requires less than 10% relative
    difference from R.

    Args:
        df (pandas.DataFrame): Raw data.
        eligible (pandas.Series): Eligibility flag.
        cols (dict): Column names.
        country (str): Country used for the check.
        reps (int): Number of replicates.

    Returns:
        dict: ``status``, ``reason``, ``python_se``, ``r_se`` and
        ``relative_disagreement``.
    """
    result={'status':'REVIEW','country':country,'reason':'Not attempted','replicates':reps}
    executable=shutil.which('Rscript')
    if not executable: result['reason']='Rscript unavailable'; return result
    try:
        mask=df[cols['country']].eq(country)
        source=df.loc[mask]
        if source.empty: result['reason']='Reference country absent'; return result
        sample=source.copy().reset_index(drop=True)
        em=eligible.loc[source.index].to_numpy(dtype=bool)
        if not em.any(): result['reason']='No eligible domain'; return result
        psu_counts=sample.groupby(cols['stratum'])[cols['psu']].nunique()
        if (psu_counts<2).any():
            result['reason']='Source stratum itself has one PSU; cannot use this validation';return result
        score,_,_=cesd8(sample)
        x=pd.DataFrame({'stratum':sample[cols['stratum']], 'psu':sample[cols['psu']],
                        'weight':pd.to_numeric(sample[cols['weight_post']],errors='coerce'),
                        'eligible':em.astype(int), 'outcome':score['eds_9'].astype(float)})
        if (x.weight<=0).any() or x.weight.isna().any():
            result['reason']='Invalid source weights';return result
        grp=[]
        for _,sub in x.groupby('stratum',sort=True):
            psus,codes=np.unique(sub.psu.to_numpy(),return_inverse=True)
            grp.append((sub.index.to_numpy(),codes,len(psus)))
        y=x.outcome.to_numpy(); w=x.weight.to_numpy(); d=x.eligible.to_numpy()
        rng=np.random.default_rng(20261015); estimates=[]
        for _ in range(reps):
            m=np.zeros(len(x))
            for idx,codes,n in grp:
                drawn=rng.integers(n,size=n-1)
                counts=np.bincount(drawn,minlength=n)
                m[idx]=counts[codes]*n/(n-1)
            wr=w*m*d; den=wr.sum()
            if den>0: estimates.append(np.dot(wr,y)/den)
        if len(estimates)!=reps:
            result['reason']='Domain empty in one or more replicates';return result
        py=float(np.std(estimates,ddof=0))
        r_code="""
suppressPackageStartupMessages(library(survey))
suppressPackageStartupMessages(library(jsonlite))
a<-commandArgs(trailingOnly=TRUE); x<-read.csv(a[1]);set.seed(20261016)
d<-svydesign(ids=~psu,strata=~stratum,weights=~weight,data=x,nest=TRUE)
r<-as.svrepdesign(d,type='subbootstrap',replicates=as.integer(a[3]))
rd<-subset(r,eligible==1)
write_json(list(se=unname(SE(svymean(~outcome,rd))[1])),a[2],auto_unbox=TRUE)
"""
        with tempfile.TemporaryDirectory(prefix='ess_domain_') as td:
            csv=Path(td)/'source.csv'; output=Path(td)/'r.json';x.to_csv(csv,index=False)
            res=subprocess.run([executable,'-e',r_code,str(csv),str(output),str(reps)],
                               capture_output=True,text=True,timeout=240)
            if res.returncode:
                result['reason']='R domain reference failed: '+res.stderr[-700:];return result
            rse=float(json.loads(output.read_text())['se'])
        rel=abs(py-rse)/rse if rse>0 else float('inf')
        result.update(python_se=py,r_se=rse,relative_disagreement=rel)
        if np.isfinite(rel) and rel<0.10:
            result.update(status='PASS',reason='Source-design Rao-Wu then domain subset agrees with R within 10%')
        else: result['reason']='R/Python domain discrepancy >=10%; review before sign-off'
    except Exception as exc:
        result['reason']=f'Domain variance validation unavailable: {type(exc).__name__}: {exc}'
    return result


def _technical_probe():
    """Inspect library APIs; optionally fit TabPFN to small synthetic CPU data.

    Opt in with ESS_AUDIT_TABPFN_CPU_SMOKE=1. This trial can download pretrained
    weights and is bounded by ESS_AUDIT_TABPFN_TIMEOUT_SECONDS (default 120).
    No ESS observations are used. A successful import is not a successful fit.

    The synthetic fit runs in a separate process with GPUs hidden, so a crash
    or time-out cannot stop the audit.

    Returns:
        dict: Installed versions, whether ``TabPFNClassifier.fit`` accepts
        ``sample_weight``, the trial result and a ``status``.
    """
    versions = package_versions()['packages']
    result = {'status': 'REVIEW',
              'versions': {k: versions.get(k, 'not-installed') for k in
                  ('scikit-learn','xgboost','lightgbm','catboost','torch','tabpfn')},
              'tabpfn_sample_weight': None, 'resource_trial': 'NOT_RUN',
              'device': 'cpu', 'reason': ''}
    if versions.get('tabpfn') == 'not-installed':
        result['reason'] = 'TabPFN not installed; synthetic CPU fit not run'
        return result
    try:
        from tabpfn import TabPFNClassifier
        result['tabpfn_sample_weight'] = ('sample_weight' in
            inspect.signature(TabPFNClassifier.fit).parameters)
    except Exception as exc:
        result['reason'] = f'TabPFN API import failed: {type(exc).__name__}: {exc}'
        return result
    if os.environ.get('ESS_AUDIT_TABPFN_CPU_SMOKE') != '1':
        result['reason'] = ('API inspection complete; set ESS_AUDIT_TABPFN_CPU_SMOKE=1 '
                            'to run bounded synthetic fit on CPU (may download weights)')
        return result
    try:
        timeout = max(15, min(900, int(os.environ.get('ESS_AUDIT_TABPFN_TIMEOUT_SECONDS','120'))))
    except ValueError:
        result['reason'] = 'ESS_AUDIT_TABPFN_TIMEOUT_SECONDS must be an integer'
        return result
    script = r'''
import json, time, numpy as np, torch
from tabpfn import TabPFNClassifier
rng = np.random.default_rng(20261013)
X = rng.normal(size=(64, 6)).astype('float32')
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype('int64')
torch.set_num_threads(2)
t0 = time.monotonic()
model = TabPFNClassifier(device='cpu')
model.fit(X[:48], y[:48])
pred = model.predict_proba(X[48:])
assert pred.shape == (16, 2) and np.isfinite(pred).all()
print(json.dumps({'fit_predict_seconds': round(time.monotonic()-t0,3),
                  'training_n': 48, 'test_n': 16, 'features': 6,
                  'cpu_only': True}))
'''
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['OMP_NUM_THREADS'] = '2'
    env['MKL_NUM_THREADS'] = '2'
    started = time.monotonic()
    try:
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                   text=True, timeout=timeout, env=env)
        result['wall_seconds'] = round(time.monotonic()-started, 3)
        if completed.returncode:
            result['resource_trial'] = 'FAILED'
            result['reason'] = ('TabPFN CPU synthetic trial failed: '
                                + (completed.stderr or completed.stdout)[-1200:])
        else:
            # Some libraries log to stdout; accept the last standalone JSON line.
            trial = next((json.loads(line) for line in reversed(completed.stdout.splitlines())
                          if line.strip().startswith('{') and line.strip().endswith('}')), None)
            if not trial or trial.get('cpu_only') is not True:
                raise ValueError('No verified CPU fit summary in subprocess output')
            result['resource_trial'] = 'PASS'
            result['trial'] = trial
            result['status'] = 'PASS'
            result['reason'] = 'Synthetic TabPFN CPU fit and prediction succeeded; full-scale feasibility not established'
    except subprocess.TimeoutExpired:
        result['resource_trial'] = 'TIMEOUT'
        result['reason'] = f'TabPFN synthetic CPU run exceeded {timeout}s; do not infer full-scale feasibility'
    except Exception as exc:
        result['resource_trial'] = 'FAILED'
        result['reason'] = f'TabPFN CPU trial assessment failed: {type(exc).__name__}: {exc}'
    return result


def _audit_acceptance_evidence(df,eligible,cols,routing_checks,config=None):
    """Evidence for existing Study Design gates. Never self-approve based on flags.

    Args:
        df (pandas.DataFrame): Raw data.
        eligible (pandas.Series): Eligibility flag.
        cols (dict): Column names.
        routing_checks (list[dict]): Output of :func:`appendix_c_diagnostics`.
        config (dict, optional): Audit configuration.

    Returns:
        dict: The gate ``checks`` and all supporting evidence (sign-off
        record, singleton policy, bootstrap references, technical probe,
        preprocessing cross-check, routing review and alcohol filter counts).
    """
    singletons=_singleton_policy(df,eligible,cols)
    validation=_rao_wu_reference(df,eligible,cols)
    domain_validation=_domain_bootstrap_reference(df,eligible,cols)
    technical=_technical_probe()
    preprocessing=_preprocessing_reference(config, routing_checks, df, eligible)
    alcohol_missing=[]
    if {'alcfreq','alcwkdy','alcwknd','alcbnge'}.issubset(df.columns):
        missing_filter=pd.to_numeric(df['alcfreq'],errors='coerce').isna()
        mode=df['mode'] if 'mode' in df else pd.Series('NOT_AVAILABLE',index=df.index)
        for (nation,md),idx in df.loc[missing_filter].groupby([cols['country'],mode.loc[missing_filter]],dropna=False).groups.items():
            row={'country':str(nation),'mode':str(md),'filter_missing_n':len(idx)}
            for dep,codes in [('alcwkdy',{6666,7777,8888,9999}),('alcwknd',{6666,7777,8888,9999}),('alcbnge',{6,7,8,9})]:
                y=pd.to_numeric(df.loc[idx,dep],errors='coerce')
                row[dep+'_substantive_n']=int((y.notna() & ~y.isin(codes)).sum())
            alcohol_missing.append(row)
    flagged=sum(int(x.get('violations') or 0) for x in routing_checks)
    unresolved=[{'rule':x['rule'],'flags':x.get('violations'),
                 'status':'REVIEW' if x.get('violations') else 'DIAGNOSTIC_ZERO'}
                for x in routing_checks]
    checks={
      'Domain-design inference validated on eligibility-created singletons':{'status':domain_validation['status'] if singletons else 'PASS',
        'detail':domain_validation['reason']},
      'Independent R Rao-Wu variance reference':{'status':validation['status'],
        'detail':validation['reason']},
      'Synthetic hardware and API feasibility verified':{'status':technical['status'],
        'detail':technical['reason']},
      'Routed predictor recoding independently verified':{'status':preprocessing['status'],
        'detail':preprocessing['detail']},
    }
    signoff = load_signoff(config or {}, {**checks, 'Appendix C routing validated': {}})
    return {'checks':checks,'signoff':signoff,'singleton_policy':singletons,'bootstrap_reference':validation,'domain_bootstrap_reference':domain_validation,
            'technical_feasibility':technical,'preprocessing_reference':preprocessing,'routing_review':unresolved,
            'alcohol_missing_filter_by_country_mode':alcohol_missing,
            'policy':'A diagnostic zero, an R reference PASS, or installed libraries alone do not approve all audit gates.'}

def package_versions():
    """Record the Python version, platform and package versions.

    Returns:
        dict: ``python``, ``platform`` and ``packages`` (name -> version or
        ``"not-installed"``).
    """
    names=['pandas','numpy','scipy','pyreadstat','PyYAML','pytest','psutil','scikit-learn','tabpfn','torch','xgboost','lightgbm','catboost']
    d={}
    for x in names:
        try:d[x]=importlib.metadata.version(x)
        except importlib.metadata.PackageNotFoundError:d[x]='not-installed'
    return {'python':sys.version.split()[0],'platform':platform.platform(),'packages':d}

def write_results(data,folder):
    """Write the audit summary (JSON) and its main tables (CSV).

    Args:
        data (dict): Output of :func:`audit_dataframe` plus metadata.
        folder (str or pathlib.Path): Output folder; created if needed.

    Returns:
        pathlib.Path: Path of ``audit_summary.json``.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    # Summary only, never individual-level information
    (folder/'audit_summary.json').write_text(json.dumps(data,indent=2,default=str,ensure_ascii=False))
    for key in ['checks','countries','variable_audit','flow','routing_checks','singleton_strata']:
        pd.DataFrame(data.get(key,[])).to_csv(folder/(key+'.csv'),index=False)
    return folder/'audit_summary.json'
