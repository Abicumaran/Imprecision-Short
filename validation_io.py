"""Input and ordering rules shared by the four validated app packages."""
from pathlib import Path
import hashlib
import io
import re
import numpy as np
import pandas as pd

ANALYTE_ORDER = ['RBC', 'WBC_2', 'PLT_3', 'HCT', 'HGB', 'MCV_3', 'RDW_3',
                 'MCH', 'MCHC', 'NEUT_2', 'LYMPH_2', 'MXD_2']
FALLBACKS = {'WBC_2': ['WBC'], 'PLT_3': ['PLT'], 'MCV_3': ['MCV'],
             'RDW_3': ['RDW'], 'NEUT_2': ['NEU_2', 'NEUT', 'NEU'],
             'LYMPH_2': ['LYMPH'], 'MXD_2': ['MXD']}

def default_analytes(columns):
    columns = set(columns)
    return [next(c for c in [a] + FALLBACKS.get(a, []) if c in columns)
            for a in ANALYTE_ORDER if any(c in columns for c in [a] + FALLBACKS.get(a, []))]

def identifier(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value).strip()

def canonical_rows(df):
    """Unique positional index and deterministic rows; never silently deduplicate."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if out.columns.duplicated().any():
        raise ValueError('Duplicate column names after trimming spaces.')
    keys = [c for c in ['Condition', 'Interferent', 'Level', 'bloodSampleId',
                        'Blood Sample ID', 'sample_id', 'donor', 'specimen_type',
                        'Day', 'Device', 'deviceId', 'Replicate', 'batch_id'] if c in out]
    for c in keys:
        if c not in ['Replicate']:
            out[c] = out[c].map(identifier)
    if keys:
        # A string sort keeps IDs stable across mixed Excel and CSV dtypes.
        order = out[keys].fillna('').astype(str).sort_values(keys, kind='stable').index
        # Duplicate source indices are common after concatenating workbooks.
        tmp = out.reset_index(drop=True)
        order = tmp[keys].fillna('').astype(str).sort_values(keys, kind='stable').index
        out = tmp.loc[order]
    return out.reset_index(drop=True)

def read_input(source):
    name = str(getattr(source, 'name', source)).lower()
    if hasattr(source, 'seek'):
        source.seek(0)
    if name.endswith('.csv'):
        df = pd.read_csv(source, float_precision='round_trip')
    elif name.endswith('.xlsx'):
        df = pd.read_excel(source, engine='openpyxl')
    else:
        raise ValueError('Upload an .xlsx or .csv file; convert legacy .xls to .xlsx first.')
    return canonical_rows(df)

def stable_rng(seed, *key):
    digest = hashlib.sha256('|'.join(map(str, key)).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([int(seed), int.from_bytes(digest[:8], 'little')]))

def finite_numeric(values):
    return pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
