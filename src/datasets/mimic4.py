"""MIMIC-IV dataset for in-hospital mortality prediction.

Cohort: Hospital admissions with at least one ICU stay.
Target: In-hospital mortality (hospital_expire_flag).

Feature groups:
  A. Demographics: age, gender
  B. Admission: type, location, insurance, marital status, race (grouped), ED indicators
  C. Diagnoses: count + ICD chapter group counts
  D. Procedures: count + procedure group counts
  E. Medications: unique drug count, unique route count
  F. Lab results: mean/min/max for 15 key labs
  G. ICU: LOS, number of stays, care unit (one-hot)
"""

import os
import torch
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

GLOBAL_TEST_SIZE = 0.2

# 15 common lab items (itemid -> short name)
LAB_ITEMS = {
    51222: 'hemoglobin',
    50912: 'creatinine',
    51006: 'bun',
    51301: 'wbc',
    51265: 'platelets',
    50983: 'sodium',
    50971: 'potassium',
    50931: 'glucose',
    50882: 'bicarbonate',
    50813: 'lactate',
    50868: 'anion_gap',
    51237: 'inr',
    50861: 'alt',
    50878: 'ast',
    51003: 'troponin',
}

# Race -> grouped label
RACE_MAP = {
    'WHITE': 'white', 'WHITE - OTHER EUROPEAN': 'white',
    'WHITE - RUSSIAN': 'white', 'WHITE - EASTERN EUROPEAN': 'white',
    'WHITE - BRAZILIAN': 'white', 'PORTUGUESE': 'white',
    'BLACK/AFRICAN AMERICAN': 'black', 'BLACK/CAPE VERDEAN': 'black',
    'BLACK/CARIBBEAN ISLAND': 'black', 'BLACK/AFRICAN': 'black',
    'HISPANIC/LATINO - PUERTO RICAN': 'hispanic', 'HISPANIC OR LATINO': 'hispanic',
    'HISPANIC/LATINO - DOMINICAN': 'hispanic', 'HISPANIC/LATINO - GUATEMALAN': 'hispanic',
    'HISPANIC/LATINO - SALVADORAN': 'hispanic', 'HISPANIC/LATINO - COLUMBIAN': 'hispanic',
    'HISPANIC/LATINO - MEXICAN': 'hispanic', 'HISPANIC/LATINO - HONDURAN': 'hispanic',
    'HISPANIC/LATINO - CUBAN': 'hispanic', 'HISPANIC/LATINO - CENTRAL AMERICAN': 'hispanic',
    'SOUTH AMERICAN': 'hispanic',
    'ASIAN': 'asian', 'ASIAN - CHINESE': 'asian',
    'ASIAN - SOUTH EAST ASIAN': 'asian', 'ASIAN - ASIAN INDIAN': 'asian',
    'ASIAN - KOREAN': 'asian',
}

# ICU care unit -> grouped label
CAREUNIT_MAP = {
    'Medical Intensive Care Unit (MICU)': 'MICU',
    'Medical/Surgical Intensive Care Unit (MICU/SICU)': 'MICU_SICU',
    'Cardiac Vascular Intensive Care Unit (CVICU)': 'CVICU',
    'Surgical Intensive Care Unit (SICU)': 'SICU',
    'Coronary Care Unit (CCU)': 'CCU',
    'Trauma SICU (TSICU)': 'TSICU',
    'Neuro Intermediate': 'Neuro',
    'Neuro Surgical Intensive Care Unit (Neuro SICU)': 'Neuro',
    'Neuro Stepdown': 'Neuro',
}

# ICD-9 diagnosis numeric-prefix ranges -> chapter
_ICD9_DX_RANGES = [
    (1, 139, 'infectious'), (140, 239, 'neoplasms'), (240, 279, 'endocrine'),
    (280, 289, 'blood'), (290, 319, 'mental'), (320, 389, 'nervous'),
    (390, 459, 'circulatory'), (460, 519, 'respiratory'), (520, 579, 'digestive'),
    (580, 629, 'genitourinary'), (630, 679, 'pregnancy'), (680, 709, 'skin'),
    (710, 739, 'musculoskeletal'), (740, 759, 'congenital'), (760, 779, 'perinatal'),
    (780, 799, 'symptoms'), (800, 999, 'injury'),
]

# ICD-10 diagnosis first-letter -> chapter
_ICD10_DX_LETTER = {
    'A': 'infectious', 'B': 'infectious', 'C': 'neoplasms',
    'E': 'endocrine', 'F': 'mental', 'G': 'nervous', 'H': 'nervous',
    'I': 'circulatory', 'J': 'respiratory', 'K': 'digestive',
    'L': 'skin', 'M': 'musculoskeletal', 'N': 'genitourinary',
    'O': 'pregnancy', 'P': 'perinatal', 'Q': 'congenital',
    'R': 'symptoms', 'S': 'injury', 'T': 'injury',
    'V': 'external', 'W': 'external', 'X': 'external', 'Y': 'external',
    'Z': 'supplementary',
}

# ICD-9 procedure first-2-digit ranges -> group
_ICD9_PX_RANGES = [
    (1, 5, 'nervous_proc'), (30, 34, 'respiratory_proc'),
    (35, 39, 'cardiovascular_proc'), (42, 54, 'digestive_proc'),
    (55, 71, 'genitourinary_proc'), (76, 84, 'musculoskeletal_proc'),
]

# ICD-10 PCS section-0 body-system char -> group
_ICD10_PCS_BODY = {
    '0': 'nervous_proc', '1': 'nervous_proc',
    '2': 'cardiovascular_proc', '3': 'cardiovascular_proc',
    '4': 'cardiovascular_proc', '5': 'cardiovascular_proc',
    '6': 'cardiovascular_proc',
    'B': 'respiratory_proc',
    'C': 'digestive_proc', 'D': 'digestive_proc', 'F': 'digestive_proc',
    'T': 'genitourinary_proc', 'U': 'genitourinary_proc',
    'V': 'genitourinary_proc',
    'L': 'musculoskeletal_proc', 'M': 'musculoskeletal_proc',
    'N': 'musculoskeletal_proc', 'P': 'musculoskeletal_proc',
    'Q': 'musculoskeletal_proc', 'R': 'musculoskeletal_proc',
    'S': 'musculoskeletal_proc',
}


# ============================================================
# Lab itemid migration drift  (LOINC re-coding)
# ============================================================
#
# Real-world narrative
# --------------------
# Alongside the 2015 ICD-10 rollout, many US hospitals transitioned their
# laboratory subsystems from local itemids to the LOINC standard.  Because
# LOINC uses a different item taxonomy than the legacy itemid schemas,
# records aggregated under the old itemid columns end up attributed to
# different LOINC columns after migration.  The chemical content of the
# measurement is unchanged — a sodium value is still a sodium value — but
# the column in which that value is stored (and which the model's learned
# weights multiply) shifts.
#
# The rename is scoped to **lab columns** (`lab_*`).  Diagnosis chapters
# (`dx_*`), procedures (`px_*`), demographics, care unit, and admission
# metadata live in separate EHR subsystems that were not part of the lab
# migration and are unchanged.
#
# The rename is universal across patients: once the hospital migrates, every
# new admission is coded under the LOINC schema.  This is deliberate — see
# earlier revisions of this comment for the argument — but in short: if
# only some patients were recoded the model would have to serve two
# contradictory label-to-column mappings with shared weights
# (impossible for linear models) and would settle on ignoring the affected
# columns.  Applying the rename universally gives the model a single new
# target mapping to re-learn: w_new = P @ w_old, where P is the permutation
# applied to the renamed columns.
#
# Why labs (not dx)?
# ------------------
# An earlier version of this drift permuted only the `dx_*` columns.  Weight
# inspection on a trained LogReg showed those columns carry ~3.5% of the
# total weight mass (labs carry ~60%) — the model had learned to ignore
# diagnosis chapter distributions entirely.  Permuting dead weights
# produced a weak, homogeneous drift that client-selection policies could
# not meaningfully differentiate on.  Labs are the dominant mortality
# predictors in this LogReg, so permuting them is what actually exercises
# the drift-recovery capabilities of the selection algorithms.
#
# Why only some labs?
# -------------------
# A full lab permutation (all 45 columns) hits every patient's feature
# vector with roughly equal magnitude — labs are dense — which dampens
# the per-care-unit severity gradient that client-selection algorithms
# need to differentiate on.  Narrowing the scope to four labs whose
# values are sharply distributed by care unit (troponin / cardiac;
# lactate / trauma; BUN, creatinine / chronic kidney disease in
# medical ICU populations) restores the structural per-client gradient:
# CCU patients have characteristically high troponin, TSICU patients
# have characteristically high lactate, etc., so when those columns get
# shuffled the magnitude of the resulting logit shift varies sharply
# by care unit.  Narrative: "the LOINC migration was partial — chemistry
# and cardiac panels migrated first; coags, hematology, and the rest
# stayed on legacy itemids."
#
# Pattern matching
# ----------------
# `_CORRUPT_COLUMN_PATTERNS` are substrings; a feature column is included
# in the permutation if any pattern occurs anywhere in its name.  That
# matches `lab_mean_troponin`, `lab_max_troponin`, `lab_min_troponin` for
# the pattern "troponin", and likewise for the other names.
_CORRUPT_COLUMN_PATTERNS = ('troponin', 'lactate', 'bun', 'creatinine')

# Magnitude amplifier applied to renamed columns.  A realistic crosswalk
# wouldn't change magnitudes, but a small amp here forces the model to
# actually fit the permuted weights rather than down-weight the affected
# region.
_CORRUPT_AMP = 4.0


def _build_corruption_perm(feature_cols, seed=0):
    """Build a column permutation that shuffles every column whose name
    contains a `_CORRUPT_COLUMN_PATTERNS` substring within itself, leaving
    every other column untouched.  Deterministic given ``seed``.

    Returns (perm, shuffled_indices):
        perm[i] = source column index fed into destination column i
        shuffled_indices = destination indices affected by the shuffle
    """
    rng = np.random.default_rng(seed)
    icd_idx = [i for i, c in enumerate(feature_cols)
               if any(p in c for p in _CORRUPT_COLUMN_PATTERNS)]
    if not icd_idx:
        return list(range(len(feature_cols))), []
    shuffled = icd_idx.copy()
    # derangement-ish: reshuffle until <10% columns self-map
    for _ in range(32):
        rng.shuffle(shuffled)
        self_maps = sum(1 for a, b in zip(icd_idx, shuffled) if a == b)
        if self_maps <= max(1, len(icd_idx) // 10):
            break
    perm = list(range(len(feature_cols)))
    for dst, src in zip(icd_idx, shuffled):
        perm[dst] = src
    return perm, icd_idx


# ============================================================
# Dataset class
# ============================================================

class MIMIC4(torch.utils.data.Dataset):
    """MIMIC-IV tabular dataset for in-hospital mortality prediction.

    Holds the pre-rename (``inputs_normal``) and post-rename
    (``inputs_corrupt``) feature matrices of identical shape.  While
    ``drift_active`` is ``False`` every admission is served from
    ``inputs_normal`` (old ICD chapter taxonomy).  Once the server flips
    ``drift_active = True`` at ``drift_start`` every admission is served
    from ``inputs_corrupt`` (new ICD chapter taxonomy — same patients, same
    outcomes, diagnosis columns renamed).
    """

    def __init__(self, identifier, inputs_normal, inputs_corrupt,
                 targets, scaler=None, care_units=None):
        self.identifier = identifier
        self.inputs_normal = inputs_normal
        self.inputs_corrupt = inputs_corrupt
        self.targets = targets
        self.scaler = scaler
        self.care_units = care_units  # raw labels for FL partitioning
        self.drift_active = False

    def __len__(self):
        return len(self.inputs_normal)

    def __getitem__(self, index):
        x = self.inputs_corrupt[index] if self.drift_active else self.inputs_normal[index]
        return (
            torch.tensor(x).float(),
            torch.tensor(self.targets[index]).long(),
        )

    def __repr__(self):
        return self.identifier


# ============================================================
# ICD mapping helpers
# ============================================================

def _dx_chapter(code, version):
    """Map an ICD diagnosis code to a unified clinical chapter."""
    code = str(code).strip()
    if version == 9:
        if code.startswith('E'):
            return 'external'
        if code.startswith('V'):
            return 'supplementary'
        try:
            n = int(code[:3])
        except ValueError:
            return 'other'
        for lo, hi, ch in _ICD9_DX_RANGES:
            if lo <= n <= hi:
                return ch
        return 'other'
    else:
        if not code:
            return 'other'
        letter = code[0].upper()
        if letter == 'D':
            try:
                return 'neoplasms' if int(code[1:3]) < 50 else 'blood'
            except (ValueError, IndexError):
                return 'neoplasms'
        return _ICD10_DX_LETTER.get(letter, 'other')


def _px_chapter(code, version):
    """Map an ICD procedure code to a broad group."""
    code = str(code).strip()
    if version == 9:
        try:
            n = int(code[:2])
        except ValueError:
            return 'other_proc'
        for lo, hi, ch in _ICD9_PX_RANGES:
            if lo <= n <= hi:
                return ch
        return 'other_proc'
    else:
        if len(code) < 2 or code[0] != '0':
            return 'other_proc'
        return _ICD10_PCS_BODY.get(code[1].upper(), 'other_proc')


# ============================================================
# Feature construction
# ============================================================

def _build_features(mimic_path, rawsmpl=1.0, seed=42):
    """Build the full feature DataFrame from raw MIMIC-IV tables.

    Args:
        mimic_path: Path to the MIMIC-IV directory.
        rawsmpl: Fraction of the ICU cohort to keep (0.0, 1.0].  Speeds up
            loading substantially because the lab-events table (158 M rows)
            is filtered to the subsampled cohort.
        seed: Random seed for reproducible subsampling.

    Returns a DataFrame with one row per ICU admission, containing
    ``hadm_id``, ``hospital_expire_flag``, ``first_careunit`` (raw label),
    and all numeric feature columns.
    """
    hosp = os.path.join(mimic_path, 'hosp')
    icu_dir = os.path.join(mimic_path, 'icu')

    # ------------------------------------------------------------------
    # 1. Core tables
    # ------------------------------------------------------------------
    logger.info('[LOAD] [MIMIC4] Loading core tables...')
    patients = pd.read_csv(
        os.path.join(hosp, 'patients.csv.gz'),
        usecols=['subject_id', 'gender', 'anchor_age'],
    )
    admissions = pd.read_csv(
        os.path.join(hosp, 'admissions.csv.gz'),
        usecols=[
            'subject_id', 'hadm_id', 'admission_type', 'admission_location',
            'insurance', 'marital_status', 'race',
            'edregtime', 'edouttime', 'hospital_expire_flag',
        ],
    )
    icustays = pd.read_csv(
        os.path.join(icu_dir, 'icustays.csv.gz'),
        usecols=['subject_id', 'hadm_id', 'stay_id', 'first_careunit',
                 'intime', 'los'],
    )

    # ------------------------------------------------------------------
    # 2. ICU cohort — first ICU stay per admission
    # ------------------------------------------------------------------
    logger.info('[LOAD] [MIMIC4] Building ICU cohort...')
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    first_icu = (
        icustays.sort_values('intime')
        .groupby('hadm_id')
        .first()
        .reset_index()
    )
    cohort = (
        admissions
        .merge(patients, on='subject_id', how='inner')
        .merge(first_icu[['hadm_id', 'first_careunit', 'los']],
               on='hadm_id', how='inner')
    )
    if rawsmpl < 1.0:
        cohort = cohort.sample(frac=rawsmpl, random_state=seed).reset_index(drop=True)
        logger.info(f'[LOAD] [MIMIC4] Subsampled to {len(cohort)} ICU admissions '
                     f'(rawsmpl={rawsmpl})')
    cohort_ids = set(cohort['hadm_id'])
    logger.info(f'[LOAD] [MIMIC4] Cohort: {len(cohort)} ICU admissions')

    # Initialise output DataFrame
    df = cohort[['hadm_id', 'hospital_expire_flag', 'first_careunit']].copy()

    # ------------------------------------------------------------------
    # 3. Group A — Demographics
    # ------------------------------------------------------------------
    logger.info('[LOAD] [MIMIC4] Group A: demographics')
    df['age'] = cohort['anchor_age'].values
    df['gender_m'] = (cohort['gender'].values == 'M').astype(np.int8)

    # ------------------------------------------------------------------
    # 4. Group B — Admission info
    # ------------------------------------------------------------------
    logger.info('[LOAD] [MIMIC4] Group B: admission info')
    for col, prefix in [('admission_type', 'admtype'), ('admission_location', 'admloc'),
                        ('insurance', 'ins'), ('marital_status', 'marital')]:
        dummies = pd.get_dummies(cohort[col].fillna('UNKNOWN'),
                                 prefix=prefix, dtype=np.int8)
        for c in dummies.columns:
            df[c] = dummies[c].values

    race_grouped = cohort['race'].map(RACE_MAP).fillna('other')
    race_dummies = pd.get_dummies(race_grouped, prefix='race', dtype=np.int8)
    for c in race_dummies.columns:
        df[c] = race_dummies[c].values

    edreg = pd.to_datetime(cohort['edregtime'])
    edout = pd.to_datetime(cohort['edouttime'])
    df['has_ed'] = edreg.notna().astype(np.int8).values
    df['ed_duration_hours'] = (
        (edout - edreg).dt.total_seconds() / 3600
    ).fillna(0).clip(lower=0).values

    # ------------------------------------------------------------------
    # 5. Group C — Diagnosis counts + chapter counts
    # ------------------------------------------------------------------
    logger.info('[LOAD] [MIMIC4] Group C: diagnoses')
    diagnoses = pd.read_csv(
        os.path.join(hosp, 'diagnoses_icd.csv.gz'),
        usecols=['hadm_id', 'icd_code', 'icd_version'],
    )
    diagnoses = diagnoses[diagnoses['hadm_id'].isin(cohort_ids)]

    dx_counts = diagnoses.groupby('hadm_id').size().rename('num_diagnoses')

    # Map unique codes to chapters, then broadcast
    unique_dx = diagnoses[['icd_code', 'icd_version']].drop_duplicates()
    unique_dx['chapter'] = unique_dx.apply(
        lambda r: _dx_chapter(r['icd_code'], r['icd_version']), axis=1,
    )
    dx_map = dict(zip(
        zip(unique_dx['icd_code'], unique_dx['icd_version']),
        unique_dx['chapter'],
    ))
    diagnoses['chapter'] = list(
        dx_map[k] for k in zip(diagnoses['icd_code'], diagnoses['icd_version'])
    )

    dx_chapter = (
        diagnoses.groupby(['hadm_id', 'chapter']).size()
        .unstack(fill_value=0)
    )
    dx_chapter.columns = [f'dx_{c}' for c in dx_chapter.columns]

    # ------------------------------------------------------------------
    # 6. Group D — Procedure counts + group counts
    # ------------------------------------------------------------------
    logger.info('[LOAD] [MIMIC4] Group D: procedures')
    procedures = pd.read_csv(
        os.path.join(hosp, 'procedures_icd.csv.gz'),
        usecols=['hadm_id', 'icd_code', 'icd_version'],
    )
    procedures = procedures[procedures['hadm_id'].isin(cohort_ids)]

    px_counts = procedures.groupby('hadm_id').size().rename('num_procedures')

    unique_px = procedures[['icd_code', 'icd_version']].drop_duplicates()
    unique_px['chapter'] = unique_px.apply(
        lambda r: _px_chapter(r['icd_code'], r['icd_version']), axis=1,
    )
    px_map = dict(zip(
        zip(unique_px['icd_code'], unique_px['icd_version']),
        unique_px['chapter'],
    ))
    procedures['chapter'] = list(
        px_map[k] for k in zip(procedures['icd_code'], procedures['icd_version'])
    )

    px_chapter = (
        procedures.groupby(['hadm_id', 'chapter']).size()
        .unstack(fill_value=0)
    )
    px_chapter.columns = [f'px_{c}' for c in px_chapter.columns]

    # ------------------------------------------------------------------
    # 7. Group E — Medication counts
    # ------------------------------------------------------------------
    logger.info('[LOAD] [MIMIC4] Group E: medications')
    prescriptions = pd.read_csv(
        os.path.join(hosp, 'prescriptions.csv.gz'),
        usecols=['hadm_id', 'drug', 'route'],
    )
    prescriptions = prescriptions[prescriptions['hadm_id'].isin(cohort_ids)]

    med_feats = prescriptions.groupby('hadm_id').agg(
        num_medications=('drug', 'nunique'),
        num_med_routes=('route', 'nunique'),
    )

    # ------------------------------------------------------------------
    # 8. Group F — Lab results (chunked read for memory)
    # ------------------------------------------------------------------
    logger.info('[LOAD] [MIMIC4] Group F: lab results (this may take a few minutes)...')
    lab_item_ids = set(LAB_ITEMS.keys())
    lab_chunks = []
    for i, chunk in enumerate(pd.read_csv(
        os.path.join(hosp, 'labevents.csv.gz'),
        usecols=['hadm_id', 'itemid', 'valuenum'],
        chunksize=5_000_000,
    )):
        chunk = chunk[
            chunk['hadm_id'].isin(cohort_ids) & chunk['itemid'].isin(lab_item_ids)
        ].dropna(subset=['valuenum'])
        if len(chunk) > 0:
            lab_chunks.append(chunk)
        if (i + 1) % 5 == 0:
            logger.info(f'[LOAD] [MIMIC4]   ...processed {(i + 1) * 5_000_000:,} lab rows')

    if lab_chunks:
        labs = pd.concat(lab_chunks, ignore_index=True)
        labs['lab_name'] = labs['itemid'].map(LAB_ITEMS)
        lab_stats = (
            labs.groupby(['hadm_id', 'lab_name'])['valuenum']
            .agg(['mean', 'min', 'max'])
        )
        lab_wide = lab_stats.unstack('lab_name')
        lab_wide.columns = [f'lab_{stat}_{name}' for stat, name in lab_wide.columns]
    else:
        lab_wide = pd.DataFrame()
    logger.info('[LOAD] [MIMIC4] ...lab results done')

    # ------------------------------------------------------------------
    # 9. Group G — ICU features
    # ------------------------------------------------------------------
    logger.info('[LOAD] [MIMIC4] Group G: ICU features')
    num_icu = icustays.groupby('hadm_id').size().rename('num_icu_stays')
    df['icu_los'] = cohort['los'].values

    careunit_grouped = cohort['first_careunit'].map(CAREUNIT_MAP).fillna('Other')
    cu_dummies = pd.get_dummies(careunit_grouped, prefix='careunit', dtype=np.int8)
    for c in cu_dummies.columns:
        df[c] = cu_dummies[c].values

    # ------------------------------------------------------------------
    # 10. Merge all features
    # ------------------------------------------------------------------
    logger.info('[LOAD] [MIMIC4] Assembling feature matrix...')
    df = df.merge(dx_counts, on='hadm_id', how='left')
    df = df.merge(dx_chapter, on='hadm_id', how='left')
    df = df.merge(px_counts, on='hadm_id', how='left')
    df = df.merge(px_chapter, on='hadm_id', how='left')
    df = df.merge(med_feats, on='hadm_id', how='left')
    df = df.merge(num_icu, on='hadm_id', how='left')
    if not lab_wide.empty:
        df = df.merge(lab_wide, on='hadm_id', how='left')

    # Fill NaN counts/indicators with 0 (no diagnoses/procedures/meds/labs)
    fill_cols = [c for c in df.columns
                 if c not in ('hadm_id', 'hospital_expire_flag', 'first_careunit')]
    df[fill_cols] = df[fill_cols].fillna(0)

    logger.info(f'[LOAD] [MIMIC4] Final feature matrix: {df.shape[0]} rows, '
                f'{len(fill_cols)} features')
    return df


# ============================================================
# Public API
# ============================================================

def fetch_mimic4(args, root):
    """Fetch MIMIC-IV dataset for in-hospital mortality prediction.

    Returns ``(raw_train, raw_test, args)`` for use with the standard
    ``simulate_split`` / ``_construct_dataset`` pipeline.
    """
    mimic_path = os.path.join(root, 'MIMIC-IV')
    rawsmpl = getattr(args, 'rawsmpl', 1.0) or 1.0

    # Cache file includes sample fraction so different fractions don't collide
    smpl_tag = '' if rawsmpl >= 1.0 else f'_smpl{rawsmpl}'
    cache_file = os.path.join(mimic_path, f'.mimic4_features{smpl_tag}.csv.gz')

    # Build or load cached features
    if os.path.exists(cache_file):
        logger.info('[LOAD] [MIMIC4] Loading cached features...')
        df = pd.read_csv(cache_file)
        logger.info(f'[LOAD] [MIMIC4] Loaded {len(df)} rows from cache')
    else:
        df = _build_features(mimic_path, rawsmpl=rawsmpl, seed=args.seed)
        logger.info('[LOAD] [MIMIC4] Caching features...')
        df.to_csv(cache_file, index=False)

    # Separate targets, care units, features
    targets = df['hospital_expire_flag'].values.astype(np.int64)
    # Use the grouped label so the partition axis matches the one-hot columns
    # the model sees.  Admissions whose raw careunit does not map to any known
    # group (fallback "Other") are dropped: they're too few to form a useful
    # client under strict care-unit partitioning, and loss-based selectors
    # waste slots on the resulting n≈46 client.
    careunit_grouped = pd.Series(df['first_careunit'].values).map(CAREUNIT_MAP)
    keep_mask = careunit_grouped.notna().values
    dropped = int((~keep_mask).sum())
    if dropped:
        logger.info(
            '[LOAD] [MIMIC4] dropping %d admission(s) with unmapped '
            'first_careunit (cohort: %d -> %d)', dropped, len(df), int(keep_mask.sum()),
        )
    df = df.loc[keep_mask].reset_index(drop=True)
    targets = targets[keep_mask]
    care_units = careunit_grouped.loc[keep_mask].values
    feature_cols = sorted(
        c for c in df.columns
        if c not in ('hadm_id', 'hospital_expire_flag', 'first_careunit')
    )

    # Canonical feature matrix — what the model learns on, and what every
    # admission looks like before `drift_start`.
    inputs = df[feature_cols].values.astype(np.float32)

    # Global train / test split (stratified by target)
    train_idx, test_idx = train_test_split(
        np.arange(len(inputs)),
        test_size=GLOBAL_TEST_SIZE,
        random_state=args.seed,
        stratify=targets,
    )

    # Impute NaN with training-set column means
    train_means = np.nanmean(inputs[train_idx], axis=0)
    train_means = np.nan_to_num(train_means, nan=0.0)
    nan_mask = np.isnan(inputs)
    if nan_mask.any():
        inputs[nan_mask] = np.take(train_means, np.where(nan_mask)[1])

    # Fit the scaler on the canonical training set, apply to the full matrix.
    scaler = StandardScaler()
    train_normal = scaler.fit_transform(inputs[train_idx])
    test_normal = scaler.transform(inputs[test_idx])

    # Build the post-rename feature matrix: permute every column whose name
    # contains a `_CORRUPT_COLUMN_PATTERNS` substring within that group, leave
    # every other column fixed, then amplify only the permuted columns.  At
    # drift_start every admission switches to this view.
    perm, shuffled_idx = _build_corruption_perm(feature_cols, seed=args.seed)
    perm = np.asarray(perm, dtype=np.int64)
    amp = np.ones(len(feature_cols), dtype=np.float32)
    amp[shuffled_idx] = _CORRUPT_AMP
    train_corrupt = (train_normal[:, perm] * amp).astype(np.float32)
    test_corrupt = (test_normal[:, perm] * amp).astype(np.float32)

    logger.info(
        '[LOAD] [MIMIC4] post-drift column rename: permuted %d column(s) '
        '(scope: %s), magnitude amp x%.1f; applied universally to all %d admissions.',
        len(shuffled_idx),
        ' | '.join(_CORRUPT_COLUMN_PATTERNS),
        _CORRUPT_AMP, len(inputs),
    )

    raw_train = MIMIC4(
        '[MIMIC4] Mortality (train)',
        train_normal, train_corrupt,
        targets[train_idx], scaler, care_units[train_idx],
    )
    raw_test = MIMIC4(
        '[MIMIC4] Mortality (test)',
        test_normal, test_corrupt,
        targets[test_idx], scaler, care_units[test_idx],
    )

    # Store feature names for interpretability
    raw_train.feature_names = feature_cols
    raw_test.feature_names = feature_cols

    args.in_features = train_normal.shape[1]
    args.num_classes = 2

    logger.info(f'[LOAD] [MIMIC4] in_features={args.in_features}, '
                f'train={len(raw_train)}, test={len(raw_test)}, '
                f'mortality_rate={targets.mean():.3f}')
    return raw_train, raw_test, args
