# MIMIC-IV Heterogeneity and Concept Drift Implementation

## Cohort

- **ICU only** — first ICU stay per hospital admission, joined via `hadm_id`
  inner-merge with `icustays`.
- **Target:** `hospital_expire_flag` (in-hospital mortality), ~11% positive rate.
- **Features:** ~125-dim mixed vector — demographics, admission info, grouped
  care unit one-hots, 15 key labs (mean/min/max), ICD-chapter counts (`dx_*`),
  procedure-group counts (`px_*`), medication counts, ICU LOS. Both ICD-9 and
  ICD-10 codes are harmonised into the same fixed chapter taxonomy so the model
  input dimensionality is invariant to the coding-system switch.
- Admissions whose `first_careunit` does not map to one of the seven grouped
  units below are dropped before splitting (too few to form a useful client and
  loss-based selectors waste slots on the resulting tiny client).

## Heterogeneity — strict care-unit partition

Each client is assigned the records of **exactly one** grouped ICU unit
`Z ∈ {MICU, MICU_SICU, CVICU, SICU, CCU, TSICU, Neuro}`. There is no
mixing — a client whose unit is, say, `CVICU` sees only CVICU patients.

Multiple clients are allocated to each care unit by the largest-remainder
method: every unit gets a floor of one client, and the remaining `K - 7`
clients are distributed proportional to the unit's sample count. Within a
unit the records are shuffled and split evenly across the clients allocated
to it. The split requires `K ≥ 7`.

`--cncntrtn` (Dirichlet `α`) is **unused** for MIMIC under `--split_type
custom`; it is preserved on the CLI only because the CheXpert custom split
still consumes it.

**Rationale.** Care unit is a real, clinically meaningful hospital
subdivision — it mirrors how a federation of hospitals (or services within
one hospital) would differ in patient mix. Strict partitioning maximises
the per-client drift-severity gradient: the lab columns the drift mechanism
permutes (`troponin`, `lactate`, `bun`, `creatinine`) have characteristically
unit-specific distributions, so a client whose only patients are CCU sees
the troponin permutation as a large logit shift, while a client whose
patients are e.g. Neuro is barely perturbed by the same permutation. A
Dirichlet mixture would dilute that gradient by averaging exposures across
units inside each client; strict partitioning preserves it.

## Concept drift — LOINC lab itemid migration

**Real-world event.** Around the same window as the ICD-10 transition,
many US hospitals migrated their laboratory information systems from local
itemid taxonomies to the LOINC standard (driven by HITECH / Meaningful Use
Stage 2 & 3 incentives). Because LOINC organises tests by analyte +
specimen + method, records that were aggregated under one local itemid
column often end up split across, or merged into, different LOINC columns
after migration. The **chemical content** of the measurement is unchanged
— a sodium value is still a sodium value — but the **column** the value
lives in (and which the model's learned weights multiply) changes.

**Corruption mechanism.** The dataset holds two pre-built feature matrices
of identical shape: `inputs_normal` (pre-migration) and `inputs_corrupt`
(post-migration). Before `drift_start`, every admission is served from
`inputs_normal`. At round `== drift_start` the server flips a single
`drift_active` flag on the shared `MIMIC4` instance. From that round
onward every admission is served from `inputs_corrupt`.

`inputs_corrupt` is built by:

1. Selecting every feature column whose name contains one of the substrings
   `troponin`, `lactate`, `bun`, `creatinine`. Under the current feature
   builder this matches `lab_{mean,min,max}_<name>` for each — 12 columns
   in total.
2. Drawing a derangement-ish permutation over those 12 indices (seeded on
   `--seed`; up to 32 reshuffles to keep self-maps below 10%) and writing
   the permuted source columns into the same 12 destination indices. All
   other columns are left untouched.
3. Multiplying the 12 permuted columns by `4×` (`_CORRUPT_AMP`). A
   realistic crosswalk wouldn't change magnitudes, but the amplifier
   forces the model to actually fit the renamed columns rather than
   silently down-weighting the affected region.

**Why the rename is universal across patients (not a "corrupt subgroup").**
A LOINC migration is hospital-wide: once the LIS cuts over, every new
admission is coded under the new schema. If only some patients were recoded
the model would have to serve two contradictory label-to-column mappings
through shared weights — impossible for a linear model — and the optimiser
would settle on ignoring the affected columns entirely. Universal rename
gives the model a single new target mapping to re-learn:
`w_new = P @ w_old`, where `P` is the permutation over the renamed columns.

**Why labs (not ICD chapters).** An earlier revision of this drift
permuted only the `dx_*` columns. Weight inspection on a trained LogReg
showed those columns carry roughly 3.5% of the total weight mass (labs
carry ~60%) — the model had learned to ignore diagnosis chapter
distributions almost entirely. Permuting near-dead weights produced a
weak, homogeneous drift that client-selection policies could not
meaningfully differentiate on. Labs are the dominant mortality predictors
in this LogReg, so permuting them is what actually exercises drift
recovery.

**Why only four labs (not all 45 lab columns).** A full lab permutation
hits every patient's feature vector with roughly equal magnitude — labs
are dense — which dampens the per-care-unit severity gradient. Narrowing
the scope to four analytes whose values are sharply distributed by care
unit restores the structural per-client gradient: CCU patients have
characteristically high troponin, TSICU patients have characteristically
high lactate, MICU patients have characteristically high BUN/creatinine
(chronic kidney disease comorbidity), etc. Narrative-wise this matches
a partial migration: "the LIS migration started with chemistry and
cardiac panels; coags, hematology, and the rest stayed on legacy
itemids."

## How heterogeneity and drift compose

Strict care-unit partitioning + lab-name-scoped permutation means each
client experiences a drift magnitude determined by its assigned unit's
distribution over the four affected analytes. CCU clients absorb most of
the troponin shift; TSICU clients most of the lactate shift; MICU clients
most of the BUN / creatinine shift; Neuro clients are comparatively
insulated because none of the four analytes is a top mortality signal in
that population. The selection algorithm thus has a continuous,
clinically interpretable per-client drift-severity gradient that lines up
exactly with the partition axis.

## Command-line interface

```bash
--split_type custom                   # strict care-unit partition (--cncntrtn ignored)
--concept_drift                       # enable drift
--drift_mode custom                   # dispatches to MIMIC LOINC-style permutation
--drift_start 400                     # round at which drift_active is flipped
--drift_duration 0                    # one-off event (idempotent after flip)
--K 20                                # K >= 7 (one client per care unit minimum)
```

## Real-world parallels for the simulated drift

The drift in this experiment is a stylised simulation of a class of
documented, datable events in clinical-data engineering: a hospital
**re-coding the columns** under which lab measurements arrive, without
changing the underlying chemistry. Real-world instances:

- **LOINC adoption (HITECH / Meaningful Use, ~2011–2018).** Federal
  incentive programmes (Meaningful Use Stage 2, then 21st Century Cures
  Act / ONC interoperability rules) progressively required LOINC for
  laboratory result exchange. Hospitals on local itemid taxonomies cut
  over to LOINC in batches — usually one analyte panel at a time
  (chemistry first, then haematology, coags, microbiology). Models
  trained on pre-cutover features see a column-rename event at the
  moment of each panel migration.
- **Hospital-wide LIS / EHR vendor swap.** Replacing the laboratory
  information system (Cerner Millennium → Epic Beaker, Sunquest → Epic,
  Meditech → Epic, etc.) re-keys every result under the new vendor's
  identifier scheme. Multi-hospital systems that consolidate onto a
  single LIS produce the same effect at acquisition time.
- **Analyser / reagent vendor change.** When a chemistry analyser
  (Roche Cobas → Abbott Architect, Beckman → Siemens, etc.) is
  swapped, the LIS often emits the test under a new instrument-tied
  itemid even though the analyte is identical. Reference-range tweaks
  bundled with the swap can also rescale the column.
- **MIMIC-III → MIMIC-IV itemid renumbering.** The MIMIC project
  itself re-keyed labs and chartevents between versions: itemids that
  identified a measurement in MIMIC-III do not necessarily identify
  the same measurement in MIMIC-IV. A model trained on MIMIC-III
  features and inferred on MIMIC-IV without a translation table
  experiences exactly this drift.
- **COVID-era LOINC additions (2020–2021).** New SARS-CoV-2 PCR /
  antigen / antibody LOINC codes were added rapidly during the
  pandemic. Hospitals that had been logging COVID assays under
  ad-hoc local codes re-mapped them to the new standard codes once
  released, generating a column-rename for that subset of tests.
- **Unit / reference-method standardisation.** Vitamin D reporting
  was reorganised when LOINC distinguished 25-OH-D2 vs D3 vs total;
  HbA1c reporting moved from %DCCT to mmol/mol IFCC in many systems.
  These re-keys are narrower than a full LIS migration but the
  model-side effect is the same: a previously stable column starts
  receiving the values that used to live in a sibling column.
- **Pandemic-era ICD-10-CM additions for pulmonary embolism, MIS-C,
  long-COVID** (2020-04, 2020-10, 2021-10 quarterly updates). The
  ICD analogue of the lab story — already covered by the previous
  revision of this document, retained here as a reminder that the
  rename pattern recurs across coding subsystems whenever standards
  bodies push out a new edition.

In all of these, the model-side observation is identical: at some
calendar moment, a subset of feature columns starts receiving the
numeric content that used to live in (different) sibling columns. The
labels do not change. The patient population does not change. Only the
mapping from "thing measured" to "column index" changes.

## Empirical footprint

The drift's quantitative impact (centralised AUROC pre/post, per-care-unit
breakdown) is measured by the experiment runs themselves rather than
asserted in this document — the figures shift whenever the corrupt-column
list, amplifier, or feature builder change. See the run logs and
`plots/` outputs for the current numbers.

## Primary references

- **Nestor, B. et al. (2019).** *Feature Robustness in Non-stationary
  Health Records: Caveats to Deployable Model Performance in Common
  Clinical Machine Learning Tasks.* Proceedings of Machine Learning for
  Healthcare (MLHC 2019), PMLR 106. arXiv:1908.00690.
  → The canonical empirical demonstration that EHR coding-system
  transitions cause measurable degradation in MIMIC mortality /
  phenotyping models trained on pre-transition data. The "EHR-coded
  features are the non-stationary ones" framing used here is taken
  directly from this paper.

- **McDonald, C. J., Huff, S. M., Suico, J. G., et al. (2003).**
  *LOINC, a Universal Standard for Identifying Laboratory Observations:
  A 5-Year Update.* Clinical Chemistry 49(4), 624–633.
  → Foundational LOINC reference; explains why mapping local lab
  itemids onto LOINC is many-to-one / one-to-many and therefore
  produces column reshuffles when a hospital migrates.

- **Lin, M. C., Vreeman, D. J., McDonald, C. J., Huff, S. M. (2012).**
  *Auditing consistency and usefulness of LOINC use among three large
  institutions — using version spaces for grouping LOINC codes.*
  Journal of Biomedical Informatics 45(4), 658–666.
  → Documents inter-hospital inconsistency in LOINC mapping for the
  same underlying assay — the structural reason a "post-migration"
  feature schema differs across institutions even after both have
  nominally adopted LOINC.

- **Hsu, T.-M. H., Qi, H. & Brown, M. (2019).** *Measuring the Effects
  of Non-Identical Data Distribution for Federated Visual
  Classification.* arXiv:1909.06335.
  → The origin of the `Dir(α · 1_K)` per-category partitioning used
  for the CheXpert custom split. The MIMIC custom split is *not*
  Dirichlet-based; it is a strict-partition specialisation, but the
  original Hsu et al. recipe is the conceptual ancestor.

- **Johnson, A. E. W. et al. (2023).** *MIMIC-IV, a freely accessible
  electronic health record dataset.* Scientific Data 10, 1.
  → Dataset citation — MIMIC-IV v2.x, the ICU cohort used throughout.

- **Harutyunyan, H., Khachatrian, H., Kale, D. C., Ver Steeg, G., &
  Galstyan, A. (2019).** *Multitask learning and benchmarking with
  clinical time series data.* Scientific Data 6, 96.
  → Establishes the in-hospital-mortality-from-first-ICU-stay
  benchmark cohort on MIMIC-III that MIMIC-IV work inherits; our
  cohort definition follows this convention.
