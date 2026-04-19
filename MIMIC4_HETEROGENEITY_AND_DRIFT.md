# MIMIC-IV Heterogeneity and Concept Drift Implementation

## Cohort

- **ICU only** — first ICU stay per hospital admission, joined via `hadm_id` inner-merge with `icustays` (≈65k admissions at full sample, ≈21k at `rawsmpl=0.25`).
- **Target:** `hospital_expire_flag` (in-hospital mortality), ~11% positive rate.
- **Features:** 126-dim mixed vector — demographics, admission info, grouped care unit one-hots, 15 key labs (mean/min/max), ICD-chapter counts (`dx_*`), procedure-group counts (`px_*`), medication counts, ICU LOS. Both ICD-9 and ICD-10 codes are harmonized into the same fixed chapter taxonomy so the model input dimensionality is invariant to the coding-system switch.

## Heterogeneity — Dirichlet over care units

Each client's data is a non-IID mixture over the grouped ICU unit `Z ∈ {MICU, MICU_SICU, CVICU, SICU, CCU, TSICU, Neuro, Other}`.

For each care unit `c`, draw `q_c ~ Dir(α · 1_K)` and distribute that unit's records across `K` clients proportional to `q_c`. Small `α` (`--cncntrtn 0.3`) yields strongly skewed mixes: each client is dominated by 1–2 care units while still seeing some minority cases. `α` is the single knob controlling how lopsided the partition is.

**Rationale.** Care unit is a real, clinically meaningful hospital subdivision — it mirrors how a federation of hospitals would differ in patient mix. It is also the axis the drift mechanism keys off (below), so the data-distribution gradient and the drift-severity gradient align on the same variable, giving the selection algorithm a single axis to exploit.

## Concept drift — ICD-9 → ICD-10 crosswalk corruption

**Real-world event.** On 2015-10-01 US hospitals switched from ICD-9 (~14k codes) to ICD-10 (~70k codes) with finer granularity (laterality, episode of care, D-code split, F-code expansion). Different IT vendors shipped different crosswalk implementations; some hospitals applied the CMS GEMs correctly, others produced scrambled chapter assignments — particularly for the chapters whose ICD-9→ICD-10 remaps were the messiest.

**Deterministic corrupt-group predicate** (computed from features, not randomly assigned):

```
is_corrupt(patient) = (dx_mental > 0) OR (dx_external > 0) OR (dx_injury > 0)
```

A patient falls in the corrupt group iff they had ≥1 diagnosis in the chapters with the worst crosswalks (F-codes expanded ~3×, E/V → V/W/X/Y with laterality, S/T with episode-of-care). On the sampled cohort this is 73% of admissions; per care unit it ranges from 62% (CVICU) to 85% (TSICU).

**Corruption mechanism.** Before `drift_start`, every admission sees the canonical feature vector. At round == `drift_start`, the server flips a single `drift_active` flag on the shared `MIMIC4` instance. From that round onward:

- **Corrupt patients** receive their scaled feature vector with all 29 ICD-derived columns (`dx_*`, `px_*`, `num_diagnoses`, `num_procedures`) **shuffled** via a fixed deterministic random permutation (seeded on `--seed`), then **amplified by 4×**. This simulates a crosswalk that both scrambles chapter assignments and inflates counts through duplicated code mappings.
- **Non-corrupt patients** remain on the canonical view (their pipeline was patched correctly).
- All non-ICD features (labs, demographics, insurance, care unit one-hots) are left untouched — the corruption is confined to the ICD coding pipeline.

## How heterogeneity and drift compose

Because the corrupt predicate is feature-based and care units differ in their chapter mix, the two mechanisms compose automatically:

| Care unit | Corrupt % | Effective drift severity |
|-----------|-----------|--------------------------|
| TSICU | 85% | highest |
| MICU / MICU_SICU | 79% | high |
| SICU | 71% | moderate |
| CCU | 69% | moderate |
| Neuro | 62% | lower |
| CVICU | 61% | lowest |

A client whose Dirichlet draw concentrates on TSICU sees ~85% of its data go corrupt at `drift_start`; a client concentrated on CVICU sees only ~61%. The selection algorithm thus has a continuous per-client drift-severity gradient, correlated with but not trivially redundant with the care-unit partition (since `α=0.3` produces mixed distributions rather than pure single-unit clients).

## Empirical footprint

A centralized LogReg trained on the canonical features achieves **AUROC 0.932 / Acc1 0.925** on the holdout. After flipping the corrupt group to the shuffled+amplified view (labels unchanged): **AUROC 0.740 / Acc1 0.872** — a 19-point AUROC drop driven primarily by the corrupt subset (acc 0.925 → 0.849) while the clean subset is stable (acc 0.933). Acc1 is structurally insensitive at the 11% positive base rate, so AUROC is reported as the primary drift metric.

## Command-line interface

```bash
--split_type custom --cncntrtn 0.3      # Dirichlet-over-care-unit
--concept_drift                          # enable drift
--drift_mode custom                      # dispatches to MIMIC crosswalk corruption
--drift_start 50                         # round at which drift_active is flipped
--drift_duration 0                       # one-off event (idempotent after flip)
```

## Real-world relation

The drift in this experiment is a stylised simulation of a documented, datable
real-world event in US clinical data.

- **The coding-system switch (2015-10-01).** US hospitals were mandated to
  switch from ICD-9-CM (~14k diagnosis codes, ~4k procedure codes) to
  ICD-10-CM / ICD-10-PCS (~70k diagnosis codes, ~72k procedure codes) on
  October 1, 2015. Structural changes included alphanumeric codes, required
  laterality (left/right/bilateral), an episode-of-care qualifier for
  injuries, a three-to-four-fold expansion of the F-code (mental/behavioural)
  chapter, and a reorganisation of external-cause codes (E/V → V/W/X/Y/Z).
- **The CMS GEMs crosswalks.** CMS published General Equivalence Mappings
  (GEMs) as a bidirectional crosswalk between ICD-9 and ICD-10 codes. The
  GEMs explicitly flag many mappings as "approximate" or "no map", and a
  substantial fraction of ICD-9 codes expand to multiple ICD-10 codes (or
  vice versa). This ambiguity is the real-world source of the "messy
  crosswalk" narrative used here.
- **Uneven hospital implementation.** Different EHR and billing vendors
  implemented GEMs application differently; some hospitals applied the
  forward maps correctly, others produced systematically distorted chapter
  counts in the period immediately after the cutover (code duplication
  across chapters, miscategorised remaps, increased coding depth per
  admission).
- **Why MIMIC-IV contains both eras.** MIMIC-IV spans 2008–2019, straddling
  the cutover: pre-2015 admissions are natively coded in ICD-9, post-2015
  in ICD-10. The feature builder harmonises both into a single chapter
  taxonomy, so the canonical feature vector already represents the "good
  crosswalk" outcome. Our corrupted view is what the same feature vector
  would look like under a *bad* crosswalk — the counterfactual scenario
  for a hospital whose pipeline was mis-configured during the transition.
- **Differential exposure by care unit.** Trauma units (S/T codes, external
  causes) and medical ICUs (F-codes, sepsis, broader diagnostic mix) were
  structurally more exposed to the messy chapters than cardiac ICUs (whose
  I-code circulatory diagnoses map relatively cleanly between ICD-9 and
  ICD-10). This is why the corrupt-group fraction varies from ~61% (CVICU)
  to ~85% (TSICU) in our implementation — the unevenness is a direct
  consequence of real chapter-level crosswalk complexity, not an imposed
  per-client parameter.

## Primary references

- **Nestor, B. et al. (2019).** *Feature Robustness in Non-stationary Health
  Records: Caveats to Deployable Model Performance in Common Clinical
  Machine Learning Tasks.* Proceedings of Machine Learning for Healthcare
  (MLHC 2019), PMLR 106. arXiv:1908.00690.
  → The canonical empirical demonstration that the ICD-9→ICD-10 transition
  and adjacent EHR policy changes cause measurable degradation in MIMIC
  mortality / phenotyping models trained on pre-transition data. The drift
  narrative and the "ICD-coded features are the non-stationary ones" framing
  used here are taken directly from this paper.

- **Hsu, T.-M. H., Qi, H. & Brown, M. (2019).** *Measuring the Effects of
  Non-Identical Data Distribution for Federated Visual Classification.*
  arXiv:1909.06335.
  → The origin of the `Dir(α · 1_K)` per-category partitioning used here
  (the `--cncntrtn` flag is exactly this α). We apply it over care units
  instead of classes, but the partitioning recipe and the role of α as the
  sole knob controlling non-IID-ness are unchanged.

- **Boyd, A. D. et al. (2013).** *A Method to Determine the Compatibility of
  Information Models by Analysing the ICD-9 to ICD-10-CM Mapping.* AMIA
  Annual Symposium Proceedings.
  → Documents the many-to-many nature of ICD-9↔ICD-10 GEMs mappings and
  the chapter-level unevenness of remap ambiguity — the structural basis
  for why mental-health, external-cause and injury chapters are singled
  out in the corrupt-group predicate.

- **Johnson, A. E. W. et al. (2023).** *MIMIC-IV, a freely accessible
  electronic health record dataset.* Scientific Data 10, 1.
  → Dataset citation — MIMIC-IV v2.x, the ICU cohort used throughout.

- **Harutyunyan, H., Khachatrian, H., Kale, D. C., Ver Steeg, G., &
  Galstyan, A. (2019).** *Multitask learning and benchmarking with
  clinical time series data.* Scientific Data 6, 96.
  → Establishes the in-hospital-mortality-from-first-ICU-stay benchmark
  cohort on MIMIC-III that MIMIC-IV work inherits; our cohort definition
  (first ICU stay per admission, `hospital_expire_flag` label, ≈11%
  mortality) follows this convention.
