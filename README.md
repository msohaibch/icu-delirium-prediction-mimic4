# ICU Delirium Prediction — MIMIC-IV
---

## Cohort

| Step | N |
|---|---|
| Total ICU stays in MIMIC-IV | 94,444 |
| CAM-ICU screened | 72,930 |
| Exclude prevalent delirium (first 24h) | -14,429 |
| Restrict to first ICU stay per patient | -17,759 |
| **Final cohort** | **40,742** |

- **Outcome:** New-onset CAM-ICU confirmed delirium after hour 24 (`outcome_incident_cam_24h`)
- **Event rate:** 13.1% (5,342 cases)
- **Train/Test:** 80/20 stratified split

---

## Files

### code/

| File | What it does |
|---|---|
| `00_check_csv_columns.py` | Utility — run first to verify CSV loaded correctly, check column names and outcome distribution |
| `01_main_prediction_pipeline.py` | Main pipeline — LASSO feature selection, logistic regression, XGBoost with calibration, SHAP, dementia subgroup analysis |
| `02_temporal_validation_and_subgroup_analyses.py` | Additional analyses — temporal validation (2008-2018 train / 2019-2022 test), race stratification, dementia subgroup detail |

### results/

Pre-generated outputs from the main pipeline run:
- `model_performance_summary.csv` — AUROC, CV AUROC, PR-AUC, Brier for both models
- `lasso_coefficients.csv` — all LASSO coefficients, selected vs zeroed
- `shap_mean_abs_values.csv` — SHAP feature importance ranking
- `dementia_subgroup_results.csv` — AUROC stratified by dementia status
- `missing_data_summary.csv` — missingness by variable before imputation
- Plots: ROC/PR curves, SHAP beeswarm, SHAP bar, dementia subgroup ROC, calibration, LASSO plot

---

## How to Run

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn scipy
```

### Step 1 — Update paths
In `01_main_prediction_pipeline.py`, set:
```python
CSV_PATH   = r"path/to/delirium_model_v2.csv"      # primary cohort (first stays)
CSV_FULL   = r"path/to/delirium_model_full.csv"     # full cohort (all screened stays)
OUTPUT_DIR = r"path/to/your/output/folder"
```

In `02_temporal_validation_and_subgroup_analyses.py`, set:
```python
CSV_FULL   = r"path/to/delirium_model_full.csv"
OUTPUT_DIR = r"path/to/your/output/folder"
```

### Step 2 — Check columns
```bash
python 00_check_csv_columns.py
```

### Step 3 — Run main pipeline
```bash
python 01_main_prediction_pipeline.py
```

### Step 4 — Run additional analyses
```bash
python 02_temporal_validation_and_subgroup_analyses.py
```

## Features Used

All features measured from first 24h of ICU admission (~90 variables):
- Demographics: age, sex, race, insurance, marital status
- Admission: ICU type, urgency, prior ICU stay, nighttime admission
- Severity: SOFA (6 components, derived from raw tables), GCS
- Sedation: RASS mean/min/max, agitation/sedation flags
- Vitals: HR, SBP, MAP, SpO2, RR, temperature + binary flags
- Labs: renal, electrolyte, CBC, coagulation, lactate
- Medications: vasopressors, sedatives, analgesics (binary yes/no)
- Fluid balance: net fluid in first 24h
- Pain scores: mean and max pain score first 24h
- Comorbidities: 23 ICD-coded conditions including dementia subtypes

### Excluded variables (leakage or >50% missing)
```python
LEAKAGE_COLS = [
    'rass_assessment_count',   # nurse assessment frequency driven by suspicion
    'antipsychotic_any',       # prescribed reactively to treat delirium
    'haloperidol',
    'quetiapine',
]
DROP_HIGH_MISSING = ['albumin_min', 'bilirubin_max', 'alt_max']  # >50% missing
```

---

## Notes on AUROC

The AUROC of 0.949 is higher than most published delirium prediction models (typically 0.78–0.87). A few things to be aware of:

1. **Leakage variables are excluded in code** — `rass_assessment_count`, `antipsychotic_any`, `haloperidol`, `quetiapine` are all excluded via `LEAKAGE_COLS` in the pipeline
2. **`rass_ever_agitated` is borderline** — this flag is still included as a feature (correlation 0.40 with outcome). Some agitation in first 24h may be early/undiagnosed delirium. Happy to discuss whether this should be removed
3. **Random split not temporal** — the main pipeline uses random 80/20. 
4. **Single site** — BIDMC only. High within-site AUROC is expected; external validation is the next step

Run the temporal validation script and compare — that will answer whether 0.949 is real or an artifact of the random split.

---

## Contact

Sohaib Ahmed — msohaibch77@gmail.com
Supervisor: Dr. Wenhui (Vivian) Zhang — Emory School of Nursing
