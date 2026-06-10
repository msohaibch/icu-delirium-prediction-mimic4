"""
=============================================================================
MLHC 2026 Additional Analyses Script
=============================================================================
Purpose: Generate missing analyses for Clinical Abstract submission
1. Temporal validation (2008-2018 train → 2019-2022 test)
2. Race/ethnicity stratified performance
3. Dementia subgroup detailed statistics (TEST SET - corrected)
4. Calibration plots stratified by dementia status
5. CNS SOFA and RASS distribution by dementia status

Run this AFTER your main pipeline to get additional results needed for MLHC.

Requirements: Same as main pipeline (pandas, numpy, sklearn, xgboost, etc.)
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             brier_score_loss, roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import xgboost as xgb
from scipy import stats

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Path to your FULL dataset CSV (exported from PostgreSQL)
CSV_FULL = r"F:\MIMIC\delirium_model_full.csv"

# Output directory
OUTPUT_DIR = r"F:\MIMIC\mlhc_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("MLHC 2026 ADDITIONAL ANALYSES")
print("=" * 80)
print(f"Data: {CSV_FULL}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

# =============================================================================
# SECTION 1: LOAD DATA AND BASIC PREP
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 1: Loading data")
print("=" * 80)

df = pd.read_csv(CSV_FULL)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# Convert admission time to datetime
if 'admittime' in df.columns:
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['admit_year'] = df['admittime'].dt.year
    print(f"Admission years range: {df['admit_year'].min()} to {df['admit_year'].max()}")
else:
    print("WARNING: 'admittime' column not found - cannot do temporal validation")
    print("Available columns:", df.columns.tolist()[:20], "...")

# Check for required columns
required_cols = ['outcome_incident_cam_24h', 'dementia_icd', 'race']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    print("Please ensure CSV export includes outcome, dementia_icd, and race")
else:
    print("✓ All required columns present")

# Outcome
y = df['outcome_incident_cam_24h'].values
print(f"Outcome distribution: {y.sum()} delirium cases ({100*y.mean():.1f}%)")

# Dementia distribution
if 'dementia_icd' in df.columns:
    print(f"Dementia: {df['dementia_icd'].sum()} cases ({100*df['dementia_icd'].mean():.1f}%)")

# Race distribution
if 'race' in df.columns:
    print("\nRace distribution:")
    print(df['race'].value_counts())

# =============================================================================
# SECTION 2: PREPARE FEATURES (same as main pipeline)
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 2: Feature preparation")
print("=" * 80)

# Columns to EXCLUDE from modeling
exclude_cols = [
    'subject_id', 'hadm_id', 'stay_id', 'admittime', 'dischtime', 'icu_intime', 
    'icu_outtime', 'admit_year',
    'outcome_incident_cam_24h',  # outcome
    'cam_positive', 'cam_screened', 'delirium_icd',  # outcomes/labels
    'concordance_icd_cam', 'is_first_icu_stay'  # study design variables
]

# Get feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Feature columns: {len(feature_cols)}")

X = df[feature_cols].copy()

# Identify binary vs continuous
binary_cols = [col for col in feature_cols if X[col].dropna().isin([0, 1]).all()]
continuous_cols = [col for col in feature_cols if col not in binary_cols]

print(f"Binary features: {len(binary_cols)}")
print(f"Continuous features: {len(continuous_cols)}")

# Force to float
X = X.astype(float)

# Imputation
X_imputed = X.copy()
if continuous_cols:
    cont_imputer = SimpleImputer(strategy='median')
    X_imputed[continuous_cols] = cont_imputer.fit_transform(X[continuous_cols])

if binary_cols:
    bin_imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed[binary_cols] = bin_imputer.fit_transform(X[binary_cols])

print("✓ Imputation complete")

# =============================================================================
# SECTION 3: TEMPORAL VALIDATION (2008-2018 train → 2019-2022 test)
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 3: TEMPORAL VALIDATION")
print("=" * 80)

if 'admit_year' not in df.columns:
    print("SKIPPING: No admit_year column available")
else:
    # Define temporal split
    temporal_train_mask = df['admit_year'] < 2019
    temporal_test_mask = df['admit_year'] >= 2019
    
    X_train_temp = X_imputed[temporal_train_mask].values
    y_train_temp = y[temporal_train_mask]
    X_test_temp = X_imputed[temporal_test_mask].values
    y_test_temp = y[temporal_test_mask]
    
    print(f"Temporal train (2008-2018): N={len(y_train_temp)}, events={y_train_temp.sum()} ({100*y_train_temp.mean():.1f}%)")
    print(f"Temporal test (2019-2022): N={len(y_test_temp)}, events={y_test_temp.sum()} ({100*y_test_temp.mean():.1f}%)")
    
    # LASSO feature selection on temporal train
    print("\nRunning LASSO on temporal train set...")
    alphas = np.logspace(-4, 1, 30)
    lasso_cv_temp = LogisticRegressionCV(
        Cs=1.0/alphas,
        cv=5,
        penalty='l1',
        solver='liblinear',
        max_iter=1000,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1
    )
    
    # Scale features
    scaler_temp = StandardScaler()
    X_train_temp_scaled = scaler_temp.fit_transform(X_train_temp)
    X_test_temp_scaled = scaler_temp.transform(X_test_temp)
    
    lasso_cv_temp.fit(X_train_temp_scaled, y_train_temp)
    
    # Selected features
    coef_temp = lasso_cv_temp.coef_[0]
    selected_mask_temp = np.abs(coef_temp) > 0
    n_selected_temp = selected_mask_temp.sum()
    print(f"LASSO selected {n_selected_temp} features")
    
    selected_features_temp = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask_temp[i]]
    
    X_train_temp_sel = X_train_temp_scaled[:, selected_mask_temp]
    X_test_temp_sel = X_test_temp_scaled[:, selected_mask_temp]
    
    # Train logistic regression
    print("\nTraining Logistic Regression on temporal split...")
    lr_temp = LogisticRegression(penalty='l2', max_iter=1000, solver='liblinear', random_state=42)
    lr_temp.fit(X_train_temp_sel, y_train_temp)
    lr_proba_temp = lr_temp.predict_proba(X_test_temp_sel)[:, 1]
    lr_auc_temp = roc_auc_score(y_test_temp, lr_proba_temp)
    lr_ap_temp = average_precision_score(y_test_temp, lr_proba_temp)
    
    print(f"Temporal LR - AUROC: {lr_auc_temp:.3f}, PR-AUC: {lr_ap_temp:.3f}")
    
    # Train XGBoost
    print("\nTraining XGBoost on temporal split...")
    scale_pos_weight_temp = (y_train_temp == 0).sum() / (y_train_temp == 1).sum()
    
    xgb_temp = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_temp,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    # Use 20% of train for early stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_temp_sel, y_train_temp, test_size=0.2, stratify=y_train_temp, random_state=42
    )
    
    xgb_temp.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    xgb_proba_temp_raw = xgb_temp.predict_proba(X_test_temp_sel)[:, 1]
    
    # Calibrate XGBoost
    print("Calibrating XGBoost...")
    from sklearn.isotonic import IsotonicRegression
    iso_reg_temp = IsotonicRegression(out_of_bounds='clip')
    
    # Use validation set for calibration
    xgb_proba_val = xgb_temp.predict_proba(X_val)[:, 1]
    iso_reg_temp.fit(xgb_proba_val, y_val)
    
    xgb_proba_temp = iso_reg_temp.transform(xgb_proba_temp_raw)
    
    xgb_auc_temp = roc_auc_score(y_test_temp, xgb_proba_temp)
    xgb_ap_temp = average_precision_score(y_test_temp, xgb_proba_temp)
    xgb_brier_temp = brier_score_loss(y_test_temp, xgb_proba_temp)
    
    print(f"Temporal XGBoost - AUROC: {xgb_auc_temp:.3f}, PR-AUC: {xgb_ap_temp:.3f}, Brier: {xgb_brier_temp:.3f}")
    
    # Save results
    temporal_results = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost'],
        'Train_Period': ['2008-2018', '2008-2018'],
        'Test_Period': ['2019-2022', '2019-2022'],
        'N_Train': [len(y_train_temp), len(y_train_temp)],
        'N_Test': [len(y_test_temp), len(y_test_temp)],
        'AUROC': [lr_auc_temp, xgb_auc_temp],
        'PR_AUC': [lr_ap_temp, xgb_ap_temp],
        'Brier': [np.nan, xgb_brier_temp]
    })
    
    temporal_results.to_csv(os.path.join(OUTPUT_DIR, 'temporal_validation_results.csv'), index=False)
    print(f"\n✓ Saved: temporal_validation_results.csv")

# =============================================================================
# SECTION 4: RACE/ETHNICITY STRATIFIED ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 4: RACE/ETHNICITY STRATIFIED PERFORMANCE")
print("=" * 80)

# For this section, we need the ORIGINAL test set predictions from main pipeline
# If you don't have them saved, you'll need to re-run the 80/20 split here
# I'll assume you have a test set with predictions

# OPTION A: If you saved test predictions from main pipeline, load them
# test_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'))

# OPTION B: Re-create 80/20 split (use same random_state as main pipeline)
print("Recreating 80/20 split to match main pipeline...")
from sklearn.model_selection import train_test_split

# Same split as main pipeline (random_state=42, stratified)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_imputed, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
)

print(f"Test set: N={len(y_test)}, events={y_test.sum()} ({100*y_test.mean():.1f}%)")

# Get race for test set
test_race = df.iloc[idx_test]['race'].values
test_dementia = df.iloc[idx_test]['dementia_icd'].values

# We need to re-train models to get predictions (or load saved model)
# For simplicity, I'll show the analysis structure - you can plug in your saved predictions

print("\nNOTE: To get predictions, either:")
print("1. Load saved model and predict on X_test, OR")
print("2. Copy xgb_proba_test and lr_proba_test from your main pipeline results")
print("\nFor now, creating placeholder - REPLACE with actual predictions\n")

# PLACEHOLDER - REPLACE WITH YOUR ACTUAL PREDICTIONS
# xgb_proba_test = your_saved_xgb_model.predict_proba(X_test)[:, 1]
# lr_proba_test = your_saved_lr_model.predict_proba(X_test)[:, 1]

# Create test results DataFrame for analysis
test_results_df = pd.DataFrame({
    'y_true': y_test,
    'race': test_race,
    'dementia_icd': test_dementia,
    # 'xgb_proba': xgb_proba_test,  # ADD YOUR PREDICTIONS HERE
    # 'lr_proba': lr_proba_test      # ADD YOUR PREDICTIONS HERE
})

# Save test indices and metadata for later
test_metadata = df.iloc[idx_test][['subject_id', 'stay_id', 'race', 'dementia_icd', 'age', 
                                     'sex_female', 'sofa_cns', 'rass_mean', 'gcs_first']].copy()
test_metadata['y_true'] = y_test
test_metadata.to_csv(os.path.join(OUTPUT_DIR, 'test_set_metadata.csv'), index=False)
print(f"✓ Saved test_set_metadata.csv (N={len(test_metadata)})")

# Race stratification analysis
print("\nRace/Ethnicity Performance:")
print("-" * 60)
race_results = []

for race_group in test_results_df['race'].unique():
    if pd.isna(race_group):
        race_label = 'Unknown'
    else:
        race_label = str(race_group)
    
    race_mask = (test_results_df['race'] == race_group) if not pd.isna(race_group) else test_results_df['race'].isna()
    race_subset = test_results_df[race_mask]
    
    n = len(race_subset)
    n_events = race_subset['y_true'].sum()
    prev = 100 * n_events / n if n > 0 else 0
    
    print(f"{race_label:20s}: N={n:5d}, Events={n_events:4d} ({prev:5.1f}%)", end='')
    
    if n < 30:
        print(" - SKIPPED (N too small)")
        continue
    
    # PLACEHOLDER - compute AUROC when you have predictions
    # auc_xgb = roc_auc_score(race_subset['y_true'], race_subset['xgb_proba'])
    # auc_lr = roc_auc_score(race_subset['y_true'], race_subset['lr_proba'])
    # print(f", XGB AUROC={auc_xgb:.3f}, LR AUROC={auc_lr:.3f}")
    print(" - Add predictions to compute AUROC")
    
    race_results.append({
        'Race': race_label,
        'N': n,
        'Events': int(n_events),
        'Prevalence': prev,
        # 'XGB_AUROC': auc_xgb,
        # 'LR_AUROC': auc_lr
    })

race_results_df = pd.DataFrame(race_results)
race_results_df.to_csv(os.path.join(OUTPUT_DIR, 'race_stratified_performance.csv'), index=False)
print(f"\n✓ Saved: race_stratified_performance.csv")

# =============================================================================
# SECTION 5: DEMENTIA SUBGROUP - CORRECTED TEST SET STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 5: DEMENTIA SUBGROUP ANALYSIS (TEST SET)")
print("=" * 80)

# Demographics by dementia status (TEST SET ONLY)
dementia_test = test_results_df[test_results_df['dementia_icd'] == 1]
no_dementia_test = test_results_df[test_results_df['dementia_icd'] == 0]

print(f"\nTest Set Dementia Subgroups:")
print(f"  No dementia: N={len(no_dementia_test)}, Events={no_dementia_test['y_true'].sum()} ({100*no_dementia_test['y_true'].mean():.1f}%)")
print(f"  Dementia:    N={len(dementia_test)}, Events={dementia_test['y_true'].sum()} ({100*dementia_test['y_true'].mean():.1f}%)")

# Get detailed demographics for test set
test_demo = df.iloc[idx_test].copy()
test_demo['y_true'] = y_test

# Table 1 for abstract: Demographics by dementia (TEST SET)
demo_vars = ['age', 'sex_female', 'sofa_cns', 'rass_mean', 'gcs_first', 'comorbidity_count']
available_demo_vars = [v for v in demo_vars if v in test_demo.columns]

demographics_table = []
for var in available_demo_vars:
    no_dem_vals = test_demo[test_demo['dementia_icd'] == 0][var].dropna()
    dem_vals = test_demo[test_demo['dementia_icd'] == 1][var].dropna()
    
    if var in ['sex_female']:  # binary
        no_dem_stat = f"{100*no_dem_vals.mean():.1f}%"
        dem_stat = f"{100*dem_vals.mean():.1f}%"
    else:  # continuous
        no_dem_stat = f"{no_dem_vals.mean():.1f} ({no_dem_vals.std():.1f})"
        dem_stat = f"{dem_vals.mean():.1f} ({dem_vals.std():.1f})"
    
    demographics_table.append({
        'Variable': var,
        'No_Dementia': no_dem_stat,
        'Dementia': dem_stat
    })

demographics_df = pd.DataFrame(demographics_table)
demographics_df.to_csv(os.path.join(OUTPUT_DIR, 'test_demographics_by_dementia.csv'), index=False)
print(f"\n✓ Saved: test_demographics_by_dementia.csv")
print("\nDemographics by Dementia Status (TEST SET):")
print(demographics_df.to_string(index=False))

# Performance by dementia (PLACEHOLDER - add predictions)
print("\n" + "-" * 60)
print("Performance by Dementia Status (TEST SET):")
print("-" * 60)
print("No dementia: N={}, Events={}".format(len(no_dementia_test), no_dementia_test['y_true'].sum()))
print("Dementia:    N={}, Events={}".format(len(dementia_test), dementia_test['y_true'].sum()))
print("\nADD XGBoost and LR predictions to compute AUROC by dementia status")

# =============================================================================
# SECTION 6: CALIBRATION PLOTS BY DEMENTIA STATUS
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 6: CALIBRATION PLOTS BY DEMENTIA")
print("=" * 80)

print("PLACEHOLDER: Need XGBoost predictions to generate calibration plots")
print("After you add predictions, this will generate 2-panel calibration plot:")
print("  Left panel: No dementia calibration curve")
print("  Right panel: Dementia calibration curve")

# PLACEHOLDER CODE - uncomment after adding predictions
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# No dementia calibration
no_dem_y = no_dementia_test['y_true'].values
no_dem_proba = no_dementia_test['xgb_proba'].values  # ADD THIS
prob_true_no_dem, prob_pred_no_dem = calibration_curve(no_dem_y, no_dem_proba, n_bins=10, strategy='quantile')

axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
axes[0].plot(prob_pred_no_dem, prob_true_no_dem, 's-', label='XGBoost', linewidth=2, markersize=8)
axes[0].set_xlabel('Predicted Probability', fontsize=12)
axes[0].set_ylabel('Observed Frequency', fontsize=12)
axes[0].set_title(f'No Dementia (N={len(no_dementia_test)})', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Dementia calibration
dem_y = dementia_test['y_true'].values
dem_proba = dementia_test['xgb_proba'].values  # ADD THIS
prob_true_dem, prob_pred_dem = calibration_curve(dem_y, dem_proba, n_bins=5, strategy='quantile')

axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
axes[1].plot(prob_pred_dem, prob_true_dem, 's-', label='XGBoost', linewidth=2, markersize=8, color='C1')
axes[1].set_xlabel('Predicted Probability', fontsize=12)
axes[1].set_ylabel('Observed Frequency', fontsize=12)
axes[1].set_title(f'Dementia (N={len(dementia_test)})', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_by_dementia.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: calibration_by_dementia.png")
"""

# =============================================================================
# SECTION 7: CNS SOFA AND RASS DISTRIBUTIONS BY DEMENTIA
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 7: CNS SOFA AND RASS BY DEMENTIA (TEST SET)")
print("=" * 80)

if 'sofa_cns' in test_demo.columns and 'rass_mean' in test_demo.columns:
    print("\nCNS SOFA by Dementia Status:")
    print(f"  No dementia: median={test_demo[test_demo['dementia_icd']==0]['sofa_cns'].median():.1f}, IQR=[{test_demo[test_demo['dementia_icd']==0]['sofa_cns'].quantile(0.25):.1f}, {test_demo[test_demo['dementia_icd']==0]['sofa_cns'].quantile(0.75):.1f}]")
    print(f"  Dementia:    median={test_demo[test_demo['dementia_icd']==1]['sofa_cns'].median():.1f}, IQR=[{test_demo[test_demo['dementia_icd']==1]['sofa_cns'].quantile(0.25):.1f}, {test_demo[test_demo['dementia_icd']==1]['sofa_cns'].quantile(0.75):.1f}]")
    
    print("\nRASS Mean by Dementia Status:")
    print(f"  No dementia: median={test_demo[test_demo['dementia_icd']==0]['rass_mean'].median():.2f}, IQR=[{test_demo[test_demo['dementia_icd']==0]['rass_mean'].quantile(0.25):.2f}, {test_demo[test_demo['dementia_icd']==0]['rass_mean'].quantile(0.75):.2f}]")
    print(f"  Dementia:    median={test_demo[test_demo['dementia_icd']==1]['rass_mean'].median():.2f}, IQR=[{test_demo[test_demo['dementia_icd']==1]['rass_mean'].quantile(0.25):.2f}, {test_demo[test_demo['dementia_icd']==1]['rass_mean'].quantile(0.75):.2f}]")
    
    # Statistical test
    from scipy.stats import mannwhitneyu
    stat_cns, p_cns = mannwhitneyu(
        test_demo[test_demo['dementia_icd']==0]['sofa_cns'].dropna(),
        test_demo[test_demo['dementia_icd']==1]['sofa_cns'].dropna()
    )
    stat_rass, p_rass = mannwhitneyu(
        test_demo[test_demo['dementia_icd']==0]['rass_mean'].dropna(),
        test_demo[test_demo['dementia_icd']==1]['rass_mean'].dropna()
    )
    print(f"\nMann-Whitney U test:")
    print(f"  CNS SOFA: p={p_cns:.4f}")
    print(f"  RASS mean: p={p_rass:.4f}")

# =============================================================================
# DONE
# =============================================================================

print("\n" + "=" * 80)
print("MLHC ANALYSES COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. temporal_validation_results.csv")
print("  2. race_stratified_performance.csv")
print("  3. test_set_metadata.csv")
print("  4. test_demographics_by_dementia.csv")
print("\nTO COMPLETE:")
print("  - Add your XGBoost and LR predictions to test_results_df")
print("  - Re-run Sections 4-6 to get AUROC by race and dementia")
print("  - Uncomment calibration plot code in Section 6")
print("\nThese results are needed for your MLHC abstract!")
print("=" * 80)
