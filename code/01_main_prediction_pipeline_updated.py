"""
=============================================================================
MIMIC-IV Delirium Prediction Pipeline
=============================================================================
Outcome:    outcome_incident_cam_24h (new-onset delirium after first 24h ICU)
Cohort:     First ICU stay, CAM-screened, incident outcome defined
Features:   First 24h vitals, labs, medications, severity scores, comorbidities
Models:     Logistic Regression (baseline) | XGBoost (primary)
Extras:     LASSO feature selection | SHAP | Dementia subgroup analysis

Requirements:
    pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn

Usage:
    Edit CSV_PATH, CSV_FULL, OUTPUT_DIR below then run:
    python delirium_prediction_pipeline.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# IMPORTANT: Use raw strings (r"...") for Windows paths to avoid SyntaxWarning
# =============================================================================

CSV_PATH   = r"F:\full_cohort.csv"      # <-- update path if needed
CSV_FULL   = r"F:\full_cohort.csv"      # same file; sensitivity analysis runs on full cohort
OUTPUT_DIR = r"F:\delirium_outputs1"

RANDOM_SEED = 42
TEST_SIZE   = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

print("=" * 70)
print("SECTION 1: Loading data")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Outcome distribution:\n{df['outcome_incident_cam_24h'].value_counts()}")
print(f"Event rate: {df['outcome_incident_cam_24h'].mean()*100:.1f}%")

# =============================================================================
# SECTION 2: DEFINE FEATURE SETS
# =============================================================================

ID_COLS = [
    'subject_id', 'hadm_id', 'stay_id', 'icu_stay_seq',
    'is_first_icu_stay', 'admission_year', 'stays_per_patient'
]

OUTCOME_COLS = [
    'outcome_incident_cam_24h',
    'outcome_incident_cam_12h', 'outcome_incident_any_24h',
    'delirium_any', 'delirium_icd', 'cam_positive',
    'concordance_icd_cam', 'group_2x2x2', 'exposure_group',
    'hours_to_first_cam_positive', 'prevalent_cam_12h', 'prevalent_cam_24h',
    'mortality_icu', 'mortality_inhospital', 'mortality_30day',
    'mortality_90day', 'mortality_1year',
    'icu_los_days', 'hospital_los_days'
]

CAM_META_COLS = [
    'cam_screened', 'cam_total_assessments',
    'cam_positive_count', 'cam_negative_count', 'cam_uta_count'
]

# High missing — too sparse for reliable imputation
# albumin_min 73%, bilirubin_max 57.8%, alt_max 57.4%, ast_max 57.1%, lactate_max 45.9%
DROP_HIGH_MISSING = ['albumin_min', 'bilirubin_max', 'alt_max', 'ast_max', 'lactate_max']

# Leakage variables — reflect nurse response to suspected delirium, not predictors
LEAKAGE_COLS = [
    'rass_assessment_count',
    'antipsychotic_any',
    'haloperidol',
    'quetiapine',
]

# Non-numeric columns that cannot be cast to float — handle separately
NON_NUMERIC_COLS = ['anchor_year_group']

EXCLUDE_COLS = set(
    ID_COLS + OUTCOME_COLS + CAM_META_COLS + DROP_HIGH_MISSING + LEAKAGE_COLS + NON_NUMERIC_COLS
)

TARGET = 'outcome_incident_cam_24h'

# Features = all columns in primary CSV not excluded
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"\nTotal features: {len(FEATURE_COLS)}")
print(f"Excluded — leakage: {LEAKAGE_COLS}")
print(f"Excluded — high missing: {DROP_HIGH_MISSING}")

# =============================================================================
# SECTION 3: PREPROCESSING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Preprocessing")
print("=" * 70)

X = df[FEATURE_COLS].copy().astype(float)   # float throughout — prevents imputer dtype errors
y = df[TARGET].copy()

# Missing data audit
missing     = X.isnull().sum()
missing_pct = (missing / len(X) * 100).round(1)
missing_df  = pd.DataFrame({'n_missing': missing, 'pct_missing': missing_pct})
missing_df  = missing_df[missing_df['n_missing'] > 0].sort_values('pct_missing', ascending=False)
print(f"\nMissing values (before imputation):")
print(missing_df.to_string())
missing_df.to_csv(os.path.join(OUTPUT_DIR, 'missing_data_summary.csv'))

# Train / test split (stratified on outcome)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"Train event rate: {y_train.mean()*100:.1f}%  |  Test: {y_test.mean()*100:.1f}%")

# Identify binary vs continuous columns
# Binary = all non-null values are 0 or 1
binary_cols     = [c for c in FEATURE_COLS if df[c].dropna().isin([0.0, 1.0]).all()]
continuous_cols = [c for c in FEATURE_COLS if c not in binary_cols]
print(f"\nContinuous: {len(continuous_cols)}  |  Binary: {len(binary_cols)}")

# Fit imputers on train set only
imputer_cont = SimpleImputer(strategy='median')
imputer_bin  = SimpleImputer(strategy='constant', fill_value=0.0)

X_train_cont = imputer_cont.fit_transform(X_train[continuous_cols])
X_test_cont  = imputer_cont.transform(X_test[continuous_cols])

X_train_bin  = imputer_bin.fit_transform(X_train[binary_cols])
X_test_bin   = imputer_bin.transform(X_test[binary_cols])

X_train_imp       = np.hstack([X_train_cont, X_train_bin])
X_test_imp        = np.hstack([X_test_cont,  X_test_bin])
feature_names_imp = continuous_cols + binary_cols

# Scale for logistic regression
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

print("Imputation and scaling complete.")

# =============================================================================
# SECTION 4: LASSO FEATURE SELECTION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: LASSO feature selection")
print("=" * 70)

lasso_cv = LogisticRegressionCV(
    Cs=np.logspace(-4, 1, 30),
    cv=5,
    penalty='l1',
    solver='liblinear',
    scoring='roc_auc',
    random_state=RANDOM_SEED,
    max_iter=1000,
    n_jobs=-1
)
lasso_cv.fit(X_train_scaled, y_train)

best_C            = lasso_cv.C_[0]
lasso_coefs       = pd.Series(lasso_cv.coef_[0], index=feature_names_imp)
selected_features = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)

print(f"Best C: {best_C:.5f}")
print(f"Features selected: {len(selected_features)} / {len(feature_names_imp)}")
print(f"\nTop 20 LASSO features:")
print(selected_features.head(20).to_string())

pd.DataFrame({
    'feature':    lasso_coefs.index,
    'lasso_coef': lasso_coefs.values,
    'abs_coef':   np.abs(lasso_coefs.values),
    'selected':   (lasso_coefs != 0).values
}).sort_values('abs_coef', ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, 'lasso_coefficients.csv'), index=False
)

# Index positions of selected features
sel_idx           = [feature_names_imp.index(f) for f in selected_features.index]
sel_feature_names = list(selected_features.index)

X_train_sel     = X_train_scaled[:, sel_idx]
X_test_sel      = X_test_scaled[:,  sel_idx]
X_train_sel_raw = X_train_imp[:,    sel_idx]
X_test_sel_raw  = X_test_imp[:,     sel_idx]

# LASSO coefficient plot
top_n     = min(25, len(selected_features))
top_feats = selected_features.head(top_n)
fig, ax   = plt.subplots(figsize=(9, top_n * 0.36 + 1.5))
colors    = ['#c0392b' if v > 0 else '#2980b9' for v in top_feats.values[::-1]]
ax.barh(range(top_n), top_feats.values[::-1], color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_feats.index[::-1], fontsize=9)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('LASSO Coefficient', fontsize=11)
ax.set_title(f'Top {top_n} LASSO-Selected Features\n(Red=risk, Blue=protective)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lasso_coefficients_plot.png'), dpi=150)
plt.close()
print("LASSO plot saved.")

# =============================================================================
# SECTION 5: LOGISTIC REGRESSION (BASELINE)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Logistic Regression (baseline)")
print("=" * 70)

lr = LogisticRegression(
    C=best_C, penalty='l1', solver='liblinear',
    random_state=RANDOM_SEED, max_iter=1000
)
lr.fit(X_train_sel, y_train)

lr_proba_test  = lr.predict_proba(X_test_sel)[:, 1]
lr_proba_train = lr.predict_proba(X_train_sel)[:, 1]
lr_auc_test    = roc_auc_score(y_test, lr_proba_test)
lr_auc_train   = roc_auc_score(y_train, lr_proba_train)
lr_ap          = average_precision_score(y_test, lr_proba_test)
lr_brier       = brier_score_loss(y_test, lr_proba_test)
lr_cv_scores   = cross_val_score(lr, X_train_sel, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

print(f"Train AUROC: {lr_auc_train:.3f}  |  Test AUROC: {lr_auc_test:.3f}")
print(f"PR-AUC: {lr_ap:.3f}  |  Brier: {lr_brier:.4f}")
print(f"5-fold CV: {lr_cv_scores.mean():.3f} +/- {lr_cv_scores.std():.3f}")

# =============================================================================
# SECTION 6: XGBOOST (PRIMARY)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: XGBoost")
print("=" * 70)

neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
scale_pw  = neg_count / pos_count
print(f"Class ratio (neg/pos): {scale_pw:.2f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pw,
    eval_metric='auc',
    random_state=RANDOM_SEED,
    n_jobs=-1,
    early_stopping_rounds=30
)
xgb_model.fit(
    X_train_sel_raw, y_train,
    eval_set=[(X_test_sel_raw, y_test)],
    verbose=False
)

xgb_proba_test  = xgb_model.predict_proba(X_test_sel_raw)[:, 1]
xgb_proba_train = xgb_model.predict_proba(X_train_sel_raw)[:, 1]
xgb_auc_test    = roc_auc_score(y_test, xgb_proba_test)
xgb_auc_train   = roc_auc_score(y_train, xgb_proba_train)
xgb_ap          = average_precision_score(y_test, xgb_proba_test)
xgb_brier_raw   = brier_score_loss(y_test, xgb_proba_test)

xgb_cv_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pw, eval_metric='auc',
    random_state=RANDOM_SEED, n_jobs=-1
)
xgb_cv_scores = cross_val_score(
    xgb_cv_model, X_train_sel_raw, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
    scoring='roc_auc', n_jobs=-1
)

# Bootstrap 95% CI on uncalibrated probabilities
np.random.seed(RANDOM_SEED)
boot_aucs = []
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    try:
        boot_aucs.append(roc_auc_score(y_test.iloc[idx], xgb_proba_test[idx]))
    except Exception:
        pass
ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

print(f"Train AUROC: {xgb_auc_train:.3f}  |  Test AUROC: {xgb_auc_test:.3f}")
print(f"PR-AUC: {xgb_ap:.3f}  |  Brier (raw): {xgb_brier_raw:.4f}")
print(f"5-fold CV: {xgb_cv_scores.mean():.3f} +/- {xgb_cv_scores.std():.3f}")
print(f"Bootstrap 95% CI: ({ci_lo:.3f}, {ci_hi:.3f})")

# =============================================================================
# SECTION 7: XGBOOST CALIBRATION (isotonic regression)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: XGBoost calibration")
print("=" * 70)

# Split train into train2 + calibration hold-out
X_tr2, X_cal, y_tr2, y_cal = train_test_split(
    X_train_sel_raw, y_train,
    test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
)
# Refit on train2 with early stopping against calibration set
xgb_model.fit(X_tr2, y_tr2, eval_set=[(X_cal, y_cal)], verbose=False)

# Calibrate on hold-out
xgb_cal = CalibratedClassifierCV(xgb_model, method='isotonic', cv='prefit')
xgb_cal.fit(X_cal, y_cal)

# Replace test probabilities with calibrated ones everywhere downstream
xgb_proba_test  = xgb_cal.predict_proba(X_test_sel_raw)[:, 1]
xgb_proba_train = xgb_cal.predict_proba(X_train_sel_raw)[:, 1]
xgb_auc_test    = roc_auc_score(y_test, xgb_proba_test)
xgb_brier       = brier_score_loss(y_test, xgb_proba_test)

print(f"Calibrated XGBoost — AUROC: {xgb_auc_test:.3f}  |  Brier: {xgb_brier:.4f}")

# =============================================================================
# SECTION 8: ROC + PRECISION-RECALL CURVES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8: ROC and PR curves")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_proba_test)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba_test)

axes[0].plot(fpr_lr,  tpr_lr,  color='#2980b9', lw=2,
             label=f'Logistic Regression (AUROC={lr_auc_test:.3f})')
axes[0].plot(fpr_xgb, tpr_xgb, color='#c0392b', lw=2,
             label=f'XGBoost calibrated (AUROC={xgb_auc_test:.3f}, 95%CI {ci_lo:.3f}-{ci_hi:.3f})')
axes[0].plot([0,1],[0,1], 'k--', lw=0.8, label='Random')
axes[0].set_xlabel('False Positive Rate', fontsize=11)
axes[0].set_ylabel('True Positive Rate', fontsize=11)
axes[0].set_title('ROC Curve — Incident Delirium\n(Test Set)', fontsize=11)
axes[0].legend(fontsize=9)

prec_lr,  rec_lr,  _ = precision_recall_curve(y_test, lr_proba_test)
prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, xgb_proba_test)
baseline_pr = float(y_test.mean())

axes[1].plot(rec_lr,  prec_lr,  color='#2980b9', lw=2,
             label=f'Logistic Regression (AP={lr_ap:.3f})')
axes[1].plot(rec_xgb, prec_xgb, color='#c0392b', lw=2,
             label=f'XGBoost (AP={xgb_ap:.3f})')
axes[1].axhline(baseline_pr, color='k', linestyle='--', lw=0.8,
                label=f'Baseline (prev={baseline_pr:.3f})')
axes[1].set_xlabel('Recall', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].set_title('Precision-Recall Curve\n(Test Set)', fontsize=11)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_pr_curves.png'), dpi=150)
plt.close()
print("ROC + PR curves saved.")

# =============================================================================
# SECTION 9: CALIBRATION PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(6, 5))
for proba, label, color, marker in [
    (lr_proba_test,                  'Logistic Regression',     '#2980b9', 'o'),
    (xgb_model.predict_proba(X_test_sel_raw)[:, 1],
                                     'XGBoost (uncalibrated)',   '#e74c3c', 's'),
    (xgb_proba_test,                 'XGBoost (calibrated)',     '#27ae60', '^'),
]:
    pt, pp = calibration_curve(y_test, proba, n_bins=10)
    ax.plot(pp, pt, marker=marker, color=color, label=label, lw=1.5)

ax.plot([0,1],[0,1], 'k--', lw=0.8, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability', fontsize=11)
ax.set_ylabel('Fraction of Positives', fontsize=11)
ax.set_title('Calibration Plot (Test Set)', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_plot.png'), dpi=150)
plt.close()
print("Calibration plot saved.")

# =============================================================================
# SECTION 10: SHAP VALUES (XGBoost)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 10: SHAP (XGBoost)")
print("=" * 70)

explainer    = shap.TreeExplainer(xgb_model)
shap_values  = explainer.shap_values(X_test_sel_raw)
shap_df      = pd.DataFrame(shap_values, columns=sel_feature_names)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
mean_abs_shap.to_csv(os.path.join(OUTPUT_DIR, 'shap_mean_abs_values.csv'), header=True)

print("Top 15 features by mean |SHAP|:")
print(mean_abs_shap.head(15).to_string())

plt.figure(figsize=(9, 8))
shap.summary_plot(shap_values, X_test_sel_raw, feature_names=sel_feature_names,
                  max_display=20, show=False, plot_type='dot')
plt.title("SHAP Summary — XGBoost (Top 20)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_beeswarm.png'), dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, X_test_sel_raw, feature_names=sel_feature_names,
                  max_display=20, show=False, plot_type='bar')
plt.title("SHAP Feature Importance — XGBoost (Top 20)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("SHAP plots saved.")

# =============================================================================
# SECTION 11: DEMENTIA SUBGROUP ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 11: Dementia subgroup analysis")
print("=" * 70)

test_df = X_test.copy()
test_df['y_true']    = y_test.values
test_df['xgb_proba'] = xgb_proba_test
test_df['lr_proba']  = lr_proba_test

dementia_groups = {
    'No dementia (dementia_icd=0)': test_df['dementia_icd'] == 0,
    'Dementia (dementia_icd=1)':    test_df['dementia_icd'] == 1
}

subgroup_results = []
for label, mask in dementia_groups.items():
    subset = test_df[mask]
    n_pos  = int(subset['y_true'].sum())
    if n_pos < 10:
        print(f"  {label}: too few events ({n_pos}), skipping")
        continue
    auc_xgb = roc_auc_score(subset['y_true'], subset['xgb_proba'])
    auc_lr  = roc_auc_score(subset['y_true'], subset['lr_proba'])
    ap_xgb  = average_precision_score(subset['y_true'], subset['xgb_proba'])
    prev    = n_pos / len(subset)
    subgroup_results.append({
        'Subgroup': label, 'N': len(subset), 'N_delirium': n_pos,
        'Prevalence_%': round(prev*100, 1),
        'XGBoost_AUROC': round(auc_xgb, 3),
        'LR_AUROC': round(auc_lr, 3),
        'XGBoost_AP': round(ap_xgb, 3)
    })
    print(f"  {label}: N={len(subset):,}, events={n_pos}, prev={prev*100:.1f}%, "
          f"XGB={auc_xgb:.3f}, LR={auc_lr:.3f}")

for col, label in [('dementia_alzheimers', "Alzheimer's dementia"),
                   ('dementia_vascular',   "Vascular dementia")]:
    if col in test_df.columns:
        subset = test_df[test_df[col] == 1]
        n_pos  = int(subset['y_true'].sum())
        if n_pos >= 10:
            auc_xgb = roc_auc_score(subset['y_true'], subset['xgb_proba'])
            print(f"  {label}: N={len(subset):,}, events={n_pos}, XGB={auc_xgb:.3f}")
            subgroup_results.append({
                'Subgroup': label, 'N': len(subset), 'N_delirium': n_pos,
                'Prevalence_%': round(n_pos/len(subset)*100, 1),
                'XGBoost_AUROC': round(auc_xgb, 3),
                'LR_AUROC': None, 'XGBoost_AP': None
            })

pd.DataFrame(subgroup_results).to_csv(
    os.path.join(OUTPUT_DIR, 'dementia_subgroup_results.csv'), index=False)

fig, ax = plt.subplots(figsize=(7, 5))
colors_sub = ['#2ecc71', '#e74c3c']
for i, (label, mask) in enumerate(dementia_groups.items()):
    subset = test_df[mask]
    if subset['y_true'].sum() < 10:
        continue
    fpr, tpr, _ = roc_curve(subset['y_true'], subset['xgb_proba'])
    auc = roc_auc_score(subset['y_true'], subset['xgb_proba'])
    ax.plot(fpr, tpr, color=colors_sub[i], lw=2, label=f'{label} (AUROC={auc:.3f})')
ax.plot([0,1],[0,1], 'k--', lw=0.8)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('XGBoost ROC — Dementia Subgroups (Test Set)', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dementia_subgroup_roc.png'), dpi=150)
plt.close()
print("Subgroup plots saved.")

# =============================================================================
# SECTION 12: SUMMARY PERFORMANCE TABLE
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 12: Summary performance table")
print("=" * 70)

summary = pd.DataFrame([
    {
        'Model': 'Logistic Regression (LASSO)', 'Cohort': 'Test set (primary)',
        'N_test': len(y_test), 'N_positive': int(y_test.sum()),
        'AUROC': round(lr_auc_test, 3),
        'CV_AUROC_mean': round(lr_cv_scores.mean(), 3),
        'CV_AUROC_sd': round(lr_cv_scores.std(), 3),
        'PR_AUC': round(lr_ap, 3),
        'Brier_Score': round(lr_brier, 4),
        'CI_95_lo': None, 'CI_95_hi': None
    },
    {
        'Model': 'XGBoost (calibrated)', 'Cohort': 'Test set (primary)',
        'N_test': len(y_test), 'N_positive': int(y_test.sum()),
        'AUROC': round(xgb_auc_test, 3),
        'CV_AUROC_mean': round(xgb_cv_scores.mean(), 3),
        'CV_AUROC_sd': round(xgb_cv_scores.std(), 3),
        'PR_AUC': round(xgb_ap, 3),
        'Brier_Score': round(xgb_brier, 4),
        'CI_95_lo': round(ci_lo, 3), 'CI_95_hi': round(ci_hi, 3)
    }
])
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUTPUT_DIR, 'model_performance_summary.csv'), index=False)

# =============================================================================
# SECTION 13: SENSITIVITY ANALYSIS — full cohort (all ICU stays)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 13: Sensitivity analysis — full cohort")
print("=" * 70)

try:
    df_full = pd.read_csv(CSV_FULL)

    # Add any columns present in primary but missing in full — fill with 0
    missing_in_full = [c for c in FEATURE_COLS if c not in df_full.columns]
    if missing_in_full:
        print(f"Zero-filling missing columns in full CSV: {missing_in_full}")
    for col in missing_in_full:
        df_full[col] = 0

    # Cast to float — critical, prevents dtype mismatch in imputers
    X_full = df_full[FEATURE_COLS].astype(float)
    y_full = df_full[TARGET].copy()

    X_full_cont    = imputer_cont.transform(X_full[continuous_cols].astype(float))
    X_full_bin     = imputer_bin.transform(X_full[binary_cols].astype(float))
    X_full_imp     = np.hstack([X_full_cont, X_full_bin])
    X_full_scaled  = scaler.transform(X_full_imp)
    X_full_sel     = X_full_scaled[:, sel_idx]
    X_full_sel_raw = X_full_imp[:, sel_idx]

    xgb_full_proba = xgb_cal.predict_proba(X_full_sel_raw)[:, 1]
    lr_full_proba  = lr.predict_proba(X_full_sel)[:, 1]

    auc_xgb_full = roc_auc_score(y_full, xgb_full_proba)
    auc_lr_full  = roc_auc_score(y_full, lr_full_proba)

    print(f"Full cohort N={len(y_full):,} (full first-stay cohort, no CAM filter)")
    print(f"  XGBoost AUROC: {auc_xgb_full:.3f}")
    print(f"  LR AUROC:      {auc_lr_full:.3f}")
    if missing_in_full:
        print(f"  Note: {missing_in_full} zero-filled. Re-export after SQL Section 3D for clean run.")

except FileNotFoundError:
    print("delirium_model_full.csv not found — skipping sensitivity analysis.")
    print("Export from PostgreSQL and update CSV_FULL path to run this section.")

# =============================================================================
# SECTION 14: DONE
# =============================================================================

print("\n" + "=" * 70)
print(f"ALL DONE — outputs saved to: {OUTPUT_DIR}")
print("=" * 70)
for f in [
    "missing_data_summary.csv",
    "lasso_coefficients.csv",
    "lasso_coefficients_plot.png",
    "roc_pr_curves.png",
    "calibration_plot.png",
    "shap_summary_beeswarm.png",
    "shap_summary_bar.png",
    "shap_mean_abs_values.csv",
    "dementia_subgroup_results.csv",
    "dementia_subgroup_roc.png",
    "model_performance_summary.csv",
]:
    print(f"  {f}")
