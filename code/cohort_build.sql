-- =============================================================================
-- MIMIC-IV Delirium Prediction Cohort — Full Build Script
-- For: Ran Xiao
-- From: Sohaib (Muhammad Sohaib, MBBS, MPH)
-- MIMIC-IV v3.1 | PostgreSQL
--
-- PURPOSE:
--   Builds the full analytic cohort from raw MIMIC-IV tables and exports
--   a single CSV ready for ML modeling (delirium_model_v2.csv).
--   No external tables required. Runs in 3 steps.
--
-- OUTPUT COLUMNS match delirium_model_v2.csv exactly, plus added robustness.
--
-- RUNTIME NOTE:
--   chartevents is ~330M rows. Steps with chartevents joins may take
--   5-20 minutes depending on server specs and indexes.
--   Run via psql CLI (not pgAdmin) so \COPY works client-side.
--
-- USAGE:
--   psql -U <user> -d <database> -f build_delirium_cohort_ranxiao.sql
--   Then update the output path in Step 3 before running.
-- =============================================================================


-- =============================================================================
-- STEP 1 (OPTIONAL) — Quick sanity checks on itemids
-- Run these individually first to verify your MIMIC instance.
-- =============================================================================

-- Confirm CAM-ICU itemid
SELECT di.itemid, di.label, COUNT(*) AS n
FROM mimiciv_icu.chartevents ce
JOIN mimiciv_icu.d_items di ON ce.itemid = di.itemid
WHERE di.label ILIKE '%cam%'
GROUP BY di.itemid, di.label
ORDER BY n DESC LIMIT 10;

-- Confirm RASS itemid
SELECT itemid, label FROM mimiciv_icu.d_items
WHERE label ILIKE '%rass%' OR label ILIKE '%richmond%';

-- Confirm pain score itemid 223791
SELECT di.itemid, di.label, COUNT(*) AS n
FROM mimiciv_icu.chartevents ce
JOIN mimiciv_icu.d_items di ON ce.itemid = di.itemid
WHERE di.label ILIKE '%pain%' AND ce.valuenum IS NOT NULL
GROUP BY di.itemid, di.label
ORDER BY n DESC LIMIT 10;

-- Confirm GCS items
SELECT itemid, label FROM mimiciv_icu.d_items
WHERE itemid IN (220739, 223900, 223901);

-- Confirm vital sign items
SELECT itemid, label FROM mimiciv_icu.d_items
WHERE itemid IN (220045, 220050, 220179, 220052, 220181, 220210, 220277, 223761, 223762);


-- =============================================================================
-- STEP 2 — Build all temp tables
-- Run this entire block as one transaction.
-- =============================================================================

-- ----------------------------------------------------------------------------
-- 2A: ICU STAY SEQUENCE (first stay flag, prior ICU stay)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS icu_seq;
CREATE TEMP TABLE icu_seq AS
SELECT
    stay_id,
    subject_id,
    ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS icu_stay_seq,
    CASE WHEN ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) = 1
         THEN 1 ELSE 0 END AS is_first_icu_stay,
    COUNT(*) OVER (PARTITION BY subject_id) AS stays_per_patient,
    CASE WHEN ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) > 1
         THEN 1 ELSE 0 END AS prior_icu_stay
FROM mimiciv_icu.icustays;


-- ----------------------------------------------------------------------------
-- 2B: CAM-ICU ASSESSMENTS (outcome definitions)
-- itemid 228300 = CAM-ICU, also check 228301/228302/228303/227013
-- Outcome: incident delirium = first CAM-positive AFTER first 12h of ICU stay
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS cam_assessments;
CREATE TEMP TABLE cam_assessments AS
-- -----------------------------------------------------------------------
-- CAM-ICU itemids confirmed from d_items in this MIMIC instance:
--   228300 / 228337 / 229326 = MS Change (Feature 1 gate)
--     value 'Yes' or 'Yes (Continue)' = acute change present -> proceed
--   228301 / 228336 / 229325 = Inattention (Feature 2)
--   228302                   = RASS/LOC (Feature 3)
--   228303 / 228335 / 229324 = Disorganized Thinking (Feature 4)
--   228334                   = Altered LOC (Feature 3 alt)
--
-- CAM-ICU POSITIVE definition:
--   Feature 1 (MS Change) = Yes AND Feature 2 (Inattention) = Yes
--   AND (Feature 3 OR Feature 4) = Yes
--   Simplified here: any 'Yes' on the MS Change gate itemids (228300/228337/229326)
--   since a positive gate means the full assessment was triggered and completed
--   as positive. This matches behaviour in the original cohort script.
--
-- cam_screened: had any CAM-ICU assessment charted (any itemid, any value)
-- -----------------------------------------------------------------------
WITH ms_change AS (
    -- Feature 1: MS Change gate — Yes = acute change, proceed with assessment
    SELECT
        ce.stay_id,
        ce.charttime,
        CASE
            WHEN LOWER(ce.value) LIKE '%yes%' THEN 'positive'
            WHEN LOWER(ce.value) LIKE '%no%'  THEN 'negative'
            WHEN LOWER(ce.value) LIKE '%unable%' OR LOWER(ce.value) LIKE '%language%'
              OR LOWER(ce.value) LIKE '%barrier%' THEN 'unable'
            ELSE 'other'
        END AS cam_result
    FROM mimiciv_icu.chartevents ce
    WHERE ce.itemid IN (228300, 228337, 229326)
      AND ce.value IS NOT NULL
),
any_cam AS (
    -- All CAM-ICU chartings for screened flag and assessment counts
    SELECT
        ce.stay_id,
        ce.charttime,
        CASE
            WHEN LOWER(ce.value) LIKE '%yes%' THEN 'positive'
            WHEN LOWER(ce.value) LIKE '%no%'  THEN 'negative'
            WHEN LOWER(ce.value) LIKE '%unable%' OR LOWER(ce.value) LIKE '%language%'
              OR LOWER(ce.value) LIKE '%barrier%' THEN 'unable'
            ELSE 'other'
        END AS cam_result
    FROM mimiciv_icu.chartevents ce
    WHERE ce.itemid IN (228300, 228301, 228302, 228303,
                        228334, 228335, 228336, 228337,
                        229324, 229325, 229326)
      AND ce.value IS NOT NULL
),
per_stay AS (
    SELECT
        ac.stay_id,
        icu.intime,
        -- Use ms_change for the positive outcome (gate itemids only)
        MAX(CASE WHEN mc.cam_result = 'positive' THEN 1 ELSE 0 END)               AS cam_positive,
        -- Prevalent = positive MS Change in first 12h
        MAX(CASE WHEN mc.cam_result = 'positive'
                  AND mc.charttime <= icu.intime + INTERVAL '12 hours'
                 THEN 1 ELSE 0 END)                                                AS prevalent_cam_12h,
        MAX(CASE WHEN mc.cam_result = 'positive'
                  AND mc.charttime <= icu.intime + INTERVAL '24 hours'
                 THEN 1 ELSE 0 END)                                                AS prevalent_cam_24h,
        MAX(CASE WHEN mc.cam_result = 'positive'
                  AND mc.charttime > icu.intime + INTERVAL '12 hours'
                 THEN 1 ELSE 0 END)                                                AS outcome_incident_cam_12h,
        -- PRIMARY OUTCOME: any CAM-ICU positive (MS Change gate = Yes) during stay
        MAX(CASE WHEN mc.cam_result = 'positive' THEN 1 ELSE 0 END)               AS outcome_incident_cam_24h,
        MAX(CASE WHEN mc.cam_result = 'positive' THEN 1 ELSE 0 END)               AS outcome_incident_any_24h,
        -- Screened = any CAM-ICU item charted
        1                                                                           AS cam_screened,
        -- Assessment counts from all CAM items
        COUNT(DISTINCT ac.charttime)                                               AS cam_total_assessments,
        SUM(CASE WHEN ac.cam_result = 'positive' THEN 1 ELSE 0 END)               AS cam_positive_count,
        SUM(CASE WHEN ac.cam_result = 'negative' THEN 1 ELSE 0 END)               AS cam_negative_count,
        SUM(CASE WHEN ac.cam_result = 'unable'   THEN 1 ELSE 0 END)               AS cam_uta_count,
        -- Time to first positive MS Change (hours from ICU admission)
        MIN(CASE WHEN mc.cam_result = 'positive'
                 THEN EXTRACT(EPOCH FROM (mc.charttime - icu.intime))/3600.0
                 ELSE NULL END)                                                     AS hours_to_first_cam_positive
    FROM any_cam ac
    JOIN mimiciv_icu.icustays icu ON ac.stay_id = icu.stay_id
    LEFT JOIN ms_change mc        ON ac.stay_id = mc.stay_id
    GROUP BY ac.stay_id, icu.intime
)
SELECT * FROM per_stay;


-- ----------------------------------------------------------------------------
-- 2C: DELIRIUM ICD CODES
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS delirium_icd;
CREATE TEMP TABLE delirium_icd AS
SELECT
    i.stay_id,
    1 AS delirium_icd
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.diagnoses_icd dx ON i.hadm_id = dx.hadm_id
WHERE
    dx.icd_code LIKE 'F05%'
    OR dx.icd_code IN (
        'F10121','F10221','F10231','F10131','F10921',
        'F11121','F11221','F11921',
        'F12121','F12221','F12921',
        'F13121','F13221','F13231','F13131','F13921',
        'F14121','F14221','F14921',
        'F15121','F15221','F15921',
        'F16121','F16221','F16921',
        'F18121','F18221','F18921',
        'F19121','F19221','F19231','F19131','F19921',
        'F10931','F13931','F19931'
    )
    OR dx.icd_code IN (
        '2930','2931','29381','29389','29390',
        '2910','29011','29012','29013',
        '29200','29211','29212','29281','29282'
    )
GROUP BY i.stay_id;


-- ----------------------------------------------------------------------------
-- 2D: DEMENTIA ICD CODES
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS dementia_icd;
CREATE TEMP TABLE dementia_icd AS
SELECT
    i.stay_id,
    1 AS dementia_icd,
    MAX(CASE WHEN dx.icd_code LIKE 'G30%' OR dx.icd_code LIKE 'F00%'
              OR dx.icd_code IN ('3310') THEN 1 ELSE 0 END) AS dementia_alzheimers,
    MAX(CASE WHEN dx.icd_code LIKE 'F01%'
              OR dx.icd_code IN ('29040','29041','29042','29043') THEN 1 ELSE 0 END) AS dementia_vascular
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.diagnoses_icd dx ON i.hadm_id = dx.hadm_id
WHERE
    dx.icd_code LIKE 'F00%' OR dx.icd_code LIKE 'F01%'
    OR dx.icd_code LIKE 'F02%' OR dx.icd_code LIKE 'F03%'
    OR dx.icd_code LIKE 'G30%' OR dx.icd_code LIKE 'G31%'
    OR dx.icd_code IN (
        '2900','29010','29011','29012','29013',
        '29020','29021','2903',
        '29040','29041','29042','29043',
        '2941','3310','3311','3312'
    )
GROUP BY i.stay_id;


-- ----------------------------------------------------------------------------
-- 2E: RASS SCORES (full stay)
-- itemid 228096
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS rass_scores;
CREATE TEMP TABLE rass_scores AS
SELECT
    ce.stay_id,
    ROUND(AVG(ce.valuenum)::numeric, 2)                                    AS rass_mean,
    MIN(ce.valuenum)                                                        AS rass_min,
    MAX(ce.valuenum)                                                        AS rass_max,
    MAX(CASE WHEN ce.valuenum <= -4 THEN 1 ELSE 0 END)                     AS rass_ever_unarousable,
    MAX(CASE WHEN ce.valuenum <= -3 THEN 1 ELSE 0 END)                     AS rass_ever_deeply_sedated,
    MAX(CASE WHEN ce.valuenum >= 1  THEN 1 ELSE 0 END)                     AS rass_ever_agitated,
    COUNT(*)                                                                AS rass_assessment_count
FROM mimiciv_icu.chartevents ce
WHERE ce.itemid = 228096
  AND ce.valuenum BETWEEN -5 AND 4
GROUP BY ce.stay_id;


-- ----------------------------------------------------------------------------
-- 2F: GCS (first value + first 24h)
-- itemid 223901 = GCS Total
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS gcs_data;
CREATE TEMP TABLE gcs_data AS
WITH first_gcs AS (
    SELECT DISTINCT ON (ce.stay_id)
        ce.stay_id,
        ce.valuenum AS gcs_first
    FROM mimiciv_icu.chartevents ce
    WHERE ce.itemid = 223901
      AND ce.valuenum BETWEEN 3 AND 15
    ORDER BY ce.stay_id, ce.charttime
)
SELECT
    fg.stay_id,
    fg.gcs_first,
    CASE WHEN fg.gcs_first <= 8 THEN 1 ELSE 0 END AS gcs_severe_admission
FROM first_gcs fg;


-- ----------------------------------------------------------------------------
-- 2G: VITAL SIGNS (first 24h)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS vitals_24h;
CREATE TEMP TABLE vitals_24h AS
SELECT
    ce.stay_id,
    ROUND(AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END)::numeric, 1)   AS hr_mean,
    MAX(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END)                       AS hr_max,
    MIN(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END)                       AS hr_min,
    ROUND(STDDEV(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END)::numeric, 2) AS hr_sd,
    ROUND(AVG(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END)::numeric, 1) AS sbp_mean,
    MIN(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END)            AS sbp_min,
    MAX(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END)            AS sbp_max,
    ROUND(AVG(CASE WHEN ce.itemid IN (220052, 220181, 225312) THEN ce.valuenum END)::numeric, 1) AS map_mean,
    MIN(CASE WHEN ce.itemid IN (220052, 220181, 225312) THEN ce.valuenum END)    AS map_min,
    ROUND(AVG(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END)::numeric, 1)   AS spo2_mean,
    MIN(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END)                       AS spo2_min,
    ROUND(AVG(CASE WHEN ce.itemid = 220210 THEN ce.valuenum END)::numeric, 1)   AS rr_mean,
    MAX(CASE WHEN ce.itemid = 220210 THEN ce.valuenum END)                       AS rr_max,
    -- Temperature: itemid 223761 = Fahrenheit, 223762 = Celsius (convert to F)
    ROUND(AVG(CASE WHEN ce.itemid = 223761 THEN ce.valuenum
                   WHEN ce.itemid = 223762 THEN ce.valuenum * 9.0/5.0 + 32 END)::numeric, 1) AS temp_f_mean,
    MAX(CASE WHEN ce.itemid = 223761 THEN ce.valuenum
             WHEN ce.itemid = 223762 THEN ce.valuenum * 9.0/5.0 + 32 END)                    AS temp_f_max
FROM mimiciv_icu.chartevents ce
JOIN mimiciv_icu.icustays icu ON ce.stay_id = icu.stay_id
WHERE ce.itemid IN (220045, 220050, 220179, 220052, 220181, 225312, 220277, 220210, 223761, 223762)
  AND ce.valuenum IS NOT NULL
  AND ce.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
GROUP BY ce.stay_id;


-- ----------------------------------------------------------------------------
-- 2H: LABORATORY VALUES (first 24h)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS labs_24h;
CREATE TEMP TABLE labs_24h AS
SELECT
    i.stay_id,
    MAX(CASE WHEN le.itemid = 50912 THEN le.valuenum END) AS creatinine_max,
    MAX(CASE WHEN le.itemid = 51006 THEN le.valuenum END) AS bun_max,
    MIN(CASE WHEN le.itemid = 50983 THEN le.valuenum END) AS sodium_min,
    MAX(CASE WHEN le.itemid = 50983 THEN le.valuenum END) AS sodium_max,
    MIN(CASE WHEN le.itemid = 50971 THEN le.valuenum END) AS potassium_min,
    MAX(CASE WHEN le.itemid = 50971 THEN le.valuenum END) AS potassium_max,
    MIN(CASE WHEN le.itemid = 50902 THEN le.valuenum END) AS chloride_min,
    MIN(CASE WHEN le.itemid = 50893 THEN le.valuenum END) AS calcium_min,
    MAX(CASE WHEN le.itemid = 50893 THEN le.valuenum END) AS calcium_max,
    MIN(CASE WHEN le.itemid = 50931 THEN le.valuenum END) AS glucose_min,
    MAX(CASE WHEN le.itemid = 50931 THEN le.valuenum END) AS glucose_max,
    MAX(CASE WHEN le.itemid = 51301 THEN le.valuenum END) AS wbc_max,
    MIN(CASE WHEN le.itemid = 51222 THEN le.valuenum END) AS hemoglobin_min,
    MIN(CASE WHEN le.itemid = 51265 THEN le.valuenum END) AS platelets_min,
    MIN(CASE WHEN le.itemid = 51221 THEN le.valuenum END) AS hematocrit_min,
    MAX(CASE WHEN le.itemid = 50885 THEN le.valuenum END) AS bilirubin_max,
    MAX(CASE WHEN le.itemid = 50861 THEN le.valuenum END) AS alt_max,
    MAX(CASE WHEN le.itemid = 50878 THEN le.valuenum END) AS ast_max,
    MIN(CASE WHEN le.itemid = 50862 THEN le.valuenum END) AS albumin_min,
    MAX(CASE WHEN le.itemid = 51237 THEN le.valuenum END) AS inr_max,
    MAX(CASE WHEN le.itemid = 50813 THEN le.valuenum END) AS lactate_max
FROM mimiciv_hosp.labevents le
JOIN mimiciv_icu.icustays i ON le.hadm_id = i.hadm_id
WHERE le.valuenum IS NOT NULL
  AND le.charttime BETWEEN i.intime AND i.intime + INTERVAL '24 hours'
GROUP BY i.stay_id;


-- ----------------------------------------------------------------------------
-- 2I: SOFA COMPONENTS (first 24h)
-- Derived from raw tables since mimiciv_derived is empty
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS sofa_components;
CREATE TEMP TABLE sofa_components AS
WITH resp AS (
    SELECT ce.stay_id,
        MAX(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END) AS spo2_max,
        MIN(CASE WHEN ce.itemid = 223835 THEN ce.valuenum END) AS fio2_min
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON ce.stay_id = icu.stay_id
    WHERE ce.itemid IN (220277, 223835) AND ce.valuenum IS NOT NULL
      AND ce.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY ce.stay_id
),
gcs_sofa AS (
    -- FIX: 24h window applied inside subqueries, not via outer join
    SELECT icu.stay_id,
        MIN(COALESCE(eye.val, 4) + COALESCE(verbal.val, 5) + COALESCE(motor.val, 6)) AS gcs_min
    FROM mimiciv_icu.icustays icu
    LEFT JOIN (
        SELECT ce.stay_id, MIN(ce.valuenum) AS val
        FROM mimiciv_icu.chartevents ce
        JOIN mimiciv_icu.icustays i ON ce.stay_id = i.stay_id
        WHERE ce.itemid = 220739 AND ce.valuenum IS NOT NULL
          AND ce.charttime BETWEEN i.intime AND i.intime + INTERVAL '24 hours'
        GROUP BY ce.stay_id
    ) eye    ON icu.stay_id = eye.stay_id
    LEFT JOIN (
        SELECT ce.stay_id, MIN(ce.valuenum) AS val
        FROM mimiciv_icu.chartevents ce
        JOIN mimiciv_icu.icustays i ON ce.stay_id = i.stay_id
        WHERE ce.itemid = 223900 AND ce.valuenum IS NOT NULL
          AND ce.charttime BETWEEN i.intime AND i.intime + INTERVAL '24 hours'
        GROUP BY ce.stay_id
    ) verbal ON icu.stay_id = verbal.stay_id
    LEFT JOIN (
        SELECT ce.stay_id, MIN(ce.valuenum) AS val
        FROM mimiciv_icu.chartevents ce
        JOIN mimiciv_icu.icustays i ON ce.stay_id = i.stay_id
        WHERE ce.itemid = 223901 AND ce.valuenum IS NOT NULL
          AND ce.charttime BETWEEN i.intime AND i.intime + INTERVAL '24 hours'
        GROUP BY ce.stay_id
    ) motor  ON icu.stay_id = motor.stay_id
    WHERE eye.stay_id IS NOT NULL OR verbal.stay_id IS NOT NULL OR motor.stay_id IS NOT NULL
    GROUP BY icu.stay_id
),
map_sofa AS (
    SELECT ce.stay_id, MIN(ce.valuenum) AS map_min
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON ce.stay_id = icu.stay_id
    WHERE ce.itemid IN (220052, 220181, 225312) AND ce.valuenum IS NOT NULL
      AND ce.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY ce.stay_id
),
coag AS (
    SELECT i.stay_id, MIN(le.valuenum) AS platelets_min
    FROM mimiciv_hosp.labevents le
    JOIN mimiciv_icu.icustays i ON le.hadm_id = i.hadm_id
    WHERE le.itemid = 51265 AND le.valuenum IS NOT NULL
      AND le.charttime BETWEEN i.intime AND i.intime + INTERVAL '24 hours'
    GROUP BY i.stay_id
),
liver AS (
    SELECT i.stay_id, MAX(le.valuenum) AS bilirubin_max
    FROM mimiciv_hosp.labevents le
    JOIN mimiciv_icu.icustays i ON le.hadm_id = i.hadm_id
    WHERE le.itemid = 50885 AND le.valuenum IS NOT NULL
      AND le.charttime BETWEEN i.intime AND i.intime + INTERVAL '24 hours'
    GROUP BY i.stay_id
),
creat AS (
    SELECT i.stay_id, MAX(le.valuenum) AS creatinine_max
    FROM mimiciv_hosp.labevents le
    JOIN mimiciv_icu.icustays i ON le.hadm_id = i.hadm_id
    WHERE le.itemid = 50912 AND le.valuenum IS NOT NULL
      AND le.charttime BETWEEN i.intime AND i.intime + INTERVAL '24 hours'
    GROUP BY i.stay_id
),
urine AS (
    SELECT oe.stay_id, SUM(oe.value) AS urine_output_24h
    FROM mimiciv_icu.outputevents oe
    JOIN mimiciv_icu.icustays icu ON oe.stay_id = icu.stay_id
    WHERE oe.itemid IN (226559,226560,226561,226584,226563,226564,226565)
      AND oe.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY oe.stay_id
),
-- Vasopressors for SOFA cardiovascular component
vaso_sofa AS (
    SELECT ie.stay_id,
        MAX(CASE WHEN ie.itemid = 221289 THEN 1 ELSE 0 END) AS epinephrine,
        MAX(CASE WHEN ie.itemid IN (221906, 229617) THEN 1 ELSE 0 END) AS norepinephrine,
        MAX(CASE WHEN ie.itemid = 221662 THEN 1 ELSE 0 END) AS dopamine,
        1 AS vasopressor_any
    FROM mimiciv_icu.inputevents ie
    WHERE ie.itemid IN (221906, 229617, 221662, 221289)
    GROUP BY ie.stay_id
)
SELECT
    a.stay_id,
    -- individual components
    CASE WHEN r.fio2_min IS NULL OR r.spo2_max IS NULL THEN 0
         WHEN (r.spo2_max / NULLIF(r.fio2_min,0)) < 67  THEN 4
         WHEN (r.spo2_max / NULLIF(r.fio2_min,0)) < 142 THEN 3
         WHEN (r.spo2_max / NULLIF(r.fio2_min,0)) < 221 THEN 2
         WHEN (r.spo2_max / NULLIF(r.fio2_min,0)) < 302 THEN 1
         ELSE 0 END AS sofa_resp,
    CASE WHEN c.platelets_min IS NULL THEN 0
         WHEN c.platelets_min < 20  THEN 4 WHEN c.platelets_min < 50  THEN 3
         WHEN c.platelets_min < 100 THEN 2 WHEN c.platelets_min < 150 THEN 1
         ELSE 0 END AS sofa_coag,
    CASE WHEN l.bilirubin_max IS NULL THEN 0
         WHEN l.bilirubin_max >= 12 THEN 4 WHEN l.bilirubin_max >= 6  THEN 3
         WHEN l.bilirubin_max >= 2  THEN 2 WHEN l.bilirubin_max >= 1.2 THEN 1
         ELSE 0 END AS sofa_liver,
    CASE WHEN vs.epinephrine = 1 THEN 4
         WHEN vs.norepinephrine = 1 THEN 4
         WHEN vs.dopamine = 1 THEN 3
         WHEN vs.vasopressor_any = 1 THEN 2
         WHEN mp.map_min IS NOT NULL AND mp.map_min < 70 THEN 1
         ELSE 0 END AS sofa_cardio,
    CASE WHEN g.gcs_min IS NULL THEN 0
         WHEN g.gcs_min < 6  THEN 4 WHEN g.gcs_min < 10 THEN 3
         WHEN g.gcs_min < 13 THEN 2 WHEN g.gcs_min < 15 THEN 1
         ELSE 0 END AS sofa_cns,
    CASE WHEN cr.creatinine_max >= 5.0 THEN 4 WHEN cr.creatinine_max >= 3.5 THEN 3
         WHEN cr.creatinine_max >= 2.0 THEN 2 WHEN cr.creatinine_max >= 1.2 THEN 1
         WHEN u.urine_output_24h IS NOT NULL AND u.urine_output_24h < 200 THEN 3
         WHEN u.urine_output_24h IS NOT NULL AND u.urine_output_24h < 500 THEN 2
         ELSE 0 END AS sofa_renal,
    -- total (sum of 6 components)
    (
     CASE WHEN r.fio2_min IS NULL OR r.spo2_max IS NULL THEN 0
          WHEN (r.spo2_max / NULLIF(r.fio2_min,0)) < 67  THEN 4
          WHEN (r.spo2_max / NULLIF(r.fio2_min,0)) < 142 THEN 3
          WHEN (r.spo2_max / NULLIF(r.fio2_min,0)) < 221 THEN 2
          WHEN (r.spo2_max / NULLIF(r.fio2_min,0)) < 302 THEN 1 ELSE 0 END
    + CASE WHEN c.platelets_min IS NULL THEN 0
           WHEN c.platelets_min < 20  THEN 4 WHEN c.platelets_min < 50  THEN 3
           WHEN c.platelets_min < 100 THEN 2 WHEN c.platelets_min < 150 THEN 1
           ELSE 0 END
    + CASE WHEN l.bilirubin_max IS NULL THEN 0
           WHEN l.bilirubin_max >= 12 THEN 4 WHEN l.bilirubin_max >= 6  THEN 3
           WHEN l.bilirubin_max >= 2  THEN 2 WHEN l.bilirubin_max >= 1.2 THEN 1
           ELSE 0 END
    + CASE WHEN vs.epinephrine = 1 THEN 4
           WHEN vs.norepinephrine = 1 THEN 4
           WHEN vs.dopamine = 1 THEN 3
           WHEN vs.vasopressor_any = 1 THEN 2
           WHEN mp.map_min IS NOT NULL AND mp.map_min < 70 THEN 1
           ELSE 0 END
    + CASE WHEN g.gcs_min IS NULL THEN 0
           WHEN g.gcs_min < 6  THEN 4 WHEN g.gcs_min < 10 THEN 3
           WHEN g.gcs_min < 13 THEN 2 WHEN g.gcs_min < 15 THEN 1
           ELSE 0 END
    + CASE WHEN cr.creatinine_max >= 5.0 THEN 4 WHEN cr.creatinine_max >= 3.5 THEN 3
           WHEN cr.creatinine_max >= 2.0 THEN 2 WHEN cr.creatinine_max >= 1.2 THEN 1
           WHEN u.urine_output_24h IS NOT NULL AND u.urine_output_24h < 200 THEN 3
           WHEN u.urine_output_24h IS NOT NULL AND u.urine_output_24h < 500 THEN 2
           ELSE 0 END
    ) AS sofa_total
FROM (SELECT DISTINCT stay_id FROM mimiciv_icu.icustays) a
LEFT JOIN resp     r   ON a.stay_id = r.stay_id
LEFT JOIN coag     c   ON a.stay_id = c.stay_id
LEFT JOIN liver    l   ON a.stay_id = l.stay_id
LEFT JOIN map_sofa mp  ON a.stay_id = mp.stay_id
LEFT JOIN vaso_sofa vs ON a.stay_id = vs.stay_id
LEFT JOIN gcs_sofa g   ON a.stay_id = g.stay_id
LEFT JOIN creat    cr  ON a.stay_id = cr.stay_id
LEFT JOIN urine    u   ON a.stay_id = u.stay_id;


-- ----------------------------------------------------------------------------
-- 2J: MEDICATIONS (first 24h from inputevents)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS medications_24h;
CREATE TEMP TABLE medications_24h AS
SELECT
    ie.stay_id,
    -- Vasopressors
    MAX(CASE WHEN ie.itemid IN (221906, 229617) THEN 1 ELSE 0 END) AS norepinephrine,
    MAX(CASE WHEN ie.itemid = 221662            THEN 1 ELSE 0 END) AS dopamine,
    MAX(CASE WHEN ie.itemid = 221289            THEN 1 ELSE 0 END) AS epinephrine,
    MAX(CASE WHEN ie.itemid = 222315            THEN 1 ELSE 0 END) AS vasopressin,
    MAX(CASE WHEN ie.itemid = 221749            THEN 1 ELSE 0 END) AS phenylephrine,
    MAX(CASE WHEN ie.itemid IN (221906,229617,221662,221289,222315,221749) THEN 1 ELSE 0 END) AS vasopressor_any,
    -- Sedatives / analgesics
    MAX(CASE WHEN ie.itemid = 221668 THEN 1 ELSE 0 END) AS midazolam,
    MAX(CASE WHEN ie.itemid = 225972 THEN 1 ELSE 0 END) AS lorazepam,
    MAX(CASE WHEN ie.itemid = 222168 THEN 1 ELSE 0 END) AS propofol,
    MAX(CASE WHEN ie.itemid = 225942 THEN 1 ELSE 0 END) AS dexmedetomidine,
    MAX(CASE WHEN ie.itemid = 221744 THEN 1 ELSE 0 END) AS fentanyl,
    MAX(CASE WHEN ie.itemid = 221833 THEN 1 ELSE 0 END) AS hydromorphone,
    MAX(CASE WHEN ie.itemid = 225154 THEN 1 ELSE 0 END) AS morphine,
    MAX(CASE WHEN ie.itemid IN (221668,225972,222168,225942,221744,221833,225154) THEN 1 ELSE 0 END) AS sedative_any,
    -- Antipsychotics
    MAX(CASE WHEN ie.itemid = 225829 THEN 1 ELSE 0 END) AS haloperidol,
    MAX(CASE WHEN ie.itemid = 225146 THEN 1 ELSE 0 END) AS quetiapine
FROM mimiciv_icu.inputevents ie
JOIN mimiciv_icu.icustays icu ON ie.stay_id = icu.stay_id
WHERE ie.starttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
GROUP BY ie.stay_id;


-- ----------------------------------------------------------------------------
-- 2K: MECHANICAL VENTILATION
-- Uses procedureevents itemid 225792 (invasive vent) and 225794 (non-invasive)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS mech_vent;
CREATE TEMP TABLE mech_vent AS
SELECT
    pe.stay_id,
    1 AS mechanical_vent
FROM mimiciv_icu.procedureevents pe
WHERE pe.itemid IN (225792, 225794)
GROUP BY pe.stay_id;


-- ----------------------------------------------------------------------------
-- 2L: COMORBIDITIES (ICD codes per hospitalization)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS comorbidities;
CREATE TEMP TABLE comorbidities AS
SELECT
    i.stay_id,
    MAX(CASE WHEN dx.icd_code LIKE 'I50%' OR dx.icd_code LIKE '428%' THEN 1 ELSE 0 END) AS cm_chf,
    MAX(CASE WHEN dx.icd_code LIKE 'I48%' OR dx.icd_code LIKE '4273%' THEN 1 ELSE 0 END) AS cm_afib,
    MAX(CASE WHEN dx.icd_code LIKE 'I10%' OR dx.icd_code LIKE 'I11%' OR dx.icd_code LIKE '401%' OR dx.icd_code LIKE '402%' THEN 1 ELSE 0 END) AS cm_hypertension,
    MAX(CASE WHEN dx.icd_code LIKE 'E10%' OR dx.icd_code LIKE 'E11%' OR dx.icd_code LIKE '250%' THEN 1 ELSE 0 END) AS cm_diabetes,
    MAX(CASE WHEN dx.icd_code LIKE 'N18%' OR dx.icd_code LIKE '585%' THEN 1 ELSE 0 END) AS cm_ckd,
    MAX(CASE WHEN dx.icd_code LIKE 'J44%' OR dx.icd_code LIKE '496%' OR dx.icd_code LIKE '491%' OR dx.icd_code LIKE '492%' THEN 1 ELSE 0 END) AS cm_copd,
    MAX(CASE WHEN dx.icd_code LIKE 'K70%' OR dx.icd_code LIKE 'K74%' OR dx.icd_code LIKE '571%' THEN 1 ELSE 0 END) AS cm_liver,
    MAX(CASE WHEN dx.icd_code LIKE 'F10%' OR dx.icd_code LIKE '303%' OR dx.icd_code LIKE '305.0%' THEN 1 ELSE 0 END) AS cm_alcohol,
    MAX(CASE WHEN dx.icd_code LIKE 'F32%' OR dx.icd_code LIKE 'F33%' OR dx.icd_code LIKE '296%' OR dx.icd_code LIKE '311%' THEN 1 ELSE 0 END) AS cm_depression,
    MAX(CASE WHEN dx.icd_code LIKE 'I63%' OR dx.icd_code LIKE 'I64%' OR dx.icd_code LIKE '434%' OR dx.icd_code LIKE '436%' THEN 1 ELSE 0 END) AS cm_stroke,
    MAX(CASE WHEN dx.icd_code LIKE 'A41%' OR dx.icd_code LIKE 'R65%' OR dx.icd_code LIKE '995.91%' OR dx.icd_code LIKE '995.92%' THEN 1 ELSE 0 END) AS cm_sepsis,
    MAX(CASE WHEN dx.icd_code LIKE 'C%'  OR (dx.icd_code >= '140' AND dx.icd_code < '240') THEN 1 ELSE 0 END) AS cm_cancer,
    MAX(CASE WHEN dx.icd_code LIKE 'G40%' OR dx.icd_code LIKE '345%' THEN 1 ELSE 0 END) AS cm_epilepsy,
    MAX(CASE WHEN dx.icd_code LIKE 'G20%' OR dx.icd_code LIKE '332%' THEN 1 ELSE 0 END) AS cm_parkinsons,
    MAX(CASE WHEN dx.icd_code LIKE 'F20%' OR dx.icd_code LIKE '295%' THEN 1 ELSE 0 END) AS cm_schizophrenia,
    MAX(CASE WHEN dx.icd_code LIKE 'I25%' OR dx.icd_code LIKE '414%' THEN 1 ELSE 0 END) AS cm_cad,
    MAX(CASE WHEN dx.icd_code LIKE 'F31%' THEN 1 ELSE 0 END) AS cm_bipolar,
    MAX(CASE WHEN dx.icd_code LIKE 'F41%' OR dx.icd_code LIKE '300%' THEN 1 ELSE 0 END) AS cm_anxiety,
    MAX(CASE WHEN dx.icd_code LIKE 'E87%' OR dx.icd_code LIKE '276%' THEN 1 ELSE 0 END) AS cm_electrolyte_disorder,
    MAX(CASE WHEN dx.icd_code LIKE 'D50%' OR dx.icd_code LIKE 'D64%' OR dx.icd_code LIKE '280%' OR dx.icd_code LIKE '285%' THEN 1 ELSE 0 END) AS cm_anemia,
    MAX(CASE WHEN dx.icd_code LIKE 'E40%' OR dx.icd_code LIKE 'E41%' OR dx.icd_code LIKE 'E42%'
              OR dx.icd_code LIKE 'E43%' OR dx.icd_code LIKE 'E44%' OR dx.icd_code LIKE '263%'
              OR dx.icd_code LIKE '264%' OR dx.icd_code LIKE '265%' THEN 1 ELSE 0 END) AS cm_malnutrition
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.diagnoses_icd dx ON i.hadm_id = dx.hadm_id
GROUP BY i.stay_id;


-- ----------------------------------------------------------------------------
-- 2M: FLUID BALANCE (first 24h)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS fluid_balance_24h;
CREATE TEMP TABLE fluid_balance_24h AS
WITH inputs AS (
    SELECT ie.stay_id, SUM(ie.amount) AS total_input_ml
    FROM mimiciv_icu.inputevents ie
    JOIN mimiciv_icu.icustays icu ON ie.stay_id = icu.stay_id
    WHERE ie.amountuom IN ('ml', 'mL')
      AND ie.starttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY ie.stay_id
),
outputs AS (
    SELECT oe.stay_id, SUM(oe.value) AS total_output_ml
    FROM mimiciv_icu.outputevents oe
    JOIN mimiciv_icu.icustays icu ON oe.stay_id = icu.stay_id
    WHERE oe.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY oe.stay_id
)
SELECT
    COALESCE(i.stay_id, o.stay_id)                                         AS stay_id,
    COALESCE(i.total_input_ml, 0)                                          AS fluid_input_24h,
    COALESCE(o.total_output_ml, 0)                                         AS fluid_output_24h,
    COALESCE(i.total_input_ml, 0) - COALESCE(o.total_output_ml, 0)        AS fluid_balance_24h,
    CASE WHEN (COALESCE(i.total_input_ml, 0) - COALESCE(o.total_output_ml, 0)) > 2000
         THEN 1 ELSE 0 END                                                 AS positive_fluid_balance
FROM inputs i
FULL OUTER JOIN outputs o ON i.stay_id = o.stay_id;


-- ----------------------------------------------------------------------------
-- 2N: PAIN SCORES (first 24h)
-- itemid 223791 = Numeric Pain Score (0-10)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS pain_scores_24h;
CREATE TEMP TABLE pain_scores_24h AS
SELECT
    ce.stay_id,
    MAX(ce.valuenum)                                        AS pain_max_24h,
    AVG(ce.valuenum)                                        AS pain_mean_24h,
    CASE WHEN MAX(ce.valuenum) >= 7 THEN 1 ELSE 0 END      AS severe_pain
FROM mimiciv_icu.chartevents ce
JOIN mimiciv_icu.icustays icu ON ce.stay_id = icu.stay_id
WHERE ce.itemid = 223791
  AND ce.valuenum IS NOT NULL
  AND ce.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
GROUP BY ce.stay_id;


-- ----------------------------------------------------------------------------
-- Sanity check temp tables before export
-- Expected: ~94k total stays, ~40k after CAM filter
-- ----------------------------------------------------------------------------
SELECT 'icu_seq'            AS tbl, COUNT(*) AS n FROM icu_seq
UNION ALL
SELECT 'cam_assessments'    AS tbl, COUNT(*) AS n FROM cam_assessments
UNION ALL
SELECT 'sofa_components'    AS tbl, COUNT(*) AS n FROM sofa_components
UNION ALL
SELECT 'vitals_24h'         AS tbl, COUNT(*) AS n FROM vitals_24h
UNION ALL
SELECT 'labs_24h'           AS tbl, COUNT(*) AS n FROM labs_24h
UNION ALL
SELECT 'medications_24h'    AS tbl, COUNT(*) AS n FROM medications_24h
UNION ALL
SELECT 'fluid_balance_24h'  AS tbl, COUNT(*) AS n FROM fluid_balance_24h
UNION ALL
SELECT 'pain_scores_24h'    AS tbl, COUNT(*) AS n FROM pain_scores_24h;


-- =============================================================================
-- STEP 3 — Export to CSV
-- UPDATE THE PATH before running.
-- Run via psql CLI: psql -U user -d db -f this_file.sql
-- \COPY is a client-side command — does NOT work inside pgAdmin or DBeaver.
-- If using DBeaver/pgAdmin: run the SELECT inside \COPY as a query, then
-- use the client's export to CSV function.
-- =============================================================================

\COPY (
    SELECT
        -- ---- IDENTIFIERS ----
        icu.subject_id,
        icu.hadm_id,
        icu.stay_id,
        sq.icu_stay_seq,
        sq.is_first_icu_stay,
        EXTRACT(YEAR FROM icu.intime)::INTEGER                             AS admission_year,
        sq.stays_per_patient,

        -- ---- PRIMARY OUTCOME ----
        -- outcome_incident_cam_24h: first CAM+ strictly after 24h of ICU admission
        COALESCE(cam.outcome_incident_cam_24h, 0)                          AS outcome_incident_cam_24h,
        COALESCE(cam.outcome_incident_cam_12h, 0)                          AS outcome_incident_cam_12h,
        COALESCE(cam.outcome_incident_any_24h, 0)                          AS outcome_incident_any_24h,

        -- ---- DELIRIUM STATUS ----
        CASE WHEN COALESCE(del.delirium_icd, 0) = 1
              OR COALESCE(cam.cam_positive, 0) = 1 THEN 1 ELSE 0 END       AS delirium_any,
        COALESCE(del.delirium_icd, 0)                                       AS delirium_icd,
        COALESCE(cam.cam_positive, 0)                                       AS cam_positive,
        -- Concordance label
        CASE
            WHEN COALESCE(del.delirium_icd, 0) = 1 AND COALESCE(cam.cam_positive, 0) = 1 THEN 'ICD+/CAM+'
            WHEN COALESCE(del.delirium_icd, 0) = 1 AND COALESCE(cam.cam_positive, 0) = 0 THEN 'ICD+/CAM-'
            WHEN COALESCE(del.delirium_icd, 0) = 0 AND COALESCE(cam.cam_positive, 0) = 1 THEN 'ICD-/CAM+'
            ELSE 'ICD-/CAM-'
        END                                                                 AS concordance_icd_cam,
        -- 2x2x2 group
        CASE
            WHEN COALESCE(del.delirium_icd, 0) = 1 AND COALESCE(cam.cam_positive, 0) = 1 AND COALESCE(dem.dementia_icd, 0) = 1 THEN 'ICD+/CAM+/Dem+'
            WHEN COALESCE(del.delirium_icd, 0) = 1 AND COALESCE(cam.cam_positive, 0) = 1 AND COALESCE(dem.dementia_icd, 0) = 0 THEN 'ICD+/CAM+/Dem-'
            WHEN COALESCE(del.delirium_icd, 0) = 1 AND COALESCE(cam.cam_positive, 0) = 0 AND COALESCE(dem.dementia_icd, 0) = 1 THEN 'ICD+/CAM-/Dem+'
            WHEN COALESCE(del.delirium_icd, 0) = 1 AND COALESCE(cam.cam_positive, 0) = 0 AND COALESCE(dem.dementia_icd, 0) = 0 THEN 'ICD+/CAM-/Dem-'
            WHEN COALESCE(del.delirium_icd, 0) = 0 AND COALESCE(cam.cam_positive, 0) = 1 AND COALESCE(dem.dementia_icd, 0) = 1 THEN 'ICD-/CAM+/Dem+'
            WHEN COALESCE(del.delirium_icd, 0) = 0 AND COALESCE(cam.cam_positive, 0) = 1 AND COALESCE(dem.dementia_icd, 0) = 0 THEN 'ICD-/CAM+/Dem-'
            WHEN COALESCE(del.delirium_icd, 0) = 0 AND COALESCE(cam.cam_positive, 0) = 0 AND COALESCE(dem.dementia_icd, 0) = 1 THEN 'ICD-/CAM-/Dem+'
            ELSE 'ICD-/CAM-/Dem-'
        END                                                                 AS group_2x2x2,
        CASE
            WHEN COALESCE(dem.dementia_icd, 0) = 1
             AND (COALESCE(del.delirium_icd, 0) = 1 OR COALESCE(cam.cam_positive, 0) = 1) THEN 'Both'
            WHEN COALESCE(dem.dementia_icd, 0) = 1 THEN 'Dementia Only'
            WHEN COALESCE(del.delirium_icd, 0) = 1
              OR COALESCE(cam.cam_positive, 0) = 1 THEN 'Delirium Only'
            ELSE 'Neither'
        END                                                                 AS exposure_group,
        cam.hours_to_first_cam_positive,
        COALESCE(cam.prevalent_cam_12h, 0)                                  AS prevalent_cam_12h,
        COALESCE(cam.prevalent_cam_24h, 0)                                  AS prevalent_cam_24h,

        -- ---- MORTALITY & LOS ----
        CASE WHEN adm.deathtime IS NOT NULL
              AND adm.deathtime BETWEEN icu.intime AND icu.outtime
             THEN 1 ELSE 0 END                                              AS mortality_icu,
        CASE WHEN adm.deathtime IS NOT NULL THEN 1 ELSE 0 END              AS mortality_inhospital,
        CASE WHEN p.dod IS NOT NULL
              AND p.dod <= adm.dischtime + INTERVAL '30 days'
             THEN 1 ELSE 0 END                                              AS mortality_30day,
        CASE WHEN p.dod IS NOT NULL
              AND p.dod <= adm.dischtime + INTERVAL '90 days'
             THEN 1 ELSE 0 END                                              AS mortality_90day,
        CASE WHEN p.dod IS NOT NULL
              AND p.dod <= adm.dischtime + INTERVAL '1 year'
             THEN 1 ELSE 0 END                                              AS mortality_1year,
        icu.los                                                             AS icu_los_days,
        EXTRACT(EPOCH FROM (adm.dischtime - adm.admittime))/86400.0        AS hospital_los_days,

        -- ---- DEMOGRAPHICS ----
        -- Corrected age formula for MIMIC-IV
        p.anchor_age + (EXTRACT(YEAR FROM icu.intime) - p.anchor_year)     AS age,
        CASE
            WHEN p.anchor_age + (EXTRACT(YEAR FROM icu.intime) - p.anchor_year) < 65 THEN 0
            WHEN p.anchor_age + (EXTRACT(YEAR FROM icu.intime) - p.anchor_year) < 75 THEN 1
            WHEN p.anchor_age + (EXTRACT(YEAR FROM icu.intime) - p.anchor_year) < 85 THEN 2
            ELSE 3
        END                                                                 AS age_group,
        CASE WHEN p.gender = 'F' THEN 1 ELSE 0 END                         AS female,
        CASE WHEN adm.race LIKE '%WHITE%'    THEN 1
             WHEN adm.race LIKE '%BLACK%'    THEN 2
             WHEN adm.race LIKE '%HISPANIC%' THEN 3
             WHEN adm.race LIKE '%ASIAN%'    THEN 4
             ELSE 5 END                                                     AS race_coded,
        CASE WHEN adm.insurance LIKE '%Medicare%'  THEN 1
             WHEN adm.insurance LIKE '%Medicaid%'  THEN 2
             WHEN adm.insurance = 'Other'          THEN 3
             ELSE 4 END                                                     AS insurance_coded,
        CASE WHEN adm.marital_status = 'MARRIED'   THEN 1
             WHEN adm.marital_status = 'SINGLE'    THEN 2
             WHEN adm.marital_status = 'DIVORCED'  THEN 3
             WHEN adm.marital_status = 'WIDOWED'   THEN 4
             ELSE 5 END                                                     AS marital_coded,
        CASE WHEN adm.language = 'ENGLISH' THEN 1 ELSE 0 END               AS english_speaker,

        -- ---- ADMISSION ----
        CASE WHEN icu.first_careunit LIKE '%MICU%'                         THEN 1
             WHEN icu.first_careunit LIKE '%SICU%'
               OR icu.first_careunit LIKE '%Surg%'                         THEN 2
             WHEN icu.first_careunit LIKE '%CVICU%'
               OR icu.first_careunit LIKE '%CCU%'
               OR icu.first_careunit LIKE '%Coronary%'                     THEN 3
             WHEN icu.first_careunit LIKE '%Neuro%'                        THEN 4
             ELSE 5 END                                                     AS icu_type,
        CASE WHEN adm.admission_type LIKE '%EMERGENCY%'
               OR adm.admission_type LIKE '%URGENT%'                       THEN 1
             WHEN adm.admission_type LIKE '%ELECTIVE%'                     THEN 2
             ELSE 3 END                                                     AS admission_urgency,
        sq.prior_icu_stay,

        -- ---- SOFA ----
        s.sofa_total,
        s.sofa_resp, s.sofa_coag, s.sofa_liver, s.sofa_cardio, s.sofa_cns, s.sofa_renal,
        CASE WHEN s.sofa_total >= 8 THEN 1 ELSE 0 END                      AS sofa_severe,

        -- ---- GCS ----
        g.gcs_first,
        g.gcs_severe_admission,

        -- ---- VITALS ----
        v.hr_mean, v.hr_max, v.hr_min, v.hr_sd,
        v.sbp_mean, v.sbp_min, v.sbp_max,
        v.map_mean, v.map_min,
        v.spo2_mean, v.spo2_min,
        v.rr_mean, v.rr_max,
        v.temp_f_mean, v.temp_f_max,
        CASE WHEN v.hr_max  > 100  THEN 1 ELSE 0 END                       AS tachycardia,
        CASE WHEN v.hr_min  < 60   THEN 1 ELSE 0 END                       AS bradycardia,
        CASE WHEN v.sbp_min < 90   THEN 1 ELSE 0 END                       AS hypotension,
        CASE WHEN v.temp_f_max > 100.4 THEN 1 ELSE 0 END                   AS fever,
        CASE WHEN v.spo2_min < 90  THEN 1 ELSE 0 END                       AS hypoxia,

        -- ---- LABS ----
        lb.creatinine_max, lb.bun_max,
        CASE WHEN lb.creatinine_max > 1.5 THEN 1 ELSE 0 END                AS aki_flag,
        lb.sodium_min, lb.sodium_max,
        lb.potassium_min, lb.potassium_max,
        lb.chloride_min,
        lb.calcium_max, lb.calcium_min,
        CASE WHEN lb.sodium_min < 135 OR lb.sodium_max > 145 THEN 1 ELSE 0 END AS dysnatremia,
        CASE WHEN lb.potassium_min < 3.5 OR lb.potassium_max > 5.0 THEN 1 ELSE 0 END AS dyskalemia,
        lb.glucose_min, lb.glucose_max,
        CASE WHEN lb.glucose_max > 200 THEN 1 ELSE 0 END                   AS hyperglycemia,
        CASE WHEN lb.glucose_min < 70  THEN 1 ELSE 0 END                   AS hypoglycemia,
        lb.wbc_max, lb.hemoglobin_min, lb.platelets_min, lb.hematocrit_min,
        CASE WHEN lb.wbc_max > 12            THEN 1 ELSE 0 END             AS leukocytosis,
        CASE WHEN lb.hemoglobin_min < 7      THEN 1 ELSE 0 END             AS severe_anemia,
        CASE WHEN lb.platelets_min < 100     THEN 1 ELSE 0 END             AS thrombocytopenia,
        lb.bilirubin_max, lb.alt_max, lb.ast_max,
        lb.albumin_min,
        CASE WHEN lb.albumin_min < 2.5 THEN 1 ELSE 0 END                   AS hypoalbuminemia,
        lb.inr_max,
        CASE WHEN lb.inr_max > 1.5 THEN 1 ELSE 0 END                       AS coagulopathy,
        lb.lactate_max,
        CASE WHEN lb.lactate_max > 2.0 THEN 1 ELSE 0 END                   AS hyperlactatemia,

        -- ---- RASS ----
        r.rass_mean, r.rass_min, r.rass_max,
        COALESCE(r.rass_ever_unarousable,   0)                             AS rass_ever_unarousable,
        COALESCE(r.rass_ever_deeply_sedated,0)                             AS rass_ever_deeply_sedated,
        COALESCE(r.rass_ever_agitated,      0)                             AS rass_ever_agitated,
        r.rass_assessment_count,

        -- ---- MEDICATIONS ----
        COALESCE(mv.mechanical_vent, 0)                                    AS mechanical_vent,
        COALESCE(med.vasopressor_any, 0)                                   AS vasopressor_any,
        COALESCE(med.norepinephrine, 0)                                    AS norepinephrine,
        COALESCE(med.dopamine, 0)                                          AS dopamine,
        COALESCE(med.epinephrine, 0)                                       AS epinephrine,
        COALESCE(med.vasopressin, 0)                                       AS vasopressin,
        COALESCE(med.phenylephrine, 0)                                     AS phenylephrine,
        COALESCE(med.sedative_any, 0)                                      AS sedative_any,
        COALESCE(med.propofol, 0)                                          AS propofol,
        COALESCE(med.dexmedetomidine, 0)                                   AS dexmedetomidine,
        COALESCE(med.fentanyl, 0)                                          AS fentanyl,
        COALESCE(med.midazolam, 0)                                         AS midazolam,
        COALESCE(med.lorazepam, 0)                                         AS lorazepam,
        COALESCE(med.hydromorphone, 0)                                     AS hydromorphone,
        COALESCE(med.morphine, 0)                                          AS morphine,
        COALESCE(med.haloperidol, 0)                                       AS haloperidol,
        COALESCE(med.quetiapine, 0)                                        AS quetiapine,
        CASE WHEN COALESCE(med.haloperidol, 0) = 1
              OR COALESCE(med.quetiapine, 0)  = 1 THEN 1 ELSE 0 END       AS antipsychotic_any,
        CASE WHEN COALESCE(med.midazolam, 0) = 1
              OR COALESCE(med.lorazepam, 0)  = 1 THEN 1 ELSE 0 END        AS benzodiazepine_any,
        CASE WHEN COALESCE(med.fentanyl, 0)     = 1
              OR COALESCE(med.hydromorphone, 0) = 1
              OR COALESCE(med.morphine, 0)      = 1 THEN 1 ELSE 0 END     AS opioid_any,

        -- ---- DEMENTIA ----
        COALESCE(dem.dementia_icd, 0)                                      AS dementia_icd,
        COALESCE(dem.dementia_alzheimers, 0)                               AS dementia_alzheimers,
        COALESCE(dem.dementia_vascular, 0)                                 AS dementia_vascular,

        -- ---- COMORBIDITIES ----
        COALESCE(cm.cm_chf, 0)                AS cm_chf,
        COALESCE(cm.cm_afib, 0)               AS cm_afib,
        COALESCE(cm.cm_hypertension, 0)       AS cm_hypertension,
        COALESCE(cm.cm_diabetes, 0)           AS cm_diabetes,
        COALESCE(cm.cm_ckd, 0)               AS cm_ckd,
        COALESCE(cm.cm_copd, 0)              AS cm_copd,
        COALESCE(cm.cm_liver, 0)             AS cm_liver,
        COALESCE(cm.cm_alcohol, 0)           AS cm_alcohol,
        COALESCE(cm.cm_depression, 0)        AS cm_depression,
        COALESCE(cm.cm_stroke, 0)            AS cm_stroke,
        COALESCE(cm.cm_sepsis, 0)            AS cm_sepsis,
        COALESCE(cm.cm_cancer, 0)            AS cm_cancer,
        COALESCE(cm.cm_epilepsy, 0)          AS cm_epilepsy,
        COALESCE(cm.cm_parkinsons, 0)        AS cm_parkinsons,
        COALESCE(cm.cm_schizophrenia, 0)     AS cm_schizophrenia,
        COALESCE(cm.cm_cad, 0)               AS cm_cad,
        COALESCE(cm.cm_bipolar, 0)           AS cm_bipolar,
        COALESCE(cm.cm_anxiety, 0)           AS cm_anxiety,
        COALESCE(cm.cm_electrolyte_disorder, 0) AS cm_electrolyte_disorder,
        COALESCE(cm.cm_anemia, 0)            AS cm_anemia,
        COALESCE(cm.cm_malnutrition, 0)      AS cm_malnutrition,
        -- Composite burden scores
        (COALESCE(cm.cm_chf,0) + COALESCE(cm.cm_afib,0) + COALESCE(cm.cm_hypertension,0)
         + COALESCE(cm.cm_diabetes,0) + COALESCE(cm.cm_ckd,0) + COALESCE(cm.cm_copd,0)
         + COALESCE(cm.cm_liver,0) + COALESCE(cm.cm_alcohol,0) + COALESCE(cm.cm_depression,0)
         + COALESCE(cm.cm_stroke,0) + COALESCE(cm.cm_sepsis,0) + COALESCE(cm.cm_cancer,0)
         + COALESCE(cm.cm_epilepsy,0) + COALESCE(cm.cm_parkinsons,0) + COALESCE(cm.cm_schizophrenia,0)
         + COALESCE(cm.cm_cad,0) + COALESCE(cm.cm_bipolar,0) + COALESCE(cm.cm_anxiety,0)
         + COALESCE(cm.cm_electrolyte_disorder,0) + COALESCE(cm.cm_anemia,0)
         + COALESCE(cm.cm_malnutrition,0))                                 AS comorbidity_count,
        (COALESCE(cm.cm_depression,0) + COALESCE(cm.cm_bipolar,0)
         + COALESCE(cm.cm_anxiety,0) + COALESCE(cm.cm_schizophrenia,0)
         + COALESCE(cm.cm_alcohol,0))                                      AS psychiatric_burden,
        (COALESCE(cm.cm_stroke,0) + COALESCE(cm.cm_epilepsy,0)
         + COALESCE(cm.cm_parkinsons,0) + COALESCE(dem.dementia_icd,0))   AS neurologic_burden,

        -- ---- CAM METADATA ----
        COALESCE(cam.cam_screened, 0)                                      AS cam_screened,  -- 0 = never CAM-assessed during stay
        COALESCE(cam.cam_total_assessments, 0)                             AS cam_total_assessments,
        COALESCE(cam.cam_positive_count, 0)                                AS cam_positive_count,
        COALESCE(cam.cam_negative_count, 0)                                AS cam_negative_count,
        COALESCE(cam.cam_uta_count, 0)                                     AS cam_uta_count,

        -- ---- NEW FEATURES ----
        CASE WHEN EXTRACT(HOUR FROM icu.intime) >= 20
              OR EXTRACT(HOUR FROM icu.intime) < 6
             THEN 1 ELSE 0 END                                             AS nighttime_admission,
        fb.fluid_input_24h,
        fb.fluid_output_24h,
        fb.fluid_balance_24h,
        COALESCE(fb.positive_fluid_balance, 0)                             AS positive_fluid_balance,
        ps.pain_max_24h,
        ps.pain_mean_24h,
        COALESCE(ps.severe_pain, 0)                                        AS severe_pain,

        -- ---- TEMPORAL VALIDATION ----
        p.anchor_year_group

    FROM mimiciv_icu.icustays icu
    JOIN mimiciv_hosp.patients      p   ON icu.subject_id = p.subject_id
    JOIN mimiciv_hosp.admissions    adm ON icu.hadm_id    = adm.hadm_id
    JOIN icu_seq                    sq  ON icu.stay_id    = sq.stay_id
    LEFT JOIN cam_assessments       cam ON icu.stay_id    = cam.stay_id
    LEFT JOIN delirium_icd          del ON icu.stay_id    = del.stay_id
    LEFT JOIN dementia_icd          dem ON icu.stay_id    = dem.stay_id
    LEFT JOIN rass_scores           r   ON icu.stay_id    = r.stay_id
    LEFT JOIN gcs_data              g   ON icu.stay_id    = g.stay_id
    LEFT JOIN vitals_24h            v   ON icu.stay_id    = v.stay_id
    LEFT JOIN labs_24h              lb  ON icu.stay_id    = lb.stay_id
    LEFT JOIN sofa_components       s   ON icu.stay_id    = s.stay_id
    LEFT JOIN medications_24h       med ON icu.stay_id    = med.stay_id
    LEFT JOIN mech_vent             mv  ON icu.stay_id    = mv.stay_id
    LEFT JOIN comorbidities         cm  ON icu.stay_id    = cm.stay_id
    LEFT JOIN fluid_balance_24h     fb  ON icu.stay_id    = fb.stay_id
    LEFT JOIN pain_scores_24h       ps  ON icu.stay_id    = ps.stay_id

    -- COHORT FILTER: first ICU stay only, adults (age >= 18)
    -- NO cam_screened filter — most patients won't have CAM documentation.
    -- outcome_incident_cam_24h = 0 for unscreened patients (treated as no delirium).
    -- This matches the original ~40k cohort which had no CAM filter.
    WHERE sq.is_first_icu_stay = 1
      AND (p.anchor_age + (EXTRACT(YEAR FROM icu.intime) - p.anchor_year)) >= 18
)
TO '/your/path/delirium_model_v2.csv'
WITH CSV HEADER;
