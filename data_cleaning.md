# Data Cleaning and Preparation

This document records the cleaning and preprocessing decisions used in the BRFSS self-reported cognitive-decline classification project.

**Scope:** Fixed recoding, filtering and capping rules were applied before the train/test split. These rules do not estimate parameters from the data, although the choices of caps and exclusions involve analytical judgement. Learned preprocessing (such as imputation and scaling) was fitted inside the modelling pipelines on training folds only.

## 1. Rows retained and outcome definition

| Step | Decision |
|---|---|
| Target (`CIMEMLO1`) | Yes (1) was recoded to 1; No (2) to 0. Don't know (7), refused (9), and blank responses were excluded. |
| Age restriction | Retained adults aged 45 and above. Excluded 118 under-45 records with valid target responses, treated as a module-administration irregularity. |
| Final analytic sample | 73,448 respondents; approximately 20.65% reported cognitive decline. |
| Identical records | Retained 48 identical rows. Without respondent identifiers, identical responses cannot be established as duplicate individuals. |

The outcome is **self-reported cognitive decline**, not a clinical diagnosis.

## 2. Continuous variables

| Variable | Cleaning rule |
|---|---|
| Age (`_AGE80`) | Used as supplied; 80 indicates age 80 or older (top-coded). No sentinel recoding was needed. |
| Sleep (`SLEPTIM1` → `sleep_hours`) | Codes 77 (don't know) and 99 (refused) became missing (`NaN`). Sleep duration was capped at 16 hours; 38 implausible values were affected. |
| Poor mental-health days (`MENTHLTH` → `mhealth`) | Code 88 (none) became 0; codes 77 and 99 became `NaN`. Valid range: 0–30 days. |
| BMI (`_BMI5` → `bmi`) | Divided by 100 to restore two implied decimal places; blanks remained `NaN`. Capped at 70, affecting four values considered data-entry artefacts. |
| Drinks per week (`_DRNKWK3` → `DRNKWK3_clean`) | Corrupted representation of zero (approximately `5.4e-79`) was recoded to 0; code 99900 became `NaN`; values were divided by 100 and capped at **70 drinks per week** (raw maximum after conversion: 420). |

Capping is an explicit analytical choice, not a value learned during model fitting. The raw data and cleaning notebook should be retained for provenance.

## 3. Categorical variables

- Don't know (7), refused (9) and missing responses were generally represented as the explicit category `999` (Unknown), where applicable to the variable's codebook.
- For blood pressure and diabetes, pregnancy-only responses (code 2) were grouped with No (code 3) for this project's operational definition. Borderline responses (code 4) were retained separately where applicable. This grouping is a modelling simplification and does not imply that gestational conditions have no longer-term health relevance.
- Smoking status used `_SMOKER3` instead of `SMOKDAY2`, because `SMOKDAY2` has structural missingness for never-smokers owing to skip logic. Smoking code 9 was recoded to `999`.
- Sex required no additional recoding.
- The high-cholesterol Unknown category combines don't-knows, refusals and skip-logic blanks; it must not be interpreted as a clinical condition.

**Important:** Sentinel codes must be interpreted according to each variable's original codebook. A value such as 7, 9 or 2 does not have the same meaning for every BRFSS variable.

## 4. Variables excluded from the final predictive feature set

| Variable/group | Reason |
|---|---|
| General health (`GENHLTH`) | Removed after an ablation experiment; the previously reported ROC-AUC reduction from removing it was approximately 0.011–0.015. The original column remains in historical data files for traceability. |
| Race | Excluded as a project fairness/design choice; exclusion alone does not establish that the model is free from bias. |
| Loneliness, emotional support, life satisfaction | Approximately 55% missing due to optional-module coverage. |
| Severity follow-up items | Only administered to respondents answering Yes to the outcome question; including them would leak target information. |

The final model uses **14 predictors: five continuous and nine categorical**. `GENHLTH` may remain in intermediate cleaned CSV files but is excluded from the final `FEATURES` list used for modelling.

## 5. Missingness after cleaning

| Variable | Approximate missing share |
|---|---:|
| High cholesterol | 6.3% |
| BMI | 5.4% |
| Sleep duration | 4.5% |
| Other variables | Under 2% each |

These figures describe the cleaned analytic dataset and should not be confused with missingness after imputation.

## 6. Preprocessing fitted within model training

These are **not** pre-split cleaning operations:

- Continuous-variable missingness indicators and median imputation.
- Standard scaling where specified by the pipeline.
- One-hot encoding of the nine categorical predictors, using `handle_unknown="ignore"` for previously unseen categories.
- Cubic spline transformations for age, sleep duration, BMI, drinks per week and poor mental-health days in spline Logistic Regression. The spline specification uses four knots.

Preprocessing is included inside the scikit-learn pipelines so that each cross-validation fold fits learned transformations using only its training portion.

## 7. Data splitting and evaluation boundaries

After fixed cleaning, the 73,448-row analytic dataset was split into 58,758 training respondents (80%) and 14,690 test respondents (20%) using stratification on the binary target. Model development, five-fold cross-validation and hyperparameter tuning used the training data. The existing test sample was used for evaluation and later exploratory consolidation; it should not be described as an untouched external validation sample.

## 8. Relevant project files

| File | Purpose |
|---|---|
| `notebooks/data_clean.ipynb` | Cleaning rules and recoding details. |
| `notebooks/data_exploration.ipynb` | Initial inspection and data exploration. |
| `src/split_cogdec.py` | Train/test split. |
| `src/build_preprocessor.py` | Shared modelling feature list and preprocessing pipeline. |
| `src/build_spline_model.py` | Final spline Logistic Regression preprocessing and model. |
| `src/model_default_comparison.py` | Untuned model cross-validation comparison. |
| `src/final_test_evaluation.py` | Consolidated final model evaluation. |

The project README summarises the full workflow; this document retains the more detailed cleaning decisions.


---

## Related Documentation

- [Back to Project Overview](README.md)
- [Project Scripts and Outputs](Project_files.md)