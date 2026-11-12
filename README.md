# Predicting Self-Reported Cognitive Decline with Machine Learning

**Portfolio project | BRFSS survey data | Binary classification and public-health outreach prioritisation**

## Project overview

This educational project uses US Behavioral Risk Factor Surveillance System (BRFSS) survey data to predict **self-reported cognitive decline** among adults aged 45 and older. It explores whether machine-learning rankings could help a hypothetical public-health agency prioritise invitations for brief memory screening under a fixed budget. It also examines potentially modifiable health factors that may inform outreach conversations.

**Recommended candidate:** Spline Logistic Regression, because it achieves discrimination comparable to Gradient Boosting while offering a more transparent additive model structure. This is **not** a clinical diagnostic system or a validated deployment model.

## Data and objective

| Item | Value |
|---|---|
| Outcome | `CIMEMLO1`: self-reported cognitive decline (Yes = 1, No = 0) |
| Analytic sample | 73,448 respondents |
| Training set | 58,758 (80%) |
| Test set | 14,690 (20%) |
| Positive-class prevalence | Approximately 20.65% |
| Final predictors | 14 demographic, health-history and behavioural variables |
| Split | Stratified, random state 42 |

The data originate from the United States. All dollar calculations below are **illustrative business scenarios**.

## Analytical workflow

### 1. Cleaning and preparation

Valid target responses were recoded into binary outcomes; adults under 45 and respondents without valid target responses were excluded. Continuous variables were recoded and capped according to documented rules; unknown categorical responses were retained as an explicit category where applicable. Identical response rows were retained because no respondent identifier was available to prove duplication.

Fixed data-cleaning rules were applied before the train/test split. **Learned preprocessing**—such as imputation, scaling and categorical encoding—was fitted within training folds as part of the modelling pipelines.

See [Data cleaning and preparation](data_cleaning.md) for the detailed rules and rationale.

**Files:** `notebooks/data_clean.ipynb`, `src/split_cogdec.py`, `src/build_preprocessor.py`.

### 2. Exploratory data analysis (EDA)

Two findings informed the modelling design:

- **Class imbalance:** Around 20.65% of respondents reported decline, motivating stratified splitting and metrics beyond accuracy.
- **Potentially nonlinear bivariate patterns:** Sleep and BMI showed possible U-shaped patterns; alcohol showed a possible non-monotonic pattern; age and poor mental-health days showed changing gradients.

These are **unadjusted, descriptive patterns**, not proof of nonlinear causal relationships.

**Files:** `src/plot_continuous_patterns.py`, `src/eda_cogdec_svg.py`; exploratory notebooks under `notebooks/`.

### 3. Feature engineering

The EDA motivated **cubic spline transformations** for five continuous predictors in Logistic Regression: age, sleep duration, BMI, drinks per week and poor mental-health days. The selected spline specification uses four knots. Categorical variables are one-hot encoded. Fitted transformations remain inside the training pipeline to prevent leakage across cross-validation folds.

**Files:** `src/build_spline_model.py`, `src/build_preprocessor.py`.

### 4. Initial model comparison — before hyperparameter tuning

Predetermined model configurations were compared using **five-fold stratified cross-validation on the training set**. Spline Logistic Regression already included the EDA-motivated spline features at this stage; its `C=1` is the default Logistic Regression regularisation value, although the spline features and class weighting were deliberately specified.

| Model | CV ROC-AUC | CV Average Precision (AP) |
|---|---:|---:|
| Dummy (majority class) | 0.5000 | 0.2065 |
| Spline Logistic Regression | 0.7309 | 0.4313 |
| Random Forest | 0.6931 | 0.3746 |
| Gradient Boosting | 0.7316 | 0.4388 |
| Neural Network (MLP) | 0.7014 | 0.3964 |

The default MLP reached its 200-iteration limit without convergence. The dummy model is a **no-skill benchmark**, not a Logistic Regression model.

**File:** `src/model_default_comparison.py`.

**Outputs:** `outputs/model_comparison/default_model_cv_results.csv`, `outputs/model_comparison/default_model_cv_folds.csv`.

### 5. Hyperparameter tuning

Training-only cross-validation was used to select configurations. The reported tuned CV scores are model-selection estimates and can be slightly optimistic because the same folds contributed to selection and reporting.

| Model | Selected configuration | Tuned CV ROC-AUC | Tuned CV AP |
|---|---|---:|---:|
| Spline Logistic Regression | `C=1`; cubic splines, 4 knots | 0.7309 | 0.4313 |
| Random Forest | 300 trees; depth 10; min. leaf 5; balanced weights | 0.7280 | 0.4320 |
| Gradient Boosting | 200 trees; depth 3; learning rate 0.05 | 0.7320 | 0.4390 |
| Neural Network | Layers (32, 16); alpha 0.01; initial learning rate 0.01 | 0.7260 | 0.4320 |

Random Forest showed the largest observed overall improvement in ROC-AUC (0.6931 → 0.7280). Because several parameters changed together, this does **not** isolate one parameter as the cause.

**Files:** `src/model_random_forest.py`, `src/model_gradient_boosting.py`, `src/model_neural_network.py`, `src/build_spline_model.py`, `src/safe_save_results.py`. Historical spline-comparison experiments were retained locally rather than included in the public portfolio.

### 6. Test-set evaluation

Selected models were fitted on the full training set and scored on the 14,690-row test set. Classification thresholds were chosen from **training-only out-of-fold probabilities**, targeting at least 70% training recall. Test recall is not guaranteed to meet that target.

| Model | Test ROC-AUC | Test AP | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Original Logistic Regression (reference) | 0.723 | 0.427 | 0.712 | 0.323 | 0.445 |
| **Spline Logistic Regression** | **0.738** | 0.440 | 0.705 | 0.337 | 0.456 |
| Random Forest | 0.733 | 0.438 | 0.712 | 0.333 | 0.454 |
| **Gradient Boosting** | **0.738** | **0.446** | 0.711 | 0.337 | 0.457 |
| Neural Network | 0.733 | 0.439 | 0.644 | **0.358** | **0.460** |

**AP** means scikit-learn's **Average Precision**, not trapezoidally integrated PR-AUC. ROC-AUC and AP summarise ranking performance; recall, precision and F1 are measured at each model's selected classification threshold.

Gradient Boosting has the highest AP; spline Logistic Regression has essentially the same ROC-AUC and offers easier explanation of the additive predictor relationships. The test set was revisited during exploratory development and consolidation, so these are not results from a wholly untouched, single-use external validation.

**File:** `src/final_test_evaluation_unified.py`.

**Outputs:** `outputs/final_evaluation/all_final_model_results.csv`, `outputs/final_evaluation/all_final_test_predictions.csv`, ROC and precision–recall figures, confusion matrices and subgroup outputs.

### 7. Model interpretation and error analysis

Grouped permutation importance on held-out training folds was used to measure each original feature's contribution to ROC-AUC. Poor mental-health days, depression history and age were prominent in the selected spline model. Predictive importance is **not** a causal effect or a direction of association.

Spline Logistic Regression's test-set subgroup recall showed potential coverage gaps:

| Subgroup | Recall |
|---|---:|
| No depression history | 51.4% |
| Physically active | 63.1% |
| Age 65–74 | 63.6% |
| Overall spline LR | 70.5% |

Potentially modifiable factors for discussion include poor mental-health days, sleep, physical activity and smoking. Observational associations do not show that changing these factors will prevent cognitive decline.

**Files:** `src/feature_importance_grouped.py`, `src/association_spline_extension.py`, `src/final_test_evaluation_unified.py`.

**Selected figures:** `outputs/importance/fig_importance_spline_lr.png`, `outputs/associations/association_spline_effect_curves.png`, `outputs/final_evaluation/fig_spline_lr_segment_recall.png`.

### 8. Opportunity sizing — illustrative $250,000 annual budget

Assume $5 outreach/administration plus $50 for a brief memory screen per participant: **$55 per person**. A $250,000 budget supports **4,545 people** ($249,975 spent). This assumes full attendance and excludes counselling, referral, diagnostics, model development and ongoing operating costs.

Models rank the test respondents and select their **top 4,545**. **Precision@4,545** measures positive-case yield among those invited; **Recall@4,545** measures the fraction of all positive test respondents included in that selection.

| Strategy | Positive respondents reached | Precision@4,545 | Recall@4,545 |
|---|---:|---:|---:|
| Random selection (expected) | ~938 | 20.65% | 30.94% |
| **Spline Logistic Regression** | **1,799** | **39.58%** | **59.31%** |
| Gradient Boosting | 1,782 | 39.21% | 58.75% |
| Random Forest | 1,776 | 39.08% | 58.56% |
| Neural Network | 1,767 | 38.88% | 58.26% |

Spline LR yields approximately **1.92 times** the expected number of respondents reporting decline compared with random selection at the same capacity. These are **observed test-sample outcomes or random-selection expectations**, not confirmed diagnoses or demonstrated financial savings. The difference of 17 positives versus Gradient Boosting at this capacity is small and exploratory.

**File:** `src/recalculate_outreach_250k.py`.

**Output:** `outputs/opportunity_sizing/precision_at_4545_all_models.csv`.

### 9. Recommendation

**Preferred model for further supervised evaluation: Spline Logistic Regression.** Gradient Boosting's overall AP is slightly stronger, but spline LR provides similar discrimination and a clearer framework for communicating nonlinear predictor associations. Retain Gradient Boosting as a predictive benchmark.

**Do not deploy autonomously.** A practical next step would require population-specific external validation, calibration, subgroup performance checks, monitoring for drift, privacy safeguards and clinical/human oversight. Model-ranked invitations should supplement—not replace—ordinary screening access.

## Limitations

- The outcome is **self-reported** decline, not a clinical diagnosis.
- Observational relationships and permutation importance do not establish causality.
- Some respondent subgroups have materially lower recall.
- Screening cost assumptions are illustrative; programme ROI has not been established.

## Project Presentation

[View Final Project Presentation (PDF)](presentation/Cognitive_Decline_Prediction.pdf)

## Repository guide and reproducibility

- [Project scripts and output inventory](Project_files.md)
- [Detailed data-cleaning methodology](data_cleaning.md)

The inventory lists the scripts and generated outputs by stage. Use the documented dataset preparation steps before running model comparisons, tuning and evaluation. Dependencies and source data availability should be checked before attempting a full reproduction.

## Data source and disclaimer

Data source: US Centers for Disease Control and Prevention, **Behavioral Risk Factor Surveillance System (BRFSS)**. This is an educational data-science project, not a validated screening or diagnostic product.


- [Project Scripts and Outputs](Project_files.md)
- [Data Cleaning and Preparation](data_cleaning.md)
## Preserving tuning results

The six tuning scripts use `src/safe_save_results.py` to compare rerun results with the authoritative CSVs in `outputs/tuning/`. Matching reruns do not overwrite those CSVs; changed results are saved separately under `outputs/experiments/tuning_reruns/`, excluded from Git by default.
