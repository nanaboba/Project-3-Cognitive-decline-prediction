# BRFSS Cognitive Decline Project — Script and Output Inventory

This document explains the project's Python scripts, notebooks and selected outputs in **analytical workflow order**. It complements the [portfolio README](README.md) and [data-cleaning documentation](data_cleaning.md).

**Workflow:** fixed cleaning → train/test split → EDA → EDA-motivated spline feature engineering → untuned CV comparison → tuning → test evaluation → interpretation and subgroup analysis → opportunity sizing.

> **Scope note:** Filenames and outputs below reflect the scripts and results established in the project. Some scripts write relative to the working directory rather than to a fixed output folder; check their `to_csv`/`savefig` paths before reproducing. Excluded or retired experiments are not required to run the final pipeline.

## Stage 1 — Data preparation and split

| File | Purpose | Input | Output / use |
|---|---|---|---|
| `notebooks/data_clean.ipynb` | Inspect and clean BRFSS fields, recode target and predictors, handle sentinels, apply documented filters and caps | Raw BRFSS extract | Cleaned analytic data (`data/CogDec.csv`) |
| `notebooks/data_exploration.ipynb` | Initial exploration and checks of source fields | BRFSS extract / working data | Notebook analyses |
| `notebooks/brfss.ipynb` | Additional exploratory notebook (review for overlap before public sharing) | BRFSS working data | Notebook analyses |
| `src/split_cogdec.py` | Reproducible stratified 80/20 split (`random_state=42`) | `data/CogDec.csv` (73,448 rows) | `data/cogdec_train.csv` (58,758), `data/cogdec_test.csv` (14,690) |

Cleaning rules were applied before splitting. **Fitted** transformations such as medians, scales and one-hot categories were learned only inside training folds. The cleaned CSVs may still contain source fields excluded from final model predictors (for example `GENHLTH`).

**Detailed documentation:** [Data Cleaning and Preparation](data_cleaning.md)

## Stage 2 — EDA and modelling motivation

| File | Purpose | Input | Output / use |
|---|---|---|---|
| `src/eda_cogdec.py` | Descriptive distributions, missingness, class balance and categorical comparisons | Cleaned data | EDA plots and summary text (paths depend on script) |
| `src/eda_cogdec_svg.py` | Additional visual EDA, including original survey variables | Cleaned data | EDA charts |
| `src/plot_continuous_patterns.py` | Training-only observed positive rates across age, sleep, BMI, alcohol and mental-health days | `data/cogdec_train.csv` | `outputs/continuous_patterns/` individual/combined PNG/SVG charts and bivariate summary CSV |

**Key EDA findings:** positive class ≈20.65%; possible nonlinear patterns in five continuous variables. These motivate stratified evaluation and spline feature engineering. Observed bivariate patterns are unadjusted.

## Stage 3 — Shared preprocessing and spline feature engineering

| File | Purpose | How it is used |
|---|---|---|
| `src/build_preprocessor.py` | Defines the shared `ColumnTransformer`, predictor list (`FEATURES`) and target (`TARGET`); imputation, missingness handling, scaling and one-hot encoding | Imported by the standard-model scripts; preprocessing fitted inside each training fold |
| `src/build_spline_model.py` | Defines `build_all_spline_model()` with cubic splines for five continuous predictors, categorical encoding and regularised, class-weighted Logistic Regression | Imported by default comparison, final evaluation and grouped importance |

Spline Logistic Regression is included **from the initial model comparison**, not introduced as a post-test change. The spline transformations are feature engineering motivated by EDA; the later `C` search is hyperparameter tuning.

## Stage 4 — Initial, untuned five-fold CV comparison

| File | Purpose | Input | Outputs |
|---|---|---|---|
| `src/model_default_comparison.py` | Evaluate Dummy, spline LR, RF, GB and MLP using predetermined configurations and identical five-fold stratified training CV | `data/cogdec_train.csv` | `outputs/model_comparison/default_model_cv_results.csv`; `default_model_cv_folds.csv` |

| Model | Untuned CV ROC-AUC | Untuned CV AP |
|---|---:|---:|
| Dummy (majority class) | 0.5000 | 0.2065 |
| Spline LR | 0.7309 | 0.4313 |
| RF | 0.6931 | 0.3746 |
| GB | 0.7316 | 0.4388 |
| MLP | 0.7014 | 0.3964 |

The untuned MLP used scikit-learn's default 200 iterations and produced a convergence warning. The spline LR's `C=1` was the default regularisation value, but its spline preprocessing and balanced class weights were fixed design choices.

## Stage 5 — Hyperparameter tuning on training data

| File | Purpose | Selected configuration | Main output |
|---|---|---|---|
| `src/model_gradient_boosting.py` | GridSearchCV over trees (100/200), depth (3/5), learning rate (0.05/0.1) | 200 trees, depth 3, LR 0.05 | `outputs/tuning/gb_gridsearch_results.csv` |
| `src/model_random_forest.py` | GridSearchCV over trees (200/300), depth (10/20), min. leaf (1/5) | 300 trees, depth 10, leaf 5 | `outputs/tuning/rf_gridsearch_results.csv` |
| `src/model_neural_network.py` | MLP learning-rate sweep using fixed (32,16) architecture, alpha 0.01, early stopping | Initial LR 0.01 | `outputs/tuning/neural_network_gridsearch_results.csv`, `outputs/tuning/fig_neural_network_lr_sweep.png/.svg` |
| `src/build_spline_model.py` | Defines fixed spline LR model used after `C` selection | `C=1`, four cubic spline knots | Reusable model builder; no standalone evaluation CSV |

**Spline `C` search:** See the locally retained spline experiment for the exact candidate grid; the final selected value was `C=1`. The historical comparison script and its `outputs/spline_experiment/` results may be excluded from the public repository while retained locally. Selected CV ROC-AUC: spline LR 0.7309; RF 0.7280; GB 0.7320; MLP 0.7260. The model-selection CV estimates may be optimistic.

Other historical scripts such as `model_baselines.py` document earlier baseline development; do not confuse their outputs with the consolidated untuned comparison.

## Stage 6 — Threshold exploration and unified test evaluation

| File | Purpose | Input | Outputs |
|---|---|---|---|
| `src/explore_recall_tradeoff.py` | Training-only exploration of recall/precision trade-offs | Training CSV | Threshold exploration results (check script for save paths) |
| `src/final_test_evaluation_unified.py` | Train-only out-of-fold threshold selection; fit final configurations on full training set; evaluate original LR, spline LR, RF, GB and MLP on test; create plots and subgroup error analysis | Train and test CSVs | `outputs/final_evaluation/all_final_model_results.csv`, `all_final_test_predictions.csv`, ROC/PR plots, confusion matrices, segment error CSV and recall plots |

| Model | Test ROC-AUC | Test AP | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Original LR (reference) | 0.723 | 0.427 | 0.712 | 0.323 | 0.445 |
| Spline LR | 0.738 | 0.440 | 0.705 | 0.337 | 0.456 |
| RF | 0.733 | 0.438 | 0.712 | 0.333 | 0.454 |
| GB | 0.738 | 0.446 | 0.711 | 0.337 | 0.457 |
| MLP | 0.733 | 0.439 | 0.644 | 0.358 | 0.460 |

**Terminology:** The scripts' `pr_auc` field is calculated using `average_precision_score`, the value is labelled **Average Precision (AP)** in documentation; it is not trapezoidal PR-AUC. 

## Stage 7 — Interpretation, associations and error analysis

| File | Purpose | Input | Outputs |
|---|---|---|---|
| `src/feature_importance_grouped.py` | Five-fold held-out-training-fold permutation importance for original LR, spline LR, RF and GB; permutes original input columns | Training CSV | `outputs/importance/importance_all_models_summary.csv`, `importance_spline_lr.csv`, `fig_importance_spline_lr.png/.svg`, other model-specific files |
| `src/association_spline_extension.py` | Spline-adjusted association analysis and effect curves | Cleaned/model data per script | `outputs/associations/association_spline_adjusted_odds_ratios.csv`, `association_spline_effect_curves.csv/.png` and associated plots |
| `src/final_test_evaluation_unified.py` | Spline LR subgroup recall/precision by age, sex, depression history, stroke history and physical activity | Test predictions | `outputs/final_evaluation/spline_lr_segment_error_analysis.csv`, `fig_spline_lr_segment_recall.png/.svg` |

**Spline LR subgroup recall:** no depression history 51.4%; physically active 63.1%; age 65–74 63.6%; overall 70.5%. These are **threshold-based test** results, not Recall@4,545.

Permutation importance measures contribution to model discrimination, not causal impact. Spline effect curves and association analyses require separate interpretation.

## Stage 8 — Opportunity sizing and recommendation

| File | Purpose | Input | Output |
|---|---|---|---|
| `src/recalculate_outreach_250k.py` (if retained) | Reporting-only top-4,545 ranking analysis for all five saved test prediction series; no refitting | `outputs/final_evaluation/all_final_test_predictions.csv` | `outputs/opportunity_sizing/precision_at_4545_all_models.csv` |

**Illustrative assumptions:** $250,000 annual budget, $55 per invitation/screen, 4,545-person capacity, full attendance, counselling and referrals excluded.

| Strategy | Positive respondents reached | Precision@4,545 | Recall@4,545 |
|---|---:|---:|---:|
| Random (expected) | ~938 | 20.65% | 30.94% |
| Spline LR | 1,799 | 39.58% | 59.31% |
| GB | 1,782 | 39.21% | 58.75% |
| RF | 1,776 | 39.08% | 58.56% |
| MLP | 1,767 | 38.88% | 58.26% |

The preferred model is spline LR for **interpretability with comparable predictive performance**, not because it is proven statistically superior to GB. The recommendation is further supervised validation/piloting, not autonomous deployment.

## Retired or local-only experiments

- `src/experiment_logistic_spline_comparison.py` and `src/retired/model_logistic_spline.py` — historical spline-development scripts.
- `outputs/spline_experiment/` — intermediate spline comparisons and duplicate exploratory test outputs.
- `src/ablation_genhlth.py` — historical experiment evaluating the `GENHLTH` predictor.
- `src/retired/` — older modelling and visualisation scripts.
- `src/threshold_tuning.py`, `src/threshold_sweep.py` — earlier threshold experiments, if present.

`GENHLTH` is an original BRFSS **general-health variable**, not a project name. Its appearance in source notebooks and cleaned CSVs is expected even though it is excluded from final model predictors.

## Recommended reproducibility order

1. Review and execute cleaning notebook(s), producing `data/CogDec.csv`.
2. Run `src/split_cogdec.py` to produce train/test CSVs.
3. Review EDA (`src/plot_continuous_patterns.py` and other descriptive scripts).
4. Use `src/build_preprocessor.py` and `src/build_spline_model.py` as shared pipeline components.
5. Run `src/model_default_comparison.py` for the untuned CV benchmark.
6. Run the retained tuning scripts on **training data only**.
7. Run `src/final_test_evaluation_unified.py` for consolidated evaluation (reproduces the existing exploratory test evaluation).
8. Run `src/feature_importance_grouped.py` and the association extension for interpretation.
9. Use the saved final predictions for `src/recalculate_outreach_250k.py`, if available.

**Before running:** Confirm file paths, the presence of datasets, Python dependencies and that the exact script filenames listed here are present in your repository. This inventory describes the analysis and outputs; it does not guarantee a fresh clone runs end-to-end without obtaining the source data and environment dependencies.


---

## Related Documentation

- [Back to Project Overview](README.md)
- [Data Cleaning and Preparation](data_cleaning.md)
## Tuning-result preservation

`src/safe_save_results.py` is shared by the six tuning scripts. It keeps the original CSVs under `outputs/tuning/` unchanged when rerun results match. Different reruns are saved to `outputs/experiments/tuning_reruns/` and identical reruns are not duplicated. The neural-network sweep figures use the same protection. The experiments directory is excluded from Git by default.
