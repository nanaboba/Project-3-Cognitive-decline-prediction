
"""
model_default_comparison.py
---------------------------
Compare untuned model configurations using five-fold
stratified cross-validation on TRAINING DATA ONLY.

Models:
    1. Dummy Classifier
    2. Spline Logistic Regression
    3. Random Forest
    4. Gradient Boosting
    5. Neural Network (MLP)

Feature engineering is retained.
Estimator hyperparameters are not tuned in this script.

Outputs:
    outputs/model_comparison/default_model_cv_results.csv
    outputs/model_comparison/default_model_cv_folds.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from build_preprocessor import (
    build_preprocessor,
    FEATURES,
    TARGET
)

from build_spline_model import build_all_spline_model



# ============================================================
# 1. FILE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "cogdec_train.csv"

OUTPUT = ROOT / "outputs" / "model_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CROSS-VALIDATION CONFIGURATION
# ============================================================

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

SCORING = {
    "ROC_AUC": "roc_auc",
    "Average_Precision": "average_precision"
}


# ============================================================
# 3. DEFINE UNTUNED MODELS
# ============================================================

models = {

    # No-skill benchmark: predicts majority class.
    "Dummy Classifier": DummyClassifier(
        strategy="most_frequent"
    ),

    # Keep existing spline feature engineering.
    # C=1 is also LogisticRegression's default.
    "Spline Logistic Regression": build_all_spline_model(),

    # Default RF estimator hyperparameters.
    "Random Forest": Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            random_state=42,
            n_jobs=1
        ))
    ]),

    # Default GB estimator hyperparameters.
    "Gradient Boosting": Pipeline([
        ("prep", build_preprocessor()),
        ("clf", GradientBoostingClassifier(
            random_state=42
        ))
    ]),

    # Genuine default MLP hyperparameters.
    # random_state is fixed for reproducibility.
    "Neural Network": Pipeline([
        ("prep", build_preprocessor()),
        ("clf", MLPClassifier(
            random_state=42
        ))
    ])
}


# ============================================================
# 4. LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

X = df[FEATURES]
y = df[TARGET].astype(int)

print("=" * 65)
print("DEFAULT MODEL COMPARISON — TRAINING DATA ONLY")
print("=" * 65)

print(f"Training respondents: {len(df):,}")
print(f"Positive prevalence: {y.mean():.2%}")

print("\nCross-validation: 5-fold stratified")


# ============================================================
# 5. RUN FIVE-FOLD CROSS-VALIDATION
# ============================================================

summary_rows = []
fold_rows = []

for name, model in models.items():

    print(f"\nEvaluating: {name}")

    scores = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=CV,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
        return_train_score=False
    )

    roc_scores = scores["test_ROC_AUC"]
    ap_scores = scores["test_Average_Precision"]

    # Save individual fold results.
    for fold_number, (roc, ap) in enumerate(
        zip(roc_scores, ap_scores),
        start=1
    ):

        fold_rows.append({
            "Model": name,
            "Fold": fold_number,
            "ROC_AUC": roc,
            "Average_Precision": ap
        })

    # Save mean and standard deviation.
    summary_rows.append({
        "Model": name,
        "ROC_AUC": np.mean(roc_scores),
        "ROC_AUC_SD": np.std(roc_scores),
        "Average_Precision": np.mean(ap_scores),
        "AP_SD": np.std(ap_scores)
    })

    print(
        f"ROC-AUC: "
        f"{np.mean(roc_scores):.4f} "
        f"(SD {np.std(roc_scores):.4f})"
    )

    print(
        f"Average Precision: "
        f"{np.mean(ap_scores):.4f} "
        f"(SD {np.std(ap_scores):.4f})"
    )


# ============================================================
# 6. CREATE RESULTS TABLES
# ============================================================

summary_df = pd.DataFrame(summary_rows)

fold_df = pd.DataFrame(fold_rows)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

summary_path = (
    OUTPUT / "default_model_cv_results.csv"
)

fold_path = (
    OUTPUT / "default_model_cv_folds.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

fold_df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# 8. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("DEFAULT MODEL PERFORMANCE — FIVE-FOLD CV")
print("=" * 65)

print(
    summary_df.round(4).to_string(index=False)
)

print("\nSaved files:")
print(summary_path)
print(fold_path)

print("\nDefault model comparison completed.")
