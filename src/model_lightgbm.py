
"""
model_lightgbm.py

LightGBM model for BRFSS cognitive decline classification.

Uses 5-fold stratified cross-validation and GridSearchCV.
Training data only — the held-out test set is not used.
"""

import sys
import pandas as pd

from pathlib import Path

from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)

from build_preprocessor import (
    build_preprocessor,
    FEATURES,
    TARGET
)


# --------------------------------------------------
# File paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


# --------------------------------------------------
# Cross-validation
# --------------------------------------------------

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1"
}


# --------------------------------------------------
# Hyperparameter search
# --------------------------------------------------

PARAM_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5],
    "clf__learning_rate": [0.05, 0.1],
    "clf__num_leaves": [7, 15]
}


def main():

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(DATA_DIR / "cogdec_train.csv")
    )

    df = pd.read_csv(path)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"Loaded {len(df):,} training rows")
    print(f"Positive class: {y.mean()*100:.2f}%")


    # Build model pipeline
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LGBMClassifier(
            random_state=42,
            verbosity=-1,
            n_jobs=1
        ))
    ])


    # Hyperparameter tuning
    search = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    search.fit(X, y)

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        f"\nBest CV ROC-AUC: "
        f"{search.best_score_:.3f}"
    )


    # Evaluate selected configuration
    best = search.best_estimator_

    results = cross_validate(
        best,
        X,
        y,
        cv=CV,
        scoring=SCORING,
        n_jobs=-1
    )

    print("\nLightGBM — 5-fold CV results:")

    for metric in SCORING:
        mean_score = results[
            "test_" + metric
        ].mean()

        print(f"{metric:12s}: {mean_score:.3f}")


    # Save tuning results
    grid_df = pd.DataFrame(search.cv_results_)

    grid_df.to_csv(
        "lightgbm_gridsearch_results.csv",
        index=False
    )

    print(
        "\nSaved -> lightgbm_gridsearch_results.csv"
    )


if __name__ == "__main__":
    main()
