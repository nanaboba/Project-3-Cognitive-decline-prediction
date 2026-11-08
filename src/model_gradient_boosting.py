"""
model_gradient_boosting.py
--------------------------
Rung 2 of the model ladder: Gradient Boosting, tuned with GridSearchCV.

Tunes n_estimators, max_depth, learning_rate via 5-fold stratified CV on the
TRAINING set only. The preprocessor sits inside the pipeline, so every fold
and every grid candidate re-learns its medians/scales/categories from its own
training portion -- no leakage. The sealed test set is NOT touched here.

Usage:
    python model_gradient_boosting.py data/cogdec_train.csv
"""

import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate

from build_preprocessor import build_preprocessor, FEATURES, TARGET

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent   # .../src
PROJECT_ROOT = SCRIPT_DIR.parent               # project root
DATA_DIR = PROJECT_ROOT / "data"               # .../data



CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

SCORING = {
    "roc_auc":   "roc_auc",
    "pr_auc":    "average_precision",
    "recall":    "recall",
    "precision": "precision",
    "f1":        "f1",
}

# Tuning grid (from the project plan). 8 combinations x 5 folds = 40 fits.
PARAM_GRID = {
    "clf__n_estimators":  [100, 200],
    "clf__max_depth":     [3, 5],
    "clf__learning_rate": [0.05, 0.1],
}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "cogdec_train.csv")
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"Loaded {path}: {len(df):,} rows | positive rate "
          f"{y.mean()*100:.2f}%\n")

    # --- Pipeline: preprocessor + gradient boosting ----------------------
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])

    # --- GridSearchCV, optimising ROC-AUC (our primary metric) -----------
    search = GridSearchCV(
        pipe,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True,       # refit the best combo on all training data
        verbose=1,
    )
    search.fit(X, y)

    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.3f}\n")

    # --- Score the best estimator across all metrics (same CV) -----------
    best = search.best_estimator_
    res = cross_validate(best, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    print("Gradient Boosting (best) 5-fold CV scores:")
    for m in SCORING:
        print(f"   {m:10s}: {res['test_' + m].mean():.3f}")

    # --- Save full grid results for the appendix -------------------------
    grid_df = pd.DataFrame(search.cv_results_)
    keep = ["param_clf__n_estimators", "param_clf__max_depth",
            "param_clf__learning_rate", "mean_test_score", "rank_test_score"]
    grid_df[keep].sort_values("rank_test_score").to_csv(
        "gridsearch_results.csv", index=False)
    print("\nSaved full grid -> gridsearch_results.csv")


if __name__ == "__main__":
    main()