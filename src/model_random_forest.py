"""
model_random_forest.py
----------------------
Rung 3: Random Forest, tuned with GridSearchCV (5-fold CV, training only).
A parallel tree ensemble, to compare against the sequential boosting (rung 2).
No learning_rate (forests don't correct sequentially); trees grown deeper
because averaging controls overfitting.

Usage:
    python model_random_forest.py data/cogdec_train.csv
"""

import sys
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate

from build_preprocessor import build_preprocessor, FEATURES, TARGET
from safe_save_results import safe_save_results

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent   # .../src
PROJECT_ROOT = SCRIPT_DIR.parent               # project root
DATA_DIR = PROJECT_ROOT / "data"               # .../data

# Directory for hyperparameter tuning results
TUNING_OUTPUT = PROJECT_ROOT / "outputs" / "tuning"
TUNING_OUTPUT.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

SCORING = {
    "roc_auc": "roc_auc", "pr_auc": "average_precision",
    "recall": "recall", "precision": "precision", "f1": "f1",
}

PARAM_GRID = {
    "clf__n_estimators":     [200, 300],
    "clf__max_depth":        [10, 20],
    "clf__min_samples_leaf": [1, 5],
}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "cogdec_train.csv")
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"Loaded {path}: {len(df):,} rows | positive rate "
          f"{y.mean()*100:.2f}%\n")

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            class_weight="balanced",   # handle the 20/80 imbalance
            random_state=42,
            n_jobs=-1,
        )),
    ])

    search = GridSearchCV(
        pipe, param_grid=PARAM_GRID, scoring="roc_auc",
        cv=CV, n_jobs=-1, refit=True, verbose=1,
    )
    search.fit(X, y)

    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.3f}\n")

    best = search.best_estimator_
    res = cross_validate(best, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    print("Random Forest (best) 5-fold CV scores:")
    for m in SCORING:
        print(f"   {m:10s}: {res['test_' + m].mean():.3f}")

    grid_df = pd.DataFrame(search.cv_results_)

    keep = [
        "param_clf__n_estimators",
        "param_clf__max_depth",
        "param_clf__min_samples_leaf",
        "mean_test_score",
        "rank_test_score"
    ]

    results_df = grid_df[keep].sort_values("rank_test_score")

    safe_save_results(
        results_df,
        TUNING_OUTPUT / "rf_gridsearch_results.csv"
    )

    print("\nRandom forest grid-search results processed.")

if __name__ == "__main__":
    main()