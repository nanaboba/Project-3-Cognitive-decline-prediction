
"""
HistGradientBoosting model for BRFSS cognitive decline.

Uses training data only.
5-fold stratified cross-validation with GridSearchCV.
"""

import sys
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)

from build_preprocessor import build_preprocessor, FEATURES, TARGET

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1"
}

PARAM_GRID = {
    "clf__max_iter": [100, 200],
    "clf__max_leaf_nodes": [7, 15],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [3, 5]
}

def main():

    path = (
        sys.argv[1] if len(sys.argv) > 1
        else str(DATA_DIR / "cogdec_train.csv")
    )

    df = pd.read_csv(path)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", HistGradientBoostingClassifier(
            random_state=42
        ))
    ])

    search = GridSearchCV(
        pipe,
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
    print(f"Best CV ROC-AUC: {search.best_score_:.3f}")

    results = cross_validate(
        search.best_estimator_,
        X,
        y,
        cv=CV,
        scoring=SCORING,
        n_jobs=-1
    )

    print("\nHistGradientBoosting — 5-fold CV results:")

    for metric in SCORING:
        print(
            f"{metric:12s}: "
            f"{results['test_' + metric].mean():.3f}"
        )

    pd.DataFrame(search.cv_results_).to_csv(
        "histgb_gridsearch_results.csv",
        index=False
    )

if __name__ == "__main__":
    main()
