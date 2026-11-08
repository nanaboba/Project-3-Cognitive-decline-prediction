
import sys
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_validate
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
    "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]
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
        ("clf", LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=42
        ))
    ])

    search = GridSearchCV(
        pipe,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True
    )

    search.fit(X, y)

    print("Best parameters:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.3f}")

    best = search.best_estimator_

    results = cross_validate(
        best, X, y,
        cv=CV,
        scoring=SCORING,
        n_jobs=-1
    )

    print("\nTuned Logistic Regression — 5-fold CV:")
    for metric in SCORING:
        print(
            f"{metric:12s}: "
            f"{results['test_' + metric].mean():.3f}"
        )

    pd.DataFrame(search.cv_results_).to_csv(
        "logistic_gridsearch_results.csv",
        index=False
    )

if __name__ == "__main__":
    main()
