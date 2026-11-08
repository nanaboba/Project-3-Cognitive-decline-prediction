"""
model_baselines.py
------------------
Fits and cross-validates the first two rungs of the model ladder on the
TRAINING set only. The sealed test set is NOT touched here.

    Rung 0: DummyClassifier (majority-class)  -- the "dumb" baseline floor
    Rung 1: Logistic Regression               -- simplest real model

Scoring is 5-fold stratified cross-validation on the training data, so every
fold re-learns the preprocessor's medians/scales/categories from its own
training portion -- no leakage.

Usage:
    python model_baselines.py data/cogdec_train.csv
"""

import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from build_preprocessor import build_preprocessor, FEATURES, TARGET

from pathlib import Path

# Folder this script lives in (e.g. .../src)
SCRIPT_DIR = Path(__file__).resolve().parent
# Project root is one level up from src/
PROJECT_ROOT = SCRIPT_DIR.parent
# Default data location: <project_root>/data/
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Metrics to collect. ROC-AUC is primary (ranking quality for targeting).
# PR-AUC matters under class imbalance. Recall/precision/F1 show the
# targeting trade-off.
# ---------------------------------------------------------------------------
SCORING = {
    "roc_auc":   "roc_auc",
    "pr_auc":    "average_precision",
    "recall":    "recall",
    "precision": "precision",
    "f1":        "f1",
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate(name, model, X, y):
    """Run cross-validation and return a dict of mean scores."""
    results = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    return {
        "model":     name,
        "roc_auc":   results["test_roc_auc"].mean(),
        "pr_auc":    results["test_pr_auc"].mean(),
        "recall":    results["test_recall"].mean(),
        "precision": results["test_precision"].mean(),
        "f1":        results["test_f1"].mean(),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "cogdec_train.csv")
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"Loaded {path}: {len(df):,} rows | positive rate "
          f"{y.mean()*100:.3f}%\n")

    rows = []

    # --- Rung 0: dumb baseline -------------------------------------------
    # No preprocessing needed -- it ignores the features entirely.
    dummy = DummyClassifier(strategy="most_frequent")
    rows.append(evaluate("Rung 0: Dummy (majority)", dummy, X, y))

    # --- Rung 1: logistic regression -------------------------------------
    # Preprocessor as step 1 so medians/scales/categories are learned inside
    # each CV fold on that fold's training data only.
    logit = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(
            class_weight="balanced",   # handle the 20/80 imbalance
            max_iter=2000,
            random_state=42,
        )),
    ])
    rows.append(evaluate("Rung 1: Logistic Reg.", logit, X, y))

    # --- Results table ---------------------------------------------------
    res = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("5-fold CV results (training set only):\n")
    print(res.to_string(index=False))

    res.to_csv("model_results_baselines.csv", index=False)
    print("\nSaved -> model_results_baselines.csv")


if __name__ == "__main__":
    main()