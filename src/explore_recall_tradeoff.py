"""
explore_recall_tradeoff.py
--------------------------
TRAINING-ONLY exploration of the recall / precision trade-off.

This script NEVER loads or touches the test set -- by design. It auto-tunes
gradient boosting on the training data, then shows, via cross-validation, what
precision and false-alarm rate you get at each target recall. Use it to CHOOSE
your operating point. The actual sealed-test scoring lives in a separate file
(final_test_evaluation.py).

Edit TARGET_RECALLS freely and re-run as often as you like -- it is always safe.

Usage:
    python explore_recall_tradeoff.py data/cogdec_train.csv
"""

import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_predict)
from sklearn.metrics import (
    recall_score, precision_score, precision_recall_curve)

from build_preprocessor import build_preprocessor, FEATURES, TARGET

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent   # .../src
PROJECT_ROOT = SCRIPT_DIR.parent               # project root
DATA_DIR = PROJECT_ROOT / "data"               # .../data


# Explore any target recalls you like -- edit and re-run freely, always safe.
TARGET_RECALLS = [0.60, 0.70, 0.80]

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
PARAM_GRID = {
    "clf__n_estimators":  [100, 200],
    "clf__max_depth":     [3, 5],
    "clf__learning_rate": [0.05, 0.1],
}


def threshold_for_recall(y_true, proba, target_recall):
    """Lowest-false-alarm threshold achieving >= target recall (training only)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    rec, prec, thr = rec[:-1], prec[:-1], thr
    ok = rec >= target_recall
    if not ok.any():
        print(f"WARNING: target recall {target_recall} not reachable; using 0.5")   # NEW
        return 0.5
    idx = np.argmax(np.where(ok, prec, -1))
    return thr[idx]


def main():
    # NOTE: only a TRAINING path is accepted. There is no test argument.
    train_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR / "cogdec_train.csv"
    tr = pd.read_csv(train_path)
    
    X_tr, y_tr = tr[FEATURES], tr[TARGET].astype(int)

    print(f"Loaded training data: {len(tr):,} rows | "
          f"positive rate {y_tr.mean()*100:.2f}%\n")

        # --- Auto-tune hyperparameters on TRAINING set -----------------------
    base = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])
    search = GridSearchCV(base, PARAM_GRID, scoring="roc_auc",
                          cv=CV, n_jobs=-1, refit=True)
    search.fit(X_tr, y_tr)
    p = search.best_params_
    print(f"Auto-tuned best params: learning_rate={p['clf__learning_rate']}, "
          f"max_depth={p['clf__max_depth']}, "
          f"n_estimators={p['clf__n_estimators']}")
    print("(Training-only exploration. The test set is never loaded by this "
          "script.)\n")

    # --- NEW: check tuned params match the ones used in other scripts ---
    expected = {"clf__learning_rate": 0.05, "clf__max_depth": 3,
                "clf__n_estimators": 200}
    if p != expected:
        print(f"WARNING: tuned params {p} differ from hard-coded {expected}. "
              "Thresholds won't match final_test_evaluation.py.\n")


    # --- Out-of-fold training probabilities (no leakage) -----------------
    oof = cross_val_predict(search.best_estimator_, X_tr, y_tr, cv=CV,
                            method="predict_proba")[:, 1]

    # --- Sweep target recalls on TRAINING CV -----------------------------
    rows = []
    for tr_recall in TARGET_RECALLS:
        thr = threshold_for_recall(y_tr, oof, tr_recall)
        pred = (oof >= thr).astype(int)
        rec = recall_score(y_tr, pred)
        prec = precision_score(y_tr, pred)
        flagged = pred.mean() * 1000
        # false alarms = flagged people who are actually negative
        false_alarms = flagged * (1 - prec)
        rows.append({
            "target_recall":         tr_recall,
            "threshold":             thr,
            "actual_recall":         rec,
            "precision":             prec,
            "flagged_per_1000":      round(flagged),
            "false_alarms_per_1000": round(false_alarms),
        })

    sweep = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("RECALL / PRECISION TRADE-OFF  (all from TRAINING cross-validation)")
    print(sweep.to_string(index=False))
    out_path = DATA_DIR / "recall_tradeoff_training.csv"     
    sweep.to_csv(out_path, index=False)                      
    print(f"\nSaved -> {out_path}")             
    print("\nWhen you have chosen a target recall, put it into "
          "final_test_evaluation.py for the single sealed-test scoring.")


if __name__ == "__main__":
    main()