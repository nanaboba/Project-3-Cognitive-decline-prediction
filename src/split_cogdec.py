"""
Stratified train/test split for the BRFSS Cognitive Decline project.

Run AFTER deterministic cleaning (filtering, sentinel recoding, rescaling)
and BEFORE any parameter-learning step (imputation, scaling, encoding fit).

Usage:
    python split_cogdec.py <input_csv> <output_dir>
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------
TARGET = "CIMEMLO1"

CONTINUOUS = ["_AGE80", "sleep_hours", "DRNKWK3_clean", "bmi", "mhealth"]

CATEGORICAL = [
    "SEXVAR", "CVDINFR4", "CVDSTRK3", "BPHIGH6", "TOLDHI3",
    "ADDEPEV3", "DIABETE4", "GENHLTH", "EXERANY2", "_SMOKER3",
]

FEATURES = CONTINUOUS + CATEGORICAL

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main(in_path, out_dir):
    # ------------------------------------------------------------------
    # 2. Load and select the final analytic columns
    # ------------------------------------------------------------------
    df = pd.read_csv('data/CogDec.csv')
    cols = FEATURES + [TARGET]
    df = df[cols].copy()

    # Enforce a clean label: drop any row whose target is not 0/1.
    # (DK/Refused/Blank were already removed upstream, but this is a safeguard.)
    df = df[df[TARGET].isin([0, 1])]
    df[TARGET] = df[TARGET].astype(int)

    X = df[FEATURES]
    y = df[TARGET]

    # ------------------------------------------------------------------
    # 3. Stratified 80/20 split  -> preserves the ~20.6% positive rate
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # 4. Report + save
    # ------------------------------------------------------------------
    def rate(s):
        return f"{s.mean() * 100:.2f}% positive  (n={len(s)})"

    print("Split complete")
    print(f"  Full : {rate(y)}")
    print(f"  Train: {rate(y_train)}")
    print(f"  Test : {rate(y_test)}")

    train = X_train.copy(); train[TARGET] = y_train
    test = X_test.copy();   test[TARGET] = y_test

    train.to_csv(f"{out_dir}/cogdec_train.csv", index=False)
    test.to_csv(f"{out_dir}/cogdec_test.csv", index=False)
    print(f"\nSaved:\n  {out_dir}/cogdec_train.csv\n  {out_dir}/cogdec_test.csv")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "CogDec.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    main(in_path, out_dir)


