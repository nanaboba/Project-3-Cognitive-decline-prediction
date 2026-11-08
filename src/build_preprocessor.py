"""
build_preprocessor.py
----------------------
Builds the ColumnTransformer that preprocesses the BRFSS cognitive-decline
features, ready to be plugged in as the first step of a model Pipeline.

Two lanes:
  - CONTINUOUS : record was-missing flags -> median impute -> standard scale
  - CATEGORICAL: one-hot encode (handle_unknown='ignore')

IMPORTANT (leakage guard):
  This transformer LEARNS its parameters (medians, scaler stats, category
  lists) only when .fit() is called. In the real modelling scripts it is the
  FIRST step inside each model's Pipeline, so fitting happens on the training
  fold only -- never on the sealed test set.

Run directly to see a demo of the output shape on the training data:
    python build_preprocessor.py data/cogdec_train.csv
"""

import sys
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from pathlib import Path

# Folder this script lives in (e.g. .../src)
SCRIPT_DIR = Path(__file__).resolve().parent
# Project root is one level up from src/
PROJECT_ROOT = SCRIPT_DIR.parent
# Default data location: <project_root>/data/
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Feature lists  (must match the split script)
# ---------------------------------------------------------------------------
CONTINUOUS = ["_AGE80", "sleep_hours", "DRNKWK3_clean", "bmi", "mhealth"]

CATEGORICAL = [
    "SEXVAR", "CVDINFR4", "CVDSTRK3", "BPHIGH6", "TOLDHI3",
    "ADDEPEV3", "DIABETE4", "EXERANY2", "_SMOKER3",
]

FEATURES = CONTINUOUS + CATEGORICAL
TARGET = "CIMEMLO1"


# ---------------------------------------------------------------------------
# Custom transformer: add "was-missing" indicator columns BEFORE imputation
# ---------------------------------------------------------------------------
class AddMissingIndicator(BaseEstimator, TransformerMixin):
    """Appends one 0/1 column per input feature flagging where values were NaN.

    Must run BEFORE the imputer, because imputation erases the missingness
    signal. Learns nothing from the data (stateless), so it is leakage-safe.
    """

    def __init__(self, columns):
        self.columns = columns          # store EXACTLY as received (clone-safe)

    def fit(self, X, y=None):
        self.columns_ = list(self.columns)   # list-conversion happens HERE, not in __init__
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        flags = X.isna().astype(int)
        flags.columns = [f"{c}__was_missing" for c in self.columns_]
        out = pd.concat([X.reset_index(drop=True),
                         flags.reset_index(drop=True)], axis=1)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(
            list(self.columns_) + [f"{c}__was_missing" for c in self.columns_]
        )

    

# ---------------------------------------------------------------------------
# Continuous lane:  missing-flag -> median impute -> standard scale
# ---------------------------------------------------------------------------
continuous_pipe = Pipeline(steps=[
    ("missing_flag", AddMissingIndicator(CONTINUOUS)),
    ("impute",       SimpleImputer(strategy="median")),
    ("scale",        StandardScaler()),
])


# ---------------------------------------------------------------------------
# Categorical lane:  one-hot encode
# ---------------------------------------------------------------------------
# sparse_output=False keeps the output as a plain dense array, which is easier
# to inspect and works fine at this data size.
categorical_pipe = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])


# ---------------------------------------------------------------------------
# Assemble the ColumnTransformer
# ---------------------------------------------------------------------------
def build_preprocessor():
    """Return an unfitted ColumnTransformer for the BRFSS feature set."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", continuous_pipe, CONTINUOUS),
            ("cat",  categorical_pipe, CATEGORICAL),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Demo when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "cogdec_train.csv"
    df = pd.read_csv(path)

    X = df[FEATURES]
    y = df[TARGET] if TARGET in df.columns else None

    pre = build_preprocessor()
    Xt = pre.fit_transform(X)   # fit here is for DEMONSTRATION ONLY

    print(f"Loaded          : {path}")
    print(f"Input shape     : {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Transformed     : {Xt.shape[0]} rows x {Xt.shape[1]} columns")
    print(f"Any NaN left?   : {np.isnan(Xt).any()}")
    print(f"All numeric?    : {Xt.dtype}")

    # Show the expanded column names
    names = pre.get_feature_names_out()
    print(f"\nFirst 15 output columns:")
    for n in names[:15]:
        print("   ", n)
    print(f"   ... ({len(names)} columns total)")