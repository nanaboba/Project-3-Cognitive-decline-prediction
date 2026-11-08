"""Reusable, unfitted BRFSS all-continuous-spline Logistic Regression pipeline."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import SplineTransformer, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from build_preprocessor import CONTINUOUS, CATEGORICAL


def missing_flag(values):
    return np.isnan(np.asarray(values, dtype=float)).astype(float)


def spline_branch():
    spline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('spline', SplineTransformer(n_knots=4, degree=3, knots='quantile', include_bias=False)),
        ('scale', StandardScaler()),
    ])
    missing = Pipeline([
        ('flag', FunctionTransformer(missing_flag, validate=False)),
        ('scale', StandardScaler()),
    ])
    return FeatureUnion([('spline_features', spline), ('missing_indicator', missing)])


def build_all_spline_model():
    transforms = [(f'spline_{name.lstrip("_")}', spline_branch(), [name]) for name in CONTINUOUS]
    transforms.append(('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL))
    return Pipeline([
        ('prep', ColumnTransformer(transforms)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)),
    ])
