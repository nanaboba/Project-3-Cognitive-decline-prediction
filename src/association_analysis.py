
    
  
"""Exploratory adjusted associations for BRFSS CIMEMLO1 (training data only).

Outputs are sample-based, NOT BRFSS survey-design-adjusted estimates.
Run: python src/association_analysis.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from build_preprocessor import CONTINUOUS, CATEGORICAL, TARGET
from association_spline_extension import fit_spline_associations

DATA = Path(__file__).resolve().parent.parent / 'data' / 'cogdec_train.csv'
OUTPUT = Path(__file__).resolve().parent.parent / 'outputs'


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    y = df[TARGET].astype(int)
    x = df[CONTINUOUS + CATEGORICAL].copy()

    # Keep all training rows; make missing continuous values explicit.
    for col in CONTINUOUS:
        x[col] = pd.to_numeric(x[col], errors='coerce')
        x[col + '__missing'] = x[col].isna().astype(int)
        x[col] = x[col].fillna(x[col].median())

    # One reference category per categorical feature avoids exact dummy collinearity.
    # Codes 999 remain an explicit Unknown category, not an actual diagnosis.
    references = {'SEXVAR': 2, 'CVDINFR4': 2, 'CVDSTRK3': 2,
                  'BPHIGH6': 3, 'TOLDHI3': 2, 'ADDEPEV3': 2,
                  'DIABETE4': 3, 'EXERANY2': 2, '_SMOKER3': 4}
    for col in CATEGORICAL:
        s = pd.to_numeric(x.pop(col), errors='coerce').fillna(999).astype(int)
        levels = sorted(s.unique().tolist())
        ref = references[col]
        if ref not in levels:
            raise ValueError(f'{col}: expected reference code {ref} not found; check recoding.')
        for level in levels:
            if level != ref:
                x[f'{col}[{level} vs {ref}]'] = (s == level).astype(int)

    # Drop missingness flags that are always zero; these carry no information.
    x = x.loc[:, x.nunique(dropna=False) > 1].astype(float)
    design = sm.add_constant(x, has_constant='add')

    # A single design-matrix diagnostic, including all encoded categories.
    # VIF is computed from each predictor's R-squared with the other predictors.
    # The correlation-matrix pseudoinverse is used only if matrix is singular;
    # singularity is reported rather than interpreting unreliable VIFs.
    z = (x - x.mean()) / x.std(ddof=0)
    corr = np.corrcoef(z.to_numpy(), rowvar=False)
    rank = np.linalg.matrix_rank(corr)
    if rank < len(x.columns):
        print(f'WARNING: encoded predictors are rank-deficient ({rank}/{len(x.columns)}).')
        print('VIF values are not reported because exact collinearity is present.')
        vif = pd.DataFrame({'feature': x.columns, 'VIF': np.nan})
    else:
        vif = pd.DataFrame({'feature': x.columns,
                            'VIF': np.diag(np.linalg.inv(corr))})
    vif.sort_values('VIF', ascending=False).to_csv(OUTPUT / 'association_vif.csv', index=False)

    # Binomial GLM gives log-odds coefficients; HC3 gives heteroskedasticity-
    # robust standard errors, but NOT complex-survey design-based errors.
    fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(cov_type='HC3')
    ci = fit.conf_int()
    results = pd.DataFrame({
        'feature': fit.params.index,
        'adjusted_OR': np.exp(fit.params),
        'CI_low': np.exp(ci[0]),
        'CI_high': np.exp(ci[1]),
        'p_value': fit.pvalues,
    })
    results = results[results['feature'] != 'const'].copy()
    results.to_csv(OUTPUT / 'association_linear_adjusted_odds_ratios.csv', index=False)

    # Compact forest plot for the 15 strongest absolute log-odds associations.
    # Rankings here are for display only, not causal or predictive importance.
    plotted = results.copy()
    plotted['distance'] = np.abs(np.log(plotted['adjusted_OR']))
    plotted = plotted.sort_values('distance', ascending=False).head(15)
    plotted = plotted.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(plotted['adjusted_OR'], plotted['feature'],
                xerr=[plotted['adjusted_OR'] - plotted['CI_low'],
                      plotted['CI_high'] - plotted['adjusted_OR']],
                fmt='o', capsize=3)
    ax.axvline(1, linestyle='--', color='gray')
    ax.set_xscale('log')
    ax.set_xlabel('Adjusted odds ratio (log scale), 95% HC3 CI')
    ax.set_title('Exploratory associations — training sample only')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'association_linear_forest_plot.png', dpi=160)
    plt.close(fig)

    print(f'Rows: {len(df):,} | Encoded predictors: {x.shape[1]}')
    print('Highest VIFs:')
    print(vif.sort_values('VIF', ascending=False).head(10).to_string(index=False))
    print('\nSaved to:', OUTPUT)
    print('  association_vif.csv')
    print('  association_adjusted_odds_ratios.csv')
    print('  association_forest_plot.png')
    print('NOTE: Not survey-weighted; do not interpret as nationally representative.')

    # Second specification: cubic splines for all five continuous predictors.
    fit_spline_associations(df, y, OUTPUT)


if __name__ == '__main__':
    main()

