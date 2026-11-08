"""Spline-adjusted, sample-based associations (not survey-weighted).

This module is imported by association_analysis.py and called from main().
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer
from build_preprocessor import CONTINUOUS, CATEGORICAL

REFERENCES = {'SEXVAR':2,'CVDINFR4':2,'CVDSTRK3':2,'BPHIGH6':3,
              'TOLDHI3':2,'ADDEPEV3':2,'DIABETE4':3,'EXERANY2':2,'_SMOKER3':4}
CONTRASTS = {
    '_AGE80':(55,[65,75,80]),
    'sleep_hours':(7,[5,9,10]),
    'DRNKWK3_clean':(0,[7,14]),
    'bmi':(25,[18.5,30,35]),
    'mhealth':(0,[7,14,21]),
}


def fit_spline_associations(df,y,output):
    """Estimate spline GLM and clinically interpretable odds-ratio contrasts."""
    design_parts=[]
    spline_info={}
    medians={}
    for col in CONTINUOUS:
        numeric=pd.to_numeric(df[col],errors='coerce')
        missing=numeric.isna().astype(float)
        median=float(numeric.median())
        medians[col]=median
        filled=numeric.fillna(median).to_numpy().reshape(-1,1)
        transformer=SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False)
        basis=transformer.fit_transform(filled)
        names=[f'{col}__spline_{i}' for i in range(basis.shape[1])]
        design_parts.append(pd.DataFrame(basis,columns=names,index=df.index))
        design_parts.append(pd.DataFrame({f'{col}__missing':missing},index=df.index))
        spline_info[col]=(transformer,names)
    for col in CATEGORICAL:
        values=pd.to_numeric(df[col],errors='coerce').fillna(999).astype(int)
        levels=sorted(values.unique())
        ref=REFERENCES[col]
        if ref not in levels:
            raise ValueError(f'Missing reference category {ref} in {col}')
        columns={f'{col}[{level} vs {ref}]':(values==level).astype(float)
                 for level in levels if level!=ref}
        design_parts.append(pd.DataFrame(columns,index=df.index))
    design=pd.concat(design_parts,axis=1)
    design=design.loc[:,design.nunique(dropna=False)>1].astype(float)
    design=sm.add_constant(design,has_constant='add')
    fit=sm.GLM(y,design,family=sm.families.Binomial()).fit(cov_type='HC3')
    beta=fit.params.to_numpy()
    covariance=fit.cov_params().to_numpy()
    columns=list(design.columns)
    def contrast(delta):
        log_or=float(delta@beta)
        se=float(np.sqrt(max(delta@covariance@delta,0)))
        return np.exp(log_or),np.exp(log_or-1.96*se),np.exp(log_or+1.96*se),2*norm.sf(abs(log_or/se)) if se>0 else np.nan
    rows=[]
    # Category comparisons: one coefficient is an adjusted log-OR vs its reference.
    for col in CATEGORICAL:
        for feature in columns:
            if feature.startswith(col+'['):
                delta=np.zeros(len(columns));delta[columns.index(feature)]=1
                estimate,lo,hi,p=contrast(delta)
                rows.append({'feature':feature,'adjusted_OR':estimate,'CI_low':lo,'CI_high':hi,'p_value':p,'kind':'categorical'})
    curves=[]
    for col,(transformer,names) in spline_info.items():
        reference,comparisons=CONTRASTS[col]
        ref_basis=transformer.transform(np.array([[reference]]))[0]
        def delta_for(value):
            delta=np.zeros(len(columns))
            basis=transformer.transform(np.array([[value]]))[0]-ref_basis
            for name,v in zip(names,basis):
                if name in columns: delta[columns.index(name)]=v
            return delta
        for value in comparisons:
            estimate,lo,hi,p=contrast(delta_for(value))
            rows.append({'feature':f'{col}[{value} vs {reference}]','adjusted_OR':estimate,
                         'CI_low':lo,'CI_high':hi,'p_value':p,'kind':'spline contrast'})
        numeric=pd.to_numeric(df[col],errors='coerce').dropna()
        grid=np.linspace(numeric.quantile(.01),numeric.quantile(.99),65)
        for value in grid:
            estimate,lo,hi,_=contrast(delta_for(value))
            curves.append({'variable':col,'value':value,'reference':reference,
                           'adjusted_OR':estimate,'CI_low':lo,'CI_high':hi})
    results=pd.DataFrame(rows)
    results.to_csv(output/'association_spline_adjusted_odds_ratios.csv',index=False)
    curves=pd.DataFrame(curves)
    curves.to_csv(output/'association_spline_effect_curves.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    for ax,col in zip(axes.flat,CONTINUOUS):
        sub=curves[curves.variable==col]
        ax.plot(sub.value,sub.adjusted_OR,color='#277ca5')
        ax.fill_between(sub.value,sub.CI_low,sub.CI_high,color='#277ca5',alpha=.18)
        ax.axhline(1,ls='--',color='gray')
        ax.set_title(col)
        ax.set_xlabel(f'{col} (reference {CONTRASTS[col][0]})')
        ax.set_ylabel('Adjusted odds ratio (95% HC3 CI)')
    fig.delaxes(axes.flat[-1]);fig.tight_layout()
    fig.savefig(output/'association_spline_effect_curves.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    # Forest plot: interpretable categories and selected spline contrasts.
    plotted=results[(results.kind=='spline contrast') | (~results.feature.str.contains('999|__missing',regex=True))].copy()
    plotted=plotted.sort_values('feature')
    fig,ax=plt.subplots(figsize=(12,max(7,.32*len(plotted))))
    yloc=np.arange(len(plotted))
    ax.errorbar(plotted.adjusted_OR,yloc,
                xerr=[plotted.adjusted_OR-plotted.CI_low,plotted.CI_high-plotted.adjusted_OR],
                fmt='o',capsize=2,color='#277ca5')
    ax.set_yticks(yloc,plotted.feature,fontsize=8)
    ax.axvline(1,ls='--',color='gray')
    ax.set_xscale('log')
    ax.set_xlabel('Adjusted OR with 95% HC3 CI (training sample; unweighted)')
    ax.set_title('Spline-adjusted associations: categorical contrasts and continuous comparisons')
    fig.tight_layout()
    fig.savefig(output/'association_spline_forest_plot.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Saved spline-adjusted association CSV, curves and forest plot to',output)
