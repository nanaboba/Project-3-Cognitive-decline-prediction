"""Combine previously saved test results without refitting or rescoring models.
Run: python src/consolidate_test_results.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs'/'final_reporting'
OUT.mkdir(parents=True,exist_ok=True)
original=ROOT/'final_test_results.csv'
spline=ROOT/'outputs'/'spline_experiment'/'all_spline_exploratory_test_results.csv'
if not original.exists():
    raise FileNotFoundError(f'Original saved results missing: {original}')
if not spline.exists():
    raise FileNotFoundError(f'Spline saved results missing: {spline}')
a=pd.read_csv(original)
b=pd.read_csv(spline).rename(columns={
    'Model':'model','Threshold':'threshold','ROC_AUC':'roc_auc',
    'Average_Precision':'pr_auc','Recall':'recall','Precision':'precision','F1':'f1'})
b['model']='Spline Logistic Regression (post-test revision)'
a.loc[a.model=='Logistic Regression','model']='Original Logistic Regression (historical)'
combined=pd.concat([a,b],ignore_index=True)
combined=combined[['model','threshold','roc_auc','pr_auc','recall','precision','f1']]
combined.to_csv(OUT/'four_model_test_comparison.csv',index=False)
print(combined.to_string(index=False))
# This is a metric comparison, not ROC/PR curves: the latter need per-row scores for every model.
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,metric,title in zip(axes,['roc_auc','pr_auc'],['Test ROC-AUC','Test Average Precision']):
    ordered=combined.sort_values(metric)
    ax.barh(ordered.model,ordered[metric],color='#277ca5')
    ax.set_xlim(0,max(ordered[metric].max()*1.15,.1))
    ax.set_title(title)
    for j,value in enumerate(ordered[metric]):ax.text(value+.003,j,f'{value:.3f}',va='center')
fig.tight_layout()
fig.savefig(OUT/'fig_four_model_test_metrics.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print('Saved consolidated results to',OUT)
