"""Out-of-fold original-feature permutation importance for four BRFSS models.

Run from project root: python src/feature_importance_grouped.py
Uses training data only; outputs to outputs/importance/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from build_preprocessor import build_preprocessor, FEATURES, TARGET
from build_spline_model import build_all_spline_model

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'outputs' / 'importance'
OUTPUT.mkdir(parents=True, exist_ok=True)
NAMES = {
    '_AGE80':'Age','sleep_hours':'Sleep Duration','DRNKWK3_clean':'Drinks per Week',
    'bmi':'BMI','mhealth':'Poor Mental Health Days','SEXVAR':'Sex',
    'CVDINFR4':'Heart Attack History','CVDSTRK3':'Stroke History',
    'BPHIGH6':'High Blood Pressure','TOLDHI3':'High Cholesterol',
    'ADDEPEV3':'Depression History','DIABETE4':'Diabetes',
    'EXERANY2':'Physical Activity','_SMOKER3':'Smoking Status'
}
MODELS = {
    'Original Logistic Regression': Pipeline([
        ('prep',build_preprocessor()),
        ('clf',LogisticRegression(C=.01,class_weight='balanced',max_iter=2000,random_state=42))]),
    'Spline Logistic Regression': build_all_spline_model(),
    'Random Forest': Pipeline([
        ('prep',build_preprocessor()),
        ('clf',RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=5,
                                      class_weight='balanced',random_state=42,n_jobs=-1))]),
    'Gradient Boosting': Pipeline([
        ('prep',build_preprocessor()),
        ('clf',GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=.05,random_state=42))]),
}


def main():
    df = pd.read_csv(ROOT/'data'/'cogdec_train.csv')
    X, y = df[FEATURES], df[TARGET].astype(int)
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    records=[]
    # A common set of held-out observations and permutation seeds for every model.
    for fold,(train_idx,val_idx) in enumerate(cv.split(X,y),1):
        xtrain,xval = X.iloc[train_idx],X.iloc[val_idx]
        ytrain,yval = y.iloc[train_idx],y.iloc[val_idx]
        for name,base in MODELS.items():
            print(f'Fold {fold}/5: {name}',flush=True)
            fitted=clone(base).fit(xtrain,ytrain)
            baseline=roc_auc_score(yval,fitted.predict_proba(xval)[:,1])
            pi=permutation_importance(fitted,xval,yval,scoring='roc_auc',n_repeats=5,
                                      random_state=42+fold,n_jobs=1)
            for j,feature in enumerate(FEATURES):
                for repeat,drop in enumerate(pi.importances[j],1):
                    records.append({'Model':name,'Fold':fold,'Repeat':repeat,
                                    'Variable':feature,'Feature':NAMES[feature],
                                    'Baseline_ROC_AUC':baseline,'ROC_AUC_Drop':drop})
    raw=pd.DataFrame(records)
    raw.to_csv(OUTPUT/'importance_all_models_repeats.csv',index=False)
    # Average repeats within a fold first; then summarise variation across folds.
    per_fold=raw.groupby(['Model','Fold','Variable','Feature'],as_index=False)['ROC_AUC_Drop'].mean()
    per_fold.to_csv(OUTPUT/'importance_all_models_by_fold.csv',index=False)
    summary=per_fold.groupby(['Model','Variable','Feature'],as_index=False).agg(
        importance_mean=('ROC_AUC_Drop','mean'),
        importance_fold_sd=('ROC_AUC_Drop','std'))
    summary.to_csv(OUTPUT/'importance_all_models_summary.csv',index=False)
    for name in MODELS:
        subset=summary[summary.Model==name].sort_values('importance_mean',ascending=False)
        slug={'Original Logistic Regression':'original_lr','Spline Logistic Regression':'spline_lr',
              'Random Forest':'rf','Gradient Boosting':'gb'}[name]
        subset.to_csv(OUTPUT/f'importance_{slug}.csv',index=False)
        plot=subset.sort_values('importance_mean')
        fig,ax=plt.subplots(figsize=(10,7))
        ax.barh(plot.Feature,plot.importance_mean,xerr=plot.importance_fold_sd,
                color='#277ca5',capsize=3)
        ax.axvline(0,color='gray',lw=.8)
        ax.set_xlabel('Mean decrease in held-out-fold ROC-AUC')
        ax.set_title(f'{name}\nGrouped permutation importance (5-fold CV)')
        fig.tight_layout()
        fig.savefig(OUTPUT/f'fig_importance_{slug}.png',dpi=180,bbox_inches='tight')
        fig.savefig(OUTPUT/f'fig_importance_{slug}.svg',bbox_inches='tight')
        plt.close(fig)
    print('\nSaved grouped permutation importance to',OUTPUT)

if __name__=='__main__':
    main()
