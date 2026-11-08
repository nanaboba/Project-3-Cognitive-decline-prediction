"""Reporting only: top-4545 from saved unified test predictions; no model fitting.
Place in Exploration/src/; run python src/recalculate_outreach_250k.py.
Requires outputs/final_evaluation/all_final_test_predictions.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'outputs' / 'final_evaluation' / 'all_final_test_predictions.csv'
OUT = ROOT / 'outputs' / 'opportunity_sizing'
OUT.mkdir(parents=True, exist_ok=True)
BUDGET = 250000
COST = 55
K = BUDGET // COST
if not SOURCE.exists():
    raise FileNotFoundError(f'Run the unified final evaluation first, or locate its saved predictions: {SOURCE}')
df = pd.read_csv(SOURCE)
actual = df['Actual'].to_numpy()
if len(actual) != 14690 or int(actual.sum()) != 3033:
    raise ValueError('Unexpected test sample. Verify saved prediction provenance.')
rows = []
expected = K * actual.mean()
rows.append({'Model':'Random selection (expected)','Invitations':K,'Positive cases reached':expected,'Precision@K':actual.mean(),'Recall@K':K/len(actual),'Cost per positive (SGD)':BUDGET/expected,'Lift vs random':1.0})
for col in df.columns:
    if not col.endswith('__probability'):
        continue
    scores = df[col].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f'Invalid scores: {col}')
    idx = np.argsort(-scores, kind='stable')[:K]
    hits = int(actual[idx].sum())
    rows.append({'Model':col.removesuffix('__probability'),'Invitations':K,'Positive cases reached':hits,'Precision@K':hits/K,'Recall@K':hits/actual.sum(),'Cost per positive (SGD)':BUDGET/hits if hits else np.nan,'Lift vs random':(hits/K)/actual.mean()})
res = pd.DataFrame(rows)
path = OUT / 'precision_at_4545_all_models.csv'
res.to_csv(path,index=False)
print(res.round(4).to_string(index=False))
print('Saved:',path)
