import pandas as pd

df = pd.read_csv("neural_network_gridsearch_results.csv")

comparison = df[[
    "param_clf__learning_rate_init",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]].copy()

comparison.columns = [
    "Learning Rate",
    "Mean ROC-AUC",
    "Standard Deviation",
    "Rank"
]

comparison = comparison.sort_values("Learning Rate")

print(comparison.to_string(index=False))