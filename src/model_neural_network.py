
"""
Neural network (MLP) for BRFSS cognitive decline.

Uses the existing preprocessing pipeline and
5-fold stratified cross-validation on training data only.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from safe_save_results import safe_save_results

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate
    )

from build_preprocessor import build_preprocessor, FEATURES, TARGET


# File locations
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Directory for hyperparameter tuning results
TUNING_OUTPUT = PROJECT_ROOT / "outputs" / "tuning"
TUNING_OUTPUT.mkdir(parents=True, exist_ok=True)


# Cross-validation
CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
    )


# Evaluation metrics
SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1"
}


# Small tuning grid
PARAM_GRID = {
    "clf__learning_rate_init": [0.0001, 0.0005, 0.005, 0.01]
}


def main():

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(DATA_DIR / "cogdec_train.csv")
    )

    df = pd.read_csv(path)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"Loaded {len(df):,} training rows")
    print(f"Positive class: {y.mean()*100:.2f}%")


    # Preprocessing + neural network
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.01,
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42
        ))
    ])


    # Tune network structure and regularisation
    search = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    search.fit(X, y)


    # --------------------------------------------------
    # Learning-rate sweep visualisation
    # --------------------------------------------------

    sweep_results = pd.DataFrame(search.cv_results_)

    sweep_results["Learning Rate"] = (
        sweep_results["param_clf__learning_rate_init"].astype(float)
    )

    sweep_results = sweep_results.sort_values("Learning Rate")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        sweep_results["Learning Rate"],
        sweep_results["mean_test_score"],
        yerr=sweep_results["std_test_score"],
        marker="o",
        capsize=4,
        linewidth=2
    )

    ax.set_xscale("log")
    ax.set_xlabel("Initial Learning Rate")
    ax.set_ylabel("Cross-Validated ROC-AUC")
    ax.set_title("Neural Network — Learning-Rate Sweep")

    ax.grid(alpha=0.3)

    plt.tight_layout()

    from safe_save_results import safe_save_figure
    safe_save_figure(fig, TUNING_OUTPUT / "fig_neural_network_lr_sweep.png", dpi=150, bbox_inches="tight")
    safe_save_figure(fig, TUNING_OUTPUT / "fig_neural_network_lr_sweep.svg", bbox_inches="tight")

    plt.close()

    print("\nBest parameters:")
    print(search.best_params_)


    print(
        f"Best CV ROC-AUC: {search.best_score_:.3f}"
    )


    # Evaluate selected configuration
    results = cross_validate(
        search.best_estimator_,
        X,
        y,
        cv=CV,
        scoring=SCORING,
        n_jobs=-1
    )

    print("\nNeural Network — 5-fold CV results:")

    for metric in SCORING:
        print(
            f"{metric:12s}: "
            f"{results['test_' + metric].mean():.3f}"
        )

    # Save tuning results
    results_df = pd.DataFrame(search.cv_results_)

    safe_save_results(
        results_df,
        TUNING_OUTPUT / "neural_network_gridsearch_results.csv"
    )
    
   
    print("\nNeural network grid-search results processed.")


if __name__ == "__main__":
    main()
