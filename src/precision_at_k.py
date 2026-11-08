
"""
precision_at_k.py
-----------------
Fixed-budget cognitive-health outreach evaluation.

Evaluates:
1. Random selection (expected benchmark)
2. Gradient Boosting
3. Random Forest
4. Spline Logistic Regression

Also generates updated Precision-Recall curves.

Uses existing tuned parameters.
No hyperparameter tuning or threshold optimisation.

IMPORTANT:
This is a post-test operational analysis using the
previously evaluated held-out test sample.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)

from build_preprocessor import (
    build_preprocessor,
    FEATURES,
    TARGET
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

OUTPUT = ROOT / "outputs" / "opportunity_sizing"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "cogdec_train.csv"
TEST_FILE = DATA_DIR / "cogdec_test.csv"

SPLINE_FILE = (
    ROOT / "outputs" / "spline_experiment"
    / "all_spline_test_predictions.csv"
)

K = 4000

BUDGET_SGD = 220_000
COST_PER_PERSON_SGD = 55

RANDOM_STATE = 42

GB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=RANDOM_STATE
)

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ============================================================
# 2. TOP-K METRICS
# ============================================================

def calculate_top_k(name, actual, scores, k):

    actual = np.asarray(actual)
    scores = np.asarray(scores, dtype=float)

    if len(actual) != len(scores):
        raise ValueError(
            f"{name}: prediction length mismatch."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            f"{name}: non-finite prediction scores."
        )

    # Stable sorting makes ties deterministic.
    ranked_indices = np.argsort(
        -scores,
        kind="stable"
    )

    selected_indices = ranked_indices[:k]

    positive_cases_reached = int(
        actual[selected_indices].sum()
    )

    total_positive_cases = int(actual.sum())

    precision_at_k = positive_cases_reached / k

    recall_at_k = (
        positive_cases_reached / total_positive_cases
    )

    cost_per_positive = (
        BUDGET_SGD / positive_cases_reached
        if positive_cases_reached > 0
        else np.nan
    )

    return {
        "Model": name,
        "Invitations": k,
        "Positive Cases Reached": positive_cases_reached,
        "Precision@K": precision_at_k,
        "Recall@K": recall_at_k,
        "Lift vs Random": (
            precision_at_k / actual.mean()
        ),
        "Cost per Positive (SGD)": cost_per_positive,
        "Score Cutoff": scores[ranked_indices[k - 1]]
    }


# ============================================================
# 3. LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

X_test = test[FEATURES]
y_test = test[TARGET].astype(int)

if not 0 < K <= len(test):
    raise ValueError("K must be within test sample size.")

if K * COST_PER_PERSON_SGD != BUDGET_SGD:
    raise ValueError(
        "Budget must equal K times unit cost."
    )

print(f"Training respondents: {len(train):,}")
print(f"Test respondents: {len(test):,}")
print(f"Positive prevalence: {y_test.mean():.2%}")


# ============================================================
# 4. RANDOM SELECTION BENCHMARK
# ============================================================

prevalence = y_test.mean()

expected_positive_cases = K * prevalence

results = [{
    "Model": "Random Selection (Expected)",
    "Invitations": K,
    "Positive Cases Reached": expected_positive_cases,
    "Precision@K": prevalence,
    "Recall@K": K / len(y_test),
    "Lift vs Random": 1.0,
    "Cost per Positive (SGD)": (
        BUDGET_SGD / expected_positive_cases
    ),
    "Score Cutoff": np.nan
}]

probabilities = {}


# ============================================================
# 5. GRADIENT BOOSTING
# ============================================================

print("\nFitting Gradient Boosting...")

gradient_boosting = Pipeline([
    ("prep", build_preprocessor()),
    ("clf", GradientBoostingClassifier(**GB_PARAMS))
])

gradient_boosting.fit(X_train, y_train)

gb_probabilities = gradient_boosting.predict_proba(
    X_test
)[:, 1]

probabilities["Gradient Boosting"] = gb_probabilities

results.append(
    calculate_top_k(
        "Gradient Boosting",
        y_test,
        gb_probabilities,
        K
    )
)


# ============================================================
# 6. RANDOM FOREST
# ============================================================

print("Fitting Random Forest...")

random_forest = Pipeline([
    ("prep", build_preprocessor()),
    ("clf", RandomForestClassifier(**RF_PARAMS))
])

random_forest.fit(X_train, y_train)

rf_probabilities = random_forest.predict_proba(
    X_test
)[:, 1]

probabilities["Random Forest"] = rf_probabilities

results.append(
    calculate_top_k(
        "Random Forest",
        y_test,
        rf_probabilities,
        K
    )
)


# ============================================================
# VERIFY REFITTED MODELS AGAINST ORIGINAL TEST RESULTS
# ============================================================

gb_test_auc = roc_auc_score(y_test, gb_probabilities)
rf_test_auc = roc_auc_score(y_test, rf_probabilities)

print(f"\nGB test ROC-AUC: {gb_test_auc:.4f} (previously ~0.738)")
print(f"RF test ROC-AUC: {rf_test_auc:.4f} (previously ~0.733)")

if not np.isclose(gb_test_auc, 0.738, atol=0.001):
    print("WARNING: GB ROC-AUC differs from previous results.")

if not np.isclose(rf_test_auc, 0.733, atol=0.001):
    print("WARNING: RF ROC-AUC differs from previous results.")


# ============================================================
# 7. SPLINE LOGISTIC REGRESSION
# ============================================================

print("Loading saved spline predictions...")

if not SPLINE_FILE.exists():
    raise FileNotFoundError(
        f"Missing spline prediction file: {SPLINE_FILE}"
    )

spline_predictions = pd.read_csv(SPLINE_FILE)

required_columns = [
    "Actual",
    "Predicted_Probability"
]

if not all(
    col in spline_predictions.columns
    for col in required_columns
):
    raise ValueError(
        "Spline prediction CSV has missing columns."
    )

if len(spline_predictions) != len(test):
    raise ValueError(
        "Spline prediction row count does not match test."
    )

if not np.array_equal(
    spline_predictions["Actual"].to_numpy(),
    y_test.to_numpy()
):
    raise ValueError(
        "Spline target sequence does not match test data."
    )

# This verifies target sequence, not unique row identity.
# The test CSV must be unchanged since spline evaluation.

spline_probabilities = spline_predictions[
    "Predicted_Probability"
].to_numpy()

probabilities[
    "Spline Logistic Regression"
] = spline_probabilities

results.append(
    calculate_top_k(
        "Spline Logistic Regression",
        y_test,
        spline_probabilities,
        K
    )
)


# ============================================================
# 8. SAVE OPPORTUNITY-SIZING RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "precision_at_4000_results.csv",
    index=False
)

print("\n" + "=" * 65)
print("OPPORTUNITY SIZING — TOP 4,000 RESPONDENTS")
print("=" * 65)

print(
    results_df.round(3).to_string(index=False)
)


# ============================================================
# 9. SAVE TEST PROBABILITIES
# ============================================================

prediction_table = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    **probabilities
})

prediction_table.to_csv(
    OUTPUT / "final_model_test_probabilities.csv",
    index=False
)


# ============================================================
# 10. UPDATED PRECISION-RECALL CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))


# Plot each model's PR curve and its top-K operating point
model_colours = {
    "Gradient Boosting": "#E07B00",
    "Random Forest": "#2E8B3D",
    "Spline Logistic Regression": "#1F4E9C"
}

for name, scores in probabilities.items():

    precision, recall, _ = precision_recall_curve(
        y_test,
        scores
    )

    average_precision = average_precision_score(
        y_test,
        scores
    )

    colour = model_colours[name]

    # Full Precision-Recall curve
    ax.plot(
        recall,
        precision,
        linewidth=2,
        color=colour,
        label=f"{name} (AP = {average_precision:.3f})"
    )

    # Find this model's top-4,000 results
    model_result = results_df.loc[
        results_df["Model"] == name
    ].iloc[0]

    capacity_recall = model_result["Recall@K"]
    capacity_precision = model_result["Precision@K"]

    # Mark the actual top-K operating point
    ax.scatter(
    capacity_recall,
    capacity_precision,
    s=30,
    color=colour,
    edgecolor="white",
    linewidth=0.6,
    zorder=5
)

ax.axhline(
    prevalence,
    color="grey",
    linestyle="--",
    label=f"Random baseline ({prevalence:.3f})"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")

ax.set_title(
    "Precision-Recall Curves — Final Model Candidates"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend()
ax.grid(alpha=0.2)


# Show exact top-4,000 results separately
# because the three operating points overlap.

capacity_rows = results_df[
    results_df["Model"] != "Random Selection (Expected)"
]

capacity_text = "TOP 4,000 INVITED\n\n"

for _, row in capacity_rows.iterrows():

    short_name = {
        "Gradient Boosting": "GB",
        "Random Forest": "RF",
        "Spline Logistic Regression": "Spline LR"
    }[row["Model"]]

    capacity_text += (
        f"{short_name}: "
        f"P={row['Precision@K']:.3f}, "
        f"R={row['Recall@K']:.3f}\n"
    )

ax.text(
    0.98,
    0.68,
    capacity_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    family="monospace",
    bbox={
        "boxstyle": "round,pad=0.5",
        "facecolor": "white",
        "edgecolor": "#CCCCCC",
        "alpha": 0.95
    }
)

plt.tight_layout()

fig.savefig(
    OUTPUT / "fig_pr_curves_final.png",
    dpi=180,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT / "fig_pr_curves_final.svg",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 11. OPPORTUNITY-SIZING COMPARISON CHART
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))


# Highlight the recommended model
bar_colours = {
    "Random Selection (Expected)": "#A6A6A6",
    "Gradient Boosting": "#9DB7C8",
    "Random Forest": "#9DB7C8",
    "Spline Logistic Regression": "#1F4E9C"
}

colours = [
    bar_colours[name]
    for name in results_df["Model"]
]

bars = ax.bar(
    results_df["Model"],
    results_df["Positive Cases Reached"],
    color=colours
)

ax.bar_label(
    bars,
    labels=[
        f"{value:,.0f}"
        for value in results_df["Positive Cases Reached"]
    ],
    padding=4
)

ax.set_ylabel(
    "Respondents Reporting Decline Reached"
)

ax.set_title(
    "Model-Ranked Outreach Reaches ~2x More Respondents\n"
    "Illustrative Budget: S$220,000 | 4,000 Participants"
)

ax.set_ylim(
    0,
    results_df["Positive Cases Reached"].max() * 1.2
)

ax.tick_params(axis="x", labelrotation=15)

fig.text(
    0.5, 0.01,
    "Random = expected count; model results = observed test-sample counts. "
    "Outcome = self-reported decline, not clinical diagnosis.",
    ha="center",
    fontsize=9,
    color="#555555"
)

plt.tight_layout(rect=[0, 0.07, 1, 1])

plt.tight_layout()

fig.savefig(
    OUTPUT / "fig_outreach_opportunity.png",
    dpi=180,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT / "fig_outreach_opportunity.svg",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 12. COMPLETION
# ============================================================

print("\nSaved results to:")
print(OUTPUT)
print("\nOpportunity-sizing analysis completed.")
