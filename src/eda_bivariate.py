
"""
BRFSS Cognitive Decline — Descriptive and Bivariate EDA

Uses the full cleaned dataset (73,448 respondents).

Produces:
- Target distribution
- Continuous descriptive statistics
- Categorical frequency tables
- Bivariate decline rates for all 14 predictors
- Missingness summary
- Six presentation-ready figures

No imputation, scaling, model fitting or hyperparameter tuning.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from build_preprocessor import FEATURES, TARGET


# ============================================================
# 1. FILE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "CogDec.csv"

OUTPUT = ROOT / "outputs" / "eda"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

# Use only the final predictors and target
df = df[FEATURES + [TARGET]].copy()

# Ensure target contains valid binary responses
df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

print("=" * 60)
print("BRFSS — FULL DATASET DESCRIPTIVE EDA")
print("=" * 60)

print(f"Observations: {len(df):,}")
print(f"Predictors: {len(FEATURES)}")
print(f"Positive class: {df[TARGET].mean()*100:.2f}%")


# ============================================================
# 2. READABLE VARIABLE AND CATEGORY NAMES
# ============================================================

NAMES = {
    "_AGE80": "Age",
    "SEXVAR": "Sex",
    "CVDINFR4": "Heart Attack History",
    "CVDSTRK3": "Stroke History",
    "BPHIGH6": "High Blood Pressure",
    "TOLDHI3": "High Cholesterol",
    "ADDEPEV3": "Depression History",
    "DIABETE4": "Diabetes",
    "_SMOKER3": "Smoking Status",
    "DRNKWK3_clean": "Drinks per Week",
    "EXERANY2": "Physical Activity",
    "bmi": "Body Mass Index",
    "sleep_hours": "Sleep Duration",
    "mhealth": "Poor Mental Health Days"
}

YES_NO = {
    1: "Yes",
    2: "No",
    999: "Unknown"
}

CATEGORIES = {
    "SEXVAR": {
        1: "Male",
        2: "Female"
    },

    "CVDINFR4": YES_NO,
    "CVDSTRK3": YES_NO,
    "ADDEPEV3": YES_NO,
    "EXERANY2": YES_NO,
    "TOLDHI3": YES_NO,

    "BPHIGH6": {
        1: "Diagnosed high BP",
        2: "Pregnancy only",
        3: "No",
        4: "Borderline/elevated",
        999: "Unknown"
    },

    "DIABETE4": {
        1: "Diagnosed diabetes",
        2: "Pregnancy only",
        3: "No",
        4: "Pre-diabetes",
        999: "Unknown"
    },

    "_SMOKER3": {
        1: "Current daily",
        2: "Current some days",
        3: "Former smoker",
        4: "Never smoked",
        999: "Unknown"
    }
}

CONTINUOUS = [
    "_AGE80",
    "sleep_hours",
    "DRNKWK3_clean",
    "bmi",
    "mhealth"
]

CATEGORICAL = [
    variable for variable in FEATURES
    if variable not in CONTINUOUS
]


# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

target_summary = (
    df[TARGET]
    .value_counts()
    .sort_index()
    .rename_axis("Target")
    .reset_index(name="Respondents")
)

target_summary["Percentage"] = (
    target_summary["Respondents"] / len(df) * 100
)

target_summary["Label"] = target_summary["Target"].map({
    0: "No Decline",
    1: "Decline"
})

target_summary.to_csv(
    OUTPUT / "target_distribution.csv",
    index=False
)

print("\nTARGET DISTRIBUTION")
print(target_summary.to_string(index=False))


# ============================================================
# 4. CONTINUOUS DESCRIPTIVE STATISTICS
# ============================================================

continuous_summary = (
    df[CONTINUOUS]
    .describe()
    .T
    .reset_index()
    .rename(columns={"index": "Variable"})
)

continuous_summary["Missing"] = [
    df[column].isna().sum()
    for column in CONTINUOUS
]

continuous_summary["Missing_Percentage"] = (
    continuous_summary["Missing"] / len(df) * 100
)

continuous_summary["Variable"] = (
    continuous_summary["Variable"].map(NAMES)
)

continuous_summary.to_csv(
    OUTPUT / "continuous_summary.csv",
    index=False
)

# Continuous statistics separately for decline/no decline
continuous_by_target = (
    df.groupby(TARGET)[CONTINUOUS]
    .agg(["count", "mean", "median", "std"])
)

continuous_by_target.to_csv(
    OUTPUT / "continuous_by_target.csv"
)


# ============================================================
# 5. CATEGORICAL DESCRIPTIVE AND BIVARIATE TABLES
# ============================================================

categorical_results = []

for variable in CATEGORICAL:

    labels = df[variable].map(CATEGORIES[variable])

    temp = pd.DataFrame({
        "Category": labels.fillna("Other / Missing"),
        "Decline": df[TARGET]
    })

    summary = (
        temp.groupby("Category", dropna=False)
        .agg(
            Respondents=("Decline", "size"),
            Positive_Cases=("Decline", "sum"),
            Decline_Rate=("Decline", "mean")
        )
        .reset_index()
    )

    summary["Variable"] = NAMES[variable]
    summary["Original_Variable"] = variable

    summary["Decline_Rate"] *= 100

    summary["Percentage_of_Sample"] = (
        summary["Respondents"] / len(df) * 100
    )

    categorical_results.append(summary)

categorical_table = pd.concat(
    categorical_results,
    ignore_index=True
)

categorical_table.to_csv(
    OUTPUT / "categorical_bivariate_summary.csv",
    index=False
)


# ============================================================
# 6. GROUP CONTINUOUS VARIABLES FOR BIVARIATE EDA
# ============================================================

groups = {}

groups["Age Group"] = pd.cut(
    df["_AGE80"],
    bins=[44, 54, 64, 74, np.inf],
    labels=["45–54", "55–64", "65–74", "75+"]
)

groups["Sleep Duration"] = pd.cut(
    df["sleep_hours"],
    bins=[-np.inf, 5, 6, 8, 9, np.inf],
    labels=["≤5h", ">5–6h", ">6–8h", ">8–9h", ">9h"]
)

groups["BMI Category"] = pd.cut(
    df["bmi"],
    bins=[-np.inf, 18.5, 25, 30, np.inf],
    labels=[
        "Underweight",
        "Normal",
        "Overweight",
        "Obesity"
    ],
    right=False
)

groups["Poor Mental Health Days"] = pd.cut(
    df["mhealth"],
    bins=[-0.1, 0, 7, 14, 30],
    labels=["0", "1–7", "8–14", "15–30"]
)

groups["Drinks per Week"] = pd.cut(
    df["DRNKWK3_clean"],
    bins=[-0.1, 0, 7, 14, np.inf],
    labels=["0", ">0–7", ">7–14", ">14"]
)


# ============================================================
# 7. GROUPED CONTINUOUS BIVARIATE TABLES
# ============================================================

group_results = []

for name, categories in groups.items():

    temp = pd.DataFrame({
        "Category": categories,
        "Decline": df[TARGET]
    })

    # Explicitly retain missing values in the table
    temp["Category"] = (
        temp["Category"]
        .astype("object")
        .fillna("Missing")
    )

    summary = (
        temp.groupby("Category", dropna=False)
        .agg(
            Respondents=("Decline", "size"),
            Positive_Cases=("Decline", "sum"),
            Decline_Rate=("Decline", "mean")
        )
        .reset_index()
    )

    summary["Variable"] = name
    summary["Decline_Rate"] *= 100

    group_results.append(summary)

grouped_table = pd.concat(
    group_results,
    ignore_index=True
)

grouped_table.to_csv(
    OUTPUT / "continuous_group_bivariate_summary.csv",
    index=False
)


# ============================================================
# 8. REUSABLE CHART FUNCTION
# ============================================================

def plot_decline_rates(data, title, ax):

    data = data.copy()

    bars = ax.bar(
        data["Category"].astype(str),
        data["Decline_Rate"],
        color="#397EAB"
    )

    for bar, (_, row) in zip(bars, data.iterrows()):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{row["Decline_Rate"]:.1f}%\n'
            f'(n={row["Respondents"]:,})',
            ha="center",
            va="bottom",
            fontsize=8
        )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Reporting Cognitive Decline (%)")

    ax.set_ylim(
        0,
        max(data["Decline_Rate"].max() * 1.35, 5)
    )

    ax.tick_params(
        axis="x",
        labelrotation=25,
        labelsize=9
    )


# ============================================================
# 9. FIGURE 1 — DEMOGRAPHIC ASSOCIATIONS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

age_data = grouped_table[
    (grouped_table["Variable"] == "Age Group") &
    (grouped_table["Category"] != "Missing")
]

sex_data = categorical_table[
    categorical_table["Variable"] == "Sex"
]

plot_decline_rates(
    age_data,
    "Cognitive Decline by Age",
    axes[0]
)

plot_decline_rates(
    sex_data,
    "Cognitive Decline by Sex",
    axes[1]
)

plt.tight_layout()

fig.savefig(
    OUTPUT / "fig_eda_demographics.png",
    dpi=180,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT / "fig_eda_demographics.svg",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 10. FIGURE 2 — CLINICAL FACTORS
# ============================================================

clinical = [
    "Heart Attack History",
    "Stroke History",
    "High Blood Pressure",
    "High Cholesterol",
    "Depression History",
    "Diabetes"
]

fig, axes = plt.subplots(3, 2, figsize=(14, 14))

for ax, variable in zip(axes.flat, clinical):

    subset = categorical_table[
        (categorical_table["Variable"] == variable) &
        (~categorical_table["Category"].isin([
            "Unknown",
            "Other / Missing"
        ]))
    ]

    plot_decline_rates(
        subset,
        variable,
        ax
    )

plt.tight_layout()

fig.savefig(
    OUTPUT / "fig_eda_clinical.png",
    dpi=180,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT / "fig_eda_clinical.svg",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 11. FIGURE 3 — BEHAVIOURAL FACTORS
# ============================================================

behavioural = [
    "Smoking Status",
    "Physical Activity"
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, variable in zip(axes.flat, behavioural):

    subset = categorical_table[
        (categorical_table["Variable"] == variable) &
        (~categorical_table["Category"].isin([
            "Unknown",
            "Other / Missing"
        ]))
    ]

    plot_decline_rates(
        subset,
        variable,
        ax
    )

plt.tight_layout()

fig.savefig(
    OUTPUT / "fig_eda_behaviours.png",
    dpi=180,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT / "fig_eda_behaviours.svg",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 12. FIGURE 4 — CONTINUOUS HEALTH FACTORS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

variables_to_plot = [
    "Sleep Duration",
    "BMI Category",
    "Poor Mental Health Days",
    "Drinks per Week"
]

for ax, variable in zip(axes.flat, variables_to_plot):

    subset = grouped_table[
        (grouped_table["Variable"] == variable) &
        (grouped_table["Category"] != "Missing")
    ]

    plot_decline_rates(
        subset,
        variable,
        ax
    )

plt.tight_layout()

fig.savefig(
    OUTPUT / "fig_eda_continuous_groups.png",
    dpi=180,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT / "fig_eda_continuous_groups.svg",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 13. FIGURE 5 — TARGET DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(
    ["No Decline", "Decline"],
    [
        (df[TARGET] == 0).sum(),
        (df[TARGET] == 1).sum()
    ],
    color=["#7D96AF", "#397EAB"]
)

ax.bar_label(bars, fmt="%.0f", padding=4)

ax.set_ylabel("Respondents")
ax.set_title("Target Class Distribution")

plt.tight_layout()

fig.savefig(
    OUTPUT / "fig_eda_target.png",
    dpi=180,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT / "fig_eda_target.svg",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 14. FIGURE 6 — MISSINGNESS
# ============================================================

missing = []

for variable in FEATURES:

    if variable in CONTINUOUS:
        count = df[variable].isna().sum()
    else:
        count = (df[variable] == 999).sum()

    missing.append({
        "Variable": NAMES[variable],
        "Missing_Count": count,
        "Missing_Percentage": count / len(df) * 100
    })

missing_df = pd.DataFrame(missing).sort_values(
    "Missing_Percentage"
)

missing_df.to_csv(
    OUTPUT / "missingness_summary.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    missing_df["Variable"],
    missing_df["Missing_Percentage"],
    color="#397EAB"
)

ax.set_xlabel("Missing or Unknown Responses (%)")
ax.set_title("Predictor Missingness")

plt.tight_layout()

fig.savefig(
    OUTPUT / "fig_eda_missingness.png",
    dpi=180,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT / "fig_eda_missingness.svg",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 15. PRINT KEY FINDINGS FOR REVIEW
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL BIVARIATE SUMMARY")
print("=" * 60)

print(
    categorical_table[
        [
            "Variable",
            "Category",
            "Respondents",
            "Positive_Cases",
            "Decline_Rate"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("GROUPED CONTINUOUS BIVARIATE SUMMARY")
print("=" * 60)

print(
    grouped_table[
        [
            "Variable",
            "Category",
            "Respondents",
            "Positive_Cases",
            "Decline_Rate"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print(f"Results saved to: {OUTPUT}")
print(f"Overall decline rate: {df[TARGET].mean()*100:.2f}%")
