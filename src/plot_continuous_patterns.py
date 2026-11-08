
"""
plot_continuous_patterns.py
---------------------------
Bivariate EDA of five continuous predictors:
    Age
    Sleep duration
    Poor mental-health days
    BMI
    Alcohol consumption

Shows observed percentages reporting subjective cognitive
decline across predictor groups.

Uses TRAINING DATA ONLY.

Charts:
    - No error bars
    - No overall prevalence reference line
    - Individual titles and axis labels
    - Combined figure and five separate figures

No predictive models are fitted.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter
from statsmodels.stats.proportion import proportion_confint


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"

OUTPUT = ROOT / "outputs" / "continuous_patterns"
OUTPUT.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "cogdec_train.csv"

TARGET = "CIMEMLO1"

MIN_BIN_N = 100

LINE_COLOUR = "#267DA8"


# ============================================================
# 2. VARIABLES AND BIN DEFINITIONS
# ============================================================

VARIABLES = {

    "_AGE80": {
        "title": "Age and Reported Cognitive Decline",
        "short_title": "Age",
        "xlabel": "Age (years; 80 = 80 or older)",
        "filename": "age",
        "bins": [
            44.5, 49.5, 54.5, 59.5,
            64.5, 69.5, 74.5, 79.5, 80.5
        ]
    },

    "sleep_hours": {
        "title": "Sleep Duration and Reported Cognitive Decline",
        "short_title": "Sleep Duration",
        "xlabel": "Sleep duration (hours per 24 hours)",
        "filename": "sleep",
        "bins": [0, 4, 5, 6, 7, 8, 9, 10, 16]
    },

    "mhealth": {
        "title": "Poor Mental-Health Days and Reported Cognitive Decline",
        "short_title": "Poor Mental-Health Days",
        "xlabel": "Poor mental-health days (past 30 days)",
        "filename": "mental_health",
        "bins": [
            -0.5, 0.5, 2.5, 5.5,
            10.5, 14.5, 20.5, 29.5, 30.5
        ]
    },

    "bmi": {
        "title": "BMI and Reported Cognitive Decline",
        "short_title": "Body Mass Index",
        "xlabel": "Body mass index (kg/m²)",
        "filename": "bmi",
        "bins": [10, 20, 25, 30, 35, 40, 70]
    },

    "DRNKWK3_clean": {
        "title": "Alcohol Consumption and Reported Cognitive Decline",
        "short_title": "Alcohol Consumption",
        "xlabel": "Drinks per week",
        "filename": "alcohol",
        "bins": [
            -0.5, 0.5, 3.5, 7.5,
            14.5, 28.5, 70.5
        ]
    }
}


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

required_columns = [TARGET] + list(VARIABLES.keys())

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

df = df[df[TARGET].isin([0, 1])].copy()

print(f"Training respondents: {len(df):,}")
print(f"Overall positive rate: {df[TARGET].mean():.2%}")


# ============================================================
# 4. CALCULATE OBSERVED RATES
# ============================================================

all_summaries = []
plot_data = {}

for variable, config in VARIABLES.items():

    d = df[[variable, TARGET]].dropna().copy()

    d["bin"] = pd.cut(
        d[variable],
        bins=config["bins"],
        include_lowest=True
    )

    d = d.dropna(subset=["bin"])

    grouped = d.groupby(
        "bin",
        observed=True
    )

    summary = grouped[TARGET].agg(
        positive_cases="sum",
        respondents="count"
    )

    summary["x_position"] = grouped[
        variable
    ].median()

    summary["decline_rate"] = (
        summary["positive_cases"]
        / summary["respondents"]
    )

    # Retain confidence intervals in the CSV only.
    # They are NOT displayed on the figures.
    ci_low, ci_high = proportion_confint(
        count=summary["positive_cases"],
        nobs=summary["respondents"],
        alpha=0.05,
        method="wilson"
    )

    summary["ci_low"] = ci_low
    summary["ci_high"] = ci_high

    summary["displayed"] = (
        summary["respondents"] >= MIN_BIN_N
    )

    # Save all bins, including those not displayed.
    for interval, row in summary.iterrows():

        all_summaries.append({
            "Variable": variable,
            "Feature": config["short_title"],
            "Bin": str(interval),
            "Respondents": int(row["respondents"]),
            "Positive_Cases": int(row["positive_cases"]),
            "Observed_Median": row["x_position"],
            "Decline_Rate": row["decline_rate"],
            "CI_Low": row["ci_low"],
            "CI_High": row["ci_high"],
            "Displayed": bool(row["displayed"])
        })

    display = summary[
        summary["displayed"]
    ].sort_values("x_position").copy()

    plot_data[variable] = {
        "data": display,
        "n": len(d),
        "config": config
    }


# ============================================================
# 5. SHARED PLOTTING FUNCTION
# ============================================================

def draw_pattern(ax, variable, individual=False):

    item = plot_data[variable]

    data = item["data"]
    config = item["config"]

    if data.empty:
        ax.text(
            0.5, 0.5,
            "No groups meet the minimum sample size",
            ha="center",
            va="center",
            transform=ax.transAxes
        )
        return

    x = data["x_position"].to_numpy()
    y = data["decline_rate"].to_numpy()

    ax.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        color=LINE_COLOUR,
        markersize=7 if individual else 5,
        linewidth=2.5 if individual else 2
    )

    ax.set_title(
        config["title"] if individual
        else config["short_title"],
        fontsize=15 if individual else 12,
        fontweight="bold",
        pad=12
    )

    ax.set_xlabel(
        config["xlabel"],
        fontsize=11 if individual else 10
    )

    ax.set_ylabel(
        "Respondents reporting cognitive decline (%)",
        fontsize=11 if individual else 9
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    ax.grid(alpha=0.2)

    ax.set_axisbelow(True)

    # Each chart carries its own sample size.
    ax.text(
        0.98,
        0.97,
        f"n = {item['n']:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10 if individual else 9,
        color="#666666"
    )


# ============================================================
# 6. SAVE FIVE INDIVIDUAL CHARTS
# ============================================================

for variable, config in VARIABLES.items():

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    draw_pattern(
        ax,
        variable,
        individual=True
    )

    fig.text(
        0.5,
        0.015,
        (
            "Training data only | Unadjusted observed rates | "
            f"Groups with fewer than {MIN_BIN_N} "
            "respondents omitted"
        ),
        ha="center",
        fontsize=9,
        color="#666666"
    )

    plt.tight_layout(
        rect=[0, 0.06, 1, 1]
    )

    filename = (
        f"fig_eda_{config['filename']}"
    )

    fig.savefig(
        OUTPUT / f"{filename}.png",
        dpi=180,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTPUT / f"{filename}.svg",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {filename}.png / .svg")


# ============================================================
# 7. SAVE COMBINED FIVE-PANEL CHART
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 9),
    sharey=True
)

axes = axes.flatten()

for ax, variable in zip(
    axes,
    VARIABLES.keys()
):
    draw_pattern(
        ax,
        variable,
        individual=False
    )

# Sixth panel: explain the modelling implication.
axes[-1].axis("off")

axes[-1].text(
    0.08,
    0.90,
    (
        "WHY INVESTIGATE SPLINES?\n\n"
        "Sleep and BMI:\n"
        "Possible U-shaped patterns.\n\n"
        "Alcohol:\n"
        "Possible non-monotonic relationship.\n\n"
        "Mental-health days:\n"
        "Increasing, potentially nonlinear gradient.\n\n"
        "Age:\n"
        "Steeper increase at older ages."
    ),
    transform=axes[-1].transAxes,
    ha="left",
    va="top",
    fontsize=12,
    linespacing=1.4,
    color="#17365D"
)

fig.suptitle(
    "EDA — Continuous Predictors and Cognitive Decline",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

fig.text(
    0.5,
    0.025,
    (
        "Training data only (n = 58,758). "
        "Unadjusted observed decline rates. "
        f"Groups with fewer than {MIN_BIN_N} "
        "respondents omitted. "
        "Associations are not causal."
    ),
    ha="center",
    fontsize=9,
    color="#555555"
)

plt.tight_layout(
    rect=[0, 0.06, 1, 0.94]
)

fig.savefig(
    OUTPUT / "fig_eda_continuous_combined.png",
    dpi=180,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT / "fig_eda_continuous_combined.svg",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 8. SAVE NUMERICAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(all_summaries)

summary_path = (
    OUTPUT / "continuous_train_bivariate_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print(f"\nSaved summary: {summary_path}")


# ============================================================
# 9. COMPLETION
# ============================================================

print("\nAll EDA figures generated successfully.")
print(f"Output directory: {OUTPUT}")
