"""
EDA for BRFSS Cognitive Decline Prediction Project
Target: CIMEMLO1 (1 = subjective cognitive decline, 0 = none)

Run:  python eda_cogdec.py /path/to/CogDec.csv /path/to/output_dir
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "CogDec.csv"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

NEG_COLOR = "#4C72B0"   # No decline
POS_COLOR = "#C44E52"   # Decline

df = pd.read_csv('data/CogDec.csv')

TARGET = "CIMEMLO1"

CONT = ["_AGE80", "sleep_hours", "DRNKWK3_clean", "bmi", "mhealth"]

# categorical features actually used in modelling (exclude dropped psychosocial)
CAT = ["SEXVAR", "CVDINFR4", "CVDSTRK3", "BPHIGH6", "TOLDHI3",
       "ADDEPEV3", "DIABETE4", "GENHLTH", "EXERANY2", "_SMOKER3"]

# Human-readable labels for category codes (for nicer axis ticks)
CAT_LABELS = {
    "SEXVAR":   {1: "Male", 2: "Female"},
    "CVDINFR4": {1: "Yes", 2: "No", 999: "Unknown"},
    "CVDSTRK3": {1: "Yes", 2: "No", 999: "Unknown"},
    "BPHIGH6":  {1: "Yes", 3: "No", 4: "Borderline", 999: "Unknown"},
    "TOLDHI3":  {1: "Yes", 2: "No", 999: "Unknown"},
    "ADDEPEV3": {1: "Yes", 2: "No", 999: "Unknown"},
    "DIABETE4": {1: "Yes", 3: "No", 4: "Borderline", 999: "Unknown"},
    "GENHLTH":  {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor", 999: "Unknown"},
    "EXERANY2": {1: "Yes", 2: "No", 999: "Unknown"},
    "_SMOKER3": {1: "Everyday", 2: "Someday", 3: "Former", 4: "Never", 999: "Unknown"},
}

CONT_LABELS = {
    "_AGE80": "Age (years, capped 80)",
    "sleep_hours": "Sleep hours / 24h",
    "DRNKWK3_clean": "Drinks per week",
    "bmi": "BMI",
    "mhealth": "Poor mental-health days (of 30)",
}

CAT_TITLES = {
    "SEXVAR":   "Sex",
    "CVDINFR4": "Heart attack (ever told)",
    "CVDSTRK3": "Stroke (ever told)",
    "BPHIGH6":  "High blood pressure",
    "TOLDHI3":  "High cholesterol",
    "ADDEPEV3": "Depressive disorder",
    "DIABETE4": "Diabetes",
    "GENHLTH":  "General health",
    "EXERANY2": "Any exercise (past 30d)",
    "_SMOKER3": "Smoking status",
}

overall_rate = df[TARGET].mean()
print(f"Rows: {len(df):,} | Overall positive rate: {overall_rate:.3%}")

# ---------------------------------------------------------------------------
# 1. TARGET DISTRIBUTION (class imbalance)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
counts = df[TARGET].value_counts().sort_index()
bars = ax.bar(["No decline (0)", "Decline (1)"], counts.values,
              color=[NEG_COLOR, POS_COLOR])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 600,
            f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Respondents")
ax.set_title("Target class distribution (Cognitive Decline)\nImbalanced ~21% positive -> use class_weight / stratify")
ax.set_ylim(0, counts.max() * 1.18)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_target_distribution.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. CONTINUOUS FEATURES BY CLASS (overlaid density + boxplot)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(CONT), 2, figsize=(11, 3.1 * len(CONT)))
summary_rows = []
for i, col in enumerate(CONT):
    neg = df.loc[df[TARGET] == 0, col].dropna()
    pos = df.loc[df[TARGET] == 1, col].dropna()

    # histogram (density) overlay
    axd = axes[i, 0]
    bins = np.histogram_bin_edges(pd.concat([neg, pos]), bins=30)
    axd.hist(neg, bins=bins, density=True, alpha=0.55, color=NEG_COLOR, label="No decline")
    axd.hist(pos, bins=bins, density=True, alpha=0.55, color=POS_COLOR, label="Decline")
    axd.axvline(neg.median(), color=NEG_COLOR, ls="--", lw=1)
    axd.axvline(pos.median(), color=POS_COLOR, ls="--", lw=1)
    axd.set_xlabel(CONT_LABELS[col]); axd.set_ylabel("Density")
    if i == 0:
        axd.legend(fontsize=8)
    axd.set_title(f"{CONT_LABELS[col]}: distribution by class")

    # boxplot
    axb = axes[i, 1]
    import matplotlib as _mpl
    _lblkw = {"tick_labels": ["No", "Decline"]} if tuple(int(x) for x in _mpl.__version__.split(".")[:2]) >= (3, 9) else {"labels": ["No", "Decline"]}
    bp = axb.boxplot([neg, pos], patch_artist=True, **_lblkw,
                     showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], [NEG_COLOR, POS_COLOR]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    axb.set_ylabel(CONT_LABELS[col])
    axb.set_title(f"{CONT_LABELS[col]}\nmedian No={neg.median():.1f} / Decline={pos.median():.1f}")

    summary_rows.append({
        "feature": CONT_LABELS[col],
        "median_No": round(neg.median(), 2),
        "median_Decline": round(pos.median(), 2),
        "mean_No": round(neg.mean(), 2),
        "mean_Decline": round(pos.mean(), 2),
    })
fig.suptitle("Continuous features: No-decline vs Decline", y=1.002, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_continuous_by_class.png"))
plt.close(fig)

cont_summary = pd.DataFrame(summary_rows)

# ---------------------------------------------------------------------------
# 3. CATEGORICAL FEATURES: positive rate by category
#    (horizontal bars, with overall-rate reference line)
# ---------------------------------------------------------------------------
ncol = 2
nrow = int(np.ceil(len(CAT) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.7 * nrow))
axes = axes.flatten()
cat_signal = []
for i, col in enumerate(CAT):
    ax = axes[i]
    grp = df.groupby(col)[TARGET].agg(["mean", "size"])
    labels = [CAT_LABELS.get(col, {}).get(int(c), str(int(c))) for c in grp.index]
    bars = ax.barh(labels, grp["mean"].values, color="#55A868", alpha=0.85)
    ax.axvline(overall_rate, color="black", ls="--", lw=1)
    for b, (m, n) in zip(bars, grp[["mean", "size"]].values):
        ax.text(m + 0.005, b.get_y() + b.get_height()/2,
                f"{m:.0%} (n={int(n):,})", va="center", fontsize=7.5)
    ax.set_xlim(0, max(grp["mean"].max() * 1.35, overall_rate * 1.4))
    ax.set_title(CAT_TITLES.get(col, col), fontsize=10)
    ax.set_xlabel("P(decline)")
    cat_signal.append({
        "feature": CAT_TITLES.get(col, col),
        "rate_spread": round(grp["mean"].max() - grp["mean"].min(), 3),
        "max_rate": round(grp["mean"].max(), 3),
        "min_rate": round(grp["mean"].min(), 3),
    })
for j in range(len(CAT), len(axes)):
    axes[j].axis("off")
fig.suptitle(f"Decline rate by category (dashed = overall {overall_rate:.1%})",
             y=1.003, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_categorical_positive_rate.png"))
plt.close(fig)

cat_summary = pd.DataFrame(cat_signal).sort_values("rate_spread", ascending=False)

# ---------------------------------------------------------------------------
# 4. MISSINGNESS (all columns, highlight modelling features)
# ---------------------------------------------------------------------------
miss = (df.isna().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0]

ALL_LABELS = {**CONT_LABELS, **CAT_TITLES,
              "SDLONELY": "Felt lonely (freq)",
              "EMTSUPRT": "Emotional support (freq)",
              "LSATISFY": "Life satisfaction",
              # ...add any others you care about
}
miss.index = [ALL_LABELS.get(c, c) for c in miss.index]

fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(miss))))
colors = ["#C44E52" if c in (CONT + CAT) else "#BBBBBB" for c in miss.index]
ax.barh(miss.index[::-1], miss.values[::-1], color=colors[::-1])
for i, (name, v) in enumerate(zip(miss.index[::-1], miss.values[::-1])):
    ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("% missing")
ax.set_title("Missingness by column\n(red = used in model -> median impute + was-missing flag; grey = dropped)")
ax.set_xlim(0, miss.max() * 1.15)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_missingness.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. CORRELATION / COLLINEARITY among continuous features
# ---------------------------------------------------------------------------
corr = df[CONT].corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
cont_tick_labels = [CONT_LABELS[c] for c in CONT]
ax.set_xticks(range(len(CONT))); ax.set_xticklabels(cont_tick_labels, rotation=45, ha="right")
ax.set_yticks(range(len(CONT))); ax.set_yticklabels(cont_tick_labels)
for i in range(len(CONT)):
    for j in range(len(CONT)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black", fontsize=9)
fig.colorbar(im, fraction=0.046, pad=0.04)
ax.set_title("Continuous feature correlation (Pearson)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_continuous_correlation.png"))
plt.close(fig)

# 5b. Collinearity check for the flagged BPHIGH6 x TOLDHI3 pair (Cramer's V)
def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    chi2 = _chi2(tab)
    n = tab.values.sum()
    r, k = tab.shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1)))

def _chi2(tab):
    obs = tab.values.astype(float)
    row = obs.sum(1, keepdims=True); col = obs.sum(0, keepdims=True)
    exp = row @ col / obs.sum()
    return ((obs - exp) ** 2 / exp).sum()

pairs = [("BPHIGH6", "TOLDHI3"), ("DIABETE4", "BPHIGH6"), ("GENHLTH", "ADDEPEV3")]
cv_rows = [{"pair": f"{CAT_TITLES.get(a, a)} x {CAT_TITLES.get(b, b)}",
            "cramers_v": round(cramers_v(df[a], df[b]), 3)}
           for a, b in pairs]
cv_df = pd.DataFrame(cv_rows)

# ---------------------------------------------------------------------------
# 6. OUTLIER INSPECTION: drinks per week
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
d = df["DRNKWK3_clean"].dropna()
axes[0].hist(d, bins=50, color="#8172B3")
axes[0].set_title(f"Drinks/week (max={d.max():.0f}, {(d>14).mean():.1%} > 14/wk)")
axes[0].set_xlabel("Drinks per week"); axes[0].set_ylabel("Count")
axes[0].set_yscale("log")
# zoom 0-20
axes[1].hist(d[d <= 20], bins=20, color="#8172B3")
axes[1].set_title("Drinks/week (zoom 0-20)")
axes[1].set_xlabel("Drinks per week"); axes[1].set_ylabel("Count")
fig.suptitle("Outlier inspection: weekly alcohol (right-skewed, already capped at 70)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06_drinks_outliers.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Save a text summary of key numbers
# ---------------------------------------------------------------------------


with open(os.path.join(OUT_DIR, "eda_summary.txt"), "w") as f:
    f.write("BRFSS COGNITIVE DECLINE — EDA KEY NUMBERS\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Rows: {len(df):,}\nOverall positive rate: {overall_rate:.3%}\n\n")
    f.write("CONTINUOUS FEATURES — median/mean by class\n")
    f.write(cont_summary.to_string(index=False) + "\n\n")
    f.write("CATEGORICAL FEATURES — decline-rate spread (signal strength)\n")
    f.write(cat_summary.to_string(index=False) + "\n\n")
    f.write("COLLINEARITY (Cramer's V) for flagged pairs\n")
    f.write(cv_df.to_string(index=False) + "\n\n")
    f.write("MISSINGNESS (%)\n")
    f.write(miss.round(2).to_string() + "\n")

print("\n--- CONTINUOUS summary ---")
print(cont_summary.to_string(index=False))
print("\n--- CATEGORICAL signal (rate spread) ---")
print(cat_summary.to_string(index=False))
print("\n--- COLLINEARITY (Cramer's V) ---")
print(cv_df.to_string(index=False))
print("\nSaved plots + eda_summary.txt to:", OUT_DIR)
