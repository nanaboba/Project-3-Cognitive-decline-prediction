"""Unified exploratory/reproducibility evaluation of five fixed models.
Training-only out-of-fold threshold selection (target recall 70%), followed
by full-training fit and scoring on the existing, previously used test set.
No hyperparameter tuning. Re-running is NOT a new independent final test.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score, classification_report,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier

from build_preprocessor import build_preprocessor, FEATURES, TARGET
from build_spline_model import build_all_spline_model

OUTPUT = PROJECT_ROOT / "outputs" / "final_evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)



CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

TARGET_RECALL = 0.70


def threshold_for_recall(y_true, proba, target_recall):
    """Find the threshold giving at least the requested recall."""

    precision, recall, thresholds = precision_recall_curve(
        y_true, proba
    )

    precision = precision[:-1]
    recall = recall[:-1]

    eligible = recall >= target_recall

    if not eligible.any():
        raise ValueError("Requested recall cannot be achieved.")

    idx = np.argmax(
        np.where(eligible, precision, -1)
    )

    return thresholds[idx]




def score(name, model, X_tr, y_tr, X_te, y_te):

    # Obtain out-of-fold probabilities using training data only
    oof_proba = cross_val_predict(
        model,
        X_tr,
        y_tr,
        cv=CV,
        method="predict_proba"
    )[:, 1]

    # Choose classification threshold on training data
    threshold = threshold_for_recall(
        y_tr, oof_proba, TARGET_RECALL
    )

    # Fit final model on all training observations
    model.fit(X_tr, y_tr)

    # Generate test-set probabilities
    proba = model.predict_proba(X_te)[:, 1]

    # Convert probabilities into binary predictions
    pred = (proba >= threshold).astype(int)

    # Calculate final test metrics
    metrics = {
        "model": name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "recall": recall_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "f1": f1_score(y_te, pred)
    }

    return metrics, proba, pred


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "cogdec_train.csv")
    test_path  = sys.argv[2] if len(sys.argv) > 2 else str(DATA_DIR / "cogdec_test.csv")

    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    X_tr, y_tr = tr[FEATURES], tr[TARGET].astype(int)
    X_te, y_te = te[FEATURES], te[TARGET].astype(int)

    print(f"Train: {len(tr):,} rows | Test: {len(te):,} rows "
          f"(positive rate {y_te.mean()*100:.2f}%)\n")

    # --- Model 1: Logistic Regression (tuned) --------------------
    logit = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ])

    # --- Model 2: gradient boosting (best performer, tuned) --------------
    gb = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42)),
    ])


    # Model 3: Random Forest
    rf = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # --- Model 4: Neural Network (tuned) ---------------------

    mlp = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.01,
            activation="relu",
            solver="adam",
            learning_rate_init=0.01,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42
        ))
    ])        
    
    spline_lr = build_all_spline_model()

    models = {
        "Logistic Regression": logit,
        "Spline Logistic Regression": spline_lr,
        "Gradient Boosting": gb,
        "Random Forest": rf,
        "Neural Network": mlp
    }

    rows = []
    test_proba = {}
    test_pred = {}
    thresholds = {}

    for name, model in models.items():

        metrics, proba, pred = score(
            name, model, X_tr, y_tr, X_te, y_te
        )

        rows.append(metrics)
        test_proba[name] = proba
        test_pred[name] = pred
        thresholds[name] = metrics["threshold"]


    res = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("UNIFIED TEST-SET PERFORMANCE (EXISTING TEST SET):\n")
    print(res.to_string(index=False))

    
    # Preserve aligned probabilities and predictions for later reporting.
    prediction_table = pd.DataFrame({"Actual": y_te.to_numpy()})
    for name in models:
        prediction_table[f"{name}__probability"] = test_proba[name]
        prediction_table[f"{name}__prediction"] = test_pred[name]
    prediction_table.to_csv(OUTPUT / "all_final_test_predictions.csv", index=False)

    # ============================================================
    # DETAILED CLASSIFICATION REPORTS
    # ============================================================

    for name, pred in test_pred.items():

        print(f"\n{name} -- detailed classification report:")

        print(
            classification_report(
                y_te,
                pred,
                target_names=["No decline", "Decline"]
            )
        )

    # ============================================================
    # ERROR ANALYSIS BY RESPONDENT SEGMENT
    # ============================================================

    # Use Spline Logistic Regression test predictions
    errors = X_te.copy()

    errors["Actual"] = y_te.to_numpy()
    errors["Predicted"] = test_pred["Spline Logistic Regression"]

    # Age groups
    errors["Age Group"] = pd.cut(
        errors["_AGE80"],
        bins=[44, 54, 64, 74, float("inf")],
        labels=["45–54", "55–64", "65–74", "75+"]
    )

    # Sex
    errors["Sex"] = errors["SEXVAR"].map({
        1: "Male",
        2: "Female"
    })

    # Depression history
    errors["Depression History"] = errors["ADDEPEV3"].map({
        1: "Yes",
        2: "No",
        999: "Unknown"
    })

    # Stroke history
    errors["Stroke History"] = errors["CVDSTRK3"].map({
        1: "Yes",
        2: "No",
        999: "Unknown"
    })

    # Physical activity
    errors["Physical Activity"] = errors["EXERANY2"].map({
        1: "Yes",
        2: "No",
        999: "Unknown"
    })

    # Calculate metrics for each segment
    results = []

    for segment in [
        "Age Group",
        "Sex",
        "Depression History",
        "Stroke History",
        "Physical Activity"
    ]:

        for category, group in errors.groupby(
            segment, observed=True
        ):

            results.append({
    "Segment": segment,
    "Category": category,
    "Respondents": len(group),

    # Actual respondents reporting cognitive decline
    "Positive Cases": int(group["Actual"].sum()),

    # Correctly identified positive cases
    "True Positives": int(
        ((group["Actual"] == 1) &
         (group["Predicted"] == 1)).sum()
    ),

    "Recall": recall_score(
        group["Actual"],
        group["Predicted"],
        zero_division=0
    ),

    "Precision": precision_score(
        group["Actual"],
        group["Predicted"],
        zero_division=0
    )
})

    segment_results = pd.DataFrame(results)

    print("\nSpline Logistic Regression — Error Analysis by Segment")
    print(segment_results.to_string(index=False))

    segment_results.to_csv(
    OUTPUT / "spline_lr_segment_error_analysis.csv",
    index=False
    )
    

    # ============================================================
    # VISUALISATION: RECALL BY SEGMENT
    # ============================================================

    fig, ax = plt.subplots(figsize=(10, 6))

    for segment in segment_results["Segment"].unique():

        subset = segment_results[
            segment_results["Segment"] == segment
        ]

        labels = [
            f"{segment}: {category}"
            for category in subset["Category"]
        ]

        ax.barh(labels, subset["Recall"])

    ax.axvline(
        0.70,
        color="red",
        linestyle="--",
        label="Target recall: 70%"
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_title("Spline Logistic Regression — Recall by Respondent Segment")
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()

    fig.savefig(
        OUTPUT / "fig_spline_lr_segment_recall.png",
        dpi=180,
        bbox_inches="tight"
    )


    fig.savefig(
        OUTPUT / "fig_spline_lr_segment_recall.svg",
        bbox_inches="tight"
    )

    plt.close(fig)

    # ============================================================
    # SAVE FINAL MODEL RESULTS
    # ============================================================

    res.to_csv(OUTPUT / "all_final_model_results.csv", index=False)
    print("Saved -> all_final_model_results.csv")



    # ============================================================
    # FINAL TEST-SET VISUALISATIONS
    # ============================================================

    colours = {
        "Logistic Regression": "#1f4e9c",
        "Gradient Boosting": "#e07b00",
        "Random Forest": "#2e8b3d",
        "Spline Logistic Regression": "#7054a3",
        "Neural Network": "#b75c91"
    }

    # ------------------------------------------------------------
    # FIGURE 1: ROC CURVES
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    for name, proba in test_proba.items():

        fpr, tpr, _ = roc_curve(y_te, proba)
        auc_value = roc_auc_score(y_te, proba)

        ax.plot(
            fpr, tpr,
            color=colours[name],
            linewidth=2,
            label=f"{name} (AUC = {auc_value:.3f})"
        )

    ax.plot(
        [0, 1], [0, 1],
        "k--",
        label="Random classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — Cognitive Decline")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT / "fig1_roc_curves.pdf")
    plt.savefig(OUTPUT / "fig1_roc_curves.png", dpi=150)
    plt.close()


    # ------------------------------------------------------------
    # FIGURE 2: PRECISION–RECALL CURVES
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    for name, proba in test_proba.items():

        precision, recall, _ = precision_recall_curve(
            y_te, proba
        )

        ap = average_precision_score(y_te, proba)

        ax.plot(
            recall, precision,
            color=colours[name],
            linewidth=2,
            label=f"{name} (AP = {ap:.3f})"
        )

    ax.axhline(
        y_te.mean(),
        color="grey",
        linestyle="--",
        label="Baseline prevalence"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT / "fig2_pr_curves.pdf")
    plt.savefig(OUTPUT / "fig2_pr_curves.png", dpi=150)
    plt.close()


    # ------------------------------------------------------------
    # FIGURE 3: CONFUSION MATRICES
    # ------------------------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (name, pred) in zip(axes, test_pred.items()):

        cm = confusion_matrix(y_te, pred)

        ax.imshow(cm, cmap="Blues")

        ax.set_title(
            f"{name}\nThreshold = {thresholds[name]:.3f}"
        )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["No Decline", "Decline"])
        ax.set_yticklabels(["No Decline", "Decline"])

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12
                )

    for ax in axes[len(test_pred):]:
        ax.axis("off")

    fig.suptitle(
        "Confusion Matrices — Target Recall 70%"
    )

    plt.tight_layout()
    plt.savefig(OUTPUT / "fig3_confusion_matrices.pdf")
    plt.savefig(OUTPUT / "fig3_confusion_matrices.png", dpi=150)
    plt.close()

    print("\nSaved final test visualisations.")



if __name__ == "__main__":
    main()