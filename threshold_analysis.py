import torch
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. LOAD PREDICTIONS
# ==========================================

print("=" * 60)
print("GRAPHSAGE THRESHOLD ANALYSIS")
print("=" * 60)

predictions_path = "results/metrics/graphsage_test_predictions.csv"

df = pd.read_csv(predictions_path)

y_true = df["actual_label"].values
y_prob = df["fraud_probability"].values

print(f"\nLoaded {len(df)} test predictions")


# ==========================================
# 2. TEST DIFFERENT THRESHOLDS
# ==========================================

thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

results = []

print("\nTesting thresholds...\n")

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    })


# ==========================================
# 3. DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("=" * 90)
print("THRESHOLD COMPARISON")
print("=" * 90)

print(
    results_df[
        [
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "False_Positives",
            "False_Negatives"
        ]
    ].round(4).to_string(index=False)
)


# ==========================================
# 4. FIND BEST THRESHOLD
# ==========================================

best_f1_row = results_df.loc[
    results_df["F1_Score"].idxmax()
]

best_threshold = best_f1_row["Threshold"]

print("\n" + "=" * 60)
print("BEST THRESHOLD BASED ON F1 SCORE")
print("=" * 60)

print(f"Threshold : {best_threshold}")
print(f"Precision : {best_f1_row['Precision']:.4f}")
print(f"Recall    : {best_f1_row['Recall']:.4f}")
print(f"F1 Score  : {best_f1_row['F1_Score']:.4f}")
print(f"Accuracy  : {best_f1_row['Accuracy']:.4f}")


# ==========================================
# 5. SAVE RESULTS
# ==========================================

output_path = "results/metrics/threshold_analysis.csv"

results_df.to_csv(
    output_path,
    index=False
)

print(f"\nResults saved to: {output_path}")


# ==========================================
# 6. FRAUD DETECTION COMPARISON
# ==========================================

print("\n" + "=" * 60)
print("FRAUD DETECTION COMPARISON")
print("=" * 60)

for _, row in results_df.iterrows():

    print(
        f"Threshold {row['Threshold']:.1f} | "
        f"Fraud detected: {int(row['True_Positives'])} | "
        f"Missed: {int(row['False_Negatives'])} | "
        f"False alarms: {int(row['False_Positives'])} | "
        f"F1: {row['F1_Score']:.4f}"
    )


print("\nThreshold analysis completed successfully!")