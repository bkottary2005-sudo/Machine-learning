import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from graphsage_model import GraphSAGE


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "Data/processed/graphsage_data.pt"
MODEL_PATH = "results/models/graphsage_model.pth"

PLOT_DIR = "results/plots"
METRICS_DIR = "results/metrics"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD GRAPH DATA
# ============================================================

print("=" * 60)
print("GRAPHSAGE MODEL EVALUATION")
print("=" * 60)

print("\nLoading graph data...")

data = torch.load(
    DATA_PATH,
    map_location="cpu",
    weights_only=False
)

print("Graph data loaded successfully!")
print(f"Number of nodes: {data.x.shape[0]}")
print(f"Number of features: {data.x.shape[1]}")
print(f"Number of edges: {data.edge_index.shape[1]}")


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained GraphSAGE model...")

model = GraphSAGE(
    input_dim=data.x.shape[1],
    hidden_dim=64
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)

model.eval()

print("Model loaded successfully!")


# ============================================================
# 4. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

with torch.no_grad():

    logits = model(
        data.x,
        data.edge_index
    )

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.squeeze()

# Test data
test_mask = data.test_mask

y_true = data.y[test_mask].numpy()

y_prob = probabilities[test_mask].numpy()

# Classification threshold
threshold = 0.5

y_pred = (y_prob >= threshold).astype(int)

print("Predictions generated successfully!")


# ============================================================
# 5. CALCULATE METRICS
# ============================================================

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

fpr, tpr, roc_thresholds = roc_curve(
    y_true,
    y_prob
)

roc_auc = auc(fpr, tpr)

cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)


# ============================================================
# 7. ROC CURVE
# ============================================================

print("\nGenerating ROC curve...")

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"GraphSAGE (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("GraphSAGE ROC Curve")

plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "graphsage_roc_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("ROC curve saved!")


# ============================================================
# 8. PRECISION-RECALL CURVE
# ============================================================

print("\nGenerating Precision-Recall curve...")

pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    pr_recall,
    pr_precision
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("GraphSAGE Precision-Recall Curve")

plt.grid(True)

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "graphsage_precision_recall_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Precision-Recall curve saved!")


# ============================================================
# 9. CONFUSION MATRIX VISUALIZATION
# ============================================================

print("\nGenerating confusion matrix...")

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("GraphSAGE Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

plt.colorbar()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "graphsage_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix saved!")


# ============================================================
# 10. SAVE TEST PREDICTIONS
# ============================================================

print("\nSaving predictions...")

test_indices = torch.where(test_mask)[0].numpy()

predictions_df = pd.DataFrame({
    "node_index": test_indices,
    "actual_label": y_true,
    "predicted_label": y_pred,
    "fraud_probability": y_prob
})

predictions_df.to_csv(
    os.path.join(
        METRICS_DIR,
        "graphsage_test_predictions.csv"
    ),
    index=False
)

print("Test predictions saved!")


# ============================================================
# 11. SAVE EVALUATION METRICS
# ============================================================

metrics_df = pd.DataFrame([{
    "Model": "GraphSAGE",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc,
    "Test_Nodes": len(y_true),
    "Fraud_Nodes": int(np.sum(y_true)),
    "Legitimate_Nodes": int(np.sum(y_true == 0)),
    "Threshold": threshold
}])

metrics_df.to_csv(
    os.path.join(
        METRICS_DIR,
        "graphsage_evaluation_summary.csv"
    ),
    index=False
)

print("Evaluation metrics saved!")


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("GRAPHSAGE EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nMetrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nGenerated Files:")

print("\nPlots:")
print(f"  {PLOT_DIR}/graphsage_roc_curve.png")
print(f"  {PLOT_DIR}/graphsage_precision_recall_curve.png")
print(f"  {PLOT_DIR}/graphsage_confusion_matrix.png")

print("\nMetrics:")
print(f"  {METRICS_DIR}/graphsage_test_predictions.csv")
print(f"  {METRICS_DIR}/graphsage_evaluation_summary.csv")

print("\n" + "=" * 60)