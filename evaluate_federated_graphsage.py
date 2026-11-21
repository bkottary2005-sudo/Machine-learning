import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

DATA_PATH = "Data/processed/graphsage_data.pt"
MODEL_PATH = "results/models/federated_graphsage_model.pth"
METRICS_DIR = "results/metrics"
PLOTS_DIR = "results/plots"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


class GraphSAGE(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.classifier(x)

        return x.squeeze(-1)


print("=" * 70)
print("FEDERATED GRAPHSAGE EVALUATION")
print("=" * 70)

# Load graph data
data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)

# Load model
model = GraphSAGE(
    input_dim=data.x.shape[1],
    hidden_dim=64
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)

model.eval()

print("\nModel and graph loaded successfully")

# Predictions
with torch.no_grad():

    logits = model(data.x, data.edge_index)

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= 0.5).long()

# Test set
test_mask = data.test_mask

y_true = data.y[test_mask].numpy()
y_pred = predictions[test_mask].numpy()
y_prob = probabilities[test_mask].numpy()

# Metrics
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

# Save metrics
metrics = {
    "Model": "Federated GraphSAGE",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc,
    "True_Negatives": int(cm[0, 0]),
    "False_Positives": int(cm[0, 1]),
    "False_Negatives": int(cm[1, 0]),
    "True_Positives": int(cm[1, 1])
}

metrics_df = pd.DataFrame([metrics])

metrics_path = os.path.join(
    METRICS_DIR,
    "federated_graphsage_metrics.csv"
)

metrics_df.to_csv(metrics_path, index=False)

# Save predictions
predictions_df = pd.DataFrame({
    "Actual": y_true,
    "Predicted": y_pred,
    "Fraud_Probability": y_prob
})

predictions_path = os.path.join(
    METRICS_DIR,
    "federated_graphsage_predictions.csv"
)

predictions_df.to_csv(predictions_path, index=False)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Federated GraphSAGE (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Federated GraphSAGE ROC Curve")

plt.legend()
plt.grid(True)

roc_path = os.path.join(
    PLOTS_DIR,
    "federated_graphsage_roc_curve.png"
)

plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()

# Confusion Matrix Plot
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Federated GraphSAGE Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["Legitimate", "Fraud"])
plt.yticks([0, 1], ["Legitimate", "Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

cm_path = os.path.join(
    PLOTS_DIR,
    "federated_graphsage_confusion_matrix.png"
)

plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nFiles saved:")
print(metrics_path)
print(predictions_path)
print(roc_path)
print(cm_path)

print("\nFEDERATED GRAPHSAGE EVALUATION COMPLETED!")