import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from graphsage_model import GraphSAGE


# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
LEARNING_RATE = 0.01
PATIENCE = 15

print("Device:", DEVICE)


# ==========================================
# 2. LOAD GRAPH DATA
# ==========================================

data = torch.load(
    "Data/processed/graphsage_data.pt",
    weights_only=False
).to(DEVICE)

print("Graph data loaded")
print(data)


# ==========================================
# 3. INITIALIZE MODEL
# ==========================================

model = GraphSAGE(
    input_dim=data.x.shape[1],
    hidden_dim=64
).to(DEVICE)

# Handle class imbalance
num_fraud = data.y[data.train_mask].sum()
num_legitimate = data.train_mask.sum() - num_fraud

pos_weight = num_legitimate / num_fraud

criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=5e-4
)

print("\nModel initialized")
print("Fraud accounts in training:", int(num_fraud.item()))
print("Legitimate accounts in training:", int(num_legitimate.item()))
print("Positive class weight:", pos_weight.item())


# ==========================================
# 4. TRAINING
# ==========================================

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):

    model.train()

    optimizer.zero_grad()

    output = model(data.x, data.edge_index)

    train_loss = criterion(
        output[data.train_mask],
        data.y[data.train_mask]
    )

    train_loss.backward()
    optimizer.step()


    # Validation
    model.eval()

    with torch.no_grad():

        val_output = model(data.x, data.edge_index)

        val_loss = criterion(
            val_output[data.val_mask],
            data.y[data.val_mask]
        )


    # Early stopping
    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            model.state_dict(),
            "results/models/graphsage_model.pth"
        )

    else:

        patience_counter += 1


    if epoch % 10 == 0 or epoch == 1:

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f}"
        )


    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# ==========================================
# 5. LOAD BEST MODEL
# ==========================================

model.load_state_dict(
    torch.load(
        "results/models/graphsage_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

print("\nBest epoch:", best_epoch)
print("Best validation loss:", best_val_loss)


# ==========================================
# 6. TEST EVALUATION
# ==========================================

model.eval()

with torch.no_grad():

    logits = model(data.x, data.edge_index)

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= 0.5).int()

    y_true = data.y[data.test_mask].cpu().numpy()

    y_pred = predictions[data.test_mask].cpu().numpy()

    y_prob = probabilities[data.test_mask].cpu().numpy()


# ==========================================
# 7. METRICS
# ==========================================

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

auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 50)
print("GRAPHSAGE TEST RESULTS")
print("=" * 50)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:  {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\n" + "=" * 50)


# ==========================================
# 9. SAVE METRICS
# ==========================================

import os
import json

os.makedirs("results/metrics", exist_ok=True)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(auc),
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_val_loss),
    "confusion_matrix": cm.tolist()
}

with open(
    "results/metrics/graphsage_metrics.json",
    "w"
) as file:

    json.dump(metrics, file, indent=4)


print("Metrics saved to results/metrics/graphsage_metrics.json")
print("Model saved to results/models/graphsage_model.pth")
print("GRAPH SAGE TRAINING COMPLETED!")