import torch
import pandas as pd
import numpy as np
import joblib
import os

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from federated_learning.model import AMLModel


# ==========================================
# 1. LOAD DATA
# ==========================================

bank_a = pd.read_csv("Data/bank_partitions/Bank_A_labeled.csv")
bank_b = pd.read_csv("Data/bank_partitions/Bank_B_labeled.csv")
bank_c = pd.read_csv("Data/bank_partitions/Bank_C_labeled.csv")

data = pd.concat([bank_a, bank_b, bank_c], ignore_index=True)

print("Dataset loaded:", data.shape)


# ==========================================
# 2. FEATURES AND LABELS
# ==========================================

feature_columns = [
    "value",
    "time",
    "amount_log",
    "amount_deviation",
    "time_step",
    "sender_transaction_count",
    "receiver_transaction_count",
    "sender_total_amount",
    "receiver_total_amount"
]

X = data[feature_columns].fillna(0).values
y = data["isFraud"].values


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))


# ==========================================
# 4. SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 5. TENSOR CONVERSION
# ==========================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(
    y_train.reshape(-1, 1),
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.reshape(-1, 1),
    dtype=torch.float32
)


# ==========================================
# 6. MINI-BATCH DATA LOADER
# ==========================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)


# ==========================================
# 7. MODEL
# ==========================================

model = AMLModel(input_size=9)

criterion = torch.nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.00005
)


# ==========================================
# 8. TRAINING
# ==========================================

epochs = 20

training_losses = []

print("\nStarting improved centralized training...\n")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    training_losses.append(average_loss)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Training Loss: {average_loss:.4f}"
    )


# ==========================================
# 9. EVALUATION
# ==========================================

model.eval()

with torch.no_grad():

    probabilities = model(X_test_tensor).numpy().flatten()

predictions = (probabilities >= 0.5).astype(int)


accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

print("\n==========================================")
print("CENTRALIZED MODEL RESULTS")
print("==========================================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))


# ==========================================
# 10. SAVE MODEL AND SCALER
# ==========================================

os.makedirs("results/models", exist_ok=True)

torch.save(
    model.state_dict(),
    "results/models/centralized_model.pth"
)

joblib.dump(
    scaler,
    "results/models/centralized_scaler.pkl"
)

print("\nModel saved successfully!")
print("Scaler saved successfully!")


# ==========================================
# 11. SAVE RESULTS
# ==========================================

results = pd.DataFrame({
    "model": ["Centralized"],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1]
})

results.to_csv(
    "results/centralized_results.csv",
    index=False
)


# Save training losses
loss_results = pd.DataFrame({
    "epoch": range(1, epochs + 1),
    "training_loss": training_losses
})

loss_results.to_csv(
    "results/centralized_training_losses.csv",
    index=False
)


# ==========================================
# 12. COMPLETION
# ==========================================

print("\n==========================================")
print("TRAINING COMPLETED!")
print("==========================================")

print("Saved files:")
print("- results/models/centralized_model.pth")
print("- results/models/centralized_scaler.pkl")
print("- results/centralized_results.csv")
print("- results/centralized_training_losses.csv")