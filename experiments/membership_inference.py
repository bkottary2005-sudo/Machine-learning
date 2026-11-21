import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from federated_learning.model import AMLModel

# Load data
bank_a = pd.read_csv("Data/bank_partitions/Bank_A_labeled.csv")
bank_b = pd.read_csv("Data/bank_partitions/Bank_B_labeled.csv")
bank_c = pd.read_csv("Data/bank_partitions/Bank_C_labeled.csv")

data = pd.concat([bank_a, bank_b, bank_c], ignore_index=True)

features = [
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

X = data[features].fillna(0).values
y = data["isFraud"].values.reshape(-1, 1)

# Split into members (training) and non-members (test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Train target model
model = AMLModel(input_size=9)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

criterion = torch.nn.BCELoss()

epochs = 5

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    print(f"Target Model Epoch {epoch + 1} | Loss: {loss.item():.4f}")

# Get confidence scores
model.eval()

with torch.no_grad():
    member_scores = model(X_train).numpy().flatten()
    nonmember_scores = model(X_test).numpy().flatten()

# Membership inference:
# confidence is used as the attack signal
attack_scores = np.concatenate([member_scores, nonmember_scores])

membership_labels = np.concatenate([
    np.ones(len(member_scores)),
    np.zeros(len(nonmember_scores))
])

# Threshold attack
attack_predictions = (attack_scores >= 0.5).astype(int)

accuracy = accuracy_score(membership_labels, attack_predictions)
precision = precision_score(membership_labels, attack_predictions)
recall = recall_score(membership_labels, attack_predictions)
auc = roc_auc_score(membership_labels, attack_scores)

print("\nMembership Inference Attack Results")
print(f"Attack Accuracy: {accuracy:.4f}")
print(f"Attack Precision: {precision:.4f}")
print(f"Attack Recall: {recall:.4f}")
print(f"Attack AUC: {auc:.4f}")