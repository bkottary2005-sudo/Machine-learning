import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader

from opacus import PrivacyEngine
from federated_learning.model import AMLModel

# Load bank datasets
bank_a = pd.read_csv("Data/bank_partitions/Bank_A_labeled.csv")
bank_b = pd.read_csv("Data/bank_partitions/Bank_B_labeled.csv")
bank_c = pd.read_csv("Data/bank_partitions/Bank_C_labeled.csv")

# Combine
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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Dataset and DataLoader
dataset = TensorDataset(X_train, y_train)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)

# Model
model = AMLModel(input_size=9)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

criterion = torch.nn.BCELoss()

# Differential Privacy
privacy_engine = PrivacyEngine()

model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0
)

# Training
epochs = 5

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for batch_X, batch_y in loader:
        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    epsilon = privacy_engine.get_epsilon(delta=1e-5)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Privacy epsilon: {epsilon:.4f}"
    )

# Evaluation
model.eval()

with torch.no_grad():
    outputs = model(X_test)

    test_loss = criterion(outputs, y_test)

    predictions = (outputs >= 0.5).float()

    accuracy = (predictions == y_test).float().mean().item()

print("\nDifferential Privacy Results")
print(f"Test Loss: {test_loss.item():.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Final Privacy Epsilon: {epsilon:.4f}")
print(f"Delta: 1e-5")