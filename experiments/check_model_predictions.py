import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from federated_learning.model import AMLModel


# Load datasets
bank_a = pd.read_csv("Data/bank_partitions/Bank_A_labeled.csv")
bank_b = pd.read_csv("Data/bank_partitions/Bank_B_labeled.csv")
bank_c = pd.read_csv("Data/bank_partitions/Bank_C_labeled.csv")

data = pd.concat([bank_a, bank_b, bank_c], ignore_index=True)


# Features
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


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert to tensors
X_test = torch.tensor(X_test, dtype=torch.float32)


# Load model
model = AMLModel(input_size=9)

model.load_state_dict(
    torch.load(
        "results/models/centralized_model.pth",
        map_location="cpu"
    )
)

model.eval()


# Predictions
with torch.no_grad():
    probabilities = model(X_test).numpy().flatten()

predictions = (probabilities >= 0.5).astype(int)


# Results
print("\nPrediction Distribution:")
print(pd.Series(predictions).value_counts())

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nProbability Statistics:")
print("Minimum:", probabilities.min())
print("Maximum:", probabilities.max())
print("Mean:", probabilities.mean())