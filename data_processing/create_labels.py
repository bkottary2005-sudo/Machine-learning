import pandas as pd
from pathlib import Path

# File paths
transactions_path = Path("Data/processed/transactions_features.csv")
nodes_path = Path("Data/raw/nodes.csv")
output_path = Path("Data/processed/transactions_labeled.csv")

# Load datasets
transactions = pd.read_csv(transactions_path)
nodes = pd.read_csv(nodes_path)

# Create account fraud mapping
fraud_mapping = nodes.set_index("nodeid")["isFraud"].to_dict()

# Label transaction if sender OR receiver is fraudulent
transactions["sender_is_fraud"] = (
    transactions["sourceNodeId"].map(fraud_mapping).fillna(0).astype(int)
)

transactions["receiver_is_fraud"] = (
    transactions["targetNodeId"].map(fraud_mapping).fillna(0).astype(int)
)

transactions["isFraud"] = (
    (transactions["sender_is_fraud"] == 1) |
    (transactions["receiver_is_fraud"] == 1)
).astype(int)

# Save labeled data
transactions.to_csv(output_path, index=False)

print("Labeled dataset created successfully!")
print("Shape:", transactions.shape)
print("\nFraud distribution:")
print(transactions["isFraud"].value_counts())
print("\nFraud percentage:")
print(transactions["isFraud"].value_counts(normalize=True) * 100)
print("\nSaved to:", output_path)