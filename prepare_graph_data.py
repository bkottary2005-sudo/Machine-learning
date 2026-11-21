import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# ==========================================
# 1. LOAD DATA
# ==========================================

transactions = pd.read_csv("Data/processed/transactions_labeled.csv")
features_df = pd.read_csv("Data/processed/graph_features.csv")

print("Transactions loaded:", len(transactions))
print("Graph features loaded:", len(features_df))


# ==========================================
# 2. MAP ACCOUNT IDS TO INDEXES
# ==========================================

account_ids = features_df["account_id"].tolist()

account_to_index = {
    account_id: index
    for index, account_id in enumerate(account_ids)
}

print("Total accounts:", len(account_to_index))


# ==========================================
# 3. PREPARE NODE FEATURES
# ==========================================

feature_columns = [
    "in_degree",
    "out_degree",
    "total_degree",
    "unique_in_neighbors",
    "unique_out_neighbors",
    "unique_neighbors",
    "total_incoming_amount",
    "total_outgoing_amount",
    "avg_incoming_amount",
    "avg_outgoing_amount"
]

X = features_df[feature_columns].values.astype(np.float32)


# ==========================================
# 4. PREPARE NODE LABELS
# ==========================================

y = torch.tensor(
    features_df["account_is_fraud"].values,
    dtype=torch.float
)


# ==========================================
# 5. PREPARE GRAPH EDGES
# ==========================================

source_indices = transactions["sourceNodeId"].map(account_to_index)
target_indices = transactions["targetNodeId"].map(account_to_index)

edge_index = torch.tensor(
    np.vstack([
        source_indices.values,
        target_indices.values
    ]),
    dtype=torch.long
)


# ==========================================
# 6. CREATE PYTORCH GEOMETRIC DATA
# ==========================================

data = Data(
    edge_index=edge_index,
    y=y
)

print("\nGraph structure prepared!")
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)


# ==========================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

indices = np.arange(len(y))

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    stratify=y.numpy(),
    random_state=42
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    stratify=y.numpy()[temp_indices],
    random_state=42
)

data.train_mask = torch.zeros(len(y), dtype=torch.bool)
data.val_mask = torch.zeros(len(y), dtype=torch.bool)
data.test_mask = torch.zeros(len(y), dtype=torch.bool)

data.train_mask[train_indices] = True
data.val_mask[val_indices] = True
data.test_mask[test_indices] = True


# ==========================================
# 8. SCALE FEATURES USING TRAINING DATA ONLY
# ==========================================

scaler = StandardScaler()

# Fit scaler only on training nodes
X_train = scaler.fit_transform(X[train_indices])

# Transform all nodes using training statistics
X_scaled = scaler.transform(X)

x = torch.tensor(X_scaled, dtype=torch.float)

data.x = x

print("\nFeatures scaled using training data only!")


# ==========================================
# 9. DISPLAY SPLIT STATISTICS
# ==========================================

print("\nDataset split:")
print("Training nodes:", data.train_mask.sum().item())
print("Validation nodes:", data.val_mask.sum().item())
print("Testing nodes:", data.test_mask.sum().item())

print("\nFraud distribution:")

for name, mask in [
    ("Train", data.train_mask),
    ("Validation", data.val_mask),
    ("Test", data.test_mask)
]:
    fraud_count = data.y[mask].sum().item()
    total = mask.sum().item()

    print(
        f"{name}: {int(fraud_count)} fraud / "
        f"{total} total nodes"
    )


# ==========================================
# 10. SAVE GRAPH DATA
# ==========================================

torch.save(data, "Data/processed/graphsage_data.pt")

print("\nSaved to: Data/processed/graphsage_data.pt")
print("SUCCESS: Graph data preparation completed!")