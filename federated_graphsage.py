import os
import copy
import torch
import pandas as pd
import numpy as np

from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


# ============================================================
# 1. CONFIGURATION
# ============================================================

NUM_BANKS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.01
HIDDEN_DIM = 64

DATA_PATH = "Data/processed/graphsage_data.pt"

PARTITION_DIR = "Data/processed/federated"
MODEL_DIR = "results/models"
METRICS_DIR = "results/metrics"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ============================================================
# 2. GRAPH SAGE MODEL
# ============================================================

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


# ============================================================
# 3. LOAD GRAPH DATA
# ============================================================

print("=" * 70)
print("FEDERATED GRAPHSAGE TRAINING")
print("=" * 70)

data = torch.load(
    DATA_PATH,
    map_location="cpu",
    weights_only=False
)

print("\nGraph loaded")
print("Nodes:", data.num_nodes)
print("Features:", data.x.shape[1])
print("Edges:", data.edge_index.shape[1])


# ============================================================
# 4. LOAD BANK PARTITIONS
# ============================================================

bank_nodes = {}

for bank_id in range(NUM_BANKS):

    bank_name = f"Bank_{bank_id + 1}"

    file_path = os.path.join(
        PARTITION_DIR,
        f"{bank_name}_nodes.csv"
    )

    df = pd.read_csv(file_path)

    bank_nodes[bank_name] = df["node_index"].values

    print(
        f"{bank_name}: "
        f"{len(bank_nodes[bank_name])} nodes loaded"
    )


# ============================================================
# 5. DEFINE LOCAL TRAINING NODES
# ============================================================

global_train_nodes = torch.where(data.train_mask)[0].numpy()

bank_train_nodes = {}

for bank_name, nodes in bank_nodes.items():

    local_train = np.intersect1d(
        nodes,
        global_train_nodes
    )

    bank_train_nodes[bank_name] = torch.tensor(
        local_train,
        dtype=torch.long
    )

    print(
        f"{bank_name} local training nodes: "
        f"{len(local_train)}"
    )


# ============================================================
# 6. INITIALIZE GLOBAL MODEL
# ============================================================

global_model = GraphSAGE(
    input_dim=data.x.shape[1],
    hidden_dim=HIDDEN_DIM
)

global_weights = global_model.state_dict()


# ============================================================
# 7. LOCAL TRAINING FUNCTION
# ============================================================

def local_train(bank_name, global_weights):

    print(f"\nTraining {bank_name}...")

    local_model = GraphSAGE(
        input_dim=data.x.shape[1],
        hidden_dim=HIDDEN_DIM
    )

    local_model.load_state_dict(global_weights)

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    local_nodes = bank_train_nodes[bank_name]

    local_labels = data.y[local_nodes]

    fraud_count = int(local_labels.sum())

    legit_count = int(len(local_labels) - fraud_count)

    pos_weight = torch.tensor(
        [legit_count / max(fraud_count, 1)],
        dtype=torch.float
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    for epoch in range(LOCAL_EPOCHS):

        local_model.train()

        optimizer.zero_grad()

        logits = local_model(
            data.x,
            data.edge_index
        )

        local_loss = criterion(
            logits[local_nodes],
            data.y[local_nodes]
        )

        local_loss.backward()

        optimizer.step()

        print(
            f"{bank_name} | "
            f"Epoch {epoch + 1}/{LOCAL_EPOCHS} | "
            f"Loss: {local_loss.item():.4f}"
        )

    return (
        local_model.state_dict(),
        len(local_nodes)
    )


# ============================================================
# 8. FEDERATED AVERAGING
# ============================================================

def fedavg(local_weights, local_sizes):

    total_samples = sum(local_sizes)

    averaged_weights = copy.deepcopy(
        local_weights[0]
    )

    for key in averaged_weights.keys():

        averaged_weights[key] = torch.zeros_like(
            local_weights[0][key]
        )

        for bank_id in range(NUM_BANKS):

            weight = (
                local_sizes[bank_id] /
                total_samples
            )

            averaged_weights[key] += (
                local_weights[bank_id][key] * weight
            )

    return averaged_weights


# ============================================================
# 9. FEDERATED TRAINING LOOP
# ============================================================

round_results = []

for round_id in range(NUM_ROUNDS):

    print("\n" + "=" * 70)

    print(
        f"FEDERATED ROUND "
        f"{round_id + 1}/{NUM_ROUNDS}"
    )

    print("=" * 70)

    local_weights = []
    local_sizes = []

    # Each bank trains locally
    for bank_name in bank_train_nodes:

        weights, size = local_train(
            bank_name,
            global_weights
        )

        local_weights.append(weights)
        local_sizes.append(size)

    # Aggregate updates
    global_weights = fedavg(
        local_weights,
        local_sizes
    )

    global_model.load_state_dict(
        global_weights
    )

    print("\nFedAvg aggregation completed!")

    round_results.append({
        "Round": round_id + 1,
        "Total_Local_Training_Nodes": sum(local_sizes)
    })


# ============================================================
# 10. SAVE FINAL MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "federated_graphsage_model.pth"
)

torch.save(
    global_model.state_dict(),
    model_path
)

print("\nFederated model saved:")
print(model_path)


# ============================================================
# 11. SAVE ROUND METRICS
# ============================================================

round_df = pd.DataFrame(round_results)

round_df.to_csv(
    os.path.join(
        METRICS_DIR,
        "federated_graphsage_rounds.csv"
    ),
    index=False
)


# ============================================================
# 12. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("FEDERATED GRAPHSAGE TRAINING COMPLETED")
print("=" * 70)

print("\nBanks:", NUM_BANKS)
print("Rounds:", NUM_ROUNDS)
print("Local epochs:", LOCAL_EPOCHS)

print("\nSaved files:")
print(model_path)

print(
    os.path.join(
        METRICS_DIR,
        "federated_graphsage_rounds.csv"
    )
)