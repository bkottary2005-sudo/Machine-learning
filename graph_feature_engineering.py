import pandas as pd
import networkx as nx
import os

# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = "Data/processed/transactions_labeled.csv"

df = pd.read_csv(DATA_PATH)

print("Transaction dataset loaded:", df.shape)


# ==========================================
# 2. BUILD MULTI-DIRECTED GRAPH
# ==========================================

G = nx.MultiDiGraph()

accounts = set(df["sourceNodeId"]) | set(df["targetNodeId"])

G.add_nodes_from(accounts)

for transaction_id, row in df.iterrows():

    G.add_edge(
        row["sourceNodeId"],
        row["targetNodeId"],
        key=transaction_id,
        value=float(row["value"]),
        isFraud=int(row["isFraud"])
    )

print("Graph created!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# ==========================================
# 3. CALCULATE STRUCTURAL FEATURES
# ==========================================

print("\nCalculating graph features...")

# Number of incoming transactions
in_degree = dict(G.in_degree())

# Number of outgoing transactions
out_degree = dict(G.out_degree())

# Total degree
total_degree = dict(G.degree())

# Unique incoming neighbors
in_neighbors = {
    node: len(set(G.predecessors(node)))
    for node in G.nodes()
}

# Unique outgoing neighbors
out_neighbors = {
    node: len(set(G.successors(node)))
    for node in G.nodes()
}

# Total unique neighbors
unique_neighbors = {
    node: len(set(G.predecessors(node)) | set(G.successors(node)))
    for node in G.nodes()
}


# ==========================================
# 4. CALCULATE TRANSACTION AMOUNTS
# ==========================================

# Total outgoing amount per account
outgoing_amount = (
    df.groupby("sourceNodeId")["value"]
    .sum()
    .to_dict()
)

# Total incoming amount per account
incoming_amount = (
    df.groupby("targetNodeId")["value"]
    .sum()
    .to_dict()
)

# Average outgoing transaction amount
avg_outgoing_amount = (
    df.groupby("sourceNodeId")["value"]
    .mean()
    .to_dict()
)

# Average incoming transaction amount
avg_incoming_amount = (
    df.groupby("targetNodeId")["value"]
    .mean()
    .to_dict()
)


# ==========================================
# 5. CALCULATE FRAUD-RELATED FEATURES
# ==========================================

# Fraud transactions sent by each account
fraud_sent = (
    df[df["isFraud"] == 1]
    .groupby("sourceNodeId")
    .size()
    .to_dict()
)

# Fraud transactions received by each account
fraud_received = (
    df[df["isFraud"] == 1]
    .groupby("targetNodeId")
    .size()
    .to_dict()
)


# ==========================================
# 6. CREATE FEATURE DATAFRAME
# ==========================================

feature_data = []

for node in G.nodes():

    feature_data.append({
        "account_id": node,

        "in_degree": in_degree.get(node, 0),
        "out_degree": out_degree.get(node, 0),
        "total_degree": total_degree.get(node, 0),

        "unique_in_neighbors": in_neighbors.get(node, 0),
        "unique_out_neighbors": out_neighbors.get(node, 0),
        "unique_neighbors": unique_neighbors.get(node, 0),

        "total_incoming_amount": incoming_amount.get(node, 0),
        "total_outgoing_amount": outgoing_amount.get(node, 0),

        "avg_incoming_amount": avg_incoming_amount.get(node, 0),
        "avg_outgoing_amount": avg_outgoing_amount.get(node, 0),

        "fraud_transactions_received": fraud_received.get(node, 0),
        "fraud_transactions_sent": fraud_sent.get(node, 0)
    })


graph_features = pd.DataFrame(feature_data)


# ==========================================
# 7. ADD ACCOUNT FRAUD LABELS
# ==========================================

account_labels = {}

for _, row in df.iterrows():

    sender = row["sourceNodeId"]
    receiver = row["targetNodeId"]

    account_labels[sender] = max(
        account_labels.get(sender, 0),
        int(row["sender_is_fraud"])
    )

    account_labels[receiver] = max(
        account_labels.get(receiver, 0),
        int(row["receiver_is_fraud"])
    )


graph_features["account_is_fraud"] = (
    graph_features["account_id"]
    .map(account_labels)
    .fillna(0)
    .astype(int)
)


# ==========================================
# 8. SAVE FEATURES
# ==========================================

os.makedirs("Data/processed", exist_ok=True)

output_path = "Data/processed/graph_features.csv"

graph_features.to_csv(output_path, index=False)


# ==========================================
# 9. RESULTS
# ==========================================

print("\n" + "=" * 50)
print("GRAPH FEATURE ENGINEERING RESULTS")
print("=" * 50)

print("Accounts:", len(graph_features))
print("Features:", len(graph_features.columns))

print("\nFeature columns:")
print(graph_features.columns.tolist())

print("\nFraudulent accounts:", graph_features["account_is_fraud"].sum())
print("Legitimate accounts:", (graph_features["account_is_fraud"] == 0).sum())

print("\nSample:")
print(graph_features.head(5).to_string())

print("\nGraph features saved to:", output_path)

print("\nSUCCESS: Graph feature engineering completed!")
print("=" * 50)