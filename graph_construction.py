import pandas as pd
import networkx as nx
import os

# ==========================================
# 1. LOAD TRANSACTION DATA
# ==========================================

DATA_PATH = "Data/processed/transactions_labeled.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Total transactions:", len(df))


# ==========================================
# 2. CREATE DIRECTED MULTI-GRAPH
# ==========================================

# MultiDiGraph preserves multiple transactions
# between the same source and target accounts.

G = nx.MultiDiGraph()


# ==========================================
# 3. ADD ACCOUNT NODES
# ==========================================

accounts = set(df["sourceNodeId"]) | set(df["targetNodeId"])

G.add_nodes_from(accounts)

print("Accounts added:", G.number_of_nodes())


# ==========================================
# 4. ADD TRANSACTION EDGES
# ==========================================

for transaction_id, row in df.iterrows():

    G.add_edge(
        row["sourceNodeId"],
        row["targetNodeId"],
        key=transaction_id,
        transaction_id=transaction_id,
        value=float(row["value"]),
        time=int(row["time"]),
        amount_log=float(row["amount_log"]),
        amount_deviation=float(row["amount_deviation"]),
        time_step=int(row["time_step"]),
        sender_transaction_count=int(row["sender_transaction_count"]),
        receiver_transaction_count=int(row["receiver_transaction_count"]),
        sender_total_amount=float(row["sender_total_amount"]),
        receiver_total_amount=float(row["receiver_total_amount"]),
        sender_is_fraud=int(row["sender_is_fraud"]),
        receiver_is_fraud=int(row["receiver_is_fraud"]),
        isFraud=int(row["isFraud"])
    )


# ==========================================
# 5. GRAPH STATISTICS
# ==========================================

print("\n" + "=" * 50)
print("GRAPH CONSTRUCTION RESULTS")
print("=" * 50)

print("Graph type:", type(G).__name__)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Directed graph:", G.is_directed())
print("Multi-graph:", G.is_multigraph())
print("Self-loops:", nx.number_of_selfloops(G))

print("Original transactions:", len(df))
print("Transactions preserved:", G.number_of_edges())


# ==========================================
# 6. SAVE GRAPH
# ==========================================

os.makedirs("Data/processed", exist_ok=True)

output_path = "Data/processed/transaction_graph.graphml"

nx.write_graphml(G, output_path)

print("\nGraph saved successfully!")
print("Location:", output_path)


# ==========================================
# 7. VERIFY GRAPH
# ==========================================

if G.number_of_edges() == len(df):
    print("\nSUCCESS: All transactions are preserved!")
else:
    print("\nWARNING: Transaction count mismatch!")

print("=" * 50)