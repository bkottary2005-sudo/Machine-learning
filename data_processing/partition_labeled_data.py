import pandas as pd
from pathlib import Path

# Load labeled transactions
data = pd.read_csv("Data/processed/transactions_labeled.csv")

# Create output folder
output_dir = Path("Data/bank_partitions")
output_dir.mkdir(parents=True, exist_ok=True)

# Assign banks based on sender account partition
accounts = pd.read_csv("Data/bank_partitions/Bank_A_accounts.csv")
bank_mapping = {}

for bank in ["Bank_A", "Bank_B", "Bank_C"]:
    bank_accounts = pd.read_csv(
        f"Data/bank_partitions/{bank}_accounts.csv"
    )
    for account in bank_accounts["account_id"]:
        bank_mapping[account] = bank

# Assign each transaction to the sender's bank
data["bank"] = data["sourceNodeId"].map(bank_mapping)

# Save labeled data for each bank
for bank in ["Bank_A", "Bank_B", "Bank_C"]:
    bank_data = data[data["bank"] == bank].copy()

    output_path = output_dir / f"{bank}_labeled.csv"
    bank_data.to_csv(output_path, index=False)

    print(f"\n{bank}")
    print("Rows:", len(bank_data))
    print("Fraud labels:")
    print(bank_data["isFraud"].value_counts())

print("\nLabeled bank partitions created successfully!")