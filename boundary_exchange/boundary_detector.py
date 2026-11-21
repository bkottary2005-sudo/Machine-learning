import pandas as pd
import os


def detect_boundary_nodes():
    bank_a = pd.read_csv("Data/bank_partitions/Bank_A_transactions.csv")
    bank_b = pd.read_csv("Data/bank_partitions/Bank_B_transactions.csv")
    bank_c = pd.read_csv("Data/bank_partitions/Bank_C_transactions.csv")

    banks = {
        "Bank_A": bank_a,
        "Bank_B": bank_b,
        "Bank_C": bank_c
    }

    account_banks = {}

    for bank_name, df in banks.items():
        accounts = set(df["sourceNodeId"]).union(
            set(df["targetNodeId"])
        )

        for account in accounts:
            account_banks.setdefault(account, []).append(bank_name)

    boundary_nodes = {
        account: bank_list
        for account, bank_list in account_banks.items()
        if len(bank_list) > 1
    }

    # Convert results to DataFrame
    results = []

    for account, bank_list in boundary_nodes.items():
        results.append({
            "account_id": account,
            "banks_involved": ",".join(bank_list),
            "num_banks": len(bank_list)
        })

    boundary_df = pd.DataFrame(results)

    # Save results
    os.makedirs("Data/boundary_nodes", exist_ok=True)

    output_path = "Data/boundary_nodes/boundary_nodes.csv"

    boundary_df.to_csv(output_path, index=False)

    print("Total boundary nodes:", len(boundary_nodes))
    print("Boundary nodes saved to:", output_path)

    print("\nSample:")
    print(boundary_df.head(10))

    return boundary_nodes


if __name__ == "__main__":
    detect_boundary_nodes()