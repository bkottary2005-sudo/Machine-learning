import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "Data" / "processed" / "transactions_features.csv"


def load_feature_dataset():
    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("BANK PARTITIONING")
    print("=" * 50)

    print(f"Dataset loaded: {df.shape}")
    print(f"Total transactions: {len(df):,}")

    return df


def partition_into_banks(df):
    accounts = sorted(
        set(df["sourceNodeId"].unique()) |
        set(df["targetNodeId"].unique())
    )

    bank_accounts = {
        "Bank_A": accounts[0::3],
        "Bank_B": accounts[1::3],
        "Bank_C": accounts[2::3],
    }

    print("\nAccounts assigned to banks:")
    for bank, bank_ids in bank_accounts.items():
        print(f"{bank}: {len(bank_ids):,} accounts")

    return bank_accounts


def partition_transactions_by_bank(df, bank_accounts):
    bank_transactions = {}

    for bank, accounts in bank_accounts.items():
        account_set = set(accounts)

        bank_df = df[
            df["sourceNodeId"].isin(account_set)
        ].copy()

        bank_transactions[bank] = bank_df

        print(f"\n{bank}:")
        print(f"Transactions assigned: {len(bank_df):,}")

    return bank_transactions


if __name__ == "__main__":
    df = load_feature_dataset()

    bank_accounts = partition_into_banks(df)

    bank_transactions = partition_transactions_by_bank(
        df, bank_accounts
    )

    OUTPUT_DIR = PROJECT_ROOT / "Data" / "bank_partitions"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for bank, accounts in bank_accounts.items():
        account_path = OUTPUT_DIR / f"{bank}_accounts.csv"
        transaction_path = OUTPUT_DIR / f"{bank}_transactions.csv"

        pd.DataFrame({"account_id": accounts}).to_csv(
            account_path, index=False
        )

        bank_transactions[bank].to_csv(
            transaction_path, index=False
        )

        print(f"\n{bank} account file saved: {account_path}")
        print(f"{bank} transaction file saved: {transaction_path}")