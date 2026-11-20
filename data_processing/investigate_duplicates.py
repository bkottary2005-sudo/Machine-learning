import pandas as pd
from pathlib import Path


DATA_PATH = Path(
    r"C:\Users\Arpitha C\OneDrive\Documents\FinGraph-Intelligence\Data\raw\transactions.csv"
)


def investigate_duplicates():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DUPLICATE TRANSACTION INVESTIGATION")
    print("=" * 60)

    # Find exact duplicate rows
    duplicates = df[df.duplicated(keep=False)]

    print("\n1. DUPLICATE ROWS")
    print("Total rows involved in duplicates:", len(duplicates))

    print("\nDuplicate samples:")
    print(duplicates.head(20).to_string(index=False))

    # Count exact duplicate groups
    duplicate_groups = (
        df.groupby(["sourceNodeId", "targetNodeId", "value", "time"])
        .size()
        .reset_index(name="count")
    )

    duplicate_groups = duplicate_groups[
        duplicate_groups["count"] > 1
    ]

    print("\n2. DUPLICATE GROUPS")
    print("Number of duplicate groups:", len(duplicate_groups))

    print("\nLargest duplicate groups:")
    print(
        duplicate_groups.sort_values("count", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    # Investigate self-transactions
    self_transactions = df[
        df["sourceNodeId"] == df["targetNodeId"]
    ]

    print("\n3. SELF-TRANSACTIONS")
    print("Count:", len(self_transactions))

    print("\nSelf-transaction samples:")
    print(self_transactions.to_string(index=False))

    # Investigate repeated transactions between account pairs
    pair_counts = (
        df.groupby(["sourceNodeId", "targetNodeId"])
        .size()
        .reset_index(name="transaction_count")
    )

    repeated_pairs = pair_counts[
        pair_counts["transaction_count"] > 1
    ]

    print("\n4. REPEATED ACCOUNT PAIRS")
    print("Repeated pairs:", len(repeated_pairs))

    print("\nMost active repeated pairs:")
    print(
        repeated_pairs.sort_values(
            "transaction_count", ascending=False
        )
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    investigate_duplicates()