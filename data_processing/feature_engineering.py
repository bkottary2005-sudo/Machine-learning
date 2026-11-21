import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "Data" / "processed" / "transactions_cleaned.csv"


def load_cleaned_data():
    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("FEATURE ENGINEERING")
    print("=" * 50)

    print(f"Dataset loaded successfully: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def create_transaction_features(df):
    # Transaction amount features
    df["amount_log"] = np.log1p(df["value"])
    df["amount_deviation"] = df["value"] - df["value"].mean()

    # Temporal features
    df["time_step"] = df["time"]

    # Transaction type indicator
    df["is_self_transaction"] = (
        df["sourceNodeId"] == df["targetNodeId"]
    ).astype(int)

    print("Transaction features created successfully!")

    return df


def create_account_activity_features(df):
    # Sender transaction frequency
    sender_frequency = df.groupby("sourceNodeId").size()

    # Receiver transaction frequency
    receiver_frequency = df.groupby("targetNodeId").size()

    df["sender_transaction_count"] = (
        df["sourceNodeId"].map(sender_frequency)
    )

    df["receiver_transaction_count"] = (
        df["targetNodeId"].map(receiver_frequency)
    )

    # Sender total transaction amount
    sender_total_amount = df.groupby("sourceNodeId")["value"].sum()

    df["sender_total_amount"] = (
        df["sourceNodeId"].map(sender_total_amount)
    )

    # Receiver total transaction amount
    receiver_total_amount = df.groupby("targetNodeId")["value"].sum()

    df["receiver_total_amount"] = (
        df["targetNodeId"].map(receiver_total_amount)
    )

    print("Account activity features created successfully!")

    return df


if __name__ == "__main__":
    df = load_cleaned_data()
    df = create_transaction_features(df)
    df = create_account_activity_features(df)

    OUTPUT_PATH = PROJECT_ROOT / "Data" / "processed" / "transactions_features.csv"

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal dataset shape:", df.shape)
    print("\nNew feature columns:")
    print(df.columns.tolist())
    print(f"\nFeature dataset saved to: {OUTPUT_PATH}")