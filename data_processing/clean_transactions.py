import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "Data" / "raw" / "transactions.csv"

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

OUTPUT_PATH = PROCESSED_DIR / "transactions_cleaned.csv"


def clean_transactions():
    print("=" * 60)
    print("TRANSACTION CLEANING PIPELINE")
    print("=" * 60)

    # Load original raw dataset
    df = pd.read_csv(RAW_PATH)

    original_rows = len(df)

    print(f"\nOriginal transactions: {original_rows:,}")

    # Validate required columns
    required_columns = [
        "sourceNodeId",
        "targetNodeId",
        "value",
        "time"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert data types safely
    df["sourceNodeId"] = pd.to_numeric(
        df["sourceNodeId"], errors="raise"
    )

    df["targetNodeId"] = pd.to_numeric(
        df["targetNodeId"], errors="raise"
    )

    df["value"] = pd.to_numeric(
        df["value"], errors="raise"
    )

    df["time"] = pd.to_numeric(
        df["time"], errors="raise"
    )

    # Remove exact duplicate rows
    duplicate_count = df.duplicated().sum()
    df["is_exact_duplicate"] = df.duplicated(keep=False)

    print(f"Exact duplicate rows (additional copies): {duplicate_count:,}")
    print(f"Rows belonging to duplicate groups: {df['is_exact_duplicate'].sum():,}")
    print("Duplicates preserved for further investigation.")

    # Flag self-transactions (preserve them)
    df["is_self_transaction"] = (
        df["sourceNodeId"] == df["targetNodeId"]
    )

    self_transaction_count = df["is_self_transaction"].sum()

    print(
        "Self-transactions preserved and flagged:",
        self_transaction_count
    )

    # Flag suspicious transaction amounts
    df["is_invalid_amount"] = df["value"] <= 0

    invalid_amount_count = df["is_invalid_amount"].sum()

    print(
        "Invalid amount transactions:",
        invalid_amount_count
    )

    # Flag missing values
    df["has_missing_values"] = df.isnull().any(axis=1)

    missing_row_count = df["has_missing_values"].sum()

    print(
        "Rows with missing values:",
        missing_row_count
    )

    # Preserve transaction chronology
    df = df.sort_values(
        by=["time", "sourceNodeId", "targetNodeId"],
        kind="stable"
    ).reset_index(drop=True)

    # Create processed directory if necessary
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)

    print(f"Original rows: {original_rows:,}")
    print(f"Final rows: {len(df):,}")
    print(f"Rows removed: {original_rows - len(df):,}")
    print(f"Final columns: {len(df.columns)}")
    print(f"Output file: {OUTPUT_PATH}")

    print("\nCleaning completed successfully!")

    return df


if __name__ == "__main__":
    clean_transactions()