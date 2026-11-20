import pandas as pd


def analyze_data_quality(df):
    """
    Perform a basic quality audit on transaction data.
    """

    print("\n" + "=" * 50)
    print("        DATA QUALITY REPORT")
    print("=" * 50)

    # 1. Dataset dimensions
    print("\n[1] DATASET SIZE")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # 2. Missing values
    print("\n[2] MISSING VALUES")
    print(df.isnull().sum())

    # 3. Duplicate transactions
    print("\n[3] DUPLICATE ROWS")
    print(f"Exact duplicates: {df.duplicated().sum():,}")

    # 4. Transaction amount analysis
    print("\n[4] TRANSACTION AMOUNT ANALYSIS")
    print(df["value"].describe())

    print(f"Zero/negative amounts: {(df['value'] <= 0).sum():,}")

    # 5. Account ID validation
    print("\n[5] ACCOUNT ID VALIDATION")

    print(f"Missing source IDs: {df['sourceNodeId'].isnull().sum():,}")
    print(f"Missing target IDs: {df['targetNodeId'].isnull().sum():,}")

    print(
        "Self-transactions:",
        (df["sourceNodeId"] == df["targetNodeId"]).sum()
    )

    # 6. Time analysis
    print("\n[6] TRANSACTION TIME ANALYSIS")
    print(f"Minimum time: {df['time'].min()}")
    print(f"Maximum time: {df['time'].max()}")
    print(f"Unique time steps: {df['time'].nunique()}")

    # 7. Unique accounts
    print("\n[7] ACCOUNT STATISTICS")

    unique_accounts = pd.concat(
        [df["sourceNodeId"], df["targetNodeId"]]
    ).nunique()

    print(f"Unique accounts involved: {unique_accounts:,}")

    print("\n" + "=" * 50)
    print("DATA QUALITY AUDIT COMPLETED")
    print("=" * 50)