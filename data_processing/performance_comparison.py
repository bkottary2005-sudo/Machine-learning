import time
import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "processed" / "transactions_features.csv"


def benchmark_pandas():
    start = time.time()

    df = pd.read_csv(DATA_PATH)

    missing = df.isnull().sum().sum()
    invalid = (df["value"] <= 0).sum()
    self_tx = (df["sourceNodeId"] == df["targetNodeId"]).sum()
    rows = len(df)

    elapsed = time.time() - start

    return rows, missing, invalid, self_tx, elapsed


def benchmark_spark():
    spark = (
        SparkSession.builder
        .appName("FinGraphPerformanceComparison")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    start = time.time()

    df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)

    missing = sum(
        df.filter(col(c).isNull()).count()
        for c in df.columns
    )

    invalid = df.filter(col("value") <= 0).count()

    self_tx = df.filter(
        col("sourceNodeId") == col("targetNodeId")
    ).count()

    rows = df.count()

    elapsed = time.time() - start

    spark.stop()

    return rows, missing, invalid, self_tx, elapsed


if __name__ == "__main__":
    print("Benchmarking Pandas...")
    p = benchmark_pandas()

    print("\nBenchmarking PySpark...")
    s = benchmark_spark()

    results = pd.DataFrame([
        {
            "Framework": "Pandas",
            "Dataset_Rows": p[0],
            "Missing_Values": p[1],
            "Invalid_Amounts": p[2],
            "Self_Transactions": p[3],
            "Processing_Time_Seconds": p[4]
        },
        {
            "Framework": "PySpark",
            "Dataset_Rows": s[0],
            "Missing_Values": s[1],
            "Invalid_Amounts": s[2],
            "Self_Transactions": s[3],
            "Processing_Time_Seconds": s[4]
        }
    ])

    print("\n===== PERFORMANCE COMPARISON =====")
    print(results.to_string(index=False))

    results.to_csv(
        PROJECT_ROOT / "results" / "spark_performance_comparison.csv",
        index=False
    )

    print("\nResults saved successfully!")