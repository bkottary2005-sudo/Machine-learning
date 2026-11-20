from pathlib import Path
from pyspark.sql import SparkSession
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = str(
    PROJECT_ROOT / "Data" / "processed" / "transactions_features.csv"
)

OUTPUT_PATH = str(
    PROJECT_ROOT / "Data" / "processed" / "spark_output"
)


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("FinGraph-AML-Data-Processing")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    print("Spark session created successfully!")

    return spark


def load_data_spark(spark):
    start_time = time.time()

    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

    total_rows = df.count()

    print("\n--- PySpark Data Quality Checks ---")

    print("\nMissing values per column:")
    for column in df.columns:
        missing_count = df.filter(df[column].isNull()).count()
        print(f"{column}: {missing_count}")

    print("\nInvalid transaction amounts:")
    invalid_amounts = df.filter(df["value"] <= 0).count()
    print(invalid_amounts)

    print("\nSelf-transactions:")
    self_transactions = df.filter(
        df["sourceNodeId"] == df["targetNodeId"]
    ).count()
    print(self_transactions)

    # Save Spark processed dataset
    df.toPandas().to_csv(OUTPUT_PATH + ".csv", index=False)

    print(f"\nSpark processed dataset saved to: {OUTPUT_PATH}.csv")

    print(f"\nSpark processed dataset saved to: {OUTPUT_PATH}")

    end_time = time.time()
    processing_time = end_time - start_time

    print("\nDataset loaded using PySpark")
    print(f"Total rows: {total_rows:,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"SPARK_TIME={processing_time}")

    return df


if __name__ == "__main__":
    spark = create_spark_session()

    df = load_data_spark(spark)

    df.show(5)

    spark.stop()