import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "results" / "spark_performance_comparison.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "performance_comparison.png"


def create_performance_chart():
    df = pd.read_csv(RESULTS_PATH)

    plt.figure(figsize=(8, 5))

    plt.bar(
        df["Framework"],
        df["Processing_Time_Seconds"]
    )

    plt.ylabel("Processing Time (Seconds)")
    plt.xlabel("Framework")
    plt.title("Pandas vs PySpark Performance Comparison")

    for i, value in enumerate(df["Processing_Time_Seconds"]):
        plt.text(i, value, f"{value:.4f}s", ha="center", va="bottom")

    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=300)

    plt.show()

    print(f"\nChart saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_performance_chart()