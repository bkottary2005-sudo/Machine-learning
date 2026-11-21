import pandas as pd
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. CREATE RESULTS DATA
# ==========================================

models = {
    "Model": [
        "Centralized ML",
        "GraphSAGE"
    ],

    "Accuracy": [
        0.8559,
        0.9132
    ],

    "Precision": [
        0.7158,
        0.3233
    ],

    "Recall": [
        0.2461,
        0.7483
    ],

    "F1_Score": [
        0.3663,
        0.4515
    ]
}

df = pd.DataFrame(models)


# ==========================================
# 2. DISPLAY COMPARISON
# ==========================================

print("=" * 70)
print("CENTRALIZED ML VS GRAPHSAGE COMPARISON")
print("=" * 70)

print(df.to_string(index=False))


# ==========================================
# 3. SAVE COMPARISON TABLE
# ==========================================

os.makedirs("results/metrics", exist_ok=True)

df.to_csv(
    "results/metrics/model_comparison.csv",
    index=False
)

print("\nComparison table saved!")


# ==========================================
# 4. BAR CHART COMPARISON
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]

plt.figure(figsize=(10, 6))

x = range(len(metrics))

width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    df.iloc[0][metrics],
    width,
    label="Centralized ML"
)

plt.bar(
    [i + width / 2 for i in x],
    df.iloc[1][metrics],
    width,
    label="GraphSAGE"
)

plt.xticks(
    x,
    metrics
)

plt.ylabel("Score")

plt.title("Centralized ML vs GraphSAGE")

plt.legend()

plt.grid(axis="y")

plt.tight_layout()

os.makedirs("results/plots", exist_ok=True)

plt.savefig(
    "results/plots/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Comparison chart saved!")


# ==========================================
# 5. PERFORMANCE DIFFERENCE
# ==========================================

print("\n" + "=" * 70)
print("PERFORMANCE DIFFERENCES")
print("=" * 70)

for metric in metrics:

    centralized = df.loc[0, metric]
    graphsage = df.loc[1, metric]

    difference = graphsage - centralized

    print(
        f"{metric}: "
        f"Centralized = {centralized:.4f}, "
        f"GraphSAGE = {graphsage:.4f}, "
        f"Difference = {difference:+.4f}"
    )


# ==========================================
# 6. SUMMARY
# ==========================================

print("\nSUMMARY")

print(
    f"GraphSAGE accuracy improved by "
    f"{(df.loc[1, 'Accuracy'] - df.loc[0, 'Accuracy']) * 100:.2f} "
    "percentage points."
)

print(
    f"GraphSAGE recall improved by "
    f"{(df.loc[1, 'Recall'] - df.loc[0, 'Recall']) * 100:.2f} "
    "percentage points."
)

print(
    f"GraphSAGE F1 improved by "
    f"{(df.loc[1, 'F1_Score'] - df.loc[0, 'F1_Score']) * 100:.2f} "
    "percentage points."
)

print("\nModel comparison completed!")