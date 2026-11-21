import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

METRICS_DIR = "results/metrics"
PLOTS_DIR = "results/plots"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model performance results
data = {
    "Model": [
        "Centralized ML",
        "GraphSAGE",
        "Federated GraphSAGE"
    ],
    "Accuracy": [
        0.8559,
        0.9132,
        0.7534
    ],
    "Precision": [
        0.7158,
        0.3233,
        0.1294
    ],
    "Recall": [
        0.2461,
        0.7483,
        0.7273
    ],
    "F1_Score": [
        0.3663,
        0.4515,
        0.2196
    ],
    "ROC_AUC": [
        np.nan,
        0.9036,
        0.8237
    ]
}

df = pd.DataFrame(data)

# Save comparison CSV
csv_path = os.path.join(
    METRICS_DIR,
    "final_model_comparison.csv"
)

df.to_csv(csv_path, index=False)

print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(df.to_string(index=False))

# Plot metrics comparison
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

models = df["Model"]

x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(12, 7))

for i, model in enumerate(models):

    values = df.loc[i, metrics].astype(float)

    plt.bar(
        x + i * width,
        values,
        width,
        label=model
    )

plt.xticks(
    x + width,
    metrics
)

plt.ylabel("Score")
plt.ylim(0, 1)

plt.title("Centralized ML vs GraphSAGE vs Federated GraphSAGE")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plot_path = os.path.join(
    PLOTS_DIR,
    "final_model_comparison.png"
)

plt.savefig(plot_path, dpi=300)
plt.show()

print("\nComparison chart saved:")
print(plot_path)

print("\nComparison CSV saved:")
print(csv_path)

# Identify best model per metric
print("\n" + "=" * 70)
print("BEST MODEL PER METRIC")
print("=" * 70)

for metric in metrics:

    valid = df.dropna(subset=[metric])

    best_idx = valid[metric].idxmax()

    print(
        f"{metric}: "
        f"{df.loc[best_idx, 'Model']} "
        f"({df.loc[best_idx, metric]:.4f})"
    )

print("\nFINAL COMPARISON COMPLETED!")