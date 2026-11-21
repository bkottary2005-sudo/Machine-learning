import pandas as pd
import matplotlib.pyplot as plt

# Load MIA results
df = pd.read_csv("results/privacy/mia_results.csv")

metrics = [
    "attack_accuracy",
    "attack_precision",
    "attack_recall",
    "attack_auc"
]

values = df[metrics].iloc[0]

plt.figure(figsize=(9, 5))

plt.bar(metrics, values)

plt.axhline(
    y=0.5,
    linestyle="--",
    label="Random Guess Baseline"
)

plt.title("Membership Inference Attack Results")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/privacy/mia_visualization.png",
    dpi=300
)

plt.show()

print("MIA visualization saved successfully!")