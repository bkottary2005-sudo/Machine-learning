import pandas as pd
import matplotlib.pyplot as plt
import os

# Load experiment results
file_path = "results/federated_learning/fedavg_results.csv"
df = pd.read_csv(file_path)

# Create output directory if needed
os.makedirs("results/federated_learning", exist_ok=True)

# Plot global loss
plt.figure(figsize=(10, 6))

plt.plot(
    df["round"],
    df["aggregated_loss"],
    marker="o",
    linewidth=2,
    markersize=8,
    label="FedAvg Global Loss"
)

plt.title("Federated Learning - FedAvg Training Loss")
plt.xlabel("Federated Round")
plt.ylabel("Aggregated Loss")
plt.xticks(df["round"])
plt.grid(True)
plt.legend()

# Save graph
output_path = "results/federated_learning/fedavg_loss.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"Graph saved to: {output_path}")