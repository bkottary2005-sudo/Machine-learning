import pandas as pd
import matplotlib.pyplot as plt
import os

# Load results
fedavg = pd.read_csv("results/federated_learning/fedavg_results.csv")
fedprox = pd.read_csv("results/federated_learning/fedprox_results.csv")

# Create comparison plot
plt.figure(figsize=(10, 6))

plt.plot(
    fedavg["round"],
    fedavg["aggregated_loss"],
    marker="o",
    label="FedAvg"
)

plt.plot(
    fedprox["round"],
    fedprox["aggregated_loss"],
    marker="s",
    label="FedProx"
)

plt.xlabel("Federated Round")
plt.ylabel("Aggregated Loss")
plt.title("FedAvg vs FedProx Performance")
plt.xticks([1, 2, 3, 4, 5])
plt.legend()
plt.grid(True)

# Save graph
output_path = "results/federated_learning/fedavg_vs_fedprox.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"Comparison graph saved to: {output_path}")