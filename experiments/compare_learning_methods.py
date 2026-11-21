import pandas as pd
import matplotlib.pyplot as plt

# Load results
centralized = pd.read_csv("results/centralized_results.csv")
fedavg = pd.read_csv("results/federated_learning/fedavg_results.csv")
fedprox = pd.read_csv("results/federated_learning/fedprox_results.csv")

# Final-round federated results
comparison = pd.DataFrame({
    "Method": ["Centralized", "FedAvg", "FedProx"],
    "Loss": [
        centralized["test_loss"].iloc[0],
        fedavg["aggregated_loss"].iloc[-1],
        fedprox["aggregated_loss"].iloc[-1]
    ]
})

print("\nLearning Method Comparison")
print(comparison)

comparison.to_csv(
    "results/learning_method_comparison.csv",
    index=False
)

# Plot
plt.figure(figsize=(9, 6))

plt.bar(comparison["Method"], comparison["Loss"])

plt.ylabel("Loss")
plt.title("Centralized vs Federated Learning")
plt.grid(axis="y")

plt.savefig(
    "results/learning_method_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nComparison saved successfully!")