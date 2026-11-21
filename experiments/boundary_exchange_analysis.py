import pandas as pd
import matplotlib.pyplot as plt
import os

banks = ["Bank_A", "Bank_B", "Bank_C"]
counts = []

for bank in banks:
    file_path = f"Data/boundary_nodes/exchanged/{bank}_received.csv"

    df = pd.read_csv(file_path)

    counts.append({
        "Bank": bank,
        "Boundary_Accounts_Received": len(df)
    })

results = pd.DataFrame(counts)

print("\nBoundary Exchange Analysis")
print(results)

os.makedirs("results/boundary_exchange", exist_ok=True)

results.to_csv(
    "results/boundary_exchange/exchange_analysis.csv",
    index=False
)

plt.figure(figsize=(8, 5))

plt.bar(
    results["Bank"],
    results["Boundary_Accounts_Received"]
)

plt.title("Boundary Node Exchange Across Banks")
plt.xlabel("Bank")
plt.ylabel("Number of Boundary Accounts Received")

plt.tight_layout()

plt.savefig(
    "results/boundary_exchange/exchange_analysis.png",
    dpi=300
)

plt.show()

print("\nAnalysis saved successfully!")