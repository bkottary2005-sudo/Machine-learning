import pandas as pd
import os

results = {
    "round": [1, 2, 3, 4, 5],
    "aggregated_loss": [
        0.6973682279,
        0.6954350283,
        0.6935009939,
        0.6915657097,
        0.6896289356
    ]
}

df = pd.DataFrame(results)

output_dir = "results/federated_learning"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "fedprox_results.csv")

df.to_csv(output_path, index=False)

print(df)
print(f"\nFedProx results saved to: {output_path}")