import pandas as pd
import os

results = {
    "attack": ["Confidence-Based Membership Inference"],
    "attack_accuracy": [0.2000],
    "attack_precision": [0.0000],
    "attack_recall": [0.0000],
    "attack_auc": [0.4984]
}

df = pd.DataFrame(results)

os.makedirs("results/privacy", exist_ok=True)

output_path = "results/privacy/mia_results.csv"

df.to_csv(output_path, index=False)

print("MIA results saved to:", output_path)
print(df)