import pandas as pd
import os

results = {
    "model": ["Centralized"],
    "test_loss": [0.6288],
    "test_accuracy": [0.8309]
}

df = pd.DataFrame(results)

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "centralized_results.csv"
)

df.to_csv(output_path, index=False)

print(df)
print(f"\nResults saved to: {output_path}")