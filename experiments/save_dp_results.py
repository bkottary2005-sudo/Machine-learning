import pandas as pd
import os

results = {
    "model": ["Differential Privacy"],
    "test_loss": [0.5691],
    "test_accuracy": [0.8308],
    "epsilon": [0.6048],
    "delta": [1e-5],
    "noise_multiplier": [1.0],
    "max_grad_norm": [1.0]
}

df = pd.DataFrame(results)

os.makedirs("results/privacy", exist_ok=True)

output_path = "results/privacy/dp_results.csv"

df.to_csv(output_path, index=False)

print(df)
print(f"\nDP results saved to: {output_path}")