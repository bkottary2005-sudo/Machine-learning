import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "value",
    "time",
    "amount_log",
    "amount_deviation",
    "time_step",
    "sender_transaction_count",
    "receiver_transaction_count",
    "sender_total_amount",
    "receiver_total_amount",
]

TARGET_COLUMN = "isFraud"


def load_bank_data(bank_name):
    file_path = f"Data/bank_partitions/{bank_name}_labeled.csv"

    df = pd.read_csv(file_path)

    X = df[FEATURE_COLUMNS].fillna(0).values
    y = df[TARGET_COLUMN].values

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    return X, y


if __name__ == "__main__":
    for bank in ["Bank_A", "Bank_B", "Bank_C"]:
        X, y = load_bank_data(bank)

        print(f"\n{bank}")
        print("Features shape:", X.shape)
        print("Labels shape:", y.shape)
        print("Fraud samples:", int(y.sum().item()))
        print("Legitimate samples:", int((y == 0).sum().item()))
        print("Feature mean:", X.mean().item())
        print("Feature std:", X.std().item())