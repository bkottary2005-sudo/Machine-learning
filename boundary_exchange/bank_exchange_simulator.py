import pandas as pd
import os


def simulate_bank_exchange():

    exchange_file = "Data/boundary_nodes/exchange_data.csv"

    exchange_df = pd.read_csv(exchange_file)

    banks = ["Bank_A", "Bank_B", "Bank_C"]

    os.makedirs("Data/boundary_nodes/exchanged", exist_ok=True)

    for bank in banks:

        # Select boundary accounts relevant to this bank
        bank_data = exchange_df[
            exchange_df["banks_involved"].str.contains(bank)
        ].copy()

        bank_data["received_by"] = bank
        bank_data["exchange_status"] = "received"

        output_file = (
            f"Data/boundary_nodes/exchanged/{bank}_received.csv"
        )

        bank_data.to_csv(output_file, index=False)

        print(f"\n{bank}")
        print("Boundary accounts received:", len(bank_data))
        print("Saved to:", output_file)

    print("\nBank-to-bank exchange simulation completed!")


if __name__ == "__main__":
    simulate_bank_exchange()