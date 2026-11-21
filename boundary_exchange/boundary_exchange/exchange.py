import pandas as pd
import os


def prepare_boundary_exchange():

    boundary_file = "Data/boundary_nodes/boundary_nodes.csv"

    boundary_df = pd.read_csv(boundary_file)

    # Prepare exchange information
    exchange_data = boundary_df.copy()

    exchange_data["exchange_status"] = "prepared"

    # Create output directory
    os.makedirs("Data/boundary_nodes", exist_ok=True)

    output_file = "Data/boundary_nodes/exchange_data.csv"

    exchange_data.to_csv(output_file, index=False)

    print("Boundary exchange data prepared successfully!")
    print("Total accounts:", len(exchange_data))
    print("Saved to:", output_file)

    print("\nSample exchange data:")
    print(exchange_data.head(10))


if __name__ == "__main__":
    prepare_boundary_exchange()