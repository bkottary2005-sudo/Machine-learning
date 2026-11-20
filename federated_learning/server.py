import flwr as fl

NUM_CLIENTS = 3
NUM_ROUNDS = 5


def main():
    print("Starting FinGraph Federated Learning Server...")
    print("Waiting for 3 bank clients...")
    print(f"Federated rounds: {NUM_ROUNDS}")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()