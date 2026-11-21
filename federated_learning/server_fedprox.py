import flwr as fl

NUM_CLIENTS = 3
NUM_ROUNDS = 5

# FedProx proximal term
PROXIMAL_MU = 0.1


def main():
    print("Starting FinGraph FedProx Server...")
    print("Waiting for 3 bank clients...")
    print(f"Federated rounds: {NUM_ROUNDS}")
    print(f"Proximal mu: {PROXIMAL_MU}")

    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        proximal_mu=PROXIMAL_MU,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8081",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()