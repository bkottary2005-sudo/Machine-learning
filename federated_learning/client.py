import sys
import torch
import flwr as fl

from federated_learning.model import AMLModel
from federated_learning.train_utils import load_bank_data

BANK_NAME = sys.argv[1] if len(sys.argv) > 1 else "Bank_A"

DEVICE = torch.device("cpu")


class BankClient(fl.client.NumPyClient):

    def __init__(self):
        self.model = AMLModel(input_size=9).to(DEVICE)

        self.X, self.y = load_bank_data(BANK_NAME)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0001
        )

        self.criterion = torch.nn.BCELoss()

    def get_parameters(self, config):
        return [
            val.cpu().numpy()
            for val in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()

        new_state_dict = {
            key: torch.tensor(value)
            for key, value in zip(state_dict.keys(), parameters)
        }

        self.model.load_state_dict(new_state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()

        epochs = 2

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            outputs = self.model(self.X)

            loss = self.criterion(outputs, self.y)

            loss.backward()

            self.optimizer.step()

            print(
                f"{BANK_NAME} | Epoch {epoch + 1} "
                f"| Loss: {loss.item():.4f}"
            )

        return self.get_parameters(config), len(self.X), {
            "bank": BANK_NAME,
            "loss": float(loss.item())
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(self.X)

            loss = self.criterion(outputs, self.y)

            predictions = (outputs >= 0.5).float()

            accuracy = (
                (predictions == self.y).float().mean().item()
            )

        return float(loss.item()), len(self.X), {
            "accuracy": accuracy,
            "bank": BANK_NAME
        }


if __name__ == "__main__":
    print(f"Starting client for {BANK_NAME}")

    client = BankClient()

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )