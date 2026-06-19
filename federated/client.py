"""
Flower Federated Client
========================
Each client represents a water treatment plant that:
1. Receives global model weights from the server
2. Trains locally on its own sensor data
3. Optionally applies differential privacy to gradients
4. Returns updated model weights to the server
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from loguru import logger

import flwr as fl

from privacy.dp_engine import attach_dp_engine, get_privacy_spent


class CPSFlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated CPS intrusion detection.

    Each client holds a local dataset (partition of the full training data)
    and trains a local copy of the model. Differential privacy can optionally
    be applied to protect sensitive plant data.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        local_epochs=3,
        learning_rate=0.001,
        dp_enabled=False,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        client_id="unknown",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.dp_enabled = dp_enabled
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.client_id = client_id

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        # Attach differential privacy engine if enabled
        self.privacy_engine = None
        if dp_enabled:
            self.model, self.optimizer, self.train_loader, self.privacy_engine = (
                attach_dp_engine(
                    self.model,
                    self.optimizer,
                    self.train_loader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=max_grad_norm,
                )
            )
            logger.info(f"Client {client_id}: DP enabled (noise={noise_multiplier}, clip={max_grad_norm})")

    def get_parameters(self, config=None):
        """Return model parameters as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train the model locally for a number of epochs.

        Steps:
        1. Set global model weights received from server
        2. Train on local data
        3. Return updated weights + number of training samples
        """
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / max(num_batches, 1)

        # Log privacy budget if DP is enabled
        epsilon_spent = None
        if self.privacy_engine is not None:
            epsilon_spent = get_privacy_spent(self.privacy_engine)
            logger.info(f"Client {self.client_id}: ε = {epsilon_spent:.2f}")

        logger.info(
            f"Client {self.client_id}: trained {self.local_epochs} epochs, "
            f"avg_loss={avg_loss:.4f}"
        )

        return (
            self.get_parameters(),
            len(self.train_loader.dataset),
            {"loss": avg_loss, "epsilon": epsilon_spent},
        )

    def evaluate(self, parameters, config):
        """Evaluate the model on local validation data."""
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)

                total_loss += loss.item() * len(batch_y)
                predicted = (predictions >= 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += len(batch_y)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        logger.info(f"Client {self.client_id}: eval loss={avg_loss:.4f}, acc={accuracy:.4f}")
        return avg_loss, total, {"accuracy": accuracy}


def create_flower_client(
    model, train_loader, val_loader, device, config, client_id="A"
):
    """Factory function to create a Flower client with config settings."""
    dp_config = config.get("privacy", {})
    fed_config = config.get("federated", {})
    train_config = config.get("training", {})

    return CPSFlowerClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        local_epochs=fed_config.get("local_epochs", 3),
        learning_rate=train_config.get("learning_rate", 0.001),
        dp_enabled=dp_config.get("enabled", True),
        noise_multiplier=dp_config.get("noise_multiplier", 1.0),
        max_grad_norm=dp_config.get("max_grad_norm", 1.0),
        client_id=client_id,
    )
