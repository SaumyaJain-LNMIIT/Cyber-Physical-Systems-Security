"""
Federated Learning Server
==========================
Orchestrates federated training using Flower's simulation mode.
Runs all clients as threads on a single machine (no network needed).
"""

import copy
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from loguru import logger


def run_federated_simulation(model, client_datasets, val_loader, device, config):
    """
    Run federated learning simulation without Flower server overhead.

    This is a simplified simulation that:
    1. Initializes a global model
    2. For each round:
       a. Send global weights to all clients
       b. Each client trains locally
       c. Collect updated weights
       d. Aggregate using weighted FedAvg
    3. Return the final global model

    Args:
        model: initial PyTorch model
        client_datasets: list of SWaTWindowDataset for each client
        val_loader: validation DataLoader
        device: torch device
        config: configuration dict

    Returns:
        dict with trained model, metrics history
    """
    import torch.nn as nn
    from privacy.dp_engine import attach_dp_engine, get_privacy_spent

    fed_config = config.get("federated", {})
    dp_config = config.get("privacy", {})
    train_config = config.get("training", {})

    num_rounds = fed_config.get("num_rounds", 10)
    local_epochs = fed_config.get("local_epochs", 3)
    batch_size = train_config.get("batch_size", 64)
    lr = train_config.get("learning_rate", 0.001)
    dp_enabled = dp_config.get("enabled", True)

    criterion = nn.BCELoss()
    history = {"round": [], "val_loss": [], "val_accuracy": [], "client_losses": []}

    logger.info(f"Starting federated simulation: {num_rounds} rounds, {len(client_datasets)} clients")

    # Global model weights
    global_state = copy.deepcopy(model.state_dict())

    for round_num in range(1, num_rounds + 1):
        client_weights = []
        client_num_samples = []
        round_losses = []

        # --- Client training ---
        for client_idx, client_ds in enumerate(client_datasets):
            # Create a fresh model copy with global weights
            client_model = copy.deepcopy(model)
            client_model.load_state_dict(copy.deepcopy(global_state))
            client_model.to(device)
            client_model.train()

            client_loader = DataLoader(client_ds, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)

            # Apply DP if enabled
            privacy_engine = None
            if dp_enabled:
                try:
                    client_model, optimizer, client_loader, privacy_engine = (
                        attach_dp_engine(
                            client_model, optimizer, client_loader,
                            noise_multiplier=dp_config.get("noise_multiplier", 1.0),
                            max_grad_norm=dp_config.get("max_grad_norm", 1.0),
                        )
                    )
                except Exception as e:
                    logger.warning(f"DP attach failed for client {client_idx}: {e}")

            # Local training
            total_loss = 0.0
            n_batches = 0
            for epoch in range(local_epochs):
                for batch_x, batch_y in client_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    preds = client_model(batch_x)
                    loss = criterion(preds, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            round_losses.append(avg_loss)

            # Log DP epsilon
            if privacy_engine:
                eps = get_privacy_spent(privacy_engine)
                logger.debug(f"  Client {client_idx}: loss={avg_loss:.4f}, ε={eps:.2f}")
            else:
                logger.debug(f"  Client {client_idx}: loss={avg_loss:.4f}")

            # Collect weights — strip Opacus '_module.' prefix if present
            raw_state = copy.deepcopy(client_model.state_dict())
            clean_state = OrderedDict()
            for k, v in raw_state.items():
                # Opacus wraps model with GradSampleModule, adding '_module.' prefix
                clean_key = k.replace("_module.", "") if k.startswith("_module.") else k
                clean_state[clean_key] = v
            client_weights.append(clean_state)
            client_num_samples.append(len(client_ds))

        # --- Weighted FedAvg aggregation ---
        global_state = _weighted_fedavg(client_weights, client_num_samples)

        # --- Evaluate global model ---
        model.load_state_dict(global_state)
        model.to(device)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        history["round"].append(round_num)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["client_losses"].append(round_losses)

        logger.info(
            f"Round {round_num}/{num_rounds} — "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
            f"client_losses: {[f'{l:.4f}' for l in round_losses]}"
        )

    # Load final weights
    model.load_state_dict(global_state)
    logger.info("Federated simulation complete!")

    return {"model": model, "history": history}


def _weighted_fedavg(client_weights, client_num_samples):
    """
    Weighted Federated Averaging.
    Each client's weights are weighted by the number of training samples.
    """
    total_samples = sum(client_num_samples)
    avg_state = OrderedDict()

    for key in client_weights[0].keys():
        weighted_sum = sum(
            w[key].float() * (n / total_samples)
            for w, n in zip(client_weights, client_num_samples)
        )
        avg_state[key] = weighted_sum

    return avg_state


def _evaluate(model, data_loader, criterion, device):
    """Evaluate model on a DataLoader. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            total_loss += loss.item() * len(batch_y)
            predicted = (preds >= 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += len(batch_y)

    return total_loss / max(total, 1), correct / max(total, 1)
