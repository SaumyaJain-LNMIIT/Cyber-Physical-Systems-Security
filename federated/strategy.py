"""
Custom Federated Aggregation Strategy
=======================================
Weighted FedAvg that handles unbalanced client data sizes.
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from loguru import logger
from typing import List, Tuple, Optional, Dict


class WeightedFedAvg(FedAvg):
    """
    Weighted Federated Averaging strategy.

    Extends Flower's FedAvg to:
    - Log per-round metrics
    - Track aggregation history
    - Support configurable minimum clients
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics = []

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model updates with logging."""
        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}

        # Log client metrics
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            logger.info(
                f"Round {server_round} | Client samples: {fit_res.num_examples} | "
                f"Loss: {metrics.get('loss', 'N/A')}"
            )

        # Use parent's weighted averaging
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        logger.info(f"Round {server_round}: Aggregated {len(results)} client updates")

        self.round_metrics.append({
            "round": server_round,
            "num_clients": len(results),
            "failures": len(failures),
        })

        return aggregated_params, aggregated_metrics
