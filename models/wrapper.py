"""
Model Wrapper for ART (Adversarial Robustness Toolbox)
=======================================================
Wraps our PyTorch RNN models to be compatible with IBM's ART library.
ART expects a specific interface for generating adversarial examples.
"""

import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from loguru import logger


def create_art_classifier(model, input_shape, device="cpu", learning_rate=0.001):
    """
    Wrap a PyTorch model as an ART PyTorchClassifier.

    ART requires:
    - model: a PyTorch nn.Module
    - loss: loss function
    - optimizer: optimizer
    - input_shape: shape of a single input sample (seq_len, num_features)
    - nb_classes: number of output classes

    For binary classification, we treat it as 2-class by modifying the output.

    Args:
        model: trained PyTorch model (ElmanRNN or GRUModel)
        input_shape: (sequence_length, num_features)
        device: 'cpu' or 'cuda'
        learning_rate: for the optimizer (used by ART internally)

    Returns:
        ART PyTorchClassifier wrapper
    """

    # Create a wrapper that outputs 2-class probabilities
    # ART expects shape (batch, num_classes)
    class BinaryToMulticlass(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            prob_attack = self.base_model(x)  # (batch,)
            prob_normal = 1 - prob_attack
            return torch.stack([prob_normal, prob_attack], dim=1)  # (batch, 2)

    wrapped_model = BinaryToMulticlass(model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=learning_rate)

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=2,
        device_type=device,
    )

    logger.info(f"Created ART classifier wrapper with input_shape={input_shape}")
    return classifier


def predictions_to_labels(predictions):
    """Convert ART 2-class predictions back to binary labels."""
    return np.argmax(predictions, axis=1)
