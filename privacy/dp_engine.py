"""
Differential Privacy Engine
=============================
Integrates Opacus with our RNN models to add differential privacy
during training. This protects individual plant data from being
reconstructed from shared model updates.

How it works:
1. Gradient clipping: limits the influence of any single sample
2. Noise injection: adds calibrated Gaussian noise to gradients
3. Privacy accounting: tracks the cumulative privacy budget (ε)

The privacy guarantee is: an adversary cannot distinguish whether
any particular data sample was in the training set (ε-differential privacy).
"""

import torch
from loguru import logger

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logger.warning("Opacus not installed. DP will be disabled.")


def make_model_dp_compatible(model):
    """
    Ensure model is compatible with Opacus.

    Opacus requires:
    - No nn.RNN/LSTM/GRU layers (use manual Linear layers instead)
    - BatchNorm must be replaced with GroupNorm or LayerNorm

    Our Elman RNN and GRU models already use nn.Linear layers,
    so they should be compatible. This function validates and fixes if needed.

    Args:
        model: PyTorch model

    Returns:
        DP-compatible model
    """
    if not OPACUS_AVAILABLE:
        logger.warning("Opacus not available, returning model as-is")
        return model

    # Validate model compatibility
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.warning(f"Model has DP compatibility issues: {errors}")
        logger.info("Attempting to fix model for DP compatibility...")
        model = ModuleValidator.fix(model)
        logger.info("Model fixed for DP compatibility")
    else:
        logger.info("Model is DP-compatible")

    return model


def attach_dp_engine(
    model,
    optimizer,
    data_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    target_epsilon=None,
    target_delta=1e-5,
    epochs=None,
):
    """
    Attach Opacus PrivacyEngine to the training pipeline.

    This modifies:
    - model: wraps to track per-sample gradients
    - optimizer: adds noise to gradients
    - data_loader: wraps with Poisson sampling

    Args:
        model: PyTorch model (must be DP-compatible)
        optimizer: training optimizer
        data_loader: training DataLoader
        noise_multiplier: controls noise level (higher = more privacy)
        max_grad_norm: gradient clipping threshold
        target_epsilon: target privacy budget (optional)
        target_delta: privacy parameter delta
        epochs: total training epochs (for budget calculation)

    Returns:
        Tuple of (model, optimizer, data_loader, privacy_engine)
    """
    if not OPACUS_AVAILABLE:
        logger.warning("Opacus not available. Training without DP.")
        return model, optimizer, data_loader, None

    # Ensure model is compatible
    model = make_model_dp_compatible(model)

    privacy_engine = PrivacyEngine()

    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    logger.info(
        f"DP Engine attached — noise_multiplier: {noise_multiplier}, "
        f"max_grad_norm: {max_grad_norm}"
    )

    return model, optimizer, data_loader, privacy_engine


def get_privacy_spent(privacy_engine, delta=1e-5):
    """
    Get the current privacy budget spent (epsilon).

    Lower epsilon = stronger privacy guarantee.
    Typical values:
    - ε < 1: very strong privacy
    - ε < 10: reasonable privacy
    - ε > 10: weak privacy

    Args:
        privacy_engine: Opacus PrivacyEngine
        delta: privacy parameter

    Returns:
        epsilon value (float)
    """
    if privacy_engine is None:
        return float("inf")

    try:
        epsilon = privacy_engine.get_epsilon(delta)
        return epsilon
    except Exception as e:
        logger.warning(f"Could not compute epsilon: {e}")
        return float("inf")
