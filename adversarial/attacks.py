"""
Adversarial Attacks
====================
Implements FGSM and PGD attacks using IBM's Adversarial Robustness Toolbox.

These attacks simulate a hacker trying to fool the intrusion detection
model by slightly modifying sensor readings to make attack data look normal.

FGSM (Fast Gradient Sign Method):
  - Single-step attack
  - Adds epsilon * sign(gradient) to input
  - Fast but less effective

PGD (Projected Gradient Descent):
  - Multi-step iterative attack
  - More powerful than FGSM
  - Considered the strongest first-order attack
"""

import numpy as np
from loguru import logger

try:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    logger.warning("ART not installed. Adversarial testing will be disabled.")


def run_fgsm_attack(art_classifier, test_data, epsilon=0.1):
    """
    Run FGSM attack on test data.

    Args:
        art_classifier: ART PyTorchClassifier wrapper
        test_data: numpy array (num_samples, seq_len, num_features)
        epsilon: perturbation magnitude (higher = stronger attack)

    Returns:
        adversarial_data: perturbed numpy array, same shape as test_data
    """
    if not ART_AVAILABLE:
        logger.error("ART not available. Cannot run FGSM attack.")
        return test_data

    logger.info(f"Running FGSM attack (ε={epsilon}) on {len(test_data)} samples...")

    attack = FastGradientMethod(
        estimator=art_classifier,
        eps=epsilon,
        eps_step=epsilon,  # single step
        minimal=False,
    )

    adversarial_data = attack.generate(x=test_data)

    # Measure perturbation magnitude
    perturbation = np.abs(adversarial_data - test_data)
    logger.info(
        f"FGSM complete — mean perturbation: {perturbation.mean():.6f}, "
        f"max perturbation: {perturbation.max():.6f}"
    )

    return adversarial_data


def run_pgd_attack(art_classifier, test_data, epsilon=0.1, step_size=0.01, max_iter=40):
    """
    Run PGD attack on test data.

    PGD is an iterative version of FGSM that takes multiple smaller steps
    and projects back onto the epsilon-ball after each step.

    Args:
        art_classifier: ART PyTorchClassifier wrapper
        test_data: numpy array (num_samples, seq_len, num_features)
        epsilon: maximum perturbation magnitude
        step_size: step size per iteration
        max_iter: maximum number of iterations

    Returns:
        adversarial_data: perturbed numpy array
    """
    if not ART_AVAILABLE:
        logger.error("ART not available. Cannot run PGD attack.")
        return test_data

    logger.info(
        f"Running PGD attack (ε={epsilon}, step={step_size}, iter={max_iter}) "
        f"on {len(test_data)} samples..."
    )

    attack = ProjectedGradientDescent(
        estimator=art_classifier,
        eps=epsilon,
        eps_step=step_size,
        max_iter=max_iter,
        targeted=False,
    )

    adversarial_data = attack.generate(x=test_data)

    perturbation = np.abs(adversarial_data - test_data)
    logger.info(
        f"PGD complete — mean perturbation: {perturbation.mean():.6f}, "
        f"max perturbation: {perturbation.max():.6f}"
    )

    return adversarial_data
