"""
Adversarial Robustness Evaluation
===================================
Evaluates model performance under adversarial attacks.
Computes clean accuracy, adversarial accuracy, and attack success rate.
"""

import numpy as np
from loguru import logger

from adversarial.attacks import run_fgsm_attack, run_pgd_attack
from models.wrapper import create_art_classifier, predictions_to_labels


def evaluate_robustness(model, test_windows, test_labels, config, device="cpu"):
    """
    Complete adversarial robustness evaluation.

    Steps:
    1. Evaluate clean accuracy (no attack)
    2. Run FGSM attack and evaluate
    3. Run PGD attack and evaluate
    4. Compute attack success rates

    Args:
        model: trained PyTorch model
        test_windows: numpy array (num_samples, seq_len, num_features)
        test_labels: numpy array (num_samples,) with values 0 or 1
        config: configuration dict with adversarial settings
        device: 'cpu' or 'cuda'

    Returns:
        dict with robustness metrics
    """
    adv_config = config.get("adversarial", {})
    num_samples = min(adv_config.get("num_attack_samples", 500), len(test_windows))

    # Subsample for efficiency
    indices = np.random.choice(len(test_windows), num_samples, replace=False)
    X_test = test_windows[indices].astype(np.float32)
    y_test = test_labels[indices].astype(int)

    # Convert labels to one-hot for ART
    y_test_onehot = np.eye(2)[y_test]

    input_shape = (X_test.shape[1], X_test.shape[2])  # (seq_len, num_features)

    # Create ART classifier
    model.eval()
    art_classifier = create_art_classifier(model, input_shape, device)

    # --- Clean accuracy ---
    clean_preds = art_classifier.predict(X_test)
    clean_labels = predictions_to_labels(clean_preds)
    clean_accuracy = np.mean(clean_labels == y_test)
    logger.info(f"Clean accuracy: {clean_accuracy:.4f}")

    # --- FGSM Attack ---
    fgsm_eps = adv_config.get("fgsm_epsilon", 0.1)
    fgsm_adv = run_fgsm_attack(art_classifier, X_test, epsilon=fgsm_eps)
    fgsm_preds = art_classifier.predict(fgsm_adv)
    fgsm_labels = predictions_to_labels(fgsm_preds)
    fgsm_accuracy = np.mean(fgsm_labels == y_test)
    fgsm_success_rate = 1.0 - fgsm_accuracy
    logger.info(f"FGSM — accuracy: {fgsm_accuracy:.4f}, attack success: {fgsm_success_rate:.4f}")

    # --- PGD Attack ---
    pgd_eps = adv_config.get("pgd_epsilon", 0.1)
    pgd_step = adv_config.get("pgd_step_size", 0.01)
    pgd_iter = adv_config.get("pgd_max_iter", 40)
    pgd_adv = run_pgd_attack(art_classifier, X_test, pgd_eps, pgd_step, pgd_iter)
    pgd_preds = art_classifier.predict(pgd_adv)
    pgd_labels = predictions_to_labels(pgd_preds)
    pgd_accuracy = np.mean(pgd_labels == y_test)
    pgd_success_rate = 1.0 - pgd_accuracy
    logger.info(f"PGD — accuracy: {pgd_accuracy:.4f}, attack success: {pgd_success_rate:.4f}")

    results = {
        "num_samples": num_samples,
        "clean_accuracy": float(clean_accuracy),
        "fgsm": {
            "epsilon": fgsm_eps,
            "adversarial_accuracy": float(fgsm_accuracy),
            "attack_success_rate": float(fgsm_success_rate),
        },
        "pgd": {
            "epsilon": pgd_eps,
            "step_size": pgd_step,
            "max_iter": pgd_iter,
            "adversarial_accuracy": float(pgd_accuracy),
            "attack_success_rate": float(pgd_success_rate),
        },
    }

    logger.info("Adversarial robustness evaluation complete")
    return results
