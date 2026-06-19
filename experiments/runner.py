"""
Experiment Runner
==================
Orchestrates the complete pipeline:
  1. Load data (Kaggle / CSV / synthetic)
  2. Preprocess (normalize, window, split, partition)
  3. Create model
  4. Run federated training with DP
  5. Evaluate on test set
  6. (Optional) Adversarial robustness testing
  7. (Optional) SHAP explanations + fidelity verification
  8. Save results to MongoDB

Usage:
    python -m experiments.runner --config config/local.yaml
"""

import os
import sys
import yaml
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path="config/default.yaml", override_path=None):
    """Load config YAML, optionally merging with override file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if override_path and os.path.exists(override_path):
        with open(override_path) as f:
            override = yaml.safe_load(f)
        # Deep merge: override values take precedence
        for key, value in override.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    return config


def get_device(config):
    """Determine torch device based on config."""
    device_str = config.get("training", {}).get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def create_model(config, input_size):
    """Create model based on config settings."""
    model_type = config.get("model", {}).get("type", "elman")
    hidden_size = config.get("model", {}).get("hidden_size", 64)
    num_layers = config.get("model", {}).get("num_layers", 1)
    dropout = config.get("model", {}).get("dropout", 0.1)

    if model_type == "elman":
        from models.elman_rnn import ElmanRNN
        model = ElmanRNN(input_size, hidden_size, num_layers, dropout)
    elif model_type == "gru":
        from models.gru_model import GRUModel
        model = GRUModel(input_size, hidden_size, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Created {model_type} model: input={input_size}, hidden={hidden_size}, layers={num_layers}")
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model and return classification metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend((preds >= 0.5).astype(int))
            all_labels.extend(batch_y.numpy().astype(int))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1_score": float(f1_score(all_labels, all_preds, zero_division=0)),
    }

    logger.info(
        f"Test Metrics — Acc: {metrics['accuracy']:.4f}, "
        f"Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1_score']:.4f}"
    )
    return metrics


def run_experiment(config, experiment_name="experiment"):
    """
    Run the complete experiment pipeline.

    Returns:
        dict with model, metrics, scaler, feature_columns, test_data
    """
    logger.info(f"{'='*60}")
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"{'='*60}")

    device = get_device(config)
    logger.info(f"Device: {device}")

    # Step 1: Load data
    logger.info("Step 1: Loading data...")
    from data.loader import load_data
    df = load_data(config)

    # Step 2: Preprocess
    logger.info("Step 2: Preprocessing...")
    from preprocessing.pipeline import run_preprocessing_pipeline
    pipeline_result = run_preprocessing_pipeline(df, config)

    input_size = pipeline_result["input_size"]
    train_loader = pipeline_result["train_loader"]
    val_loader = pipeline_result["val_loader"]
    test_loader = pipeline_result["test_loader"]
    client_datasets = pipeline_result["client_datasets"]

    # Step 3: Create model
    logger.info("Step 3: Creating model...")
    model = create_model(config, input_size)
    model.to(device)

    # Step 4: Federated training
    logger.info("Step 4: Federated training with DP...")
    from federated.server import run_federated_simulation
    fed_results = run_federated_simulation(
        model, client_datasets, val_loader, device, config
    )
    model = fed_results["model"]

    # Step 5: Evaluate
    logger.info("Step 5: Evaluating on test set...")
    metrics = evaluate_model(model, test_loader, device)

    # Prepare test data for API use
    test_ds = pipeline_result["datasets"]["test"]
    test_data = {
        "windows": test_ds.windows.numpy(),
        "labels": test_ds.labels.numpy(),
    }

    # Save model weights, scaler, and config
    import joblib
    os.makedirs("experiments/results", exist_ok=True)
    
    model_path = f"experiments/results/{experiment_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    
    scaler_path = f"experiments/results/{experiment_name}_scaler.pkl"
    joblib.dump(pipeline_result["scaler"], scaler_path)
    
    config_path = f"experiments/results/{experiment_name}_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    logger.info(f"Artifacts saved: model, scaler, and config to experiments/results/")

    logger.info(f"{'='*60}")
    logger.info(f"Experiment '{experiment_name}' complete!")
    logger.info(f"{'='*60}")

    return {
        "model": model,
        "metrics": metrics,
        "history": fed_results["history"],
        "scaler": pipeline_result["scaler"],
        "feature_columns": pipeline_result["feature_columns"],
        "test_data": test_data,
    }


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CPS Security Experiment")
    parser.add_argument("--config", default="config/local.yaml", help="Config file path")
    parser.add_argument("--name", default="experiment_001", help="Experiment name")
    args = parser.parse_args()

    # Load config (merge with defaults)
    base_config = load_config("config/default.yaml")
    config = load_config("config/default.yaml", args.config)

    results = run_experiment(config, args.name)

    print("\n📊 Final Results:")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.4f}")
