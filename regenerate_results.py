import os
import sys
import yaml
import json
import torch
import joblib
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_data
from preprocessing.pipeline import run_preprocessing_pipeline
from models.elman_rnn import ElmanRNN
from experiments.runner import evaluate_model
from adversarial.evaluation import evaluate_robustness
from xai.shap_explainer import SHAPExplainer
from fidelity.verifier import FidelityVerifier

def main():
    device = torch.device("cpu")
    logger.info("Starting local regeneration of experiment results...")

    # Set Kaggle credentials from environment variables
    kaggle_user = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    if kaggle_user and kaggle_key:
        os.environ['KAGGLE_USERNAME'] = kaggle_user
        os.environ['KAGGLE_KEY'] = kaggle_key
    else:
        logger.warning("KAGGLE_USERNAME or KAGGLE_KEY not found in environment variables.")

    results_dir = os.path.join("experiments", "results")
    config_path = os.path.join(results_dir, "colab_config.yaml")
    model_path = os.path.join(results_dir, "colab_trained_model.pt")
    scaler_path = os.path.join(results_dir, "colab_scaler.pkl")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 1. Load and Subsample Data
    logger.info("Loading Kaggle dataset...")
    df = load_data(config)
    
    # We use a smaller subset of the Kaggle data (to prevent your laptop from crashing/running out of RAM during the evaluations)
    normal_df = df[df["label"] == 0].head(50000)
    attack_df = df[df["label"] == 1].head(15000)
    df = pd.concat([normal_df, attack_df]).reset_index(drop=True)

    # 2. Preprocess using the trained Colab scaler
    logger.info("Loading Colab scaler and running preprocessing pipeline...")
    colab_scaler = joblib.load(scaler_path)
    pipeline_result = run_preprocessing_pipeline(df, config, scaler=colab_scaler)

    # 3. Load Model
    logger.info("Loading Colab trained model weights...")
    model = ElmanRNN(
        input_size=pipeline_result["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Standard Evaluation
    logger.info("Evaluating clean metrics...")
    metrics = evaluate_model(model, pipeline_result["test_loader"], device)
    logger.info(f"Metrics: {metrics}")

    # 5. Robustness Testing
    logger.info("Evaluating adversarial robustness...")
    test_ds = pipeline_result["datasets"]["test"]
    test_windows = test_ds.windows.numpy()
    test_labels = test_ds.labels.numpy()
    robustness = evaluate_robustness(model, test_windows, test_labels, config, device)

    # 6. SHAP Explainability
    logger.info("Running SHAP explainability analysis...")
    feature_columns = pipeline_result["feature_columns"]
    explainer = SHAPExplainer(model, test_windows[:100], feature_columns, device)
    explanation = explainer.explain(test_windows[-50:], nsamples=100)
    
    # 7. Fidelity Verification
    logger.info("Running Fidelity verification...")
    verifier = FidelityVerifier(model, feature_columns, device)
    fidelity = verifier.verify(test_windows[-50:], explanation["shap_values"], top_k=5)

    # 8. Load old training history if available to keep it
    old_results_path = os.path.join(results_dir, "experiment_results.json")
    training_history = {}
    if os.path.exists(old_results_path):
        with open(old_results_path) as f:
            old_res = json.load(f)
            training_history = old_res.get("training_history", {})

    # 9. Save all results
    all_results = {
        "metrics": metrics,
        "robustness": robustness,
        "fidelity": {
            "fidelity_score": fidelity["fidelity_score"],
            "faithfulness_ratio": fidelity["faithfulness_ratio"],
            "top_features": fidelity["top_masked_features"]
        },
        "training_history": training_history
    }
    
    with open(old_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"✅ Successfully regenerated and saved full results to {old_results_path}")

if __name__ == "__main__":
    main()
