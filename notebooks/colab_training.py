"""
CPS Security Project — Google Colab Training Script
=====================================================
Upload this to Colab and run cells sequentially.
Each section marked with #%% is a separate Colab cell.

Instructions:
1. Upload entire cps_security_project folder to Colab
2. Set Runtime → Change runtime type → GPU (T4)
3. Run each cell in order
"""

#%% Cell 1: Setup Environment
# !pip install torch numpy pandas scikit-learn pyyaml loguru tqdm
# !pip install opacus shap adversarial-robustness-toolbox flwr kagglehub
# !pip install matplotlib plotly seaborn

#%% Cell 2: Configure Kaggle (run once)
# Upload your kaggle.json or set credentials:
# import os
# os.environ['KAGGLE_USERNAME'] = 'your_username'
# os.environ['KAGGLE_KEY'] = 'your_key'

#%% Cell 3: Import and configure
import os, sys, yaml, torch
import numpy as np

# If running from notebooks/ folder
sys.path.insert(0, os.path.join(os.getcwd(), '..'))
sys.path.insert(0, os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

#%% Cell 4: Load Colab config
COLAB_CONFIG = {
    "data": {
        "source": "kaggle",
        "kaggle_dataset": "vishala28/swat-dataset-secure-water-treatment-system",
        "synthetic_num_normal": 50000,
        "synthetic_num_attack": 15000,
        "test_size": 0.2,
        "val_size": 0.1,
        "random_seed": 42,
    },
    "model": {
        "type": "elman",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "sequence_length": 30,
    },
    "training": {
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 0.001,
        "device": "auto",
    },
    "federated": {
        "num_clients": 3,
        "num_rounds": 20,
        "local_epochs": 5,
    },
    "privacy": {
        "enabled": True,
        "noise_multiplier": 1.0,
        "max_grad_norm": 1.0,
    },
    "adversarial": {
        "fgsm_epsilon": 0.1,
        "pgd_epsilon": 0.1,
        "pgd_step_size": 0.01,
        "pgd_max_iter": 40,
        "num_attack_samples": 1000,
    },
    "xai": {
        "num_background_samples": 200,
        "num_explain_samples": 100,
        "top_k_features": 5,
    },
    "fidelity": {
        "top_k": 5,
        "mask_value": "mean",
        "significance_threshold": 0.1,
    },
}

config = COLAB_CONFIG
print("Config loaded!")

#%% Cell 5: Load Data
from data.loader import load_data
print("Loading SWaT dataset...")
df = load_data(config)
print(f"Dataset shape: {df.shape}")
print(f"Attack ratio: {df['label'].mean():.2%}")
print(f"Columns: {list(df.columns)}")

#%% Cell 6: Preprocess
from preprocessing.pipeline import run_preprocessing_pipeline
print("Preprocessing...")
pipeline = run_preprocessing_pipeline(df, config)
print(f"Input size: {pipeline['input_size']}")
print(f"Sequence length: {pipeline['sequence_length']}")
print(f"Clients: {len(pipeline['client_datasets'])}")

#%% Cell 7: Create Model
from experiments.runner import create_model
model = create_model(config, pipeline["input_size"])
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

#%% Cell 8: Federated Training with DP
from federated.server import run_federated_simulation
print("Starting federated training...")
fed_results = run_federated_simulation(
    model, pipeline["client_datasets"],
    pipeline["val_loader"], device, config
)
model = fed_results["model"]
print("Training complete!")

# Plot training curve
import matplotlib.pyplot as plt
h = fed_results["history"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(h["round"], h["val_accuracy"], 'g-o')
ax1.set_title("Validation Accuracy"); ax1.set_xlabel("Round"); ax1.set_ylabel("Accuracy")
ax2.plot(h["round"], h["val_loss"], 'r-o')
ax2.set_title("Validation Loss"); ax2.set_xlabel("Round"); ax2.set_ylabel("Loss")
plt.tight_layout(); plt.show()

#%% Cell 9: Evaluate
from experiments.runner import evaluate_model
metrics = evaluate_model(model, pipeline["test_loader"], device)
print("\n📊 Test Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

#%% Cell 10: Adversarial Robustness Testing
from adversarial.evaluation import evaluate_robustness
test_ds = pipeline["datasets"]["test"]
robustness = evaluate_robustness(
    model, test_ds.windows.numpy(), test_ds.labels.numpy(),
    config, str(device)
)
print("\n⚔️ Robustness Results:")
print(f"  Clean accuracy: {robustness['clean_accuracy']:.4f}")
print(f"  FGSM accuracy:  {robustness['fgsm']['adversarial_accuracy']:.4f}")
print(f"  PGD accuracy:   {robustness['pgd']['adversarial_accuracy']:.4f}")

#%% Cell 11: SHAP Explanations
from xai.shap_explainer import SHAPExplainer
test_windows = test_ds.windows.numpy()
bg = test_windows[:config["xai"]["num_background_samples"]]
samples = test_windows[-config["xai"]["num_explain_samples"]:]

explainer = SHAPExplainer(model, bg, pipeline["feature_columns"], str(device))
explanation = explainer.explain(samples, nsamples=100)

# Plot feature importance
from xai.visualizations import plot_feature_importance
plot_feature_importance(explanation["feature_importance"], save_path="shap_importance.png")
from IPython.display import Image, display
display(Image("shap_importance.png"))

#%% Cell 12: Fidelity Verification
from fidelity.verifier import FidelityVerifier
verifier = FidelityVerifier(model, pipeline["feature_columns"], str(device))
fidelity = verifier.verify(samples, explanation["shap_values"], top_k=5)
print(f"\n🔍 Fidelity Score: {fidelity['fidelity_score']:.4f}")
print(f"   Faithfulness:  {fidelity['faithfulness_ratio']:.1%}")
print(f"   Top features:  {fidelity['top_masked_features']}")

#%% Cell 13: Save Model
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/colab_trained_model.pt")
print("Model saved to results/colab_trained_model.pt")
print("\n✅ All done! Download the model file to use locally.")
