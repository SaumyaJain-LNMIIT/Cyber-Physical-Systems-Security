# 🚀 Google Colab — Step-by-Step Guide

## Prerequisites
- A Google account (for Colab)
- A Kaggle account (for the SWaT dataset) — optional, synthetic data works too

---

## Step 1: Zip & Upload Your Project

On your Mac terminal:
```bash
cd /Users/abhinavdogra/Desktop/Projects/ctmas-project
zip -r cps_project.zip cps_security_project/ -x "*/venv/*" "*/__pycache__/*" "*.pyc"
```

Then upload `cps_project.zip` to [Google Drive](https://drive.google.com).

---

## Step 2: Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **New Notebook**
3. Go to **Runtime → Change runtime type → GPU (T4)** → Save

---

## Step 3: Run These Cells One by One

### Cell 1 — Mount Drive & Unzip
```python
from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/cps_project.zip /content/
!unzip -q /content/cps_project.zip -d /content/
!ls /content/cps_security_project/
```

### Cell 2 — Install Dependencies
```python
!pip install -q torch numpy pandas scikit-learn pyyaml loguru tqdm
!pip install -q opacus shap adversarial-robustness-toolbox flwr kagglehub
!pip install -q matplotlib plotly seaborn fastapi pydantic pymongo
print("✅ All packages installed!")
# NOTE: Ignore protobuf warnings — they don't affect our project
```

### Cell 3 — Setup Path + Verify
```python
import sys, os

sys.path.insert(0, '/content/cps_security_project')
os.chdir('/content/cps_security_project')

# Verify
print("Files:", os.listdir('/content/cps_security_project/experiments/'))
from experiments.runner import create_model
print("✅ Import works!")
```

### Cell 4 — Check GPU
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU! Go to Runtime → Change runtime type → GPU")
```

### Cell 5 — Config + Load Data
```python
config = {
    "data": {
        "source": "kaggle",
        "kaggle_dataset": "vishala28/swat-dataset-secure-water-treatment-system",
        "test_size": 0.2, "val_size": 0.1, "random_seed": 42,
    },
    "model": {"type": "elman", "hidden_size": 128, "num_layers": 2, "dropout": 0.1, "sequence_length": 30},
    "training": {"epochs": 50, "batch_size": 128, "learning_rate": 0.001, "device": "auto"},
    "federated": {"num_clients": 3, "num_rounds": 20, "local_epochs": 5},
    "privacy": {"enabled": True, "noise_multiplier": 1.0, "max_grad_norm": 1.0},
    "adversarial": {"fgsm_epsilon": 0.1, "pgd_epsilon": 0.1, "pgd_step_size": 0.01, "pgd_max_iter": 40, "num_attack_samples": 1000},
    "xai": {"num_background_samples": 200, "num_explain_samples": 100, "top_k_features": 5},
    "fidelity": {"top_k": 5, "mask_value": "mean", "significance_threshold": 0.1},
}

import os
from google.colab import userdata

# Securely load Kaggle credentials from Google Colab's Secrets manager
try:
    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    print("✅ Kaggle credentials loaded from Colab Secrets!")
except Exception as e:
    print("⚠️ Colab Secrets not configured. Make sure KAGGLE_USERNAME and KAGGLE_KEY are set in the Secrets vault.")

from data.loader import load_data
import pandas as pd

df = load_data(config)

# --- PREVENT RAM CRASH BY SUBSAMPLING ---
normal_df = df[df["label"] == 0].head(50000)
attack_df = df[df["label"] == 1].head(15000)
df = pd.concat([normal_df, attack_df]).reset_index(drop=True)

print(f"✅ Safe Data: {df.shape[0]} rows, {df.shape[1]} cols, attack ratio: {df['label'].mean():.2%}")
```

### Cell 6 — Preprocess
```python
from preprocessing.pipeline import run_preprocessing_pipeline
pipeline = run_preprocessing_pipeline(df, config)
print(f"✅ Ready: {pipeline['input_size']} sensors, {len(pipeline['client_datasets'])} clients")
```

### Cell 7 — Create Model
```python
model = create_model(config, pipeline["input_size"])
model.to(device)
print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} params on {device}")
```

### Cell 8 — Federated Training with DP 🏋️
```python
from federated.server import run_federated_simulation
print("🏋️ Training (20 rounds, 3 clients, DP enabled)... ~5-15 min on GPU")
fed_results = run_federated_simulation(model, pipeline["client_datasets"], pipeline["val_loader"], device, config)
model = fed_results["model"]
print("✅ Training complete!")
```

### Cell 9 — Plot Training Curves 📈
```python
import matplotlib.pyplot as plt

h = fed_results["history"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(h["round"], h["val_accuracy"], 'g-o', linewidth=2, markersize=6)
ax1.set_title("Validation Accuracy per Round", fontsize=14)
ax1.set_xlabel("Federated Round"); ax1.set_ylabel("Accuracy"); ax1.grid(True, alpha=0.3)

ax2.plot(h["round"], h["val_loss"], 'r-o', linewidth=2, markersize=6)
ax2.set_title("Validation Loss per Round", fontsize=14)
ax2.set_xlabel("Federated Round"); ax2.set_ylabel("Loss"); ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.show()
print(f"Final accuracy: {h['val_accuracy'][-1]:.4f}")
```

### Cell 10 — Evaluate on Test Set 📊
```python
from experiments.runner import evaluate_model
metrics = evaluate_model(model, pipeline["test_loader"], device)
print(f"\n📊 Accuracy: {metrics['accuracy']:.4f}")
print(f"   Precision: {metrics['precision']:.4f}")
print(f"   Recall: {metrics['recall']:.4f}")
print(f"   F1 Score: {metrics['f1_score']:.4f}")

# --- EMERGENCY SAVE ---
# Instantly save to Drive so we don't lose it if it disconnects later!
import torch, yaml, joblib, shutil, os
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/colab_trained_model.pt")
joblib.dump(pipeline["scaler"], "results/colab_scaler.pkl")
with open("results/colab_config.yaml", "w") as f:
    yaml.dump(config, f)

for f in ["results/colab_trained_model.pt", "results/colab_scaler.pkl", "results/colab_config.yaml"]:
    shutil.copy(f, "/content/drive/MyDrive/")
print("✅ EMERGENCY SAVE COMPLETED TO GOOGLE DRIVE!")
```

### Cell 11 — Adversarial Robustness Testing ⚔️
```python
from adversarial.evaluation import evaluate_robustness
test_ds = pipeline["datasets"]["test"]
robustness = evaluate_robustness(
    model, test_ds.windows.numpy(), test_ds.labels.numpy(), config, "cpu"
)
print(f"\n⚔️ Clean accuracy:      {robustness['clean_accuracy']:.4f}")
print(f"   FGSM accuracy:       {robustness['fgsm']['adversarial_accuracy']:.4f}")
print(f"   FGSM attack success: {robustness['fgsm']['attack_success_rate']:.4f}")
print(f"   PGD accuracy:        {robustness['pgd']['adversarial_accuracy']:.4f}")
print(f"   PGD attack success:  {robustness['pgd']['attack_success_rate']:.4f}")
```

### Cell 12 — SHAP Explanations 🔍
```python
from xai.shap_explainer import SHAPExplainer
test_windows = test_ds.windows.numpy()
explainer = SHAPExplainer(model, test_windows[:100], pipeline["feature_columns"], "cpu")
explanation = explainer.explain(test_windows[-50:], nsamples=100)

print("\n🔍 Top Important Sensors:")
for name, imp in list(explanation["feature_importance"].items())[:10]:
    bar = "█" * int(imp * 100)
    print(f"   {name:10s} {imp:.4f} {bar}")
```

### Cell 13 — SHAP Plot
```python
from xai.visualizations import plot_feature_importance
plot_feature_importance(explanation["feature_importance"], save_path="/content/shap_plot.png")
from IPython.display import Image, display
display(Image("/content/shap_plot.png"))
```

### Cell 14 — Fidelity Verification 🧪
```python
from fidelity.verifier import FidelityVerifier
verifier = FidelityVerifier(model, pipeline["feature_columns"], "cpu")
fidelity = verifier.verify(test_windows[-50:], explanation["shap_values"], top_k=5)

print(f"\n🧪 Fidelity Score:     {fidelity['fidelity_score']:.4f}")
print(f"   Faithfulness Ratio: {fidelity['faithfulness_ratio']:.1%}")
print(f"   Top Features:       {fidelity['top_masked_features']}")
verdict = "✅ FAITHFUL" if fidelity['fidelity_score'] > 0.1 else "⚠️ LOW FIDELITY"
print(f"   Verdict: {verdict}")
```

### Cell 15 — Save & Download 💾
```python
import torch, json, shutil, yaml, joblib
os.makedirs("results", exist_ok=True)

torch.save(model.state_dict(), "results/colab_trained_model.pt")
joblib.dump(pipeline["scaler"], "results/colab_scaler.pkl")
with open("results/colab_config.yaml", "w") as f:
    yaml.dump(config, f)

all_results = {
    "metrics": metrics,
    "robustness": robustness,
    "fidelity": {"fidelity_score": fidelity["fidelity_score"],
                 "faithfulness_ratio": fidelity["faithfulness_ratio"],
                 "top_features": fidelity["top_masked_features"]},
    "training_history": fed_results["history"],
}
with open("results/experiment_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Copy to Google Drive
for f in ["results/colab_trained_model.pt", "results/colab_scaler.pkl", "results/colab_config.yaml", "results/experiment_results.json", "shap_plot.png", "training_curves.png"]:
    if os.path.exists(f):
        shutil.copy(f, "/content/drive/MyDrive/")

print("✅ Saved to Google Drive!")
print("   📦 colab_trained_model.pt")
print("   📊 experiment_results.json")
print("   📈 training_curves.png")
print("   🔍 shap_plot.png")
```

---

## Step 4: Bring Model Back to MacBook

Download from Google Drive, then:
```bash
cp ~/Downloads/colab_trained_model.pt \
   ~/Desktop/Projects/ctmas-project/cps_security_project/experiments/results/
cp ~/Downloads/colab_scaler.pkl \
   ~/Desktop/Projects/ctmas-project/cps_security_project/experiments/results/
cp ~/Downloads/colab_config.yaml \
   ~/Desktop/Projects/ctmas-project/cps_security_project/experiments/results/
cp ~/Downloads/experiment_results.json \
   ~/Desktop/Projects/ctmas-project/cps_security_project/experiments/results/
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Protobuf warnings | Safe to ignore — doesn't affect our code |
| `ModuleNotFoundError` | Re-run Cell 3 (path setup) |
| `NameError: config not defined` | Re-run Cell 5 (config + data) |
| No GPU | Runtime → Change runtime type → T4 GPU |
| Kaggle download fails | Use `source: "synthetic"` instead |
| Out of memory | Reduce `batch_size` to 64, `hidden_size` to 64 |
| Training too slow | Reduce `num_rounds` to 10, `local_epochs` to 3 |

---

## Important: After Any Session Restart

If Colab restarts, re-run cells in this order: **2 → 3 → 4 → 5 → 6 → 7** then continue.
