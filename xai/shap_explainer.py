"""
SHAP Explainer for CPS Intrusion Detection
=============================================
Uses SHAP (SHapley Additive exPlanations) to explain why the model
flagged a particular sensor reading as an attack.

For RNN models, we use KernelExplainer which works with any model
by treating it as a black box. We average SHAP values across the
time dimension to get per-sensor importance.
"""

import numpy as np
import torch
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Explanations will be disabled.")


class SHAPExplainer:
    """
    SHAP-based explainer for RNN intrusion detection models.

    Since SHAP's KernelExplainer expects 2D input (samples, features),
    we flatten the 3D RNN input (samples, seq_len, features) and
    reshape inside the prediction function.
    """

    def __init__(self, model, background_data, feature_names, device="cpu"):
        """
        Args:
            model: trained PyTorch model
            background_data: numpy array (num_bg, seq_len, num_features) for SHAP baseline
            feature_names: list of sensor names
            device: torch device
        """
        self.model = model
        self.device = device
        self.feature_names = feature_names
        self.seq_len = background_data.shape[1]
        self.num_features = background_data.shape[2]

        # Flatten background for KernelExplainer
        self.bg_flat = background_data.reshape(len(background_data), -1)

        # Create prediction function for flattened input
        def predict_fn(flat_input):
            """Takes flattened input, returns attack probabilities."""
            x = flat_input.reshape(-1, self.seq_len, self.num_features)
            x_tensor = torch.FloatTensor(x).to(device)
            self.model.eval()
            with torch.no_grad():
                probs = self.model(x_tensor).cpu().numpy()
            return probs

        self.predict_fn = predict_fn

        if SHAP_AVAILABLE:
            self.explainer = shap.KernelExplainer(predict_fn, self.bg_flat)
            logger.info(
                f"SHAP Explainer created with {len(background_data)} background samples"
            )
        else:
            self.explainer = None

    def explain(self, samples, nsamples=100):
        """
        Generate SHAP explanations for given samples.

        Args:
            samples: numpy array (num_samples, seq_len, num_features)
            nsamples: number of SHAP kernel samples (higher = more accurate but slower)

        Returns:
            dict with:
              - shap_values: (num_samples, num_features) averaged across time
              - feature_importance: dict of {sensor_name: importance_score}
              - raw_shap_values: (num_samples, seq_len * num_features) full values
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.error("SHAP not available")
            return None

        flat_samples = samples.reshape(len(samples), -1)

        logger.info(f"Computing SHAP values for {len(samples)} samples (nsamples={nsamples})...")
        raw_shap = self.explainer.shap_values(flat_samples, nsamples=nsamples)

        # raw_shap shape: (num_samples, seq_len * num_features)
        # Reshape to (num_samples, seq_len, num_features)
        shap_3d = raw_shap.reshape(len(samples), self.seq_len, self.num_features)

        # Average across time dimension to get per-sensor importance
        sensor_shap = np.abs(shap_3d).mean(axis=1)  # (num_samples, num_features)

        # Average across samples for global importance
        global_importance = sensor_shap.mean(axis=0)  # (num_features,)

        # Create importance dictionary
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_importance[name] = float(global_importance[i])

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        logger.info("Top 5 most important sensors:")
        for name, imp in list(feature_importance.items())[:5]:
            logger.info(f"  {name}: {imp:.4f}")

        return {
            "shap_values": sensor_shap,
            "feature_importance": feature_importance,
            "raw_shap_values": raw_shap,
        }

    def get_top_features(self, explanation, top_k=5):
        """Get the top-k most important feature names from an explanation."""
        if explanation is None:
            return []
        importance = explanation["feature_importance"]
        return list(importance.keys())[:top_k]
