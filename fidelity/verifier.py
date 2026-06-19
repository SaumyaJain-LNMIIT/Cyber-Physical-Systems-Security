"""
Fidelity Verifier
==================
Tests whether SHAP explanations are faithful (honest).

Procedure:
1. Get SHAP explanation → identify top-k important features
2. Mask those features (replace with mean/zero/random)
3. Re-run the model on masked input
4. If prediction changes significantly → explanation is faithful
5. If prediction doesn't change → explanation may be misleading

Fidelity Score = average prediction change when top features are masked.
Higher score = more faithful explanations.
"""

import numpy as np
import torch
from loguru import logger


class FidelityVerifier:
    """Tests explanation faithfulness by masking important features."""

    def __init__(self, model, feature_names, device="cpu"):
        self.model = model
        self.feature_names = feature_names
        self.device = device

    def verify(
        self,
        samples,
        shap_values,
        top_k=5,
        mask_value="mean",
        significance_threshold=0.1,
    ):
        """
        Verify explanation fidelity for a batch of samples.

        Args:
            samples: numpy array (num_samples, seq_len, num_features)
            shap_values: numpy array (num_samples, num_features) — per-sensor SHAP
            top_k: number of top features to mask
            mask_value: "mean", "zero", or "random"
            significance_threshold: min prediction change to consider significant

        Returns:
            dict with fidelity metrics
        """
        self.model.eval()
        num_samples = len(samples)

        # Get original predictions
        x_orig = torch.FloatTensor(samples).to(self.device)
        with torch.no_grad():
            orig_preds = self.model(x_orig).cpu().numpy()

        # For each sample, mask top-k features and get new prediction
        prediction_changes = []
        faithful_count = 0

        for i in range(num_samples):
            # Find top-k features for this sample
            sample_importance = np.abs(shap_values[i])
            top_indices = np.argsort(sample_importance)[-top_k:]

            # Create masked version
            masked = samples[i].copy()  # (seq_len, num_features)

            for feat_idx in top_indices:
                if mask_value == "mean":
                    masked[:, feat_idx] = samples[:, :, feat_idx].mean()
                elif mask_value == "zero":
                    masked[:, feat_idx] = 0.0
                elif mask_value == "random":
                    masked[:, feat_idx] = np.random.uniform(0, 1, size=masked.shape[0])

            # Get prediction on masked input
            x_masked = torch.FloatTensor(masked).unsqueeze(0).to(self.device)
            with torch.no_grad():
                masked_pred = self.model(x_masked).cpu().numpy()[0]

            # Compute prediction change
            change = abs(orig_preds[i] - masked_pred)
            prediction_changes.append(change)

            if change >= significance_threshold:
                faithful_count += 1

        # Compute fidelity score
        fidelity_score = np.mean(prediction_changes)
        faithfulness_ratio = faithful_count / max(num_samples, 1)

        # Log top features that were masked
        avg_importance = np.abs(shap_values).mean(axis=0)
        top_global = np.argsort(avg_importance)[-top_k:]
        top_names = [self.feature_names[i] for i in top_global]

        results = {
            "fidelity_score": float(fidelity_score),
            "faithfulness_ratio": float(faithfulness_ratio),
            "mean_prediction_change": float(np.mean(prediction_changes)),
            "std_prediction_change": float(np.std(prediction_changes)),
            "num_samples": num_samples,
            "top_k": top_k,
            "mask_value": mask_value,
            "top_masked_features": top_names,
            "per_sample_changes": [float(c) for c in prediction_changes],
        }

        logger.info(
            f"Fidelity verification — score: {fidelity_score:.4f}, "
            f"faithful: {faithfulness_ratio:.1%}, "
            f"top features masked: {top_names}"
        )

        return results
