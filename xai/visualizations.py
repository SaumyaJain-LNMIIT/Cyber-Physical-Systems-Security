"""
SHAP Visualization
===================
Creates publication-quality plots for SHAP explanations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import os
from loguru import logger


def plot_feature_importance(feature_importance, save_path=None, top_k=10):
    """
    Bar chart of global feature importance from SHAP.

    Args:
        feature_importance: dict {sensor_name: importance_score}
        save_path: path to save the plot (optional)
        top_k: number of features to display
    """
    names = list(feature_importance.keys())[:top_k]
    values = [feature_importance[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Sensor Importance for Attack Detection")
    ax.invert_yaxis()  # Most important at top
    plt.tight_layout()

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.close(fig)
    return fig


def plot_sample_explanation(shap_values_sample, feature_names, prediction, save_path=None):
    """
    Waterfall-style plot showing SHAP values for a single prediction.

    Args:
        shap_values_sample: (num_features,) SHAP values for one sample
        feature_names: list of sensor names
        prediction: model prediction probability
        save_path: optional save path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_idx = np.argsort(np.abs(shap_values_sample))[::-1][:10]
    names = [feature_names[i] for i in sorted_idx]
    values = [shap_values_sample[i] for i in sorted_idx]

    colors = ["#ff4444" if v > 0 else "#4444ff" for v in values]
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(f"Prediction: {'ATTACK' if prediction > 0.5 else 'NORMAL'} (prob={prediction:.3f})")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Sample explanation plot saved to {save_path}")

    plt.close(fig)
    return fig
