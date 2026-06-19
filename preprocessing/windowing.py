"""
Sliding Window Generator
=========================
Converts flat time-series data into overlapping sliding window sequences
suitable for RNN (Elman/GRU) input.

Input shape:  (num_samples, num_features)
Output shape: (num_windows, sequence_length, num_features)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger


def create_sliding_windows(features, labels, sequence_length=20):
    """
    Create sliding window sequences from time-series data.
    Uses the label at the END of each window as the window label.

    Args:
        features: (num_samples, num_features)
        labels: (num_samples,)
        sequence_length: time steps per window

    Returns:
        (windows, window_labels) numpy arrays
    """
    num_samples, num_features = features.shape
    num_windows = num_samples - sequence_length + 1

    if num_windows <= 0:
        raise ValueError(f"Need at least {sequence_length} samples, got {num_samples}")

    windows = np.zeros((num_windows, sequence_length, num_features), dtype=np.float32)
    window_labels = np.zeros(num_windows, dtype=np.float32)

    for i in range(num_windows):
        windows[i] = features[i : i + sequence_length]
        window_labels[i] = labels[i + sequence_length - 1]

    logger.info(f"Created {num_windows} windows: shape={windows.shape}, attack_ratio={window_labels.mean():.1%}")
    return windows, window_labels


class SWaTWindowDataset(Dataset):
    """PyTorch Dataset for sliding window sequences."""

    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

    @property
    def num_features(self):
        return self.windows.shape[2]

    @property
    def sequence_length(self):
        return self.windows.shape[1]
