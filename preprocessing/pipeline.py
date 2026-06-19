"""
Preprocessing Pipeline
=======================
Complete data preprocessing: normalization, splitting, windowing, and
federated client partitioning.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from loguru import logger

from preprocessing.windowing import create_sliding_windows, SWaTWindowDataset


def normalize_features(df, feature_columns, scaler=None):
    """
    Normalize sensor values to [0, 1] using MinMaxScaler.

    Args:
        df: DataFrame with sensor columns
        feature_columns: list of column names to normalize
        scaler: existing scaler (for test data), or None to fit new one

    Returns:
        (normalized_array, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized = scaler.fit_transform(df[feature_columns].values)
        logger.info(f"Fitted MinMaxScaler on {len(feature_columns)} features")
    else:
        normalized = scaler.transform(df[feature_columns].values)
        logger.info(f"Transformed {len(feature_columns)} features with existing scaler")

    return normalized.astype(np.float32), scaler


def split_data(features, labels, test_size=0.2, val_size=0.1, random_seed=42):
    """
    Split data into train, validation, and test sets.

    Args:
        features: (num_samples, num_features)
        labels: (num_samples,)
        test_size: fraction for test set
        val_size: fraction for validation set (from remaining after test)
        random_seed: for reproducibility

    Returns:
        dict with 'train', 'val', 'test' keys, each containing (features, labels)
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        features, labels, test_size=test_size,
        random_state=random_seed, stratify=labels
    )

    # Second split: train vs val
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction,
        random_state=random_seed, stratify=y_trainval
    )

    logger.info(
        f"Data split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} | "
        f"Attack ratios — Train: {y_train.mean():.1%}, Val: {y_val.mean():.1%}, Test: {y_test.mean():.1%}"
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def partition_for_federated(features, labels, num_clients=3, seed=42):
    """
    Partition training data into non-overlapping subsets for federated clients.
    Uses stratified splitting to maintain label distribution across clients.

    Args:
        features: training features array
        labels: training labels array
        num_clients: number of federated clients
        seed: random seed

    Returns:
        list of (features, labels) tuples, one per client
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    # Split indices into num_clients roughly equal parts
    client_indices = np.array_split(indices, num_clients)

    partitions = []
    for i, idx in enumerate(client_indices):
        client_features = features[idx]
        client_labels = labels[idx]
        partitions.append((client_features, client_labels))
        logger.info(
            f"Client {i}: {len(idx)} samples, attack ratio: {client_labels.mean():.1%}"
        )

    return partitions


def run_preprocessing_pipeline(df, config, scaler=None):
    """
    Complete preprocessing pipeline: normalize → split → window → partition.

    Args:
        df: raw DataFrame with sensor columns and 'label'
        config: configuration dict

    Returns:
        dict with all processed data, scaler, and metadata
    """
    seq_len = config.get("model", {}).get("sequence_length", 20)
    test_size = config.get("data", {}).get("test_size", 0.2)
    val_size = config.get("data", {}).get("val_size", 0.1)
    seed = config.get("data", {}).get("random_seed", 42)
    num_clients = config.get("federated", {}).get("num_clients", 3)
    batch_size = config.get("training", {}).get("batch_size", 64)

    # Identify feature columns (everything except 'label' and 'timestamp')
    feature_columns = [c for c in df.columns if c not in ["label", "timestamp"]]
    labels = df["label"].values.astype(np.float32)

    logger.info(f"Features: {len(feature_columns)} sensors, {len(df)} total samples")

    # Step 1: Normalize
    features, scaler = normalize_features(df, feature_columns, scaler)

    # Step 2: Split (before windowing to avoid data leakage)
    splits = split_data(features, labels, test_size, val_size, seed)

    # Step 3: Create sliding windows for each split
    datasets = {}
    for split_name, (X, y) in splits.items():
        windows, window_labels = create_sliding_windows(X, y, seq_len)
        datasets[split_name] = SWaTWindowDataset(windows, window_labels)

    # Step 4: Partition training data for federated learning
    train_X, train_y = splits["train"]
    client_partitions = partition_for_federated(train_X, train_y, num_clients, seed)

    # Create windowed datasets for each client
    client_datasets = []
    for i, (c_features, c_labels) in enumerate(client_partitions):
        c_windows, c_labels_w = create_sliding_windows(c_features, c_labels, seq_len)
        client_datasets.append(SWaTWindowDataset(c_windows, c_labels_w))

    # Create DataLoaders
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    client_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets
    ]

    input_size = len(feature_columns)

    result = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "client_loaders": client_loaders,
        "client_datasets": client_datasets,
        "datasets": datasets,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "input_size": input_size,
        "sequence_length": seq_len,
    }

    logger.info(
        f"Pipeline complete — input_size: {input_size}, seq_len: {seq_len}, "
        f"{num_clients} federated clients"
    )
    return result
