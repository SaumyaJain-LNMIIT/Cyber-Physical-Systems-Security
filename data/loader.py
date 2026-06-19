"""
Data Loader
============
Unified data loading from three sources:
  1. Kaggle (vishala28/swat-dataset-secure-water-treatment-system)
  2. Local CSV files (Normal_v0.csv, Attack_v0.csv)
  3. Synthetic generation (for development)

All sources produce a consistent DataFrame with sensor columns + 'label' column.
"""

import os
import pandas as pd
import numpy as np
from loguru import logger


def load_kaggle_data(dataset_id: str = "vishala28/swat-dataset-secure-water-treatment-system"):
    """
    Download and load the SWaT dataset from Kaggle using kagglehub.

    Requires Kaggle credentials to be configured:
      - Set KAGGLE_USERNAME and KAGGLE_KEY environment variables, OR
      - Place kaggle.json in ~/.kaggle/

    Args:
        dataset_id: Kaggle dataset identifier

    Returns:
        Tuple of (normal_df, attack_df) DataFrames
    """
    try:
        import kagglehub
        logger.info(f"Downloading SWaT dataset from Kaggle: {dataset_id}")
        path = kagglehub.dataset_download(dataset_id)
        logger.info(f"Dataset downloaded to: {path}")

        # Find CSV files in the downloaded directory
        csv_files = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))

        logger.info(f"Found CSV files: {csv_files}")

        # Try to identify normal and attack files
        normal_df = None
        attack_df = None

        for csv_path in csv_files:
            fname = os.path.basename(csv_path).lower()
            if "normal" in fname:
                normal_df = pd.read_csv(csv_path)
                logger.info(f"Loaded normal data: {csv_path} ({len(normal_df)} rows)")
            elif "attack" in fname:
                attack_df = pd.read_csv(csv_path)
                logger.info(f"Loaded attack data: {csv_path} ({len(attack_df)} rows)")

        # If we couldn't identify by name, load all CSVs
        if normal_df is None and attack_df is None:
            if len(csv_files) >= 2:
                normal_df = pd.read_csv(csv_files[0])
                attack_df = pd.read_csv(csv_files[1])
            elif len(csv_files) == 1:
                # Single file — we'll need to split by label column
                full_df = pd.read_csv(csv_files[0])
                return _process_single_csv(full_df)

        return _process_swat_dataframes(normal_df, attack_df)

    except ImportError:
        logger.error("kagglehub not installed. Install with: pip install kagglehub")
        raise
    except Exception as e:
        logger.error(f"Failed to load Kaggle dataset: {e}")
        raise


def load_csv_data(normal_path: str, attack_path: str):
    """
    Load SWaT dataset from local CSV files.

    Args:
        normal_path: Path to Normal_v0.csv
        attack_path: Path to Attack_v0.csv

    Returns:
        Combined DataFrame with sensor columns and 'label' column
    """
    logger.info(f"Loading normal data from: {normal_path}")
    normal_df = pd.read_csv(normal_path)
    logger.info(f"Normal data: {len(normal_df)} rows, {len(normal_df.columns)} columns")

    logger.info(f"Loading attack data from: {attack_path}")
    attack_df = pd.read_csv(attack_path)
    logger.info(f"Attack data: {len(attack_df)} rows, {len(attack_df.columns)} columns")

    return _process_swat_dataframes(normal_df, attack_df)


def _process_swat_dataframes(normal_df: pd.DataFrame, attack_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw SWaT DataFrames into a unified format.

    Steps:
    1. Clean column names (strip whitespace)
    2. Handle timestamp column
    3. Convert categorical columns to numeric
    4. Add label column (0=normal, 1=attack)
    5. Drop non-numeric columns

    Args:
        normal_df: Raw normal operation data
        attack_df: Raw attack scenario data

    Returns:
        Combined, cleaned DataFrame
    """
    # Clean column names
    normal_df.columns = [c.strip() for c in normal_df.columns]
    attack_df.columns = [c.strip() for c in attack_df.columns]

    # Add labels
    # Check if there's already a label/Normal/Attack column
    label_cols = [c for c in attack_df.columns if c.lower() in
                  ["label", "normal/attack", "normal_attack", "attack"]]

    if label_cols:
        label_col = label_cols[0]
        logger.info(f"Found existing label column: {label_col}")
        # Map labels: "Normal" -> 0, "Attack" -> 1, "A ttack" -> 1 (typo in real dataset)
        attack_df["label"] = attack_df[label_col].apply(
            lambda x: 1 if str(x).strip().lower() in ["attack", "a ttack"] else 0
        )
        normal_df["label"] = 0
        # Drop original label column
        if label_col != "label":
            attack_df = attack_df.drop(columns=[label_col])
            if label_col in normal_df.columns:
                normal_df = normal_df.drop(columns=[label_col])
    else:
        normal_df["label"] = 0
        attack_df["label"] = 1

    # Handle timestamp column
    timestamp_cols = [c for c in normal_df.columns if
                      "timestamp" in c.lower() or "time" in c.lower() or "date" in c.lower()]

    for col in timestamp_cols:
        if col in normal_df.columns:
            normal_df = normal_df.drop(columns=[col])
        if col in attack_df.columns:
            attack_df = attack_df.drop(columns=[col])

    # Keep only common columns
    common_cols = list(set(normal_df.columns) & set(attack_df.columns))
    normal_df = normal_df[common_cols]
    attack_df = attack_df[common_cols]

    # Convert all columns to numeric (coerce errors to NaN)
    for col in common_cols:
        if col != "label":
            normal_df[col] = pd.to_numeric(normal_df[col], errors="coerce")
            attack_df[col] = pd.to_numeric(attack_df[col], errors="coerce")

    # Combine
    full_df = pd.concat([normal_df, attack_df], ignore_index=True)

    # Drop columns with too many NaNs (>50%)
    nan_threshold = 0.5
    cols_to_keep = [c for c in full_df.columns
                    if full_df[c].isna().mean() < nan_threshold]
    full_df = full_df[cols_to_keep]

    # Fill remaining NaNs with column mean
    full_df = full_df.fillna(full_df.mean(numeric_only=True))

    logger.info(
        f"Processed SWaT data: {len(full_df)} samples, "
        f"{len(full_df.columns) - 1} features, "
        f"attack ratio: {full_df['label'].mean():.1%}"
    )
    return full_df


def _process_single_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Process a single CSV that contains both normal and attack data."""
    df.columns = [c.strip() for c in df.columns]

    label_cols = [c for c in df.columns if c.lower() in
                  ["label", "normal/attack", "normal_attack", "attack"]]

    if label_cols:
        label_col = label_cols[0]
        df["label"] = df[label_col].apply(
            lambda x: 1 if str(x).strip().lower() in ["attack", "a ttack"] else 0
        )
        if label_col != "label":
            df = df.drop(columns=[label_col])

    # Drop timestamp columns
    timestamp_cols = [c for c in df.columns if
                      "timestamp" in c.lower() or "time" in c.lower()]
    df = df.drop(columns=[c for c in timestamp_cols if c in df.columns])

    # Convert to numeric
    for col in df.columns:
        if col != "label":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(df.mean(numeric_only=True))
    return df


def load_data(config: dict) -> pd.DataFrame:
    """
    Unified data loading based on config settings.

    Args:
        config: Configuration dictionary with 'data' section

    Returns:
        DataFrame with sensor columns and 'label' column
    """
    source = config.get("data", {}).get("source", "synthetic")

    if source == "kaggle":
        dataset_id = config["data"].get(
            "kaggle_dataset",
            "vishala28/swat-dataset-secure-water-treatment-system"
        )
        return load_kaggle_data(dataset_id)

    elif source == "csv":
        normal_path = config["data"]["csv_normal_path"]
        attack_path = config["data"]["csv_attack_path"]
        return load_csv_data(normal_path, attack_path)

    elif source == "synthetic":
        from data.synthetic_generator import generate_full_dataset
        return generate_full_dataset(
            num_normal=config["data"].get("synthetic_num_normal", 10000),
            num_attack=config["data"].get("synthetic_num_attack", 3000),
            seed=config["data"].get("random_seed", 42),
        )

    else:
        raise ValueError(f"Unknown data source: {source}. Use 'kaggle', 'csv', or 'synthetic'.")
