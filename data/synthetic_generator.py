"""
Synthetic SWaT Data Generator
==============================
Generates realistic synthetic sensor data mimicking the SWaT (Secure Water Treatment)
dataset for development and testing when the real dataset is unavailable.

The SWaT testbed has 6 stages:
  - Stage 1: Raw water intake (LIT101, MV101, P101, P102, FIT101)
  - Stage 2: Chemical dosing (AIT201, AIT202, AIT203, FIT201, MV201, P201-P208)
  - Stage 3: Ultrafiltration (DPIT301, FIT301, LIT301, MV301-MV304, P301-P302)
  - Stage 4: De-chlorination (AIT401, AIT402, FIT401, LIT401, P401-P404, UV401)
  - Stage 5: Reverse Osmosis (AIT501-AIT504, FIT501-FIT504, PIT501-PIT503, P501-P502)
  - Stage 6: Backwash (FIT601, P601-P603)

We simulate a subset of the most critical sensors for a working prototype.
"""

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================
# Define sensor profiles that mimic real SWaT behavior
# Each sensor has: name, normal_range, noise_std, type
# ============================================================
SENSOR_PROFILES = [
    # Stage 1 — Raw Water Intake
    {"name": "LIT101", "normal_mean": 500, "normal_std": 30, "min": 0, "max": 1000,
     "description": "Raw water tank level"},
    {"name": "FIT101", "normal_mean": 2.5, "normal_std": 0.3, "min": 0, "max": 5,
     "description": "Raw water flow rate"},
    {"name": "MV101", "normal_mean": 1, "normal_std": 0.05, "min": 0, "max": 2,
     "description": "Motorized valve status"},
    {"name": "P101", "normal_mean": 1, "normal_std": 0.05, "min": 0, "max": 2,
     "description": "Raw water pump status"},
    {"name": "P102", "normal_mean": 1, "normal_std": 0.05, "min": 0, "max": 2,
     "description": "Raw water pump 2 status"},

    # Stage 2 — Chemical Dosing
    {"name": "AIT201", "normal_mean": 250, "normal_std": 15, "min": 0, "max": 500,
     "description": "NaCl analyzer"},
    {"name": "AIT202", "normal_mean": 7.0, "normal_std": 0.3, "min": 0, "max": 14,
     "description": "pH analyzer"},
    {"name": "AIT203", "normal_mean": 300, "normal_std": 20, "min": 0, "max": 600,
     "description": "ORP analyzer"},
    {"name": "FIT201", "normal_mean": 2.3, "normal_std": 0.25, "min": 0, "max": 5,
     "description": "Chemical flow rate"},

    # Stage 3 — Ultrafiltration
    {"name": "DPIT301", "normal_mean": 40, "normal_std": 5, "min": 0, "max": 100,
     "description": "Differential pressure"},
    {"name": "FIT301", "normal_mean": 2.0, "normal_std": 0.2, "min": 0, "max": 5,
     "description": "UF feed flow rate"},
    {"name": "LIT301", "normal_mean": 600, "normal_std": 40, "min": 0, "max": 1200,
     "description": "UF feed tank level"},

    # Stage 4 — De-chlorination
    {"name": "AIT401", "normal_mean": 200, "normal_std": 15, "min": 0, "max": 400,
     "description": "RO hardness analyzer"},
    {"name": "AIT402", "normal_mean": 0.5, "normal_std": 0.1, "min": 0, "max": 2,
     "description": "ORP analyzer (dechlorination)"},
    {"name": "FIT401", "normal_mean": 1.8, "normal_std": 0.2, "min": 0, "max": 4,
     "description": "RO feed flow rate"},
    {"name": "LIT401", "normal_mean": 400, "normal_std": 25, "min": 0, "max": 800,
     "description": "RO feed tank level"},

    # Stage 5 — Reverse Osmosis
    {"name": "AIT501", "normal_mean": 150, "normal_std": 10, "min": 0, "max": 300,
     "description": "RO inlet conductivity"},
    {"name": "AIT502", "normal_mean": 15, "normal_std": 3, "min": 0, "max": 50,
     "description": "RO permeate conductivity"},
    {"name": "FIT501", "normal_mean": 1.5, "normal_std": 0.15, "min": 0, "max": 3,
     "description": "RO inlet flow rate"},
    {"name": "FIT502", "normal_mean": 1.0, "normal_std": 0.1, "min": 0, "max": 2,
     "description": "RO permeate flow rate"},
    {"name": "PIT501", "normal_mean": 200, "normal_std": 15, "min": 0, "max": 400,
     "description": "RO inlet pressure"},
    {"name": "PIT502", "normal_mean": 180, "normal_std": 12, "min": 0, "max": 350,
     "description": "RO outlet pressure"},
    {"name": "PIT503", "normal_mean": 50, "normal_std": 5, "min": 0, "max": 100,
     "description": "RO permeate pressure"},

    # Stage 6 — Backwash
    {"name": "FIT601", "normal_mean": 1.2, "normal_std": 0.1, "min": 0, "max": 3,
     "description": "Backwash flow rate"},
]

# Column names (just sensor names)
SENSOR_NAMES = [s["name"] for s in SENSOR_PROFILES]


def get_sensor_names():
    """Return list of sensor names used in the synthetic data."""
    return SENSOR_NAMES.copy()


def generate_normal_data(num_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate normal (non-attack) sensor data with realistic temporal correlations.

    The data includes:
    - Base sensor values drawn from normal distributions
    - Temporal autocorrelation (each value depends partly on the previous)
    - Small random fluctuations mimicking sensor noise

    Args:
        num_samples: Number of time steps to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sensor columns and a 'label' column (0 = normal)
    """
    rng = np.random.RandomState(seed)
    data = {}

    for sensor in SENSOR_PROFILES:
        # Start with a base value
        values = np.zeros(num_samples)
        values[0] = sensor["normal_mean"] + rng.normal(0, sensor["normal_std"] * 0.5)

        # Generate temporally correlated data (AR(1) process)
        # Each value is ~90% of the previous + 10% new noise
        for t in range(1, num_samples):
            noise = rng.normal(0, sensor["normal_std"] * 0.3)
            values[t] = 0.9 * values[t - 1] + 0.1 * sensor["normal_mean"] + noise

        # Clip to valid range
        values = np.clip(values, sensor["min"], sensor["max"])
        data[sensor["name"]] = values

    df = pd.DataFrame(data)
    df["label"] = 0  # Normal
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=num_samples, freq="1s")

    logger.info(f"Generated {num_samples} normal samples with {len(SENSOR_PROFILES)} sensors")
    return df


def generate_attack_data(num_samples: int, seed: int = 123) -> pd.DataFrame:
    """
    Generate attack sensor data with various attack patterns.

    Attack types simulated:
    1. Single sensor manipulation (e.g., spike LIT101 to overflow)
    2. Multi-sensor coordinated attack (change valve + pump states)
    3. Stealthy drift attack (slowly shift sensor values)
    4. Replay attack (repeat old normal values while actual state changes)

    Args:
        num_samples: Number of attack time steps to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sensor columns and a 'label' column (1 = attack)
    """
    rng = np.random.RandomState(seed)

    # Start with normal-looking data
    data = {}
    for sensor in SENSOR_PROFILES:
        values = np.zeros(num_samples)
        values[0] = sensor["normal_mean"]
        for t in range(1, num_samples):
            noise = rng.normal(0, sensor["normal_std"] * 0.3)
            values[t] = 0.9 * values[t - 1] + 0.1 * sensor["normal_mean"] + noise
        data[sensor["name"]] = values

    # ---- Apply various attack patterns to different segments ----

    segment_size = num_samples // 4

    # Attack 1: Tank overflow — LIT101 rises dangerously while pump stays ON
    # (In normal operation, high LIT101 should trigger pump OFF)
    start, end = 0, segment_size
    for t in range(start, end):
        progress = (t - start) / segment_size
        # Tank level rises to dangerous levels
        data["LIT101"][t] = 500 + progress * 450 + rng.normal(0, 10)
        # But pump stays ON (should turn OFF)
        data["P101"][t] = 1.0 + rng.normal(0, 0.02)
        # Flow rate increases abnormally
        data["FIT101"][t] = 2.5 + progress * 2.0 + rng.normal(0, 0.1)

    # Attack 2: Chemical dosing manipulation — pH changed to dangerous levels
    start, end = segment_size, 2 * segment_size
    for t in range(start, end):
        progress = (t - start) / segment_size
        # pH drops to acidic levels
        data["AIT202"][t] = 7.0 - progress * 4.0 + rng.normal(0, 0.1)
        # ORP shifts abnormally
        data["AIT203"][t] = 300 + progress * 200 + rng.normal(0, 10)

    # Attack 3: Stealthy drift — UF tank level slowly drifts down
    start, end = 2 * segment_size, 3 * segment_size
    for t in range(start, end):
        progress = (t - start) / segment_size
        # Slow drift that might evade simple threshold detection
        data["LIT301"][t] = 600 - progress * 300 + rng.normal(0, 5)
        data["FIT301"][t] = 2.0 - progress * 1.0 + rng.normal(0, 0.05)

    # Attack 4: RO pressure manipulation — could damage membranes
    start, end = 3 * segment_size, num_samples
    for t in range(start, end):
        progress = (t - start) / (num_samples - start)
        # Inlet pressure spikes
        data["PIT501"][t] = 200 + progress * 180 + rng.normal(0, 8)
        # Permeate flow drops (membrane blocking)
        data["FIT502"][t] = 1.0 - progress * 0.8 + rng.normal(0, 0.03)

    # Clip all values
    for sensor in SENSOR_PROFILES:
        data[sensor["name"]] = np.clip(
            data[sensor["name"]], sensor["min"], sensor["max"]
        )

    df = pd.DataFrame(data)
    df["label"] = 1  # Attack
    df["timestamp"] = pd.date_range(
        start="2024-01-02", periods=num_samples, freq="1s"
    )

    logger.info(
        f"Generated {num_samples} attack samples with 4 attack patterns "
        f"across {len(SENSOR_PROFILES)} sensors"
    )
    return df


def generate_full_dataset(
    num_normal: int = 10000,
    num_attack: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a complete synthetic SWaT-like dataset with normal and attack data.

    Args:
        num_normal: Number of normal samples
        num_attack: Number of attack samples
        seed: Random seed

    Returns:
        Combined DataFrame with both normal and attack data, shuffled
    """
    normal_df = generate_normal_data(num_normal, seed=seed)
    attack_df = generate_attack_data(num_attack, seed=seed + 1)

    # Combine and shuffle
    full_df = pd.concat([normal_df, attack_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(
        f"Full dataset: {len(full_df)} samples "
        f"({num_normal} normal, {num_attack} attack, "
        f"attack ratio: {num_attack / len(full_df):.1%})"
    )
    return full_df
