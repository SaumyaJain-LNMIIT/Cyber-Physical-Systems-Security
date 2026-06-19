"""
Smoke Tests for CPS Security Project
======================================
Minimal tests to verify all modules work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest


class TestSyntheticData:
    """Test synthetic data generation."""

    def test_generate_normal(self):
        from data.synthetic_generator import generate_normal_data
        df = generate_normal_data(100, seed=42)
        assert len(df) == 100
        assert "label" in df.columns
        assert (df["label"] == 0).all()

    def test_generate_attack(self):
        from data.synthetic_generator import generate_attack_data
        df = generate_attack_data(100, seed=42)
        assert len(df) == 100
        assert (df["label"] == 1).all()

    def test_generate_full(self):
        from data.synthetic_generator import generate_full_dataset
        df = generate_full_dataset(500, 200, seed=42)
        assert len(df) == 700
        assert df["label"].sum() == 200


class TestPreprocessing:
    """Test preprocessing pipeline."""

    def test_sliding_windows(self):
        from preprocessing.windowing import create_sliding_windows
        features = np.random.randn(100, 10).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)
        windows, w_labels = create_sliding_windows(features, labels, sequence_length=10)
        assert windows.shape == (91, 10, 10)
        assert len(w_labels) == 91

    def test_normalization(self):
        from preprocessing.pipeline import normalize_features
        import pandas as pd
        df = pd.DataFrame({"A": [0, 50, 100], "B": [10, 20, 30]})
        normalized, scaler = normalize_features(df, ["A", "B"])
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


class TestModels:
    """Test model architectures."""

    def test_elman_rnn_forward(self):
        from models.elman_rnn import ElmanRNN
        model = ElmanRNN(input_size=10, hidden_size=32)
        x = torch.randn(4, 20, 10)  # (batch=4, seq=20, features=10)
        out = model(x)
        assert out.shape == (4,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_gru_forward(self):
        from models.gru_model import GRUModel
        model = GRUModel(input_size=10, hidden_size=32)
        x = torch.randn(4, 20, 10)
        out = model(x)
        assert out.shape == (4,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_model_training_step(self):
        from models.elman_rnn import ElmanRNN
        model = ElmanRNN(input_size=5, hidden_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()

        x = torch.randn(8, 10, 5)
        y = torch.randint(0, 2, (8,)).float()

        # Forward + backward
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


class TestFidelity:
    """Test fidelity verification."""

    def test_fidelity_verifier(self):
        from models.elman_rnn import ElmanRNN
        from fidelity.verifier import FidelityVerifier

        model = ElmanRNN(input_size=5, hidden_size=16)
        verifier = FidelityVerifier(model, ["f1", "f2", "f3", "f4", "f5"])

        samples = np.random.randn(5, 10, 5).astype(np.float32)
        shap_values = np.random.randn(5, 5).astype(np.float32)

        result = verifier.verify(samples, shap_values, top_k=2)
        assert "fidelity_score" in result
        assert "faithfulness_ratio" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
