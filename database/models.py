"""
Database Document Models
=========================
Pydantic models for MongoDB documents. Used for validation before storage.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class ExperimentDoc(BaseModel):
    name: str
    config: Dict
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "running"  # running, completed, failed
    rounds: int = 0
    final_metrics: Optional[Dict] = None


class PredictionDoc(BaseModel):
    experiment_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    input_features: Dict[str, float]
    prediction: str  # "attack" or "normal"
    probability: float
    ground_truth: Optional[str] = None


class RobustnessMetricsDoc(BaseModel):
    experiment_id: str
    attack_type: str  # "FGSM" or "PGD"
    epsilon: float
    clean_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class XAIMetricsDoc(BaseModel):
    experiment_id: str
    shap_feature_importance: Dict[str, float]
    fidelity_score: float
    faithfulness_ratio: float
    top_features: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
