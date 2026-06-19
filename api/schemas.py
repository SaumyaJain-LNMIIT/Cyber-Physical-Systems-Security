"""
API Pydantic Schemas
=====================
Request/response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class TrainRequest(BaseModel):
    config_path: str = Field(default="config/local.yaml", description="Path to config YAML")
    experiment_name: str = Field(default="experiment_001", description="Name for this experiment")


class TrainResponse(BaseModel):
    experiment_id: Optional[str] = None
    status: str
    message: str
    metrics: Optional[Dict] = None


class PredictRequest(BaseModel):
    sensor_data: List[List[float]] = Field(
        description="Sequence of sensor readings: [[sensor1, sensor2, ...], ...]"
    )


class PredictResponse(BaseModel):
    prediction: str  # "attack" or "normal"
    probability: float
    confidence: float


class ExplainResponse(BaseModel):
    feature_importance: Dict[str, float]
    top_features: List[str]
    fidelity_score: Optional[float] = None


class AttackTestRequest(BaseModel):
    attack_type: str = Field(default="fgsm", description="'fgsm' or 'pgd'")
    epsilon: float = Field(default=0.1, description="Perturbation magnitude")
    num_samples: int = Field(default=100, description="Number of samples to attack")


class AttackTestResponse(BaseModel):
    attack_type: str
    clean_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float


class MetricsResponse(BaseModel):
    experiment_id: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    attack_success_rate: Optional[Dict[str, float]] = None
    fidelity_score: Optional[float] = None
