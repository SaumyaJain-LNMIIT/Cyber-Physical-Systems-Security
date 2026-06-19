"""
MongoDB Connection Manager
============================
Handles connections to MongoDB for storing experiment results,
predictions, robustness metrics, and XAI results.
"""

from pymongo import MongoClient
from loguru import logger
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseManager:
    """MongoDB connection and CRUD operations for CPS experiments."""

    def __init__(self, uri=None, db_name=None):
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", "cps_security")
        self.client = None
        self.db = None

    def connect(self):
        """Establish connection to MongoDB."""
        try:
            import certifi
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                retryWrites=True,
                w="majority",
                tlsCAFile=certifi.where(),
            )
            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB: {self.db_name}")
            return True
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Running without database.")
            self.db = None
            return False

    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    # --- Experiments ---
    def save_experiment(self, experiment_data):
        """Save experiment metadata and config."""
        if self.db is None:
            return None
        result = self.db.experiments.insert_one(experiment_data)
        logger.info(f"Saved experiment: {result.inserted_id}")
        return str(result.inserted_id)

    def get_experiment(self, experiment_id):
        from bson import ObjectId
        if self.db is None:
            return None
        return self.db.experiments.find_one({"_id": ObjectId(experiment_id)})

    def list_experiments(self, limit=20):
        if self.db is None:
            return []
        return list(self.db.experiments.find().sort("created_at", -1).limit(limit))

    # --- Predictions ---
    def save_predictions(self, predictions_list):
        """Save batch of predictions."""
        if self.db is None:
            return 0
        result = self.db.predictions.insert_many(predictions_list)
        return len(result.inserted_ids)

    def get_predictions(self, experiment_id, limit=100):
        if self.db is None:
            return []
        return list(self.db.predictions.find(
            {"experiment_id": experiment_id}
        ).limit(limit))

    # --- Robustness Metrics ---
    def save_robustness_metrics(self, metrics_data):
        """Save adversarial robustness results."""
        if self.db is None:
            return None
        result = self.db.robustness_metrics.insert_one(metrics_data)
        return str(result.inserted_id)

    def get_robustness_metrics(self, experiment_id):
        if self.db is None:
            return []
        return list(self.db.robustness_metrics.find({"experiment_id": experiment_id}))

    # --- XAI Metrics ---
    def save_xai_metrics(self, xai_data):
        """Save SHAP explanation and fidelity results."""
        if self.db is None:
            return None
        result = self.db.xai_metrics.insert_one(xai_data)
        return str(result.inserted_id)

    def get_xai_metrics(self, experiment_id):
        if self.db is None:
            return []
        return list(self.db.xai_metrics.find({"experiment_id": experiment_id}))
