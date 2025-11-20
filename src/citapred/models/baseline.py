"""
Baseline models for citation prediction.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Simple baseline models for citation prediction.
    """

    def __init__(self, model_type: str = "linear"):
        """
        Initialize the baseline model.

        Args:
            model_type: Type of model ('linear', 'random_forest', 'mean')
        """
        self.model_type = model_type
        self.model = None
        self.mean_citations = None

        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "mean":
            # Simple mean predictor
            pass
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y):
        """
        Train the model.

        Args:
            X: Feature matrix
            y: Target values (citation counts)
        """
        if self.model_type == "mean":
            self.mean_citations = np.mean(y)
            logger.info(f"Mean baseline: {self.mean_citations:.2f}")
        else:
            self.model.fit(X, y)
            logger.info(f"{self.model_type} model trained successfully")

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted citation counts
        """
        if self.model_type == "mean":
            return np.full(len(X), self.mean_citations)
        else:
            return self.model.predict(X)

    def score(self, X, y):
        """
        Calculate R² score.

        Args:
            X: Feature matrix
            y: True citation counts

        Returns:
            R² score
        """
        if self.model_type == "mean":
            predictions = self.predict(X)
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            return self.model.score(X, y)
