"""
Baseline models for citation prediction.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)

# Import neural network (optional - only if PyTorch is available)
try:
    from citapred.models.neural import NeuralNetworkModel
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False
    logger.warning("PyTorch not available. Neural network models will be disabled.")

# Import gradient boosting models (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")


class BaselineModel:
    """
    Simple baseline models for citation prediction.
    """

    def __init__(self, model_type: str = "linear", **kwargs):
        """
        Initialize the baseline model.

        Args:
            model_type: Type of model ('linear', 'random_forest', 'xgboost', 'lightgbm', 'neural_net', 'mean')
            **kwargs: Additional arguments passed to the model
        """
        self.model_type = model_type
        self.model = None
        self.mean_citations = None

        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "random_forest":
            n_estimators = kwargs.get('n_estimators', 100)
            self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError(
                    "XGBoost is required for xgboost models. "
                    "Install it with: pip install xgboost"
                )
            # Default XGBoost parameters optimized for citation prediction
            xgb_params = {
                'n_estimators': kwargs.get('n_estimators', 200),
                'learning_rate': kwargs.get('learning_rate', 0.1),
                'max_depth': kwargs.get('max_depth', 6),
                'min_child_weight': kwargs.get('min_child_weight', 1),
                'subsample': kwargs.get('subsample', 0.8),
                'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
                'random_state': 42,
                'verbosity': 0
            }
            self.model = xgb.XGBRegressor(**xgb_params)
        elif model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError(
                    "LightGBM is required for lightgbm models. "
                    "Install it with: pip install lightgbm"
                )
            # Default LightGBM parameters optimized for citation prediction
            lgb_params = {
                'n_estimators': kwargs.get('n_estimators', 200),
                'learning_rate': kwargs.get('learning_rate', 0.1),
                'num_leaves': kwargs.get('num_leaves', 31),
                'min_child_samples': kwargs.get('min_child_samples', 20),
                'subsample': kwargs.get('subsample', 0.8),
                'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
                'random_state': 42,
                'verbosity': -1
            }
            self.model = lgb.LGBMRegressor(**lgb_params)
        elif model_type == "neural_net":
            if not NEURAL_NET_AVAILABLE:
                raise ImportError(
                    "PyTorch is required for neural network models. "
                    "Install it with: pip install torch"
                )
            self.model = NeuralNetworkModel(**kwargs)
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
