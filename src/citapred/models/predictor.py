"""
Main citation predictor class.
"""

import joblib
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from citapred.features.extractor import FeatureExtractor
from citapred.models.baseline import BaselineModel

logger = logging.getLogger(__name__)


class CitationPredictor:
    """
    Main class for citation prediction pipeline.
    """

    def __init__(self, model_type: str = "random_forest", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the citation predictor.

        Args:
            model_type: Type of model to use
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.config = config or {}

        # Initialize components
        self.feature_extractor = FeatureExtractor(
            max_features=self.config.get('max_features', 1000)
        )
        self.model = BaselineModel(model_type=model_type)

        self.is_trained = False

    def train(self, train_df: pd.DataFrame, target_col: str = 'citationCount'):
        """
        Train the citation predictor.

        Args:
            train_df: Training DataFrame with paper data
            target_col: Name of the target column (citation counts)
        """
        logger.info(f"Training citation predictor with {len(train_df)} samples")

        # Extract features
        X_train = self.feature_extractor.fit_transform(train_df)
        y_train = train_df[target_col].values

        # Train model
        self.model.fit(X_train, y_train)

        self.is_trained = True
        logger.info("Training completed successfully")

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """
        Predict citation counts for papers.

        Args:
            test_df: DataFrame with paper data

        Returns:
            Series with predicted citation counts
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Extract features
        X_test = self.feature_extractor.transform(test_df)

        # Make predictions
        predictions = self.model.predict(X_test)

        return pd.Series(predictions, index=test_df.index, name='predicted_citations')

    def evaluate(self, test_df: pd.DataFrame, target_col: str = 'citationCount') -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_df: Test DataFrame
            target_col: Name of the target column

        Returns:
            Dictionary with evaluation metrics
        """
        from citapred.evaluation.metrics import calculate_metrics

        predictions = self.predict(test_df)
        y_true = test_df[target_col].values

        metrics = calculate_metrics(y_true, predictions.values)
        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def save(self, path: str):
        """
        Save the trained model.

        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'model_type': self.model_type,
            'config': self.config
        }

        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, path: str):
        """
        Load a trained model.

        Args:
            path: Path to the saved model
        """
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.feature_extractor = model_data['feature_extractor']
        self.model_type = model_data['model_type']
        self.config = model_data['config']
        self.is_trained = True

        logger.info(f"Model loaded from {path}")
