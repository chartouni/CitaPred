"""
Improved training script with log transformation and temporal validation.

This script addresses the issues found in the basic example:
1. Uses log transformation for citation counts (reduces skewness)
2. Implements temporal validation (train on old, test on new)
3. Better evaluation with stratified metrics
4. Model comparison
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.data.preprocessor import DataPreprocessor
from citapred.features.extractor import FeatureExtractor
from citapred.evaluation.metrics import calculate_metrics, print_metrics, calculate_stratified_metrics
from citapred.utils.logger import setup_logger

logger = setup_logger()


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from JSON file."""
    logger.info(f"Loading dataset from {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    preprocessor = DataPreprocessor()
    df = preprocessor.clean_paper_data(papers)

    logger.info(f"Loaded {len(df)} papers")
    return df


def temporal_split(df: pd.DataFrame, train_end_year: int = 2018, val_end_year: int = 2019):
    """
    Split data by year for realistic evaluation.

    Train on papers from older years, validate/test on newer papers.
    This simulates real-world scenario: predict citations for new papers.
    """
    if 'year' not in df.columns:
        raise ValueError("Dataset must have 'year' column")

    train_df = df[df['year'] <= train_end_year].copy()
    val_df = df[(df['year'] > train_end_year) & (df['year'] <= val_end_year)].copy()
    test_df = df[df['year'] > val_end_year].copy()

    logger.info(f"Temporal split:")
    logger.info(f"  Train: {len(train_df)} papers (year <= {train_end_year})")
    logger.info(f"  Val:   {len(val_df)} papers ({train_end_year} < year <= {val_end_year})")
    logger.info(f"  Test:  {len(test_df)} papers (year > {val_end_year})")

    return train_df, val_df, test_df


def apply_log_transform(y: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Apply log transformation to citation counts.

    Log transform reduces skewness and makes the distribution more normal.
    We add epsilon to handle zero citations: log(citations + 1)
    """
    return np.log(y + epsilon)


def inverse_log_transform(y_log: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Inverse log transformation to get back citation counts.
    """
    return np.exp(y_log) - epsilon


def train_model(X_train, y_train, model_type='random_forest'):
    """Train a model."""
    logger.info(f"Training {model_type} model...")

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    logger.info("Training complete")

    return model


def evaluate_model(model, X_test, y_test_log, y_test_original, split_name='Test'):
    """
    Evaluate model and print metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test_log: Log-transformed test targets
        y_test_original: Original citation counts (not log-transformed)
        split_name: Name of split for logging
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation on {split_name} Set")
    logger.info(f"{'='*60}")

    # Predict in log space
    y_pred_log = model.predict(X_test)

    # Transform back to original space
    y_pred = inverse_log_transform(y_pred_log)

    # Clip negative predictions to 0
    y_pred = np.maximum(y_pred, 0)

    # Calculate metrics in original space
    metrics = calculate_metrics(y_test_original, y_pred)
    print_metrics(metrics)

    # Calculate stratified metrics
    logger.info("\nStratified Metrics (by citation count):")
    stratified = calculate_stratified_metrics(
        y_test_original,
        y_pred,
        bins=[0, 10, 50, 100, 500, float('inf')]
    )

    for bin_range, bin_metrics in stratified.items():
        logger.info(f"\n  Citation range: {bin_range}")
        logger.info(f"    Count: {bin_metrics['count']}")
        logger.info(f"    MAE: {bin_metrics['mae']:.2f}")
        logger.info(f"    RMSE: {bin_metrics['rmse']:.2f}")
        logger.info(f"    R²: {bin_metrics['r2']:.4f}")

    return metrics, y_pred


def main():
    """Main training pipeline."""

    logger.info("=== CitaPred Improved Training Script ===\n")

    # Configuration
    DATA_FILE = "data/raw/complete_dataset.json"
    TRAIN_END_YEAR = 2018
    VAL_END_YEAR = 2019

    # Check if data file exists
    if not Path(DATA_FILE).exists():
        logger.error(f"Dataset not found: {DATA_FILE}")
        logger.error("Please run: python scripts/collect_large_dataset.py first")
        return

    # Load data
    df = load_dataset(DATA_FILE)

    # Filter papers with valid citation counts
    df = df[df['citationCount'].notna()].copy()
    logger.info(f"Papers with valid citations: {len(df)}")

    if len(df) < 100:
        logger.error("Not enough papers for training. Collect more data first.")
        return

    # Show citation distribution
    logger.info("\nCitation Distribution:")
    logger.info(f"  Mean: {df['citationCount'].mean():.2f}")
    logger.info(f"  Median: {df['citationCount'].median():.2f}")
    logger.info(f"  Std: {df['citationCount'].std():.2f}")
    logger.info(f"  Min: {df['citationCount'].min()}")
    logger.info(f"  Max: {df['citationCount'].max()}")
    logger.info(f"  25th percentile: {df['citationCount'].quantile(0.25):.2f}")
    logger.info(f"  75th percentile: {df['citationCount'].quantile(0.75):.2f}")

    # Temporal split
    train_df, val_df, test_df = temporal_split(df, TRAIN_END_YEAR, VAL_END_YEAR)

    if len(test_df) == 0:
        logger.warning("No test data available with current year split. Using validation as test.")
        test_df = val_df
        val_df = pd.DataFrame()

    # Extract features
    logger.info("\nExtracting features...")
    feature_extractor = FeatureExtractor(max_features=1000)

    X_train = feature_extractor.fit_transform(train_df)
    X_test = feature_extractor.transform(test_df)

    logger.info(f"Feature matrix shape: {X_train.shape}")
    logger.info(f"Features: {list(X_train.columns)}")

    # Get targets
    y_train_original = train_df['citationCount'].values
    y_test_original = test_df['citationCount'].values

    # Apply log transformation
    logger.info("\nApplying log transformation to citation counts...")
    y_train_log = apply_log_transform(y_train_original)
    y_test_log = apply_log_transform(y_test_original)

    logger.info(f"Original citation range: [{y_train_original.min()}, {y_train_original.max()}]")
    logger.info(f"Log-transformed range: [{y_train_log.min():.2f}, {y_train_log.max():.2f}]")

    # Train models and compare
    models_to_try = ['linear', 'random_forest']
    results = {}

    for model_type in models_to_try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type.upper()} Model")
        logger.info(f"{'='*60}")

        model = train_model(X_train, y_train_log, model_type=model_type)
        metrics, predictions = evaluate_model(
            model, X_test, y_test_log, y_test_original,
            split_name=f'Test ({model_type})'
        )

        results[model_type] = {
            'model': model,
            'metrics': metrics,
            'predictions': predictions
        }

    # Compare models
    logger.info(f"\n{'='*60}")
    logger.info("Model Comparison Summary")
    logger.info(f"{'='*60}")

    comparison_df = pd.DataFrame({
        model_type: {
            'MAE': results[model_type]['metrics']['mae'],
            'RMSE': results[model_type]['metrics']['rmse'],
            'R²': results[model_type]['metrics']['r2'],
            'Pearson r': results[model_type]['metrics']['pearson_r']
        }
        for model_type in models_to_try
    }).T

    logger.info("\n" + str(comparison_df))

    # Save best model
    best_model_type = comparison_df['R²'].idxmax()
    logger.info(f"\nBest model: {best_model_type} (R² = {comparison_df.loc[best_model_type, 'R²']:.4f})")

    # Feature importance (for Random Forest)
    if best_model_type == 'random_forest':
        model = results[best_model_type]['model']
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop 10 Most Important Features:")
        logger.info("\n" + str(feature_importance.head(10)))

    logger.info("\n=== Training Complete! ===")


if __name__ == "__main__":
    main()
