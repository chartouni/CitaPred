"""
K-Fold Cross Validation training script for citation prediction.

This script uses k-fold CV to maximize training data usage and get
robust performance estimates with limited data.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.data.preprocessor import DataPreprocessor
from citapred.features.extractor import FeatureExtractor
from citapred.evaluation.metrics import calculate_metrics, print_metrics
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


def apply_log_transform(y: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Apply log transformation to citation counts."""
    return np.log(y + epsilon)


def inverse_log_transform(y_log: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Inverse log transformation to get back citation counts."""
    return np.exp(y_log) - epsilon


def train_model(X_train, y_train, model_type='random_forest'):
    """Train a model."""
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
    elif model_type == 'xgboost':
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            return None
    elif model_type == 'lightgbm':
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            )
        except ImportError:
            logger.error("LightGBM not installed. Install with: pip install lightgbm")
            return None
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def kfold_cross_validate(df: pd.DataFrame, model_type: str, n_splits: int = 5):
    """
    Perform k-fold cross validation for a model.

    Args:
        df: Full dataset
        model_type: Type of model to train
        n_splits: Number of folds

    Returns:
        Dictionary with averaged metrics and per-fold results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"K-Fold CV for {model_type.upper()} (k={n_splits})")
    logger.info(f"{'='*60}")

    # Prepare feature extractor (fit on full dataset)
    feature_extractor = FeatureExtractor(max_features=1000)
    X_full = feature_extractor.fit_transform(df)
    y_full_original = df['citationCount'].values
    y_full_log = apply_log_transform(y_full_original)

    # K-Fold split
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    fold_predictions = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_full), 1):
        logger.info(f"\nFold {fold_idx}/{n_splits}")
        logger.info(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        # Split data
        X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_train_log, y_val_log = y_full_log[train_idx], y_full_log[val_idx]
        y_val_original = y_full_original[val_idx]

        # Train model
        model = train_model(X_train, y_train_log, model_type=model_type)

        if model is None:
            return None

        # Predict in log space
        y_pred_log = model.predict(X_val)

        # Transform back to original space
        y_pred = inverse_log_transform(y_pred_log)
        y_pred = np.maximum(y_pred, 0)  # Ensure non-negative

        # Calculate metrics (including adjusted R²)
        n_features = X_val.shape[1]
        metrics = calculate_metrics(y_val_original, y_pred, n_features=n_features)
        fold_metrics.append(metrics)
        fold_predictions.append({
            'true': y_val_original,
            'pred': y_pred,
            'indices': val_idx
        })

        logger.info(f"  MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")

    # Average metrics across folds
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)

    logger.info(f"\n{'='*60}")
    logger.info(f"Averaged Results Across {n_splits} Folds")
    logger.info(f"{'='*60}")
    logger.info(f"MAE:  {avg_metrics['mae']:.2f} ± {avg_metrics['mae_std']:.2f}")
    logger.info(f"RMSE: {avg_metrics['rmse']:.2f} ± {avg_metrics['rmse_std']:.2f}")
    logger.info(f"R²:   {avg_metrics['r2']:.4f} ± {avg_metrics['r2_std']:.4f}")
    logger.info(f"Pearson r: {avg_metrics['pearson_r']:.4f} ± {avg_metrics['pearson_r_std']:.4f}")
    logger.info(f"Spearman r: {avg_metrics['spearman_r']:.4f} ± {avg_metrics['spearman_r_std']:.4f}")

    return {
        'avg_metrics': avg_metrics,
        'fold_metrics': fold_metrics,
        'fold_predictions': fold_predictions
    }


def main():
    """Main training pipeline with k-fold cross validation."""

    logger.info("=== CitaPred K-Fold Cross Validation Training ===\n")

    # Configuration
    DATA_FILE = "data/raw/complete_dataset.json"
    N_FOLDS = 5

    # Check if data file exists
    if not Path(DATA_FILE).exists():
        logger.error(f"Dataset not found at {DATA_FILE}")
        logger.error("Please run collect_large_dataset.py first")
        return

    # Load dataset
    df = load_dataset(DATA_FILE)

    # Filter papers with valid citation counts
    df = df[df['citationCount'].notna()].copy()
    logger.info(f"Papers with valid citations: {len(df)}")

    if len(df) < 50:
        logger.error("Not enough papers for k-fold CV. Need at least 50 papers with citations.")
        return

    # Show citation distribution
    logger.info("\nCitation Distribution (Original Scale):")
    logger.info(f"  Mean: {df['citationCount'].mean():.2f}")
    logger.info(f"  Median: {df['citationCount'].median():.2f}")
    logger.info(f"  Std: {df['citationCount'].std():.2f}")
    logger.info(f"  Min: {df['citationCount'].min()}")
    logger.info(f"  Max: {df['citationCount'].max()}")
    logger.info(f"  99th percentile: {df['citationCount'].quantile(0.99):.2f}")

    # Show log-transformed distribution
    log_citations = np.log(df['citationCount'] + 1)
    logger.info("\nCitation Distribution (Log-Transformed):")
    logger.info(f"  Mean: {log_citations.mean():.2f}")
    logger.info(f"  Median: {log_citations.median():.2f}")
    logger.info(f"  Std: {log_citations.std():.2f}")
    logger.info(f"  Min: {log_citations.min():.2f}")
    logger.info(f"  Max: {log_citations.max():.2f}")
    logger.info("\nNote: Models will be trained on log-transformed citations to handle skewness")


    # Train and evaluate models with k-fold CV
    models_to_try = ['linear', 'random_forest', 'xgboost', 'lightgbm']
    results = {}

    for model_type in models_to_try:
        result = kfold_cross_validate(df, model_type, n_splits=N_FOLDS)

        if result is None:
            logger.warning(f"Skipping {model_type} - library not available")
            continue

        results[model_type] = result

    # Compare models
    logger.info(f"\n{'='*60}")
    logger.info("Model Comparison Summary (K-Fold CV)")
    logger.info(f"{'='*60}")

    comparison_data = {}
    for model_type, result in results.items():
        metrics = result['avg_metrics']
        comparison_data[model_type] = {
            'MAE': f"{metrics['mae']:.2f} ± {metrics['mae_std']:.2f}",
            'RMSE': f"{metrics['rmse']:.2f} ± {metrics['rmse_std']:.2f}",
            'R²': f"{metrics['r2']:.4f} ± {metrics['r2_std']:.4f}",
            'Pearson r': f"{metrics['pearson_r']:.4f} ± {metrics['pearson_r_std']:.4f}"
        }

    comparison_df = pd.DataFrame(comparison_data).T
    logger.info("\n" + str(comparison_df))

    # Determine best model
    best_model_type = max(results.keys(), key=lambda k: results[k]['avg_metrics']['r2'])
    best_r2 = results[best_model_type]['avg_metrics']['r2']

    logger.info(f"\nBest model: {best_model_type} (R² = {best_r2:.4f})")

    # Train final model on full dataset
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Final {best_model_type.upper()} Model on Full Dataset")
    logger.info(f"{'='*60}")

    feature_extractor = FeatureExtractor(max_features=1000)
    X_full = feature_extractor.fit_transform(df)
    y_full_original = df['citationCount'].values
    y_full_log = apply_log_transform(y_full_original)

    final_model = train_model(X_full, y_full_log, model_type=best_model_type)

    if final_model is not None:
        logger.info("Final model trained successfully on all data")

        # Feature importance for tree-based models
        if best_model_type in ['random_forest', 'xgboost', 'lightgbm']:
            feature_importance = pd.DataFrame({
                'feature': X_full.columns,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"\nTop 10 Most Important Features ({best_model_type}):")
            logger.info("\n" + str(feature_importance.head(10)))

    logger.info("\n=== K-Fold Cross Validation Complete! ===")


if __name__ == "__main__":
    main()
