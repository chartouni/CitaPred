"""
Demo of CitaPred using sample data (no API calls required).

This demonstrates the full pipeline with synthetic data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.features.extractor import FeatureExtractor
from citapred.models.baseline import BaselineModel
from citapred.evaluation.metrics import calculate_metrics, print_metrics
from citapred.utils.logger import setup_logger

logger = setup_logger()


def create_sample_data(n_samples=200):
    """Create sample paper data for demonstration."""
    np.random.seed(42)

    # Generate synthetic paper data
    years = np.random.randint(2015, 2023, n_samples)
    author_counts = np.random.randint(1, 8, n_samples)
    reference_counts = np.random.randint(10, 100, n_samples)

    # Generate citation counts with some realistic correlation
    # Papers with more authors, references, and older papers tend to have more citations
    citations = (
        (2023 - years) * 20 +  # Older papers get more citations
        author_counts * 15 +    # More authors = more citations
        reference_counts * 2 +  # More references = more citations
        np.random.normal(0, 50, n_samples)  # Random noise
    )
    citations = np.maximum(0, citations).astype(int)  # No negative citations

    df = pd.DataFrame({
        'paperId': [f'paper_{i}' for i in range(n_samples)],
        'title': [f'Research Paper {i}: A Study on Machine Learning' for i in range(n_samples)],
        'abstract': [f'This paper presents novel findings in the field of machine learning and artificial intelligence. ' * 3 for i in range(n_samples)],
        'year': years,
        'citationCount': citations,
        'referenceCount': reference_counts,
        'authors': [[{'name': f'Author {j}', 'authorId': f'auth_{i}_{j}'}
                     for j in range(author_counts[i])] for i in range(n_samples)],
        'venue': ['Top Conference'] * n_samples
    })

    return df


def main():
    """Run the demo."""

    logger.info("=" * 60)
    logger.info("CitaPred Demo - Using Sample Data")
    logger.info("=" * 60)

    # Create sample data
    logger.info("\n[1/6] Generating sample paper data...")
    df = create_sample_data(n_samples=200)
    logger.info(f"Generated {len(df)} papers")
    logger.info(f"Citation statistics:")
    logger.info(f"  Mean: {df['citationCount'].mean():.2f}")
    logger.info(f"  Median: {df['citationCount'].median():.2f}")
    logger.info(f"  Min: {df['citationCount'].min()}")
    logger.info(f"  Max: {df['citationCount'].max()}")

    # Split data: 80% train, 20% test
    logger.info("\n[2/6] Splitting into train/test sets...")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    logger.info(f"Train: {len(train_df)} papers | Test: {len(test_df)} papers")

    # Extract features
    logger.info("\n[3/6] Extracting features...")
    extractor = FeatureExtractor(max_features=1000)
    X_train = extractor.fit_transform(train_df)
    X_test = extractor.transform(test_df)
    logger.info(f"Feature matrix shape: {X_train.shape}")
    logger.info(f"Features extracted: {list(X_train.columns)[:5]}...")

    # Prepare targets
    y_train = train_df['citationCount'].values
    y_test = test_df['citationCount'].values

    # Train Linear Regression model
    logger.info("\n[4/6] Training Linear Regression model...")
    lr_model = BaselineModel(model_type='linear')
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    # Train Random Forest model
    logger.info("\n[5/6] Training Random Forest model...")
    rf_model = BaselineModel(model_type='random_forest')
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Evaluate both models
    logger.info("\n[6/6] Evaluating models...")

    logger.info("\n" + "=" * 60)
    logger.info("LINEAR REGRESSION RESULTS:")
    logger.info("=" * 60)
    lr_metrics = calculate_metrics(y_test, lr_predictions)
    print_metrics(lr_metrics)

    logger.info("\n" + "=" * 60)
    logger.info("RANDOM FOREST RESULTS:")
    logger.info("=" * 60)
    rf_metrics = calculate_metrics(y_test, rf_predictions)
    print_metrics(rf_metrics)

    # Show sample predictions
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE PREDICTIONS (first 10 test papers):")
    logger.info("=" * 60)
    logger.info(f"{'Actual':>10} | {'Linear':>10} | {'RF':>10} | {'Paper'}")
    logger.info("-" * 60)
    for i in range(min(10, len(y_test))):
        logger.info(f"{y_test[i]:10d} | {lr_predictions[i]:10.0f} | "
                   f"{rf_predictions[i]:10.0f} | {test_df.iloc[i]['title'][:30]}...")

    # Model comparison
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON:")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<20} | {'Linear':>12} | {'Random Forest':>15} | {'Better'}")
    logger.info("-" * 65)

    metrics_to_compare = ['mae', 'rmse', 'r2', 'pearson_r']
    metric_names = ['MAE', 'RMSE', 'R²', 'Pearson r']

    for metric, name in zip(metrics_to_compare, metric_names):
        lr_val = lr_metrics[metric]
        rf_val = rf_metrics[metric]

        # For MAE and RMSE, lower is better
        # For R² and Pearson r, higher is better
        if metric in ['mae', 'rmse']:
            better = 'Linear' if lr_val < rf_val else 'Random Forest'
        else:
            better = 'Linear' if lr_val > rf_val else 'Random Forest'

        logger.info(f"{name:<20} | {lr_val:12.4f} | {rf_val:15.4f} | {better}")

    logger.info("\n" + "=" * 60)
    logger.info("Demo completed successfully!")
    logger.info("=" * 60)
    logger.info("\nNote: This used synthetic data. For real results:")
    logger.info("  1. Run: python scripts/collect_large_dataset.py")
    logger.info("  2. Run: python scripts/train_improved_model.py")


if __name__ == "__main__":
    main()
