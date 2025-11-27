"""
Neural network example for CitaPred.

This script demonstrates how to use the neural network model for citation prediction.
Requires PyTorch to be installed: pip install torch
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.data.collector import DataCollector
from citapred.data.preprocessor import DataPreprocessor
from citapred.models.predictor import CitationPredictor
from citapred.evaluation.metrics import print_metrics
from citapred.utils.logger import setup_logger

# Setup logging
logger = setup_logger()


def main():
    """Main function demonstrating neural network usage."""

    logger.info("Starting CitaPred Neural Network Example")

    # Step 1: Collect data
    logger.info("Step 1: Collecting sample data from Semantic Scholar")
    # Initialize collector with API key
    # You can set your API key here or use environment variable
    api_key = "0G8y90GfQIaYaqoxFYPFH5kQFkH75un23fvs0hIx"
    collector = DataCollector(api_key=api_key)

    # Search for machine learning papers (limit=100 for API stability)
    papers = collector.search_papers(query="deep learning", limit=100)
    logger.info(f"Collected {len(papers)} papers")

    if len(papers) == 0:
        logger.error("No papers collected. Check your internet connection.")
        return

    # Step 2: Preprocess data
    logger.info("Step 2: Preprocessing data")
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_paper_data(papers)
    logger.info(f"Preprocessed data shape: {df.shape}")

    # Filter papers with citation counts (for training)
    df = df[df['citationCount'].notna() & (df['citationCount'] > 0)]
    logger.info(f"Papers with citations: {len(df)}")

    if len(df) < 30:
        logger.error("Not enough papers with citation data for training")
        return

    # Step 3: Split data
    logger.info("Step 3: Splitting data into train/test")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Step 4: Train Neural Network predictor
    logger.info("Step 4: Training neural network citation predictor")

    # Configure neural network parameters
    nn_config = {
        'hidden_sizes': [128, 64, 32],  # Three hidden layers
        'dropout_rate': 0.3,             # Dropout for regularization
        'learning_rate': 0.001,          # Adam optimizer learning rate
        'batch_size': 16,                # Batch size for training
        'epochs': 50,                    # Maximum epochs
        'early_stopping_patience': 10    # Stop if no improvement after 10 epochs
    }

    predictor = CitationPredictor(model_type="neural_net", config=nn_config)
    predictor.train(train_df)

    # Step 5: Make predictions
    logger.info("Step 5: Making predictions on test data")
    predictions = predictor.predict(test_df)

    # Step 6: Evaluate
    logger.info("Step 6: Evaluating neural network model")
    metrics = predictor.evaluate(test_df)
    print_metrics(metrics)

    # Step 7: Show training history (if available)
    try:
        if hasattr(predictor.model.model, 'get_training_history'):
            history = predictor.model.model.get_training_history()
            logger.info("\nTraining History:")
            logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
            logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
            logger.info(f"Total epochs: {len(history['train_loss'])}")
    except Exception as e:
        logger.warning(f"Could not retrieve training history: {e}")

    # Step 8: Compare with Random Forest baseline
    logger.info("\n" + "="*50)
    logger.info("Step 8: Comparing with Random Forest baseline")
    logger.info("="*50)

    rf_predictor = CitationPredictor(model_type="random_forest")
    rf_predictor.train(train_df)
    rf_metrics = rf_predictor.evaluate(test_df)

    logger.info("\nRandom Forest Results:")
    print_metrics(rf_metrics)

    # Step 9: Model comparison
    logger.info("\n" + "="*50)
    logger.info("Model Comparison Summary")
    logger.info("="*50)
    logger.info(f"Neural Network - R²: {metrics.get('r2', 0):.4f}, "
                f"MAE: {metrics.get('mae', 0):.2f}")
    logger.info(f"Random Forest  - R²: {rf_metrics.get('r2', 0):.4f}, "
                f"MAE: {rf_metrics.get('mae', 0):.2f}")

    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    try:
        import torch
        main()
    except ImportError:
        print("ERROR: PyTorch is required for neural network models.")
        print("Install it with: pip install torch")
        sys.exit(1)
