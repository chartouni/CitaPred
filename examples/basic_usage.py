"""
Basic usage example for CitaPred.

This script demonstrates how to:
1. Collect paper data from Semantic Scholar
2. Preprocess the data
3. Train a citation predictor
4. Make predictions and evaluate
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
    """Main function demonstrating basic usage."""

    logger.info("Starting CitaPred basic usage example")

    # Step 1: Collect data
    logger.info("Step 1: Collecting sample data from Semantic Scholar")
    # Initialize collector with API key (optional but recommended)
    # You can set your API key here or use environment variable
    api_key = "0G8y90GfQIaYaqoxFYPFH5kQFkH75un23fvs0hIx"
    collector = DataCollector(api_key=api_key)

    # Search for machine learning papers
    papers = collector.search_papers(query="machine learning", limit=100)
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

    if len(df) < 10:
        logger.error("Not enough papers with citation data for training")
        return

    # Step 3: Split data
    logger.info("Step 3: Splitting data into train/test")
    # Simple split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Step 4: Train predictor
    logger.info("Step 4: Training citation predictor")
    predictor = CitationPredictor(model_type="random_forest")
    predictor.train(train_df)

    # Step 5: Make predictions
    logger.info("Step 5: Making predictions on test data")
    predictions = predictor.predict(test_df)

    # Step 6: Evaluate
    logger.info("Step 6: Evaluating model")
    metrics = predictor.evaluate(test_df)
    print_metrics(metrics)

    # Step 7: Save model (optional)
    # model_path = "models/citation_predictor.pkl"
    # predictor.save(model_path)
    # logger.info(f"Model saved to {model_path}")

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
