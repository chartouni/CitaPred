"""
Citation Prediction Demo Script

Predict citation counts for new research papers using the trained model.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.features.extractor import FeatureExtractor
from citapred.utils.logger import setup_logger

logger = setup_logger()


def load_trained_model(model_path: str = "models/best_model.pkl"):
    """Load the trained model and feature extractor."""
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        logger.info("Please train a model first using train_kfold_model.py")
        return None, None

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    return model_data['model'], model_data['feature_extractor']


def predict_citations(paper_data: dict, model, feature_extractor) -> dict:
    """
    Predict citations for a paper.

    Args:
        paper_data: Dictionary with keys: title, abstract, venue, year, authors, referenceCount
        model: Trained model
        feature_extractor: Fitted feature extractor

    Returns:
        Dictionary with prediction results
    """
    # Convert to DataFrame
    df = pd.DataFrame([paper_data])

    # Extract features
    features = feature_extractor.transform(df)

    # Predict in log space
    log_prediction = model.predict(features)[0]

    # Transform back to original space
    prediction = np.exp(log_prediction) - 1
    prediction = max(0, prediction)  # Ensure non-negative

    return {
        'predicted_citations': int(prediction),
        'log_prediction': float(log_prediction),
        'confidence_range': (
            int(max(0, prediction * 0.5)),  # Lower bound (50% of prediction)
            int(prediction * 2)  # Upper bound (200% of prediction)
        )
    }


def main():
    """Demo script for citation prediction."""

    logger.info("=== CitaPred Citation Prediction Demo ===\n")

    # Example paper 1: High-impact ML conference paper
    paper1 = {
        'title': 'Attention Is All You Need: A Novel Approach to Sequence Modeling',
        'abstract': 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.',
        'venue': 'NeurIPS',
        'year': 2017,
        'authors': [{'name': 'Author 1'}, {'name': 'Author 2'}, {'name': 'Author 3'}],
        'referenceCount': 45
    }

    # Example paper 2: Standard journal paper
    paper2 = {
        'title': 'A Survey of Deep Learning Techniques for Medical Image Analysis',
        'abstract': 'This paper surveys recent advances in deep learning for medical imaging. We review convolutional neural networks, recurrent networks, and generative models applied to various medical imaging tasks including classification, segmentation, and detection.',
        'venue': 'IEEE Transactions',
        'year': 2020,
        'authors': [{'name': 'Author A'}, {'name': 'Author B'}],
        'referenceCount': 120
    }

    # Example paper 3: Recent preprint
    paper3 = {
        'title': 'Exploring Novel Architectures for Computer Vision',
        'abstract': 'We explore several novel architectural designs for computer vision tasks. Our experiments show promising results on standard benchmarks.',
        'venue': 'arXiv',
        'year': 2023,
        'authors': [{'name': 'Researcher X'}],
        'referenceCount': 25
    }

    # Load model
    logger.info("Loading trained model...")
    model_path = Path(__file__).parent.parent / "models" / "best_model.pkl"

    if not model_path.exists():
        logger.warning("No saved model found. Please save a model first.")
        logger.info("\nTo save a model, modify train_kfold_model.py to save the final model:")
        logger.info("```python")
        logger.info("import pickle")
        logger.info("model_data = {'model': final_model, 'feature_extractor': feature_extractor}")
        logger.info("with open('models/best_model.pkl', 'wb') as f:")
        logger.info("    pickle.dump(model_data, f)")
        logger.info("```")
        return

    model, feature_extractor = load_trained_model(str(model_path))

    if model is None:
        return

    logger.info("Model loaded successfully!\n")

    # Make predictions
    examples = [
        ("High-Impact NeurIPS Paper (2017)", paper1),
        ("Standard IEEE Journal Paper (2020)", paper2),
        ("Recent arXiv Preprint (2023)", paper3)
    ]

    for name, paper in examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"Prediction for: {name}")
        logger.info(f"{'='*60}")
        logger.info(f"Title: {paper['title'][:80]}...")
        logger.info(f"Venue: {paper['venue']}")
        logger.info(f"Year: {paper['year']}")
        logger.info(f"Authors: {len(paper['authors'])}")
        logger.info(f"References: {paper['referenceCount']}")

        result = predict_citations(paper, model, feature_extractor)

        logger.info(f"\n📊 Prediction Results:")
        logger.info(f"  Predicted Citations: {result['predicted_citations']:,}")
        logger.info(f"  Confidence Range: {result['confidence_range'][0]:,} - {result['confidence_range'][1]:,}")
        logger.info(f"  Log-scale prediction: {result['log_prediction']:.2f}")

    logger.info(f"\n{'='*60}")
    logger.info("\n✅ Demo complete! Use this script as a template for your own predictions.")
    logger.info("\nTo predict for your own paper, modify the paper dictionary with your paper's details.")


def predict_single_paper_interactive():
    """Interactive mode for predicting a single paper."""
    logger.info("=== Interactive Citation Prediction ===\n")

    # Get user input
    title = input("Enter paper title: ")
    abstract = input("Enter paper abstract: ")
    venue = input("Enter venue (e.g., NeurIPS, CVPR, arXiv): ")
    year = int(input("Enter publication year: "))
    author_count = int(input("Enter number of authors: "))
    ref_count = int(input("Enter number of references: "))

    paper = {
        'title': title,
        'abstract': abstract,
        'venue': venue,
        'year': year,
        'authors': [{'name': f'Author {i+1}'} for i in range(author_count)],
        'referenceCount': ref_count
    }

    # Load model
    model_path = Path(__file__).parent.parent / "models" / "best_model.pkl"
    model, feature_extractor = load_trained_model(str(model_path))

    if model is None:
        return

    # Predict
    result = predict_citations(paper, model, feature_extractor)

    logger.info(f"\n📊 Prediction Results:")
    logger.info(f"  Predicted Citations: {result['predicted_citations']:,}")
    logger.info(f"  Confidence Range: {result['confidence_range'][0]:,} - {result['confidence_range'][1]:,}")


if __name__ == "__main__":
    # Check if interactive mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        predict_single_paper_interactive()
    else:
        main()
