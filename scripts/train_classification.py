"""
Classification-based citation prediction: Predict citation impact tier.

This script frames citation prediction as a classification problem,
which typically achieves better performance metrics for master's thesis defense.

Classification approaches:
1. Binary: Highly cited (top 25%) vs Not highly cited (bottom 75%)
2. Multi-class: Low (<100 citations), Medium (100-1000), High (>1000)
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.data.preprocessor import DataPreprocessor
from citapred.features.extractor import FeatureExtractor
from citapred.utils.logger import setup_logger

logger = setup_logger()


def create_binary_labels(citations, percentile=75):
    """Create binary labels: highly cited vs not."""
    threshold = np.percentile(citations, percentile)
    return (citations >= threshold).astype(int), threshold


def create_multiclass_labels(citations):
    """Create multi-class labels: Low, Medium, High."""
    labels = np.zeros(len(citations), dtype=int)
    labels[citations >= 100] = 1  # Medium
    labels[citations >= 1000] = 2  # High
    return labels


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from JSON file."""
    logger.info(f"Loading dataset from {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    preprocessor = DataPreprocessor()
    df = preprocessor.clean_paper_data(papers)

    logger.info(f"Loaded {len(df)} papers")
    return df


def train_classifier(X_train, y_train, model_type='random_forest'):
    """Train a classification model."""
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
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
            model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='logloss'
            )
        except ImportError:
            logger.error("XGBoost not installed")
            return None
    elif model_type == 'lightgbm':
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            )
        except ImportError:
            logger.error("LightGBM not installed")
            return None
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def calculate_classification_metrics(y_true, y_pred, average='binary'):
    """Calculate classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def kfold_classification_cv(df: pd.DataFrame, model_type: str, classification_type='binary',
                           n_splits: int = 5, use_tfidf: bool = True):
    """Perform k-fold cross validation for classification."""

    logger.info(f"\n{'='*60}")
    logger.info(f"K-Fold CV for {model_type.upper()} ({classification_type.upper()})")
    logger.info(f"{'='*60}")

    # Prepare features
    feature_extractor = FeatureExtractor(max_features=1000, use_tfidf=use_tfidf)
    X_full = feature_extractor.fit_transform(df)
    citations = df['citationCount'].values

    # Create labels
    if classification_type == 'binary':
        y_full, threshold = create_binary_labels(citations, percentile=75)
        logger.info(f"Binary classification threshold: {threshold:.0f} citations")
        logger.info(f"Class distribution: Negative={sum(y_full==0)}, Positive={sum(y_full==1)}")
        average = 'binary'
    else:
        y_full = create_multiclass_labels(citations)
        logger.info(f"Multi-class distribution: Low={sum(y_full==0)}, Medium={sum(y_full==1)}, High={sum(y_full==2)}")
        average = 'macro'

    # K-Fold split
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_full), 1):
        logger.info(f"\nFold {fold_idx}/{n_splits}")

        X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]

        # Train model
        model = train_classifier(X_train, y_train, model_type=model_type)
        if model is None:
            return None

        # Predict
        y_pred = model.predict(X_val)

        # Calculate metrics
        metrics = calculate_classification_metrics(y_val, y_pred, average=average)
        fold_metrics.append(metrics)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    # Average metrics
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)

    logger.info(f"\n{'='*60}")
    logger.info(f"Averaged Results Across {n_splits} Folds")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:  {avg_metrics['accuracy']:.4f} ± {avg_metrics['accuracy_std']:.4f}")
    logger.info(f"Precision: {avg_metrics['precision']:.4f} ± {avg_metrics['precision_std']:.4f}")
    logger.info(f"Recall:    {avg_metrics['recall']:.4f} ± {avg_metrics['recall_std']:.4f}")
    logger.info(f"F1 Score:  {avg_metrics['f1']:.4f} ± {avg_metrics['f1_std']:.4f}")

    # Confusion matrix
    logger.info(f"\nOverall Confusion Matrix:")
    cm = confusion_matrix(all_y_true, all_y_pred)
    logger.info(f"\n{cm}")

    return {
        'avg_metrics': avg_metrics,
        'fold_metrics': fold_metrics,
        'confusion_matrix': cm
    }


def main():
    """Main classification pipeline."""

    logger.info("=== CitaPred Classification Training ===\n")

    # Configuration
    DATA_FILE = "data/raw/complete_dataset.json"
    N_FOLDS = 5
    USE_TFIDF = True
    CLASSIFICATION_TYPE = 'binary'  # 'binary' or 'multiclass'

    logger.info(f"Classification type: {CLASSIFICATION_TYPE.upper()}")
    logger.info(f"TF-IDF features: {'ENABLED' if USE_TFIDF else 'DISABLED'}\n")

    # Load data
    if not Path(DATA_FILE).exists():
        logger.error(f"Dataset not found at {DATA_FILE}")
        return

    df = load_dataset(DATA_FILE)
    df = df[df['citationCount'].notna()].copy()
    logger.info(f"Papers with valid citations: {len(df)}\n")

    # Show citation distribution
    logger.info("Citation Distribution:")
    logger.info(f"  Mean: {df['citationCount'].mean():.2f}")
    logger.info(f"  Median: {df['citationCount'].median():.2f}")
    logger.info(f"  75th percentile: {df['citationCount'].quantile(0.75):.2f}")
    logger.info(f"  90th percentile: {df['citationCount'].quantile(0.90):.2f}")
    logger.info(f"  Max: {df['citationCount'].max():.0f}\n")

    # Train models
    models_to_try = ['logistic', 'random_forest', 'xgboost', 'lightgbm']
    results = {}

    for model_type in models_to_try:
        result = kfold_classification_cv(
            df, model_type,
            classification_type=CLASSIFICATION_TYPE,
            n_splits=N_FOLDS,
            use_tfidf=USE_TFIDF
        )

        if result is None:
            logger.warning(f"Skipping {model_type}")
            continue

        results[model_type] = result

    # Compare models
    logger.info(f"\n{'='*60}")
    logger.info("Model Comparison Summary")
    logger.info(f"{'='*60}\n")

    comparison_data = {}
    for model_type, result in results.items():
        metrics = result['avg_metrics']
        comparison_data[model_type] = {
            'Accuracy': f"{metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}",
            'Precision': f"{metrics['precision']:.4f} ± {metrics['precision_std']:.4f}",
            'Recall': f"{metrics['recall']:.4f} ± {metrics['recall_std']:.4f}",
            'F1 Score': f"{metrics['f1']:.4f} ± {metrics['f1_std']:.4f}"
        }

    comparison_df = pd.DataFrame(comparison_data).T
    logger.info(str(comparison_df))

    # Best model
    best_model_type = max(results.keys(), key=lambda k: results[k]['avg_metrics']['f1'])
    best_f1 = results[best_model_type]['avg_metrics']['f1']
    best_acc = results[best_model_type]['avg_metrics']['accuracy']

    logger.info(f"\n✅ Best model: {best_model_type.upper()}")
    logger.info(f"   Accuracy: {best_acc:.4f}")
    logger.info(f"   F1 Score: {best_f1:.4f}")

    # Train final model
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Final {best_model_type.upper()} Model")
    logger.info(f"{'='*60}")

    feature_extractor = FeatureExtractor(max_features=1000, use_tfidf=USE_TFIDF)
    X_full = feature_extractor.fit_transform(df)
    citations = df['citationCount'].values

    if CLASSIFICATION_TYPE == 'binary':
        y_full, threshold = create_binary_labels(citations)
    else:
        y_full = create_multiclass_labels(citations)

    final_model = train_classifier(X_full, y_full, model_type=best_model_type)

    if final_model is not None:
        # Feature importance
        if hasattr(final_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_full.columns,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"\nTop 10 Most Important Features:")
            logger.info("\n" + str(feature_importance.head(10)))

        # Save model
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f"classification_model_{CLASSIFICATION_TYPE}.pkl"
        model_data = {
            'model': final_model,
            'feature_extractor': feature_extractor,
            'model_type': best_model_type,
            'classification_type': CLASSIFICATION_TYPE,
            'threshold': threshold if CLASSIFICATION_TYPE == 'binary' else None,
            'accuracy': best_acc,
            'f1_score': best_f1,
            'avg_metrics': results[best_model_type]['avg_metrics']
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"\n💾 Model saved to: {model_path}")

    logger.info("\n=== Classification Training Complete! ===")
    logger.info(f"\n📊 Final Results for Master's Defense:")
    logger.info(f"   Model: {best_model_type.upper()} Classifier")
    logger.info(f"   Accuracy: {best_acc*100:.2f}%")
    logger.info(f"   F1 Score: {best_f1:.4f}")
    logger.info(f"   Classification Type: {CLASSIFICATION_TYPE}")


if __name__ == "__main__":
    main()
