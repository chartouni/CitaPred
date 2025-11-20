"""
Evaluation metrics for citation prediction.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various evaluation metrics.

    Args:
        y_true: True citation counts
        y_pred: Predicted citation counts

    Returns:
        Dictionary with metric names and values
    """
    metrics = {}

    # Regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['r2'] = r2_score(y_true, y_pred)

    # Correlation metrics
    if len(y_true) > 1:
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)

        metrics['pearson_r'] = pearson_corr
        metrics['pearson_p'] = pearson_p
        metrics['spearman_r'] = spearman_corr
        metrics['spearman_p'] = spearman_p

    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    non_zero_mask = y_true != 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        metrics['mape'] = mape

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    Pretty print evaluation metrics.

    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)

    for metric_name, value in metrics.items():
        if 'p' in metric_name and 'pearson' in metric_name or 'spearman' in metric_name:
            # P-values in scientific notation
            print(f"{metric_name:20s}: {value:.4e}")
        else:
            print(f"{metric_name:20s}: {value:.4f}")

    print("="*50 + "\n")


def calculate_stratified_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 bins: list = [0, 10, 50, 100, float('inf')]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics stratified by citation count ranges.

    Useful for understanding model performance across different citation levels.

    Args:
        y_true: True citation counts
        y_pred: Predicted citation counts
        bins: Bin edges for stratification

    Returns:
        Dictionary mapping bin ranges to metrics
    """
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    bin_indices = np.digitize(y_true, bins[:-1]) - 1

    stratified_metrics = {}

    for i, label in enumerate(bin_labels):
        mask = bin_indices == i
        if mask.sum() > 0:
            metrics = calculate_metrics(y_true[mask], y_pred[mask])
            metrics['count'] = mask.sum()
            stratified_metrics[label] = metrics

    return stratified_metrics
