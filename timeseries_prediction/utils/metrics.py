"""
Evaluation Metrics for Time Series Models

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return mean_squared_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE value
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (Coefficient of Determination)."""
    return r2_score(y_true, y_pred)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Mean Absolute Scaled Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for scaling
        seasonality: Seasonal period
        
    Returns:
        MASE value
    """
    n = len(y_train)
    d = np.abs(np.diff(y_train, n=seasonality)).sum() / (n - seasonality)
    
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy (percentage of correct direction predictions).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy (0-100)
    """
    if len(y_true) < 2:
        return 0.0
    
    true_direction = np.sign(np.diff(y_true.flatten()))
    pred_direction = np.sign(np.diff(y_pred.flatten()))
    
    return np.mean(true_direction == pred_direction) * 100


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Quantile loss for probabilistic forecasting.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Quantile level (0-1)
        
    Returns:
        Quantile loss
    """
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def coverage_probability(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> float:
    """
    Coverage probability for prediction intervals.
    
    Args:
        y_true: True values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        
    Returns:
        Coverage probability (0-1)
    """
    coverage = (y_true >= lower_bound) & (y_true <= upper_bound)
    return np.mean(coverage)


def mean_interval_width(lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """
    Mean width of prediction intervals.
    
    Args:
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        
    Returns:
        Mean interval width
    """
    return np.mean(upper_bound - lower_bound)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Forecast bias (mean of errors).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Forecast bias
    """
    return np.mean(y_pred - y_true)


def theil_u_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Theil's U statistic for forecast accuracy.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Theil's U statistic
    """
    numerator = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    
    return numerator / denominator if denominator != 0 else np.inf


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data (optional, for MASE)
        verbose: Whether to print results
        
    Returns:
        Dictionary of metric names and values
    """
    # Flatten arrays if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    metrics = {
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred),
        'R2': r2(y_true, y_pred),
        'Directional_Accuracy': directional_accuracy(y_true, y_pred),
        'Forecast_Bias': forecast_bias(y_true, y_pred),
        'Theil_U': theil_u_statistic(y_true, y_pred)
    }
    
    # Add MASE if training data is provided
    if y_train is not None:
        metrics['MASE'] = mase(y_true, y_pred, y_train.flatten())
    
    if verbose:
        print("\n" + "=" * 50)
        print("Model Evaluation Metrics")
        print("=" * 50)
        for metric_name, value in metrics.items():
            print(f"{metric_name:.<30} {value:.6f}")
        print("=" * 50 + "\n")
    
    return metrics


class MetricsTracker:
    """
    Track and store metrics over multiple evaluations.
    """
    
    def __init__(self):
        self.history = []
    
    def add(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """Add metrics for an epoch."""
        entry = metrics.copy()
        if epoch is not None:
            entry['epoch'] = epoch
        self.history.append(entry)
    
    def get_best(self, metric: str, mode: str = 'min') -> Dict[str, float]:
        """
        Get best metrics based on a specific metric.
        
        Args:
            metric: Metric to optimize
            mode: 'min' or 'max'
            
        Returns:
            Best metrics dictionary
        """
        if not self.history:
            return {}
        
        if mode == 'min':
            return min(self.history, key=lambda x: x.get(metric, float('inf')))
        else:
            return max(self.history, key=lambda x: x.get(metric, float('-inf')))
    
    def get_metric_history(self, metric: str) -> list:
        """Get history of a specific metric."""
        return [entry.get(metric) for entry in self.history if metric in entry]
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of all metrics."""
        if not self.history:
            return {}
        
        summary = {}
        metrics = set(k for entry in self.history for k in entry.keys() if k != 'epoch')
        
        for metric in metrics:
            values = self.get_metric_history(metric)
            values = [v for v in values if v is not None]
            
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same test set.
    
    Args:
        y_true: True values
        predictions: Dictionary of {model_name: predictions}
        verbose: Whether to print comparison
        
    Returns:
        Dictionary of {model_name: metrics}
    """
    results = {}
    
    for model_name, y_pred in predictions.items():
        results[model_name] = evaluate_model(y_true, y_pred, verbose=False)
    
    if verbose:
        print("\n" + "=" * 80)
        print("Model Comparison")
        print("=" * 80)
        
        # Get all metrics
        all_metrics = set()
        for metrics in results.values():
            all_metrics.update(metrics.keys())
        
        # Print header
        print(f"{'Metric':<20}", end='')
        for model_name in results.keys():
            print(f"{model_name:>15}", end='')
        print()
        print("-" * 80)
        
        # Print metrics
        for metric in sorted(all_metrics):
            print(f"{metric:<20}", end='')
            for model_name in results.keys():
                value = results[model_name].get(metric, 0)
                print(f"{value:>15.6f}", end='')
            print()
        
        print("=" * 80 + "\n")
    
    return results
