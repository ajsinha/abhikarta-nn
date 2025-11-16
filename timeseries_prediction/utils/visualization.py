"""
Visualization Utilities for Time Series

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_time_series(
    data: np.ndarray,
    title: str = 'Time Series',
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot time series data.
    
    Args:
        data: Time series data (n_samples, n_features)
        title: Plot title
        labels: Feature labels
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    if data.ndim == 1:
        plt.plot(data)
    else:
        for i in range(min(data.shape[1], 10)):  # Plot max 10 features
            label = labels[i] if labels and i < len(labels) else f'Feature {i+1}'
            plt.plot(data[:, i], label=label, alpha=0.7)
        
        if data.shape[1] <= 10:
            plt.legend()
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Predictions vs True Values',
    save_path: Optional[str] = None
):
    """
    Plot predictions against true values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series plot
    ax1.plot(y_true.flatten(), label='True', alpha=0.7)
    ax1.plot(y_pred.flatten(), label='Predicted', alpha=0.7)
    ax1.set_title(title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # Scatter plot
    ax2.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5)
    
    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Prediction Scatter Plot')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = 'Training History',
    save_path: Optional[str] = None
):
    """
    Plot training and validation losses.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=3)
    
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=3)
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot residual analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save figure
    """
    residuals = y_true.flatten() - y_pred.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals over time
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True)
    
    # Residuals distribution
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True)
    
    # Residuals vs Predicted
    axes[1, 1].scatter(y_pred.flatten(), residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_title('Residuals vs Predicted Values')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_forecast_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    title: str = 'Forecast with Prediction Intervals',
    save_path: Optional[str] = None
):
    """
    Plot forecasts with prediction intervals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    x = range(len(y_true.flatten()))
    
    plt.plot(x, y_true.flatten(), label='True', marker='o', markersize=4, alpha=0.7)
    plt.plot(x, y_pred.flatten(), label='Predicted', marker='s', markersize=4, alpha=0.7)
    
    if lower_bound is not None and upper_bound is not None:
        plt.fill_between(
            x,
            lower_bound.flatten(),
            upper_bound.flatten(),
            alpha=0.3,
            label='95% Confidence Interval'
        )
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_multiple_forecasts(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = 'Model Comparison',
    save_path: Optional[str] = None
):
    """
    Plot predictions from multiple models.
    
    Args:
        y_true: True values
        predictions: Dictionary of {model_name: predictions}
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(14, 7))
    
    plt.plot(y_true.flatten(), label='True', linewidth=2, alpha=0.8)
    
    for model_name, y_pred in predictions.items():
        plt.plot(y_pred.flatten(), label=model_name, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(
    data: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = 'Correlation Matrix',
    save_path: Optional[str] = None
):
    """
    Plot correlation matrix heatmap.
    
    Args:
        data: Time series data (n_samples, n_features)
        labels: Feature labels
        title: Plot title
        save_path: Path to save figure
    """
    corr = np.corrcoef(data.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        xticklabels=labels if labels else range(data.shape[1]),
        yticklabels=labels if labels else range(data.shape[1])
    )
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_decomposition(
    data: np.ndarray,
    period: int = 12,
    title: str = 'Time Series Decomposition',
    save_path: Optional[str] = None
):
    """
    Plot time series decomposition (trend, seasonal, residual).
    
    Args:
        data: Time series data (1D)
        period: Seasonal period
        title: Plot title
        save_path: Path to save figure
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Ensure 1D
    if data.ndim > 1:
        data = data.flatten()
    
    # Decompose
    result = seasonal_decompose(data, model='additive', period=period)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    result.observed.plot(ax=axes[0], title='Observed')
    axes[0].grid(True)
    
    result.trend.plot(ax=axes[1], title='Trend')
    axes[1].grid(True)
    
    result.seasonal.plot(ax=axes[2], title='Seasonal')
    axes[2].grid(True)
    
    result.resid.plot(ax=axes[3], title='Residual')
    axes[3].grid(True)
    
    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = 'Feature Importance',
    top_n: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot feature importance.
    
    Args:
        importance: Feature importance scores
        feature_names: Names of features
        title: Plot title
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    # Sort by importance
    indices = np.argsort(importance)[-top_n:]
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance))]
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
