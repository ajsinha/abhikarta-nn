"""
Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com

Legal Notice: This module and the associated software architecture are proprietary 
and confidential. Unauthorized copying, distribution, modification, or use is 
strictly prohibited without explicit written permission from the copyright holder.

Patent Pending: Certain architectural patterns and implementations described in 
this module may be subject to patent applications.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from pathlib import Path


class TimeSeriesModel(ABC):
    """
    Abstract base class for all time series models (statistical and deep learning).
    Provides common utilities and enforces a consistent interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the time series model.
        
        Args:
            config: Configuration dictionary for the model
        """
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.target_names = None
        self.train_history = []
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'TimeSeriesModel':
        """
        Fit the model to training data.
        
        Args:
            X: Input features as DataFrame
            y: Target values as DataFrame
            **kwargs: Additional fitting parameters
            
        Returns:
            self: Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features as DataFrame
            steps: Number of steps ahead to predict
            
        Returns:
            Predictions as numpy array
        """
        pass
    
    @abstractmethod
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for forecasting
            
        Returns:
            Forecasted values as numpy array
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Input features
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        y_true = y.values if isinstance(y, pd.DataFrame) else y
        
        # Handle shape mismatch due to sequence preparation
        # Predictions may be shorter than y_true due to sequence_length
        if len(predictions) < len(y_true):
            # Use only the last part of y_true that matches predictions
            y_true = y_true[-len(predictions):]
        elif len(predictions) > len(y_true):
            # Truncate predictions (shouldn't happen but handle it)
            predictions = predictions[:len(y_true)]
        
        metrics = {
            'mse': self._calculate_mse(y_true, predictions),
            'rmse': self._calculate_rmse(y_true, predictions),
            'mae': self._calculate_mae(y_true, predictions),
            'mape': self._calculate_mape(y_true, predictions),
            'r2': self._calculate_r2(y_true, predictions)
        }
        
        return metrics
    
    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return float(np.mean((y_true - y_pred) ** 2))
    
    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: File path to save the model
        """
        model_data = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'train_history': self.train_history,
            'model_state': self._get_model_state()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path: str) -> 'TimeSeriesModel':
        """
        Load a saved model from disk.
        
        Args:
            path: File path to load the model from
            
        Returns:
            self: Loaded model instance
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.train_history = model_data['train_history']
        self._set_model_state(model_data['model_state'])
        
        return self
    
    def _get_model_state(self) -> Dict:
        """Get model state for serialization. Override in subclasses."""
        return {'model': self.model}
    
    def _set_model_state(self, state: Dict) -> None:
        """Set model state from deserialization. Override in subclasses."""
        self.model = state.get('model')
    
    def create_sequences(self, data: np.ndarray, seq_length: int, 
                        pred_length: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input data array
            seq_length: Length of input sequences
            pred_length: Length of prediction sequences
            
        Returns:
            Tuple of (X sequences, y sequences)
        """
        X, y = [], []
        for i in range(len(data) - seq_length - pred_length + 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[(i + seq_length):(i + seq_length + pred_length)])
        return np.array(X), np.array(y)
    
    def train_test_split(self, X: pd.DataFrame, y: pd.DataFrame, 
                        test_size: float = 0.2) -> Tuple:
        """
        Split time series data into train and test sets.
        
        Args:
            X: Input features
            y: Target values
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def add_lag_features(self, df: pd.DataFrame, columns: List[str], 
                        lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features to the dataframe.
        
        Args:
            df: Input dataframe
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with added lag features
        """
        df_lagged = df.copy()
        for col in columns:
            for lag in lags:
                df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df_lagged.dropna()
    
    def add_rolling_features(self, df: pd.DataFrame, columns: List[str], 
                           windows: List[int]) -> pd.DataFrame:
        """
        Add rolling statistics features.
        
        Args:
            df: Input dataframe
            columns: Columns to calculate rolling stats for
            windows: List of window sizes
            
        Returns:
            DataFrame with added rolling features
        """
        df_rolling = df.copy()
        for col in columns:
            for window in windows:
                df_rolling[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                df_rolling[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
        return df_rolling.dropna()
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set model configuration."""
        self.config.update(config)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"
