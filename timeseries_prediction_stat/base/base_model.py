"""
Base Statistical Model Module
==============================

Abstract base class for all statistical time series models.
Provides unified interface for fitting, forecasting, and evaluation.

Author: Time Series Prediction Team
License: MIT
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import pickle
from pathlib import Path


class StatisticalTimeSeriesModel(ABC):
    """
    Abstract base class for statistical time series models.
    
    All statistical models must implement:
    - fit(): Fit model to data
    - forecast(): Generate forecasts
    - _build_model(): Model-specific setup
    
    Supports multi-output prediction natively for all models.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize base statistical model.
        
        Args:
            name: Model name for identification
        """
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        self.results = None
        self.data = None
        self.variable_names = None
        
        # Diagnostics storage
        self.diagnostics = {
            'aic': None,
            'bic': None,
            'hqic': None,
            'log_likelihood': None,
            'residuals': None
        }
        
        # Forecast history
        self.forecast_history = []
    
    @abstractmethod
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the model to data.
        
        Args:
            data: Time series data (DataFrame or array)
                  For multi-output: shape (n_obs, n_variables)
            **kwargs: Model-specific parameters
            
        Returns:
            self (fitted model)
        """
        pass
    
    @abstractmethod
    def forecast(self, steps: int, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps ahead to forecast
            **kwargs: Model-specific parameters
            
        Returns:
            DataFrame with forecasts for all variables
        """
        pass
    
    def predict(self, steps: int, **kwargs) -> pd.DataFrame:
        """
        Alias for forecast (consistent with sklearn API).
        
        Args:
            steps: Number of steps ahead to predict
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with predictions
        """
        return self.forecast(steps, **kwargs)
    
    def evaluate(self, 
                 actual: Union[pd.DataFrame, np.ndarray],
                 forecast: Union[pd.DataFrame, np.ndarray] = None,
                 metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate forecast accuracy.
        
        Args:
            actual: Actual values
            forecast: Forecasted values (if None, uses last forecast)
            metrics: List of metrics to compute
                    ['rmse', 'mae', 'mape', 'mse', 'r2']
            
        Returns:
            Dictionary of metric values
        """
        if forecast is None:
            if not self.forecast_history:
                raise ValueError("No forecasts available. Run forecast() first.")
            forecast = self.forecast_history[-1]
        
        # Convert to numpy arrays
        if isinstance(actual, pd.DataFrame):
            actual = actual.values
        if isinstance(forecast, pd.DataFrame):
            forecast = forecast.values
        
        # Ensure same shape
        min_len = min(len(actual), len(forecast))
        actual = actual[:min_len]
        forecast = forecast[:min_len]
        
        if metrics is None:
            metrics = ['rmse', 'mae', 'mape', 'r2']
        
        results = {}
        
        if 'mse' in metrics:
            results['mse'] = np.mean((actual - forecast) ** 2)
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(np.mean((actual - forecast) ** 2))
        
        if 'mae' in metrics:
            results['mae'] = np.mean(np.abs(actual - forecast))
        
        if 'mape' in metrics:
            # Avoid division by zero
            mask = actual != 0
            mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
            results['mape'] = mape
        
        if 'r2' in metrics:
            ss_res = np.sum((actual - forecast) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            results['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        
        return results
    
    def get_diagnostics(self) -> Dict:
        """
        Get model diagnostics (AIC, BIC, etc.).
        
        Returns:
            Dictionary of diagnostic values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.diagnostics.copy()
    
    def get_residuals(self) -> Union[pd.DataFrame, np.ndarray]:
        """
        Get model residuals.
        
        Returns:
            Residuals from fitted model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.diagnostics.get('residuals')
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'name': self.name,
            'results': self.results,
            'diagnostics': self.diagnostics,
            'variable_names': self.variable_names,
            'is_fitted': self.is_fitted,
            'model_class': self.__class__.__name__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Create instance
        model = cls()
        model.name = save_dict['name']
        model.results = save_dict['results']
        model.diagnostics = save_dict['diagnostics']
        model.variable_names = save_dict['variable_names']
        model.is_fitted = save_dict['is_fitted']
        
        return model
    
    def summary(self):
        """Print model summary."""
        print(f"\n{'=' * 70}")
        print(f"{self.name} Summary")
        print(f"{'=' * 70}")
        
        if self.is_fitted:
            print(f"Status: Fitted")
            if self.variable_names is not None:
                print(f"Variables: {', '.join(self.variable_names)}")
                print(f"Number of variables: {len(self.variable_names)}")
            
            if self.diagnostics['aic'] is not None:
                print(f"\nInformation Criteria:")
                print(f"  AIC:  {self.diagnostics['aic']:.4f}")
                print(f"  BIC:  {self.diagnostics['bic']:.4f}")
                print(f"  HQIC: {self.diagnostics['hqic']:.4f}")
            
            if self.diagnostics['log_likelihood'] is not None:
                print(f"\nLog Likelihood: {self.diagnostics['log_likelihood']:.4f}")
        else:
            print(f"Status: Not fitted")
        
        print(f"{'=' * 70}\n")
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name}({status})"


class MultiOutputMixin:
    """
    Mixin for handling multi-output predictions and evaluation.
    """
    
    def evaluate_per_variable(self,
                             actual: pd.DataFrame,
                             forecast: pd.DataFrame,
                             metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate each output variable separately.
        
        Args:
            actual: Actual values (DataFrame)
            forecast: Forecasted values (DataFrame)
            metrics: Metrics to compute
            
        Returns:
            Dictionary mapping variable names to metric dictionaries
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'mape', 'r2']
        
        results = {}
        
        for col in actual.columns:
            if col in forecast.columns:
                col_results = {}
                
                y_true = actual[col].values
                y_pred = forecast[col].values
                
                # Align lengths
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                
                if 'rmse' in metrics:
                    col_results['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                if 'mae' in metrics:
                    col_results['mae'] = np.mean(np.abs(y_true - y_pred))
                
                if 'mape' in metrics:
                    mask = y_true != 0
                    if mask.any():
                        col_results['mape'] = np.mean(
                            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
                        ) * 100
                
                if 'r2' in metrics:
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    col_results['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
                
                results[col] = col_results
        
        return results
