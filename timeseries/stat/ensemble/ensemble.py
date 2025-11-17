"""
Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com

Legal Notice: This module and the associated software architecture are proprietary 
and confidential. Unauthorized copying, distribution, modification, or use is 
strictly prohibited without explicit written permission from the copyright holder.

Patent Pending: Certain architectural patterns and implementations described in 
this module may be subject to patent applications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import sys
sys.path.append('/home/claude/timeseries_package')
from timeseries.model import TimeSeriesModel


class StatisticalEnsemble(TimeSeriesModel):
    """
    Ensemble model combining multiple statistical models.
    """
    
    def __init__(self, models: List[TimeSeriesModel], 
                 method: str = 'average',
                 weights: Optional[List[float]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize statistical ensemble.
        
        Args:
            models: List of statistical TimeSeriesModel instances
            method: 'average', 'weighted', or 'median'
            weights: Weights for each model
            config: Additional configuration
        """
        super().__init__(config)
        self.models = models
        self.method = method
        self.weights = weights
        
        if self.method == 'weighted' and self.weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        
        if self.weights and len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'StatisticalEnsemble':
        """Fit all statistical models in the ensemble."""
        self.feature_names = X.columns.tolist()
        self.target_names = y.columns.tolist()
        
        print(f"Training statistical ensemble of {len(self.models)} models...")
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}: {model.__class__.__name__}")
            try:
                model.fit(X, y, **kwargs)
            except Exception as e:
                print(f"Warning: Model {i+1} failed to fit: {str(e)}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """Make predictions using ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(X, steps)
                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Prediction failed for {model.__class__.__name__}: {str(e)}")
        
        if not predictions:
            raise ValueError("All models failed to predict")
        
        predictions = np.array(predictions)
        
        if self.method == 'average':
            return np.mean(predictions, axis=0)
        elif self.method == 'weighted':
            weights = np.array(self.weights[:len(predictions)]).reshape(-1, 1, 1)
            return np.sum(predictions * weights, axis=0)
        elif self.method == 'median':
            return np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast future values using ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        forecasts = []
        for model in self.models:
            try:
                forecast = model.forecast(steps, exog)
                forecasts.append(forecast)
            except Exception as e:
                print(f"Warning: Forecast failed for {model.__class__.__name__}: {str(e)}")
        
        if not forecasts:
            raise ValueError("All models failed to forecast")
        
        forecasts = np.array(forecasts)
        
        if self.method == 'average':
            return np.mean(forecasts, axis=0)
        elif self.method == 'weighted':
            weights = np.array(self.weights[:len(forecasts)]).reshape(-1, 1, 1)
            return np.sum(forecasts * weights, axis=0)
        elif self.method == 'median':
            return np.median(forecasts, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate ensemble and individual models."""
        results = {
            'ensemble': super().evaluate(X, y),
            'individual_models': []
        }
        
        for i, model in enumerate(self.models):
            try:
                model_metrics = model.evaluate(X, y)
                model_metrics['model_name'] = model.__class__.__name__
                model_metrics['model_index'] = i
                results['individual_models'].append(model_metrics)
            except Exception as e:
                print(f"Warning: Evaluation failed for model {i+1}: {str(e)}")
        
        return results
