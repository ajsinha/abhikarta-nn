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

from timeseries.model import TimeSeriesModel


class EnsembleModel(TimeSeriesModel):
    """
    Ensemble model that combines multiple time series models.
    Supports averaging, weighted averaging, and stacking.
    """
    
    def __init__(self, models: List[TimeSeriesModel], 
                 method: str = 'average',
                 weights: Optional[List[float]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize ensemble model.
        
        Args:
            models: List of TimeSeriesModel instances
            method: 'average', 'weighted', or 'voting'
            weights: Weights for each model (used with 'weighted' method)
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
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'EnsembleModel':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Input features
            y: Target values
            **kwargs: Additional fitting parameters
        """
        self.feature_names = X.columns.tolist()
        self.target_names = y.columns.tolist()
        
        print(f"Training ensemble of {len(self.models)} models...")
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}: {model.__class__.__name__}")
            model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """
        Make predictions using ensemble.
        
        Args:
            X: Input features
            steps: Number of steps ahead to predict
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get predictions from all models
        predictions = np.array([model.predict(X, steps) for model in self.models])
        
        if self.method == 'average':
            return np.mean(predictions, axis=0)
        elif self.method == 'weighted':
            weights = np.array(self.weights).reshape(-1, 1, 1)
            return np.sum(predictions * weights, axis=0)
        elif self.method == 'voting':
            # For regression, voting is similar to averaging
            return np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Forecast future values using ensemble.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables
            
        Returns:
            Ensemble forecast
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        forecasts = np.array([model.forecast(steps, exog) for model in self.models])
        
        if self.method == 'average':
            return np.mean(forecasts, axis=0)
        elif self.method == 'weighted':
            weights = np.array(self.weights).reshape(-1, 1, 1)
            return np.sum(forecasts * weights, axis=0)
        elif self.method == 'voting':
            return np.median(forecasts, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate ensemble and individual models.
        
        Args:
            X: Input features
            y: True target values
            
        Returns:
            Dictionary with ensemble and individual model metrics
        """
        results = {
            'ensemble': super().evaluate(X, y),
            'individual_models': []
        }
        
        for i, model in enumerate(self.models):
            model_metrics = model.evaluate(X, y)
            model_metrics['model_name'] = model.__class__.__name__
            model_metrics['model_index'] = i
            results['individual_models'].append(model_metrics)
        
        return results
    
    def get_model_weights(self) -> Optional[List[float]]:
        """Get the weights assigned to each model."""
        return self.weights
    
    def set_model_weights(self, weights: List[float]) -> None:
        """Set new weights for the models."""
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        self.weights = weights
        self.method = 'weighted'


class StackingEnsemble(TimeSeriesModel):
    """
    Stacking ensemble that uses a meta-model to combine base models.
    """
    
    def __init__(self, base_models: List[TimeSeriesModel], 
                 meta_model: TimeSeriesModel,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base TimeSeriesModel instances
            meta_model: Meta-model to combine base model predictions
            config: Additional configuration
        """
        super().__init__(config)
        self.base_models = base_models
        self.meta_model = meta_model
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'StackingEnsemble':
        """
        Fit stacking ensemble.
        
        Args:
            X: Input features
            y: Target values
            **kwargs: Additional fitting parameters
        """
        self.feature_names = X.columns.tolist()
        self.target_names = y.columns.tolist()
        
        print(f"Training {len(self.base_models)} base models...")
        
        # Train base models
        base_predictions = []
        for i, model in enumerate(self.base_models):
            print(f"\nTraining base model {i+1}/{len(self.base_models)}: {model.__class__.__name__}")
            model.fit(X, y, **kwargs)
            preds = model.predict(X)
            base_predictions.append(preds)
        
        # Create meta-features from base model predictions
        meta_features = np.concatenate(base_predictions, axis=1)
        meta_X = pd.DataFrame(meta_features, 
                             columns=[f'base_{i}_{j}' for i in range(len(self.base_models)) 
                                    for j in range(base_predictions[0].shape[1])])
        
        # Train meta-model
        print(f"\nTraining meta-model: {self.meta_model.__class__.__name__}")
        self.meta_model.fit(meta_X, y, **kwargs)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """Make predictions using stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Stacking ensemble not fitted. Call fit() first.")
        
        # Get predictions from base models
        base_predictions = [model.predict(X, steps) for model in self.base_models]
        
        # Create meta-features
        meta_features = np.concatenate(base_predictions, axis=1)
        meta_X = pd.DataFrame(meta_features,
                             columns=[f'base_{i}_{j}' for i in range(len(self.base_models))
                                    for j in range(base_predictions[0].shape[1])])
        
        # Get final prediction from meta-model
        return self.meta_model.predict(meta_X, steps)
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast using stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Stacking ensemble not fitted. Call fit() first.")
        
        # Get forecasts from base models
        base_forecasts = [model.forecast(steps, exog) for model in self.base_models]
        
        # Create meta-features
        meta_features = np.concatenate(base_forecasts, axis=1)
        meta_X = pd.DataFrame(meta_features,
                             columns=[f'base_{i}_{j}' for i in range(len(self.base_models))
                                    for j in range(base_forecasts[0].shape[1])])
        
        # Get final forecast from meta-model
        return self.meta_model.forecast(steps, exog)


class BaggingEnsemble(TimeSeriesModel):
    """
    Bagging ensemble for time series models.
    """
    
    def __init__(self, base_model_class, n_models: int = 10,
                 sample_ratio: float = 0.8,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize bagging ensemble.
        
        Args:
            base_model_class: Class of the base model to use
            n_models: Number of models in the ensemble
            sample_ratio: Ratio of data to sample for each model
            config: Configuration for base models
        """
        super().__init__(config)
        self.base_model_class = base_model_class
        self.n_models = n_models
        self.sample_ratio = sample_ratio
        self.models = []
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'BaggingEnsemble':
        """Fit bagging ensemble."""
        self.feature_names = X.columns.tolist()
        self.target_names = y.columns.tolist()
        
        print(f"Training bagging ensemble with {self.n_models} models...")
        
        n_samples = int(len(X) * self.sample_ratio)
        
        for i in range(self.n_models):
            print(f"\nTraining model {i+1}/{self.n_models}")
            
            # Bootstrap sample
            indices = np.random.choice(len(X), size=n_samples, replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            
            # Create and train model
            model = self.base_model_class(config=self.config)
            model.fit(X_sample, y_sample, **kwargs)
            self.models.append(model)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """Make predictions using bagging ensemble."""
        if not self.is_fitted:
            raise ValueError("Bagging ensemble not fitted. Call fit() first.")
        
        predictions = np.array([model.predict(X, steps) for model in self.models])
        return np.mean(predictions, axis=0)
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast using bagging ensemble."""
        if not self.is_fitted:
            raise ValueError("Bagging ensemble not fitted. Call fit() first.")
        
        forecasts = np.array([model.forecast(steps, exog) for model in self.models])
        return np.mean(forecasts, axis=0)
