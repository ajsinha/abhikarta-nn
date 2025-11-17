"""
Linear Regression Models for Time Series
=========================================

Various linear regression approaches for time series prediction:
- Multivariate Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Time Series Regression with lags

Author: Time Series Prediction Team
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from timeseries_prediction_stat.base.base_model import StatisticalTimeSeriesModel, MultiOutputMixin


class MultiOutputLinearRegression(StatisticalTimeSeriesModel, MultiOutputMixin):
    """
    Multi-Output Linear Regression for Time Series.
    
    Predicts multiple output variables using lagged features.
    Simple but effective baseline model.
    
    Best for:
    - Linear relationships
    - Baseline comparisons
    - Quick prototyping
    - When interpretability is key
    
    Example:
        >>> model = MultiOutputLinearRegression(lags=5)
        >>> model.fit(X, y)
        >>> forecast = model.forecast(steps=10)
    """
    
    def __init__(self, 
                 lags: int = 1,
                 normalize: bool = True,
                 fit_intercept: bool = True,
                 name: str = "LinearRegression"):
        """
        Initialize model.
        
        Args:
            lags: Number of lagged observations to use as features
            normalize: Whether to normalize features
            fit_intercept: Whether to fit intercept
        """
        super().__init__(name)
        self.lags = lags
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.scaler = StandardScaler() if normalize else None
        self.last_values = None
    
    def _create_lagged_features(self, data: np.ndarray) -> tuple:
        """Create lagged feature matrix."""
        n_samples, n_features = data.shape
        n_obs = n_samples - self.lags
        
        X = np.zeros((n_obs, self.lags * n_features))
        y = data[self.lags:]
        
        for i in range(n_obs):
            # Stack lagged observations
            lagged = data[i:i + self.lags].flatten()
            X[i] = lagged
        
        return X, y
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit linear regression model.
        
        Args:
            data: Time series data (n_obs, n_variables)
            **kwargs: Additional arguments
            
        Returns:
            self
        """
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
        
        self.data = data
        self.variable_names = list(data.columns)
        
        # Create lagged features
        X, y = self._create_lagged_features(data.values)
        
        # Normalize if requested
        if self.normalize:
            X = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X, y)
        
        # Store last values for forecasting
        self.last_values = data.values[-self.lags:].copy()
        
        # Calculate diagnostics
        predictions = self.model.predict(X)
        residuals = y - predictions
        
        self.diagnostics['residuals'] = pd.DataFrame(
            residuals,
            columns=self.variable_names
        )
        
        # Calculate R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        self.diagnostics['r2'] = np.mean(r2)
        
        self.is_fitted = True
        
        return self
    
    def forecast(self, steps: int, **kwargs) -> pd.DataFrame:
        """
        Generate multi-step ahead forecasts.
        
        Args:
            steps: Number of steps to forecast
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        forecasts = []
        current_values = self.last_values.copy()
        
        for _ in range(steps):
            # Create feature vector from last 'lags' observations
            X_forecast = current_values.flatten().reshape(1, -1)
            
            # Normalize if needed
            if self.normalize:
                X_forecast = self.scaler.transform(X_forecast)
            
            # Predict next step
            y_pred = self.model.predict(X_forecast)[0]
            forecasts.append(y_pred)
            
            # Update current_values (shift and add new prediction)
            current_values = np.vstack([current_values[1:], y_pred])
        
        # Create DataFrame
        forecast_df = pd.DataFrame(
            forecasts,
            columns=self.variable_names
        )
        
        self.forecast_history.append(forecast_df)
        
        return forecast_df


class RidgeTimeSeriesRegression(MultiOutputLinearRegression):
    """
    Ridge Regression (L2 regularization) for Time Series.
    
    Adds L2 penalty to prevent overfitting.
    Good when features are correlated.
    
    Example:
        >>> model = RidgeTimeSeriesRegression(lags=10, alpha=1.0)
        >>> model.fit(data)
        >>> forecast = model.forecast(steps=5)
    """
    
    def __init__(self, 
                 lags: int = 1,
                 alpha: float = 1.0,
                 normalize: bool = True,
                 name: str = "RidgeRegression"):
        """
        Initialize Ridge regression.
        
        Args:
            lags: Number of lags
            alpha: Regularization strength (higher = more regularization)
            normalize: Whether to normalize features
        """
        super().__init__(lags, normalize, True, name)
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)


class LassoTimeSeriesRegression(MultiOutputLinearRegression):
    """
    Lasso Regression (L1 regularization) for Time Series.
    
    Performs feature selection by driving some coefficients to zero.
    Useful for identifying most important lags.
    
    Example:
        >>> model = LassoTimeSeriesRegression(lags=20, alpha=0.1)
        >>> model.fit(data)
        >>> important_lags = model.get_nonzero_coefficients()
    """
    
    def __init__(self,
                 lags: int = 1,
                 alpha: float = 1.0,
                 normalize: bool = True,
                 name: str = "LassoRegression"):
        """
        Initialize Lasso regression.
        
        Args:
            lags: Number of lags
            alpha: Regularization strength
            normalize: Whether to normalize features
        """
        super().__init__(lags, normalize, True, name)
        self.alpha = alpha
        self.model = Lasso(alpha=alpha, fit_intercept=True, max_iter=2000)
    
    def get_nonzero_coefficients(self) -> Dict[str, List[int]]:
        """
        Get indices of non-zero coefficients (selected features).
        
        Returns:
            Dictionary mapping variable names to selected lag indices
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        coef = self.model.coef_  # Shape: (n_outputs, n_features)
        n_vars = len(self.variable_names)
        
        selected = {}
        for i, var in enumerate(self.variable_names):
            var_coef = coef[i]
            nonzero_idx = np.where(np.abs(var_coef) > 1e-10)[0]
            
            # Convert flat indices to (variable, lag) pairs
            selected_lags = [(idx // n_vars, idx % n_vars) for idx in nonzero_idx]
            selected[var] = selected_lags
        
        return selected


class ElasticNetTimeSeriesRegression(MultiOutputLinearRegression):
    """
    Elastic Net (L1 + L2) Regression for Time Series.
    
    Combines Ridge and Lasso regularization.
    Balances feature selection and coefficient shrinkage.
    
    Example:
        >>> model = ElasticNetTimeSeriesRegression(lags=15, alpha=0.5, l1_ratio=0.5)
        >>> model.fit(data)
    """
    
    def __init__(self,
                 lags: int = 1,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 normalize: bool = True,
                 name: str = "ElasticNet"):
        """
        Initialize Elastic Net regression.
        
        Args:
            lags: Number of lags
            alpha: Overall regularization strength
            l1_ratio: L1 vs L2 balance (0=Ridge, 1=Lasso)
            normalize: Whether to normalize features
        """
        super().__init__(lags, normalize, True, name)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=2000
        )


class BayesianLinearRegression(MultiOutputLinearRegression):
    """
    Bayesian Ridge Regression for Time Series.
    
    Provides probabilistic predictions with uncertainty estimates.
    
    Example:
        >>> model = BayesianLinearRegression(lags=5)
        >>> model.fit(data)
        >>> forecast, std = model.forecast_with_uncertainty(steps=10)
    """
    
    def __init__(self,
                 lags: int = 1,
                 normalize: bool = True,
                 alpha_1: float = 1e-6,
                 alpha_2: float = 1e-6,
                 lambda_1: float = 1e-6,
                 lambda_2: float = 1e-6,
                 name: str = "BayesianRidge"):
        """
        Initialize Bayesian Ridge regression.
        
        Args:
            lags: Number of lags
            normalize: Whether to normalize
            alpha_1, alpha_2: Gamma prior parameters for alpha
            lambda_1, lambda_2: Gamma prior parameters for lambda
        """
        super().__init__(lags, normalize, True, name)
        self.model = BayesianRidge(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            fit_intercept=True
        )
    
    def forecast_with_uncertainty(self, steps: int) -> tuple:
        """
        Generate forecasts with uncertainty estimates.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast_df, std_df)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        forecasts = []
        stds = []
        current_values = self.last_values.copy()
        
        for _ in range(steps):
            # Create feature vector
            X_forecast = current_values.flatten().reshape(1, -1)
            
            # Normalize if needed
            if self.normalize:
                X_forecast = self.scaler.transform(X_forecast)
            
            # Predict with uncertainty
            y_pred, y_std = self.model.predict(X_forecast, return_std=True)
            forecasts.append(y_pred[0])
            stds.append(y_std)
            
            # Update
            current_values = np.vstack([current_values[1:], y_pred[0]])
        
        # Create DataFrames
        forecast_df = pd.DataFrame(forecasts, columns=self.variable_names)
        std_df = pd.DataFrame(stds, columns=[f'{v}_std' for v in self.variable_names])
        
        return forecast_df, std_df


class RobustLinearRegression(MultiOutputLinearRegression):
    """
    Robust Regression (Huber) for Time Series.
    
    Less sensitive to outliers than ordinary least squares.
    Uses Huber loss function.
    
    Example:
        >>> model = RobustLinearRegression(lags=5, epsilon=1.35)
        >>> model.fit(data)
    """
    
    def __init__(self,
                 lags: int = 1,
                 epsilon: float = 1.35,
                 normalize: bool = True,
                 name: str = "RobustRegression"):
        """
        Initialize Robust regression.
        
        Args:
            lags: Number of lags
            epsilon: Huber loss parameter (smaller = more robust)
            normalize: Whether to normalize
        """
        super().__init__(lags, normalize, True, name)
        self.epsilon = epsilon
        self.model = HuberRegressor(epsilon=epsilon, fit_intercept=True)
