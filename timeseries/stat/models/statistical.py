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
from typing import Dict, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys

from timeseries.model import TimeSeriesModel


class ARIMAModel(TimeSeriesModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        default_config = {
            'order': (1, 1, 1),  # (p, d, q)
            'seasonal': False,
            'trend': 'c'
        }
        default_config.update(self.config)
        self.config = default_config
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'ARIMAModel':
        """
        Fit ARIMA model.
        
        Args:
            X: Input features (can be exogenous variables)
            y: Target time series
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError("statsmodels is required. Install with: pip install statsmodels")
        
        self.feature_names = X.columns.tolist() if X is not None and len(X.columns) > 0 else []
        self.target_names = y.columns.tolist()
        
        # ARIMA works with univariate series
        if len(self.target_names) > 1:
            print("Warning: ARIMA is univariate. Using first target column only.")
        
        y_series = y.iloc[:, 0]
        
        # Fit ARIMA model
        order = self.config['order']
        trend = self.config.get('trend', 'c')
        
        if X is not None and len(X.columns) > 0:
            self.model = ARIMA(y_series, exog=X, order=order, trend=trend)
        else:
            self.model = ARIMA(y_series, order=order, trend=trend)
        
        self.model_fit = self.model.fit()
        self.is_fitted = True
        
        print(f"ARIMA{order} model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """Make in-sample predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model_fit.predict(start=0, end=len(X)-1, exog=X if X is not None else None)
        return predictions.values.reshape(-1, 1)
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast future values."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.model_fit.forecast(steps=steps, exog=exog)
        return forecast.values.reshape(-1, 1)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        if not self.is_fitted:
            return {}
        return {
            'aic': self.model_fit.aic,
            'bic': self.model_fit.bic,
            'params': self.model_fit.params.to_dict()
        }


class SARIMAModel(TimeSeriesModel):
    """
    SARIMA (Seasonal ARIMA) model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        default_config = {
            'order': (1, 1, 1),  # (p, d, q)
            'seasonal_order': (1, 1, 1, 12),  # (P, D, Q, s)
            'trend': 'c'
        }
        default_config.update(self.config)
        self.config = default_config
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'SARIMAModel':
        """Fit SARIMA model."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("statsmodels is required. Install with: pip install statsmodels")
        
        self.feature_names = X.columns.tolist() if X is not None and len(X.columns) > 0 else []
        self.target_names = y.columns.tolist()
        
        if len(self.target_names) > 1:
            print("Warning: SARIMA is univariate. Using first target column only.")
        
        y_series = y.iloc[:, 0]
        
        order = self.config['order']
        seasonal_order = self.config['seasonal_order']
        trend = self.config.get('trend', 'c')
        
        if X is not None and len(X.columns) > 0:
            self.model = SARIMAX(y_series, exog=X, order=order, 
                                seasonal_order=seasonal_order, trend=trend)
        else:
            self.model = SARIMAX(y_series, order=order, 
                                seasonal_order=seasonal_order, trend=trend)
        
        self.model_fit = self.model.fit(disp=False)
        self.is_fitted = True
        
        print(f"SARIMA{order}x{seasonal_order} model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """Make in-sample predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model_fit.predict(start=0, end=len(X)-1, exog=X if X is not None else None)
        return predictions.values.reshape(-1, 1)
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast future values."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.model_fit.forecast(steps=steps, exog=exog)
        return forecast.values.reshape(-1, 1)


class VARModel(TimeSeriesModel):
    """
    VAR (Vector AutoRegression) model for multivariate time series.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        default_config = {
            'maxlags': 5,
            'ic': 'aic'  # Information criterion for lag selection
        }
        default_config.update(self.config)
        self.config = default_config
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'VARModel':
        """Fit VAR model."""
        try:
            from statsmodels.tsa.api import VAR
        except ImportError:
            raise ImportError("statsmodels is required. Install with: pip install statsmodels")
        
        self.feature_names = X.columns.tolist()
        self.target_names = y.columns.tolist()
        
        # VAR uses the target variables; X can be ignored or used differently
        self.model = VAR(y)
        
        maxlags = self.config['maxlags']
        ic = self.config['ic']
        
        self.model_fit = self.model.fit(maxlags=maxlags, ic=ic)
        self.is_fitted = True
        
        print(f"VAR model fitted with {self.model_fit.k_ar} lags")
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """Make in-sample predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # For VAR, we need the lagged values
        lag_order = self.model_fit.k_ar
        predictions = []
        
        for i in range(lag_order, len(X)):
            pred = self.model_fit.forecast(X.iloc[i-lag_order:i].values, steps=1)
            predictions.append(pred[0])
        
        return np.array(predictions)
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast future values."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get the last observations needed for forecasting
        lag_order = self.model_fit.k_ar
        last_obs = self.model.endog[-lag_order:]
        
        forecast = self.model_fit.forecast(last_obs, steps=steps)
        return forecast


class ExponentialSmoothingModel(TimeSeriesModel):
    """
    Exponential Smoothing (ETS) model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        default_config = {
            'trend': 'add',
            'seasonal': 'add',
            'seasonal_periods': 12
        }
        default_config.update(self.config)
        self.config = default_config
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'ExponentialSmoothingModel':
        """Fit Exponential Smoothing model."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            raise ImportError("statsmodels is required. Install with: pip install statsmodels")
        
        self.feature_names = X.columns.tolist() if X is not None else []
        self.target_names = y.columns.tolist()
        
        if len(self.target_names) > 1:
            print("Warning: Exponential Smoothing is univariate. Using first target column only.")
        
        y_series = y.iloc[:, 0]
        
        trend = self.config.get('trend', 'add')
        seasonal = self.config.get('seasonal', 'add')
        seasonal_periods = self.config.get('seasonal_periods', 12)
        
        self.model = ExponentialSmoothing(
            y_series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        
        self.model_fit = self.model.fit()
        self.is_fitted = True
        
        print(f"Exponential Smoothing model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """Make in-sample predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model_fit.fittedvalues
        return predictions.values.reshape(-1, 1)
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast future values."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.model_fit.forecast(steps=steps)
        return forecast.values.reshape(-1, 1)


class ProphetModel(TimeSeriesModel):
    """
    Facebook Prophet model for time series forecasting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        default_config = {
            'growth': 'linear',
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
        default_config.update(self.config)
        self.config = default_config
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'ProphetModel':
        """Fit Prophet model."""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet is required. Install with: pip install prophet")
        
        self.feature_names = X.columns.tolist() if X is not None and len(X.columns) > 0 else []
        self.target_names = y.columns.tolist()
        
        if len(self.target_names) > 1:
            print("Warning: Prophet is univariate. Using first target column only.")
        
        # Prophet expects dataframe with 'ds' (date) and 'y' (value) columns
        df = pd.DataFrame({
            'ds': X.index if hasattr(X, 'index') else range(len(X)),
            'y': y.iloc[:, 0].values
        })
        
        self.model = Prophet(
            growth=self.config['growth'],
            seasonality_mode=self.config['seasonality_mode'],
            yearly_seasonality=self.config['yearly_seasonality'],
            weekly_seasonality=self.config['weekly_seasonality'],
            daily_seasonality=self.config['daily_seasonality']
        )
        
        self.model.fit(df)
        self.is_fitted = True
        
        print(f"Prophet model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """Make in-sample predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        df = pd.DataFrame({
            'ds': X.index if hasattr(X, 'index') else range(len(X))
        })
        
        forecast = self.model.predict(df)
        return forecast['yhat'].values.reshape(-1, 1)
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast future values."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        
        # Return only the forecasted values (last 'steps' rows)
        return forecast['yhat'].iloc[-steps:].values.reshape(-1, 1)
