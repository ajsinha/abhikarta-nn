"""
Dynamic Factor Models
=====================

Implementation of Dynamic Factor Models (DFM) for dimension reduction
and forecasting of large-scale time series.

Author: Time Series Prediction Team
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings
warnings.filterwarnings('ignore')

from timeseries_prediction_stat.base.base_model import StatisticalTimeSeriesModel, MultiOutputMixin


class DynamicFactorModel(StatisticalTimeSeriesModel, MultiOutputMixin):
    """
    Dynamic Factor Model (DFM).
    
    Extracts common latent factors driving multiple time series.
    Reduces dimensionality while capturing co-movements.
    
    Model:
        y_t = Λ f_t + ε_t
        f_t = A f_{t-1} + η_t
    
    Where:
        - y_t: observed variables
        - f_t: latent factors
        - Λ: factor loadings
        - A: factor dynamics
    
    Best for:
    - Many time series (30-100+)
    - Identifying common factors
    - Dimension reduction
    - Economic/financial forecasting
    
    Example:
        >>> model = DynamicFactorModel(k_factors=3, factor_order=2)
        >>> model.fit(data)  # data with 30 variables
        >>> forecast = model.forecast(steps=10)
        >>> factors = model.get_factors()  # Extract 3 common factors
    """
    
    def __init__(self,
                 k_factors: int = 1,
                 factor_order: int = 2,
                 name: str = "DynamicFactorModel"):
        """
        Initialize DFM.
        
        Args:
            k_factors: Number of latent factors
            factor_order: AR order of factor dynamics
        """
        super().__init__(name)
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.model = None
        self.factors = None
    
    def fit(self,
            data: Union[pd.DataFrame, np.ndarray],
            **kwargs):
        """
        Fit Dynamic Factor Model.
        
        Args:
            data: Time series data (n_obs, n_variables)
                  n_variables should be much larger than k_factors
            **kwargs: Additional arguments for DynamicFactor
            
        Returns:
            self
        """
        try:
            from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        except ImportError:
            raise ImportError("Please install statsmodels: pip install statsmodels")
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
        
        self.data = data
        self.variable_names = list(data.columns)
        
        # Create and fit model
        self.model = DynamicFactor(
            data,
            k_factors=self.k_factors,
            factor_order=self.factor_order,
            **kwargs
        )
        
        # Fit with optimization
        self.results = self.model.fit(disp=False, maxiter=1000)
        
        # Extract factors
        self.factors = pd.DataFrame(
            self.results.factors.filtered,
            index=data.index,
            columns=[f'factor_{i+1}' for i in range(self.k_factors)]
        )
        
        # Store diagnostics
        self.diagnostics['aic'] = self.results.aic
        self.diagnostics['bic'] = self.results.bic
        self.diagnostics['hqic'] = self.results.hqic
        self.diagnostics['log_likelihood'] = self.results.llf
        self.diagnostics['residuals'] = pd.DataFrame(
            self.results.resid,
            columns=self.variable_names,
            index=data.index
        )
        
        self.is_fitted = True
        
        return self
    
    def forecast(self, steps: int, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps ahead
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with forecasts for all variables
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Generate forecast
        forecast_result = self.results.forecast(steps=steps, **kwargs)
        
        # Create DataFrame
        forecast_df = pd.DataFrame(
            forecast_result,
            columns=self.variable_names
        )
        
        self.forecast_history.append(forecast_df)
        
        return forecast_df
    
    def get_factors(self) -> pd.DataFrame:
        """
        Get extracted latent factors.
        
        Returns:
            DataFrame with factor values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.factors
    
    def get_factor_loadings(self) -> pd.DataFrame:
        """
        Get factor loadings (Λ matrix).
        
        Returns:
            DataFrame showing how each variable loads on each factor
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Extract loadings from model parameters
        loadings = self.results.params['loading'].values.reshape(
            len(self.variable_names), self.k_factors
        )
        
        return pd.DataFrame(
            loadings,
            index=self.variable_names,
            columns=[f'factor_{i+1}' for i in range(self.k_factors)]
        )
    
    def explained_variance_ratio(self) -> np.ndarray:
        """
        Calculate variance explained by each factor.
        
        Returns:
            Array of variance ratios
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Calculate variance of each factor
        factor_vars = np.var(self.factors.values, axis=0)
        total_var = np.sum(factor_vars)
        
        return factor_vars / total_var


class PCABasedForecaster(StatisticalTimeSeriesModel, MultiOutputMixin):
    """
    PCA-based Forecasting Model.
    
    Simpler alternative to DFM using PCA for dimension reduction,
    then forecasting principal components and transforming back.
    
    Best for:
    - Quick dimension reduction
    - When DFM is too complex
    - Exploratory analysis
    
    Example:
        >>> model = PCABasedForecaster(n_components=5, ar_order=3)
        >>> model.fit(data)
        >>> forecast = model.forecast(steps=10)
    """
    
    def __init__(self,
                 n_components: int = 3,
                 ar_order: int = 2,
                 name: str = "PCAForecaster"):
        """
        Initialize PCA-based forecaster.
        
        Args:
            n_components: Number of principal components
            ar_order: AR order for forecasting components
        """
        super().__init__(name)
        self.n_components = n_components
        self.ar_order = ar_order
        self.pca = None
        self.ar_models = []
        self.components = None
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit PCA and AR models.
        
        Args:
            data: Time series data
            **kwargs: Additional arguments
            
        Returns:
            self
        """
        from sklearn.decomposition import PCA
        from statsmodels.tsa.ar_model import AutoReg
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
        
        self.data = data
        self.variable_names = list(data.columns)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.components = self.pca.fit_transform(data.values)
        
        # Fit AR model for each component
        self.ar_models = []
        for i in range(self.n_components):
            ar_model = AutoReg(self.components[:, i], lags=self.ar_order)
            ar_result = ar_model.fit()
            self.ar_models.append(ar_result)
        
        # Store diagnostics
        self.diagnostics['variance_explained'] = self.pca.explained_variance_ratio_
        
        self.is_fitted = True
        
        return self
    
    def forecast(self, steps: int, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps ahead
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Forecast each component
        component_forecasts = []
        for ar_result in self.ar_models:
            fc = ar_result.forecast(steps=steps)
            component_forecasts.append(fc)
        
        component_forecasts = np.array(component_forecasts).T  # (steps, n_components)
        
        # Transform back to original space
        forecast_values = self.pca.inverse_transform(component_forecasts)
        
        # Create DataFrame
        forecast_df = pd.DataFrame(
            forecast_values,
            columns=self.variable_names
        )
        
        self.forecast_history.append(forecast_df)
        
        return forecast_df
    
    def get_components(self) -> pd.DataFrame:
        """
        Get principal components.
        
        Returns:
            DataFrame with component values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return pd.DataFrame(
            self.components,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
    
    def get_loadings(self) -> pd.DataFrame:
        """
        Get PCA loadings.
        
        Returns:
            DataFrame showing variable loadings on components
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return pd.DataFrame(
            self.pca.components_.T,
            index=self.variable_names,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
