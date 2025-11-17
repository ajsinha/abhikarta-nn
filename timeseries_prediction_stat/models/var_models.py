"""
Vector Autoregression Models
=============================

Implementations of VAR, VECM, and VARMA models for multi-output time series prediction.

Models:
- VAR: Vector Autoregression
- VECM: Vector Error Correction Model
- VARMA: Vector ARMA (experimental)

Author: Time Series Prediction Team
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

from timeseries_prediction_stat.base.base_model import StatisticalTimeSeriesModel, MultiOutputMixin


class VARModel(StatisticalTimeSeriesModel, MultiOutputMixin):
    """
    Vector Autoregression (VAR) Model.
    
    Models multiple time series jointly, capturing interdependencies.
    Each variable is a linear function of past values of itself and all other variables.
    
    Best for:
    - 2-10 related time series
    - Understanding Granger causality
    - Stationary time series
    - 50-200 observations
    
    Example:
        >>> model = VARModel()
        >>> model.fit(data, maxlags=5)
        >>> forecast = model.forecast(steps=10)
    """
    
    def __init__(self, name: str = "VAR"):
        """Initialize VAR model."""
        super().__init__(name)
        self.model = None
        self.maxlags = None
    
    def fit(self, 
            data: Union[pd.DataFrame, np.ndarray],
            maxlags: int = None,
            ic: str = 'aic',
            trend: str = 'c',
            **kwargs):
        """
        Fit VAR model to data.
        
        Args:
            data: Time series data (n_obs, n_variables)
            maxlags: Maximum lag order (if None, automatically selected)
            ic: Information criterion for lag selection ('aic', 'bic', 'hqic')
            trend: Trend term ('n': no trend, 'c': constant, 'ct': constant+trend)
            **kwargs: Additional arguments for VAR model
            
        Returns:
            self (fitted model)
        """
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
        except ImportError:
            raise ImportError("Please install statsmodels: pip install statsmodels")
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
        
        self.data = data
        self.variable_names = list(data.columns)
        
        # Create VAR model
        self.model = VAR(data)
        
        # Fit model
        self.results = self.model.fit(maxlags=maxlags, ic=ic, trend=trend, **kwargs)
        self.maxlags = self.results.k_ar
        
        # Store diagnostics
        self.diagnostics['aic'] = self.results.aic
        self.diagnostics['bic'] = self.results.bic
        self.diagnostics['hqic'] = self.results.hqic
        self.diagnostics['log_likelihood'] = self.results.llf
        self.diagnostics['residuals'] = pd.DataFrame(
            self.results.resid,
            columns=self.variable_names,
            index=data.index[self.maxlags:]
        )
        
        self.is_fitted = True
        
        return self
    
    def forecast(self, 
                 steps: int,
                 exog_future: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps ahead to forecast
            exog_future: Future exogenous variables (if any)
            
        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get last observations for forecasting
        last_obs = self.data.values[-self.maxlags:]
        
        # Generate forecast
        forecast_values = self.results.forecast(last_obs, steps=steps, exog_future=exog_future)
        
        # Create DataFrame
        forecast_df = pd.DataFrame(
            forecast_values,
            columns=self.variable_names
        )
        
        # Store forecast
        self.forecast_history.append(forecast_df)
        
        return forecast_df
    
    def granger_causality(self, 
                         caused: str,
                         causing: str,
                         maxlag: Optional[int] = None,
                         signif: float = 0.05) -> Dict:
        """
        Test Granger causality between variables.
        
        Args:
            caused: Dependent variable
            causing: Independent variable  
            maxlag: Maximum lag to test
            signif: Significance level
            
        Returns:
            Dictionary with test results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Prepare data
        test_data = self.data[[caused, causing]]
        
        # Run test
        maxlag = maxlag or self.maxlags
        test_results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
        
        # Extract results
        results = {}
        for lag in range(1, maxlag + 1):
            p_value = test_results[lag][0]['ssr_ftest'][1]
            results[f'lag_{lag}'] = {
                'p_value': p_value,
                'significant': p_value < signif
            }
        
        return results
    
    def impulse_response(self, 
                        periods: int = 10,
                        impulse: Optional[str] = None,
                        response: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate Impulse Response Function (IRF).
        
        Args:
            periods: Number of periods for IRF
            impulse: Variable receiving the shock (None = all)
            response: Variable responding (None = all)
            
        Returns:
            DataFrame with IRF values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        irf = self.results.irf(periods)
        
        if impulse and response:
            # Specific impulse-response pair
            impulse_idx = self.variable_names.index(impulse)
            response_idx = self.variable_names.index(response)
            irf_values = irf.irfs[:, response_idx, impulse_idx]
            
            return pd.DataFrame({
                'period': range(periods),
                'irf': irf_values
            })
        else:
            # All impulse-response combinations
            return pd.DataFrame(
                irf.irfs.reshape(periods, -1),
                columns=[f'{resp}_from_{imp}' 
                        for imp in self.variable_names 
                        for resp in self.variable_names]
            )


class VECMModel(StatisticalTimeSeriesModel, MultiOutputMixin):
    """
    Vector Error Correction Model (VECM).
    
    For cointegrated time series with long-run equilibrium relationships.
    Useful when variables have common stochastic trends.
    
    Best for:
    - Cointegrated time series
    - Long-run equilibrium relationships
    - Financial pairs trading
    - Non-stationary but mean-reverting series
    
    Example:
        >>> model = VECMModel()
        >>> model.fit(data, k_ar_diff=2, coint_rank=1)
        >>> forecast = model.forecast(steps=10)
    """
    
    def __init__(self, name: str = "VECM"):
        """Initialize VECM model."""
        super().__init__(name)
        self.model = None
        self.k_ar_diff = None
        self.coint_rank = None
    
    def fit(self,
            data: Union[pd.DataFrame, np.ndarray],
            coint_rank: int = 1,
            k_ar_diff: int = 1,
            deterministic: str = 'ci',
            **kwargs):
        """
        Fit VECM model to data.
        
        Args:
            data: Time series data (n_obs, n_variables)
            coint_rank: Cointegration rank (number of cointegrating relationships)
            k_ar_diff: Lag order for differenced variables
            deterministic: Deterministic terms ('n', 'co', 'ci', 'lo', 'li')
                          ci: constant inside cointegration relation
                          co: constant outside
            **kwargs: Additional arguments
            
        Returns:
            self (fitted model)
        """
        try:
            from statsmodels.tsa.vector_ar.vecm import VECM
        except ImportError:
            raise ImportError("Please install statsmodels: pip install statsmodels")
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
        
        self.data = data
        self.variable_names = list(data.columns)
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        
        # Create VECM model
        self.model = VECM(
            data,
            k_ar_diff=k_ar_diff,
            coint_rank=coint_rank,
            deterministic=deterministic,
            **kwargs
        )
        
        # Fit model
        self.results = self.model.fit()
        
        # Store diagnostics
        self.diagnostics['aic'] = self.results.aic
        self.diagnostics['bic'] = self.results.bic
        self.diagnostics['hqic'] = self.results.hqic
        self.diagnostics['log_likelihood'] = self.results.llf
        self.diagnostics['residuals'] = pd.DataFrame(
            self.results.resid,
            columns=self.variable_names
        )
        
        self.is_fitted = True
        
        return self
    
    def forecast(self, steps: int, exog_fc: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps ahead to forecast
            exog_fc: Future exogenous variables (if any)
            
        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Generate forecast
        forecast_values = self.results.predict(steps=steps, exog_fc=exog_fc)
        
        # Create DataFrame
        forecast_df = pd.DataFrame(
            forecast_values,
            columns=self.variable_names
        )
        
        # Store forecast
        self.forecast_history.append(forecast_df)
        
        return forecast_df
    
    def cointegration_test(self) -> Dict:
        """
        Test for cointegration (Johansen test).
        
        Returns:
            Dictionary with test statistics
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        
        # Run Johansen test
        result = coint_johansen(self.data, det_order=0, k_ar_diff=self.k_ar_diff)
        
        return {
            'trace_statistic': result.lr1,
            'max_eigenvalue_statistic': result.lr2,
            'critical_values_trace': result.cvt,
            'critical_values_max_eigen': result.cvm,
            'eigenvalues': result.eig
        }


class VARMAModel(StatisticalTimeSeriesModel, MultiOutputMixin):
    """
    Vector ARMA (VARMA) Model.
    
    Extends VAR by adding moving average components.
    More flexible but harder to estimate.
    
    Best for:
    - Complex temporal dependencies
    - When VAR residuals show autocorrelation
    - Moderate sample sizes
    
    Example:
        >>> model = VARMAModel()
        >>> model.fit(data, order=(2, 1))
        >>> forecast = model.forecast(steps=10)
    """
    
    def __init__(self, name: str = "VARMA"):
        """Initialize VARMA model."""
        super().__init__(name)
        self.model = None
        self.order = None
    
    def fit(self,
            data: Union[pd.DataFrame, np.ndarray],
            order: tuple = (1, 1),
            trend: str = 'n',
            **kwargs):
        """
        Fit VARMA model to data.
        
        Args:
            data: Time series data
            order: (p, q) for VARMA(p,q)
                   p: AR order, q: MA order
            trend: Trend specification
            **kwargs: Additional arguments
            
        Returns:
            self (fitted model)
        """
        try:
            from statsmodels.tsa.statespace.varmax import VARMAX
        except ImportError:
            raise ImportError("Please install statsmodels: pip install statsmodels")
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
        
        self.data = data
        self.variable_names = list(data.columns)
        self.order = order
        
        # Create VARMAX model (VARMA is VARMAX without exogenous variables)
        self.model = VARMAX(data, order=order, trend=trend, **kwargs)
        
        # Fit model
        self.results = self.model.fit(disp=False, **kwargs)
        
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
    
    def forecast(self, steps: int, exog: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps ahead
            exog: Future exogenous variables
            
        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Generate forecast
        forecast_values = self.results.forecast(steps=steps, exog=exog)
        
        # Create DataFrame
        forecast_df = pd.DataFrame(
            forecast_values,
            columns=self.variable_names
        )
        
        # Store forecast
        self.forecast_history.append(forecast_df)
        
        return forecast_df
