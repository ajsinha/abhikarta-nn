"""
Statistical Time Series Prediction Package
===========================================

A comprehensive package for statistical time series forecasting including:
- Vector Autoregression (VAR, VECM, VARMA)
- Linear Regression Models (OLS, Ridge, Lasso, Elastic Net, Bayesian, Robust)
- Dynamic Factor Models (DFM, PCA-based)
- Multi-output prediction support

Author: Time Series Prediction Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Time Series Prediction Team"

# Base classes
from timeseries_prediction_stat.base.base_model import (
    StatisticalTimeSeriesModel,
    MultiOutputMixin
)

# VAR models
from timeseries_prediction_stat.models.var_models import (
    VARModel,
    VECMModel,
    VARMAModel
)

# Regression models
from timeseries_prediction_stat.models.regression_models import (
    MultiOutputLinearRegression,
    RidgeTimeSeriesRegression,
    LassoTimeSeriesRegression,
    ElasticNetTimeSeriesRegression,
    BayesianLinearRegression,
    RobustLinearRegression
)

# Factor models
from timeseries_prediction_stat.models.factor_models import (
    DynamicFactorModel,
    PCABasedForecaster
)

__all__ = [
    # Base
    'StatisticalTimeSeriesModel',
    'MultiOutputMixin',
    # VAR models
    'VARModel',
    'VECMModel',
    'VARMAModel',
    # Regression
    'MultiOutputLinearRegression',
    'RidgeTimeSeriesRegression',
    'LassoTimeSeriesRegression',
    'ElasticNetTimeSeriesRegression',
    'BayesianLinearRegression',
    'RobustLinearRegression',
    # Factor models
    'DynamicFactorModel',
    'PCABasedForecaster'
]
