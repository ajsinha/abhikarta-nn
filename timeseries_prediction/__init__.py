"""
Time Series Prediction Package
===============================

A comprehensive PyTorch-based package for time series forecasting with various
neural network architectures optimized for financial and complex interrelated data.

Author: Developed for ML/Financial Time Series Applications
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Time Series Prediction Team"

# Base model
from timeseries_prediction.base.base_model import TimeSeriesModel

# RNN models
from timeseries_prediction.models.rnn_models import (
    LSTMModel, GRUModel, BiLSTMModel
)

# CNN models
from timeseries_prediction.models.cnn_models import (
    CNNModel, TCNModel
)

# Transformer models
from timeseries_prediction.models.transformer_models import (
    TransformerModel, TemporalFusionTransformer
)

# Ensemble models
from timeseries_prediction.models.ensemble_models import (
    EnsembleModel
)

# Data utilities
from timeseries_prediction.utils.data_utils import (
    download_stock_data,
    get_dow30_tickers,
    calculate_returns,
    create_sequences,
    train_val_test_split,
    normalize_data,
    denormalize_data
)

__all__ = [
    'TimeSeriesModel',
    'LSTMModel',
    'GRUModel',
    'BiLSTMModel',
    'CNNModel',
    'TCNModel',
    'TransformerModel',
    'TemporalFusionTransformer',
    'EnsembleModel',
    'download_stock_data',
    'get_dow30_tickers',
    'calculate_returns',
    'create_sequences',
    'train_val_test_split',
    'normalize_data',
    'denormalize_data'
]

