"""Base module for statistical time series models."""

from timeseries_prediction_stat.base.base_model import (
    StatisticalTimeSeriesModel,
    MultiOutputMixin
)

__all__ = [
    'StatisticalTimeSeriesModel',
    'MultiOutputMixin'
]
