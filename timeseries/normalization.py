"""
Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com

Legal Notice: This module and the associated software architecture are proprietary 
and confidential. Unauthorized copying, distribution, modification, or use is 
strictly prohibited without explicit written permission from the copyright holder.

Patent Pending: Certain architectural patterns and implementations described in 
this module may be subject to patent applications.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json


class NormalizationStrategy(ABC):
    """Abstract base class for normalization strategies."""
    
    def __init__(self):
        self.params = {}
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'NormalizationStrategy':
        """Fit the normalization parameters."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        pass
    
    @abstractmethod
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform to get original scale."""
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


class MinMaxNormalization(NormalizationStrategy):
    """Min-Max normalization to scale data to [0, 1] or custom range."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.feature_range = feature_range
    
    def fit(self, data: pd.DataFrame) -> 'MinMaxNormalization':
        self.params['min'] = data.min()
        self.params['max'] = data.max()
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        data_min = self.params['min']
        data_max = self.params['max']
        data_range = data_max - data_min
        
        # Avoid division by zero
        data_range = data_range.replace(0, 1)
        
        scaled = (data - data_min) / data_range
        scaled = scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        return scaled
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        data_min = self.params['min']
        data_max = self.params['max']
        data_range = data_max - data_min
        data_range = data_range.replace(0, 1)
        
        unscaled = (data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        unscaled = unscaled * data_range + data_min
        
        return unscaled


class ZScoreNormalization(NormalizationStrategy):
    """Z-score normalization (standardization)."""
    
    def fit(self, data: pd.DataFrame) -> 'ZScoreNormalization':
        self.params['mean'] = data.mean()
        self.params['std'] = data.std()
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        mean = self.params['mean']
        std = self.params['std']
        
        # Avoid division by zero
        std = std.replace(0, 1)
        
        return (data - mean) / std
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        mean = self.params['mean']
        std = self.params['std']
        std = std.replace(0, 1)
        
        return data * std + mean


class RobustNormalization(NormalizationStrategy):
    """Robust normalization using median and IQR."""
    
    def fit(self, data: pd.DataFrame) -> 'RobustNormalization':
        self.params['median'] = data.median()
        self.params['q1'] = data.quantile(0.25)
        self.params['q3'] = data.quantile(0.75)
        self.params['iqr'] = self.params['q3'] - self.params['q1']
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        median = self.params['median']
        iqr = self.params['iqr']
        
        # Avoid division by zero
        iqr = iqr.replace(0, 1)
        
        return (data - median) / iqr
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        median = self.params['median']
        iqr = self.params['iqr']
        iqr = iqr.replace(0, 1)
        
        return data * iqr + median


class DailyChangeNormalization(NormalizationStrategy):
    """Normalize using daily changes (differences)."""
    
    def __init__(self, periods: int = 1):
        super().__init__()
        self.periods = periods
        self.original_data = None
    
    def fit(self, data: pd.DataFrame) -> 'DailyChangeNormalization':
        self.original_data = data.iloc[0:self.periods].copy()
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        return data.diff(periods=self.periods).fillna(0)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        result = data.copy()
        for i in range(self.periods, len(result)):
            result.iloc[i] = result.iloc[i] + result.iloc[i - self.periods]
        
        # Restore initial values
        for i in range(min(self.periods, len(self.original_data))):
            result.iloc[i] = self.original_data.iloc[i]
        
        return result


class DailyFractionalChangeNormalization(NormalizationStrategy):
    """Normalize using daily fractional (percentage) changes."""
    
    def __init__(self, periods: int = 1):
        super().__init__()
        self.periods = periods
        self.original_data = None
    
    def fit(self, data: pd.DataFrame) -> 'DailyFractionalChangeNormalization':
        self.original_data = data.iloc[0:self.periods].copy()
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        return data.pct_change(periods=self.periods).fillna(0).replace([np.inf, -np.inf], 0)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        result = data.copy()
        for i in range(self.periods, len(result)):
            result.iloc[i] = result.iloc[i - self.periods] * (1 + result.iloc[i])
        
        # Restore initial values
        for i in range(min(self.periods, len(self.original_data))):
            result.iloc[i] = self.original_data.iloc[i]
        
        return result


class LogNormalization(NormalizationStrategy):
    """Log transformation for positively skewed data."""
    
    def __init__(self, base: float = np.e):
        super().__init__()
        self.base = base
    
    def fit(self, data: pd.DataFrame) -> 'LogNormalization':
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        # Add small constant to avoid log(0)
        return np.log(data + 1e-8) / np.log(self.base)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        return np.power(self.base, data) - 1e-8


class PowerTransformNormalization(NormalizationStrategy):
    """Box-Cox-like power transformation."""
    
    def __init__(self, method: str = 'yeo-johnson'):
        super().__init__()
        self.method = method
    
    def fit(self, data: pd.DataFrame) -> 'PowerTransformNormalization':
        from scipy import stats
        
        self.params['lambdas'] = {}
        for col in data.columns:
            if self.method == 'box-cox':
                _, lambda_val = stats.boxcox(data[col] + 1e-8)
            else:  # yeo-johnson
                _, lambda_val = stats.yeojohnson(data[col])
            self.params['lambdas'][col] = lambda_val
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        from scipy import stats
        result = data.copy()
        
        for col in data.columns:
            lambda_val = self.params['lambdas'][col]
            if self.method == 'box-cox':
                result[col] = stats.boxcox(data[col] + 1e-8, lmbda=lambda_val)
            else:  # yeo-johnson
                result[col] = stats.yeojohnson(data[col], lmbda=lambda_val)
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        from scipy.special import inv_boxcox
        result = data.copy()
        
        for col in data.columns:
            lambda_val = self.params['lambdas'][col]
            if self.method == 'box-cox':
                result[col] = inv_boxcox(data[col], lambda_val) - 1e-8
            else:  # yeo-johnson (approximate inverse)
                if lambda_val == 0:
                    result[col] = np.exp(data[col]) - 1
                else:
                    result[col] = np.power(lambda_val * data[col] + 1, 1.0 / lambda_val) - 1
        
        return result


class MaxAbsNormalization(NormalizationStrategy):
    """Scale by maximum absolute value to [-1, 1] range."""
    
    def fit(self, data: pd.DataFrame) -> 'MaxAbsNormalization':
        self.params['max_abs'] = data.abs().max()
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        max_abs = self.params['max_abs']
        max_abs = max_abs.replace(0, 1)
        
        return data / max_abs
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        max_abs = self.params['max_abs']
        max_abs = max_abs.replace(0, 1)
        
        return data * max_abs


class DataNormalizer:
    """
    Config-driven data normalizer that applies various normalization strategies.
    """
    
    STRATEGIES = {
        'minmax': MinMaxNormalization,
        'zscore': ZScoreNormalization,
        'robust': RobustNormalization,
        'daily_change': DailyChangeNormalization,
        'fractional_change': DailyFractionalChangeNormalization,
        'log': LogNormalization,
        'power': PowerTransformNormalization,
        'maxabs': MaxAbsNormalization
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data normalizer.
        
        Args:
            config: Configuration dictionary with strategy and parameters
                   Example: {'strategy': 'zscore'} or {'strategy': 'minmax', 'feature_range': (0, 1)}
        """
        self.config = config or {'strategy': 'zscore'}
        self.strategy = None
        self._initialize_strategy()
    
    def _initialize_strategy(self):
        """Initialize the normalization strategy based on config."""
        strategy_name = self.config.get('strategy', 'zscore')
        
        if strategy_name not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}. "
                           f"Available: {list(self.STRATEGIES.keys())}")
        
        strategy_class = self.STRATEGIES[strategy_name]
        
        # Extract strategy-specific parameters
        strategy_params = {k: v for k, v in self.config.items() if k != 'strategy'}
        
        # Initialize strategy with parameters
        try:
            self.strategy = strategy_class(**strategy_params)
        except TypeError:
            # If strategy doesn't accept parameters, initialize without them
            self.strategy = strategy_class()
    
    def fit(self, data: pd.DataFrame) -> 'DataNormalizer':
        """Fit the normalization strategy."""
        self.strategy.fit(data)
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted strategy."""
        return self.strategy.transform(data)
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform to original scale."""
        return self.strategy.inverse_transform(data)
    
    def save_config(self, path: str) -> None:
        """Save normalizer configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def from_config_file(cls, path: str) -> 'DataNormalizer':
        """Load normalizer from config file."""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(config)
    
    def get_strategy_name(self) -> str:
        """Get the name of the current strategy."""
        return self.config.get('strategy', 'unknown')
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available normalization strategies."""
        return list(DataNormalizer.STRATEGIES.keys())
