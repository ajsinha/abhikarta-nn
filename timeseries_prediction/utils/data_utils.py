"""
Data Utilities

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


# Financial Data Functions
# =========================

def download_stock_data(tickers: List[str], 
                        start_date: str, 
                        end_date: str,
                        column: str = 'Close') -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.
    
    Args:
        tickers: List of stock tickers (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        column: Column to extract ('Open', 'High', 'Low', 'Close', 'Volume')
        
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Please install yfinance: pip install yfinance")
    
    print(f"Downloading data for {len(tickers)} tickers...")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Handle single ticker case
    if len(tickers) == 1:
        df = data[[column]].copy()
        df.columns = tickers
    else:
        # Extract the specified column
        df = data[column].copy()
    
    # Drop any rows with all NaN values
    df = df.dropna(how='all')
    
    # Forward fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Downloaded {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")
    
    return df


def get_dow30_tickers() -> List[str]:
    """
    Get list of DOW 30 component tickers.
    
    Returns:
        List of DOW 30 ticker symbols
    """
    dow30 = [
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT', 'WBA'
    ]
    return dow30


def calculate_returns(prices: pd.DataFrame, 
                     method: str = 'pct_change',
                     handle_zeros: bool = True) -> pd.DataFrame:
    """
    Calculate returns from prices in a robust way.
    
    Args:
        prices: DataFrame of prices
        method: Return calculation method
                - 'pct_change': (P_t - P_{t-1}) / P_{t-1}
                - 'log': log(P_t / P_{t-1})
                - 'diff': P_t - P_{t-1}
        handle_zeros: If True, replace zeros with small epsilon to avoid division errors
        
    Returns:
        DataFrame of returns
    """
    prices_copy = prices.copy()
    
    # Handle zeros and negative prices robustly
    if handle_zeros:
        # Replace zeros with very small positive number
        epsilon = 1e-10
        prices_copy = prices_copy.replace(0, epsilon)
        # Also ensure all prices are positive for log returns
        prices_copy = prices_copy.abs()
        prices_copy = prices_copy.replace(0, epsilon)
    
    if method == 'pct_change':
        # Percentage change: (P_t - P_{t-1}) / P_{t-1}
        returns = prices_copy.pct_change()
        
    elif method == 'log':
        # Log returns: log(P_t / P_{t-1})
        returns = np.log(prices_copy / prices_copy.shift(1))
        
    elif method == 'diff':
        # Simple difference: P_t - P_{t-1}
        returns = prices_copy.diff()
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Drop first row (NaN from differencing)
    returns = returns.iloc[1:]
    
    # Replace inf values with NaN then forward fill
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return returns


def create_sequences(data: np.ndarray, 
                     lookback: int, 
                     horizon: int = 1,
                     step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Args:
        data: Input data (timesteps, features)
        lookback: Number of timesteps to look back
        horizon: Number of timesteps to predict ahead
        step: Step size for sliding window
        
    Returns:
        Tuple of (X, y) where:
        - X: (samples, lookback, features)
        - y: (samples, horizon * features) for multi-output
    """
    X, y = [], []
    
    for i in range(0, len(data) - lookback - horizon + 1, step):
        # Input sequence
        X.append(data[i:i + lookback])
        
        # Output sequence (next 'horizon' timesteps)
        if horizon == 1:
            y.append(data[i + lookback])
        else:
            # For multi-horizon, flatten the output
            y.append(data[i + lookback:i + lookback + horizon].flatten())
    
    return np.array(X), np.array(y)


def train_val_test_split(X: np.ndarray, 
                         y: np.ndarray,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15) -> Tuple:
    """
    Split data into train, validation, and test sets (time-series aware).
    
    Args:
        X: Features array
        y: Targets array
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(train_data: np.ndarray,
                   val_data: Optional[np.ndarray] = None,
                   test_data: Optional[np.ndarray] = None,
                   method: str = 'standardize') -> Tuple:
    """
    Normalize data using training statistics.
    
    Args:
        train_data: Training data
        val_data: Validation data (optional)
        test_data: Test data (optional)
        method: Normalization method ('standardize' or 'minmax')
        
    Returns:
        Tuple of normalized arrays and statistics dict
    """
    if method == 'standardize':
        mean = train_data.mean(axis=0, keepdims=True)
        std = train_data.std(axis=0, keepdims=True) + 1e-8
        
        train_normalized = (train_data - mean) / std
        stats = {'mean': mean, 'std': std, 'method': 'standardize'}
        
        val_normalized = (val_data - mean) / std if val_data is not None else None
        test_normalized = (test_data - mean) / std if test_data is not None else None
        
    elif method == 'minmax':
        min_val = train_data.min(axis=0, keepdims=True)
        max_val = train_data.max(axis=0, keepdims=True)
        range_val = max_val - min_val + 1e-8
        
        train_normalized = (train_data - min_val) / range_val
        stats = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
        val_normalized = (val_data - min_val) / range_val if val_data is not None else None
        test_normalized = (test_data - min_val) / range_val if test_data is not None else None
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return train_normalized, val_normalized, test_normalized, stats


def denormalize_data(data: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Denormalize data using saved statistics.
    
    Args:
        data: Normalized data
        stats: Statistics dictionary from normalize_data
        
    Returns:
        Denormalized data
    """
    if stats['method'] == 'standardize':
        return data * stats['std'] + stats['mean']
    elif stats['method'] == 'minmax':
        return data * (stats['max'] - stats['min']) + stats['min']
    else:
        raise ValueError(f"Unknown method: {stats['method']}")


# Original Dataset Classes
# =========================


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    
    Creates sequences from time series data for supervised learning.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_length: int,
        prediction_horizon: int = 1,
        stride: int = 1
    ):
        """
        Initialize dataset.
        
        Args:
            data: Time series data of shape (n_samples, n_features)
            seq_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
            stride: Stride for creating sequences
        """
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        
        # Create sequences
        self.X, self.y = self._create_sequences()
    
    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create input-output sequences."""
        X, y = [], []
        
        for i in range(0, len(self.data) - self.seq_length - self.prediction_horizon + 1, self.stride):
            X.append(self.data[i:i + self.seq_length])
            y.append(self.data[i + self.seq_length:i + self.seq_length + self.prediction_horizon])
        
        X = torch.stack(X)
        y = torch.stack(y)
        
        # If prediction horizon is 1, squeeze the dimension
        if self.prediction_horizon == 1:
            y = y.squeeze(1)
        
        return X, y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DataTransformer:
    """
    Data transformation utilities for time series.
    
    Supports various scaling and transformation methods.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize transformer.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust', 'log', 'diff')
        """
        self.method = method
        self.scaler = None
        self.original_data = None
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit transformer and transform data."""
        self.original_data = data.copy()
        
        if self.method in ['standard', 'minmax', 'robust']:
            return self.scaler.fit_transform(data)
        
        elif self.method == 'log':
            return np.log1p(data)
        
        elif self.method == 'diff':
            return np.diff(data, axis=0, prepend=data[0:1])
        
        return data
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if self.method in ['standard', 'minmax', 'robust']:
            return self.scaler.transform(data)
        
        elif self.method == 'log':
            return np.log1p(data)
        
        elif self.method == 'diff':
            return np.diff(data, axis=0, prepend=data[0:1])
        
        return data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if self.method in ['standard', 'minmax', 'robust']:
            return self.scaler.inverse_transform(data)
        
        elif self.method == 'log':
            return np.expm1(data)
        
        elif self.method == 'diff':
            # Cumulative sum to reverse differencing
            return np.cumsum(data, axis=0)
        
        return data


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        X: Input features
        y: Target values
        batch_size: Batch size
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        shuffle: Whether to shuffle data
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    np.random.seed(random_seed)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
    
    # Split indices
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X[train_idx]),
        torch.FloatTensor(y[train_idx])
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X[val_idx]),
        torch.FloatTensor(y[val_idx])
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X[test_idx]),
        torch.FloatTensor(y[test_idx])
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def prepare_time_series_data(
    data: np.ndarray,
    seq_length: int,
    prediction_horizon: int = 1,
    scaler_method: str = 'standard',
    train_split: float = 0.7,
    val_split: float = 0.15,
    batch_size: int = 32,
    stride: int = 1
) -> Tuple[DataLoader, DataLoader, DataLoader, DataTransformer]:
    """
    Complete pipeline for preparing time series data.
    
    Args:
        data: Raw time series data (n_samples, n_features)
        seq_length: Length of input sequences
        prediction_horizon: Number of steps to predict
        scaler_method: Scaling method to use
        train_split: Training data proportion
        val_split: Validation data proportion
        batch_size: Batch size
        stride: Stride for sequence creation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Scale data
    transformer = DataTransformer(method=scaler_method)
    scaled_data = transformer.fit_transform(data)
    
    # Create dataset
    dataset = TimeSeriesDataset(
        scaled_data,
        seq_length=seq_length,
        prediction_horizon=prediction_horizon,
        stride=stride
    )
    
    # Split into train/val/test
    n_samples = len(dataset)
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))
    
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_end))
    val_dataset = torch.utils.data.Subset(dataset, range(train_end, val_end))
    test_dataset = torch.utils.data.Subset(dataset, range(val_end, n_samples))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, transformer


def generate_synthetic_time_series(
    n_samples: int = 1000,
    n_features: int = 5,
    trend: bool = True,
    seasonality: bool = True,
    noise_level: float = 0.1,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic time series data for testing.
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features
        trend: Whether to include trend
        seasonality: Whether to include seasonality
        noise_level: Standard deviation of noise
        random_seed: Random seed
        
    Returns:
        Synthetic time series data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    t = np.arange(n_samples)
    data = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        # Base signal
        signal = np.zeros(n_samples)
        
        # Add trend
        if trend:
            signal += 0.01 * t + np.random.randn() * 10
        
        # Add seasonality
        if seasonality:
            period = np.random.randint(20, 100)
            signal += 5 * np.sin(2 * np.pi * t / period)
            
            # Add multiple seasonal components
            if np.random.rand() > 0.5:
                period2 = np.random.randint(50, 200)
                signal += 3 * np.cos(2 * np.pi * t / period2)
        
        # Add noise
        signal += noise_level * np.random.randn(n_samples)
        
        data[:, i] = signal
    
    return data


def add_missing_values(
    data: np.ndarray,
    missing_ratio: float = 0.1,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Add missing values to time series data.
    
    Args:
        data: Time series data
        missing_ratio: Proportion of values to make missing
        random_seed: Random seed
        
    Returns:
        Data with missing values (NaN)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    data_with_missing = data.copy()
    mask = np.random.rand(*data.shape) < missing_ratio
    data_with_missing[mask] = np.nan
    
    return data_with_missing


def handle_missing_values(
    data: np.ndarray,
    method: str = 'forward_fill'
) -> np.ndarray:
    """
    Handle missing values in time series.
    
    Args:
        data: Time series data with missing values
        method: Method to handle missing values
                ('forward_fill', 'backward_fill', 'interpolate', 'mean')
        
    Returns:
        Data with missing values filled
    """
    from pandas import DataFrame
    
    df = DataFrame(data)
    
    if method == 'forward_fill':
        df = df.fillna(method='ffill')
    elif method == 'backward_fill':
        df = df.fillna(method='bfill')
    elif method == 'interpolate':
        df = df.interpolate(method='linear')
    elif method == 'mean':
        df = df.fillna(df.mean())
    
    # Fill any remaining NaNs with 0
    df = df.fillna(0)
    
    return df.values
