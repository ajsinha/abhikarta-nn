# Time Series Analysis Package

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | Email: ajsinha@gmail.com**

**Legal Notice:** This module and the associated software architecture are proprietary and confidential. Unauthorized copying, distribution, modification, or use is strictly prohibited without explicit written permission from the copyright holder.

**Patent Pending:** Certain architectural patterns and implementations described in this module may be subject to patent applications.

---

## Overview

A comprehensive Python package for time series analysis and prediction, featuring both **deep learning** and **statistical** models. This package provides a unified interface for working with various time series forecasting methods, from classical statistical approaches to modern neural networks.

## Features

### Core Components

- **Abstract Base Class**: Unified interface for all time series models
- **Config-Driven Architecture**: Flexible configuration system for all components
- **Data Normalization**: Multiple normalization strategies (Z-score, Min-Max, Log, etc.)
- **Ensemble Methods**: Combine multiple models for improved predictions
- **Comprehensive Evaluation**: Built-in metrics (MSE, RMSE, MAE, MAPE, R²)

### Deep Learning Models

- **LSTM** (Long Short-Term Memory): Excellent for learning long-term dependencies
- **GRU** (Gated Recurrent Unit): Faster alternative to LSTM
- **BiLSTM** (Bidirectional LSTM): Processes sequences in both directions
- **CNN-LSTM**: Hybrid architecture combining CNNs and LSTMs
- **Transformer**: State-of-the-art attention-based architecture

### Statistical Models

- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA for data with seasonality
- **VAR**: Vector AutoRegression for multivariate time series
- **ETS**: Exponential Smoothing (Holt-Winters)
- **Prophet**: Facebook's forecasting tool for business time series

### Ensemble Models

- **Averaging Ensemble**: Simple or weighted average of predictions
- **Stacking**: Meta-model learns to combine base models
- **Bagging**: Bootstrap aggregating for reduced variance

## Installation

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn matplotlib yfinance
pip install statsmodels prophet scipy
```

### Package Structure

```
timeseries/
├── model.py                    # Abstract base class
├── normalization.py           # Data normalization strategies
├── deeplearning/
│   ├── models/
│   │   ├── lstm.py           # LSTM implementation
│   │   ├── gru.py            # GRU implementation
│   │   └── advanced.py       # BiLSTM, CNN-LSTM, Transformer
│   ├── ensemble/
│   │   └── ensemble.py       # Deep learning ensembles
│   ├── examples/
│   │   └── stock_prediction_example.py
│   └── docs/                 # Detailed model documentation
├── stat/
│   ├── models/
│   │   └── statistical.py    # ARIMA, SARIMA, VAR, ETS, Prophet
│   ├── ensemble/
│   │   └── ensemble.py       # Statistical ensembles
│   ├── examples/
│   │   └── stock_prediction_example.py
│   └── docs/                 # Detailed model documentation
└── README.md
```

## Quick Start

### Data Normalization

```python
from timeseries.normalization import DataNormalizer

# Configure normalization strategy
config = {'strategy': 'zscore'}  # or 'minmax', 'robust', 'daily_change', etc.
normalizer = DataNormalizer(config)

# Fit and transform
normalized_data = normalizer.fit_transform(data)

# Inverse transform
original_data = normalizer.inverse_transform(normalized_data)
```

Available normalization strategies:
- `zscore`: Z-score standardization
- `minmax`: Min-Max scaling to [0, 1]
- `robust`: Robust scaling using median and IQR
- `daily_change`: Daily differences
- `fractional_change`: Daily percentage changes
- `log`: Logarithmic transformation
- `power`: Box-Cox/Yeo-Johnson power transformation
- `maxabs`: Max absolute value scaling

### Deep Learning Example

```python
from timeseries.deeplearning.models.lstm import LSTMModel
from timeseries.normalization import DataNormalizer

# Prepare data
X_train, y_train = ...  # Your training data
X_test, y_test = ...    # Your test data

# Normalize
normalizer = DataNormalizer({'strategy': 'zscore'})
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

# Configure and train LSTM
config = {
    'sequence_length': 20,
    'hidden_size': 64,
    'num_layers': 2,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001
}

model = LSTMModel(config=config)
model.fit(X_train, y_train, validation_split=0.1)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

### Statistical Models Example

```python
from timeseries.stat.models.statistical import ARIMAModel, VARModel

# ARIMA for univariate time series
arima = ARIMAModel(config={'order': (2, 1, 2)})
arima.fit(X_train, y_train)
predictions = arima.predict(X_test)
forecast = arima.forecast(steps=10)

# VAR for multivariate time series
var = VARModel(config={'maxlags': 5})
var.fit(X_train, y_train)
predictions = var.predict(X_test)
```

### Ensemble Models

```python
from timeseries.deeplearning.models.lstm import LSTMModel
from timeseries.deeplearning.models.gru import GRUModel
from timeseries.deeplearning.ensemble.ensemble import EnsembleModel

# Create individual models
lstm = LSTMModel(config=config)
gru = GRUModel(config=config)

# Create ensemble
ensemble = EnsembleModel(
    models=[lstm, gru],
    method='weighted',
    weights=[0.6, 0.4]
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)
```

## Examples

### Stock Price Prediction

The package includes comprehensive examples for predicting stock prices:

#### Deep Learning Example
```bash
python timeseries/deeplearning/examples/stock_prediction_example.py
```

This example:
- Downloads data for 10 stocks from Yahoo Finance
- Uses them as features to predict BMO and C stocks
- Trains LSTM, GRU, BiLSTM, CNN-LSTM, and Transformer models
- Creates an ensemble of all models
- Evaluates and compares performance
- Generates prediction visualizations

#### Statistical Models Example
```bash
python timeseries/stat/examples/stock_prediction_example.py
```

This example:
- Uses the same stock data
- Trains ARIMA, SARIMA, VAR, ETS, and Prophet models
- Creates statistical ensemble
- Compares classical vs modern approaches
- Generates forecasts

## Model Documentation

Detailed documentation for each model is available in the `docs/` folders:

### Deep Learning Models
- `deeplearning/docs/LSTM.md` - LSTM architecture and usage
- `deeplearning/docs/GRU.md` - GRU architecture and usage
- `deeplearning/docs/BiLSTM.md` - Bidirectional LSTM details
- `deeplearning/docs/CNN-LSTM.md` - CNN-LSTM hybrid architecture
- `deeplearning/docs/Transformer.md` - Transformer for time series

### Statistical Models
- `stat/docs/ARIMA.md` - ARIMA methodology
- `stat/docs/SARIMA.md` - Seasonal ARIMA
- `stat/docs/VAR.md` - Vector AutoRegression
- `stat/docs/ETS.md` - Exponential Smoothing
- `stat/docs/Prophet.md` - Prophet forecasting

## Configuration

All models support configuration through dictionaries:

```python
config = {
    # Model architecture
    'input_size': 10,
    'hidden_size': 64,
    'num_layers': 2,
    'output_size': 2,
    
    # Training parameters
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    
    # Sequence parameters
    'sequence_length': 20,
    
    # Device
    'device': 'cuda'  # or 'cpu'
}

model = LSTMModel(config=config)
```

## Performance Tips

1. **Data Normalization**: Always normalize your data for neural networks
2. **Sequence Length**: Experiment with different sequence lengths (10-50 typically works well)
3. **Hyperparameters**: Tune learning rate, hidden size, and number of layers
4. **Ensemble Methods**: Combine multiple models for better robustness
5. **GPU Acceleration**: Use CUDA when available for faster training
6. **Early Stopping**: Monitor validation loss to prevent overfitting

## Best Practices

### For Deep Learning Models
- Use z-score normalization for better convergence
- Start with smaller networks and gradually increase complexity
- Use validation split to monitor overfitting
- Experiment with different architectures (LSTM vs GRU vs Transformer)
- Consider ensemble methods for production systems

### For Statistical Models
- Check for stationarity before using ARIMA
- Use ACF/PACF plots to determine AR/MA orders
- Consider seasonal patterns in your data
- VAR works best with correlated multivariate series
- Prophet is excellent for business time series with trends and seasonality

## API Reference

### TimeSeriesModel (Base Class)

All models inherit from this abstract base class:

```python
class TimeSeriesModel(ABC):
    def fit(X, y, **kwargs) -> TimeSeriesModel
    def predict(X, steps=1) -> np.ndarray
    def forecast(steps, exog=None) -> np.ndarray
    def evaluate(X, y) -> Dict[str, float]
    def save(path) -> None
    def load(path) -> TimeSeriesModel
```

### Evaluation Metrics

All models provide these metrics:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: R-squared (coefficient of determination)

## Contributing

This is proprietary software. Contributions are not accepted without explicit permission from the copyright holder.

## License

Copyright © 2025-2030, All Rights Reserved  
Ashutosh Sinha | Email: ajsinha@gmail.com

This software is proprietary and confidential. Unauthorized copying, distribution, modification, or use is strictly prohibited without explicit written permission from the copyright holder.

## Contact

For licensing inquiries or permissions:
- Email: ajsinha@gmail.com

## Acknowledgments

This package implements various architectures and methods from published research:
- LSTM: Hochreiter & Schmidhuber (1997)
- GRU: Cho et al. (2014)
- Transformer: Vaswani et al. (2017)
- ARIMA: Box & Jenkins (1970)
- Prophet: Taylor & Letham (2018)

---

**Patent Pending**: Certain architectural patterns and implementations described in this package may be subject to patent applications.
