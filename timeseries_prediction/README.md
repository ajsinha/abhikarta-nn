# Time Series Prediction Package 📈

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | Email: ajsinha@gmail.com**

A comprehensive PyTorch-based package for time series forecasting with **native multi-output prediction support**, optimized for financial and complex interrelated data.

## 🎯 Key Features

### ✅ **Multi-Output Prediction - Confirmed and Implemented!**

**YES, this package fully supports predicting multiple variables simultaneously!**

Simply set `output_size > 1` when creating any model:

```python
model = LSTMModel(
    input_size=30,      # 30 features (e.g., DOW30 stocks)
    output_size=2,      # Predict 2 variables (e.g., BMO + JPM)
    hidden_size=128
)
```

All models in this package support multi-output prediction out of the box!

## 🚀 Quick Start: Multi-Output Prediction

### Predict BMO and JPM using DOW30 Features

```python
from timeseries_prediction.models.rnn_models import LSTMModel
from timeseries_prediction.utils.data_utils import *

# 1. Download data
dow30 = download_stock_data(get_dow30_tickers(), '2020-01-01', '2024-01-01')
targets = download_stock_data(['BMO', 'JPM'], '2020-01-01', '2024-01-01')

# 2. Calculate returns (robust to division by zero!)
dow30_returns = calculate_returns(dow30, handle_zeros=True)
target_returns = calculate_returns(targets, handle_zeros=True)

# 3. Create sequences
X, y = create_sequences(dow30_returns.values, lookback=20)
_, y_targets = create_sequences(target_returns.values, lookback=20)

# 4. Split and normalize
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y_targets)

# 5. Create multi-output model
model = LSTMModel(input_size=30, output_size=2, hidden_size=128)

# 6. Train
model.fit(X_train, y_train, X_val, y_val, epochs=100)

# 7. Evaluate with per-output metrics
metrics = model.evaluate(X_test, y_test, per_output=True)
print(f"BMO R²: {metrics['r2_output_0']:.4f}")  # R² for BMO
print(f"JPM R²: {metrics['r2_output_1']:.4f}")  # R² for JPM
```

## 📊 Complete Examples Included

### Example 1: Multi-Output Prediction
**File**: `examples/multi_output_prediction.py`

- Downloads DOW30 stocks as features
- Downloads BMO and JPM as targets
- Robust return calculation with zero handling
- Trains LSTM to predict both stocks simultaneously
- Per-stock evaluation metrics
- Visualization

**Run it:**
```bash
cd examples
python multi_output_prediction.py
```

### Example 2: Ensemble Multi-Output Prediction  
**File**: `examples/ensemble_multi_output_prediction.py`

- Ensemble of LSTM + GRU + Transformer
- Predicts BMO and JPM using DOW30 features
- Compares individual vs ensemble performance
- Shows performance improvement

**Run it:**
```bash
cd examples
python ensemble_multi_output_prediction.py
```

Both examples handle real-world challenges:
- ✅ Division by zero in return calculations
- ✅ Missing data
- ✅ Date alignment
- ✅ Multi-output training and evaluation

## 🧠 Available Models (All Support Multi-Output!)

| Category | Models | Best For |
|----------|--------|----------|
| **RNN** | LSTM, GRU, BiLSTM | Long-term dependencies, sequential patterns |
| **CNN** | CNN1D, TCN | Fast training, local patterns |
| **Transformer** | Transformer, TFT | Global dependencies, attention mechanisms |
| **Attention** | AttentionLSTM, SelfAttention | Important feature selection |
| **Graph** | GNN, GAT | Relationship modeling (stocks, networks) |
| **Hybrid** | CNN-LSTM, CNN-Transformer | Combined benefits |
| **Probabilistic** | DeepAR, VAE | Uncertainty quantification |
| **Ensemble** | Mean, Weighted, Stacking | Best overall performance |
| **Hierarchical** | Multi-scale | Different time horizons |

## 📦 Installation

```bash
# Install dependencies
pip install torch numpy pandas scikit-learn yfinance matplotlib

# Install package
cd timeseries_prediction
python setup.py install

# Or for development
pip install -e .
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Pandas >= 1.2.0
- scikit-learn >= 0.24.0
- yfinance >= 0.1.63

## 🎓 How Multi-Output Works

### Architecture

All models output predictions of shape `(batch_size, output_size)`:

```python
# Single output (output_size=1)
predictions.shape = (100, 1)  # Predict 1 variable

# Multi output (output_size=2)
predictions.shape = (100, 2)  # Predict 2 variables (e.g., BMO, JPM)

# Multi output (output_size=30)
predictions.shape = (100, 30)  # Predict 30 variables (e.g., all DOW30)
```

### Per-Output Metrics

Evaluate each output separately:

```python
metrics = model.evaluate(X_test, y_test, per_output=True)

# Overall metrics
print(metrics['rmse'])      # Overall RMSE
print(metrics['r2'])         # Overall R²

# Per-output metrics
print(metrics['rmse_output_0'])  # RMSE for output 0
print(metrics['rmse_output_1'])  # RMSE for output 1
print(metrics['r2_output_0'])    # R² for output 0
print(metrics['r2_output_1'])    # R² for output 1
```

## 🛠️ Robust Data Processing

### Handle Division by Zero

The package includes robust return calculation:

```python
# Automatic handling of zeros and edge cases
returns = calculate_returns(
    prices,
    method='pct_change',  # or 'log', 'diff'
    handle_zeros=True     # Replaces zeros with epsilon (1e-10)
)
```

Methods:
- `pct_change`: (P_t - P_{t-1}) / P_{t-1} - **handles zeros!**
- `log`: log(P_t / P_{t-1}) - **handles zeros and negatives!**
- `diff`: P_t - P_{t-1}

### Sequence Creation

```python
X, y = create_sequences(
    data,
    lookback=20,   # Input: last 20 timesteps
    horizon=1,     # Output: predict next 1 timestep
    step=1         # Sliding window step
)
```

## 🎯 Ensemble Methods

Combine multiple models for robust predictions:

```python
from timeseries_prediction.models.ensemble_models import EnsembleModel

# Create individual models
models = [
    LSTMModel(input_size=30, output_size=2),
    GRUModel(input_size=30, output_size=2),
    TransformerModel(input_size=30, output_size=2)
]

# Create ensemble
ensemble = EnsembleModel(
    models=models,
    model_names=['LSTM', 'GRU', 'Transformer'],
    ensemble_method='mean'  # or 'median', 'weighted', 'stacking'
)

# Train all models
histories = ensemble.fit(X_train, y_train, X_val, y_val)

# Predict (automatically combines predictions)
predictions = ensemble.predict(X_test)
```

Ensemble methods:
- `mean`: Average predictions
- `median`: Median of predictions
- `weighted`: Learned weights via softmax
- `stacking`: Meta-learner on top

## 📈 Common Use Cases

### 1. Stock Price Prediction
```python
# Predict multiple stocks using market features
model = LSTMModel(input_size=50, output_size=10)  # 10 stocks
```

### 2. Portfolio Optimization
```python
# Predict returns of portfolio constituents
model = TransformerModel(input_size=30, output_size=5)  # 5 assets
```

### 3. Risk Management
```python
# Predict multiple risk factors simultaneously
model = EnsembleModel(..., output_size=3)  # VaR, CVaR, Volatility
```

### 4. Multi-Horizon Forecasting
```python
# Predict next 5 timesteps for 2 variables
# output_size = 5 * 2 = 10
model = CNNLSTMModel(input_size=30, output_size=10)
```

## 🔧 Model Configuration

### Common Parameters

```python
model = LSTMModel(
    input_size=30,           # Number of input features
    output_size=2,           # Number of outputs to predict
    hidden_size=128,         # Hidden layer size
    num_layers=2,            # Number of layers
    dropout=0.2,             # Dropout probability
    device='cuda'            # 'cuda' or 'cpu'
)
```

### Training Configuration

```python
history = model.fit(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    optimizer_name='adam',           # 'adam', 'adamw', 'sgd'
    early_stopping_patience=10,
    verbose=True
)
```

## 📊 Evaluation

```python
# Evaluate model
metrics = model.evaluate(
    X_test, y_test,
    metrics=['mse', 'rmse', 'mae', 'mape', 'r2'],
    per_output=True  # Get metrics for each output
)

# Print results
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
```

## 💾 Save/Load Models

```python
# Save
model.save('my_model.pth')

# Load
from timeseries_prediction.models.rnn_models import LSTMModel
loaded_model = LSTMModel.load('my_model.pth')

# Use loaded model
predictions = loaded_model.predict(X_test)
```

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce `batch_size` or `hidden_size`

### Issue: NaN Loss
**Solution**: 
- Check data for zeros/infinities (use `handle_zeros=True`)
- Reduce learning rate
- Gradient clipping is enabled by default

### Issue: Poor Performance
**Solution**:
- Normalize input data
- Increase model capacity (`hidden_size`, `num_layers`)
- Use ensemble methods
- Tune hyperparameters

## 📚 Documentation

For detailed documentation, see:
- `DOCUMENTATION.md` - Comprehensive theory and examples
- `examples/` - Working code examples
- Docstrings in source code

## 🎯 Performance Tips

1. **Always normalize data** - Use `normalize_data()` utility
2. **Use GPU** - Automatic if available, 10-50x faster
3. **Ensemble for production** - More robust than single models
4. **Start simple** - Begin with LSTM, add complexity as needed
5. **Use validation set** - Enable early stopping

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

Built for researchers and practitioners working with:
- Financial time series
- Multi-variate forecasting
- Complex interrelated data
- Production ML systems

---

## Quick Reference

```python
# Download data
from timeseries_prediction.utils.data_utils import *
prices = download_stock_data(['AAPL', 'MSFT'], '2020-01-01', '2024-01-01')
returns = calculate_returns(prices, handle_zeros=True)

# Create sequences
X, y = create_sequences(returns.values, lookback=20)

# Split data
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

# Build model (MULTI-OUTPUT READY!)
from timeseries_prediction.models.rnn_models import LSTMModel
model = LSTMModel(input_size=2, output_size=2, hidden_size=64)

# Train
model.fit(X_train, y_train, X_val, y_val, epochs=50)

# Evaluate
metrics = model.evaluate(X_test, y_test, per_output=True)

# Predict
predictions = model.predict(X_test)

# Save
model.save('model.pth')
```

**That's it! You're ready to predict multiple time series simultaneously!** 🚀
