# Time Series Neural Network with Ratio Scaling

## Overview
This package provides deep learning code for time series prediction using neural networks (LSTM/GRU) with **ratio-based scaling** where values at time `t` are divided by values at time `t-1`.

## Files Included

1. **timeseries_deep_learning.py** - Complete, production-ready implementation with:
   - Custom ratio preprocessing class
   - LSTM and GRU model builders
   - Training with early stopping and model checkpointing
   - Visualization of results
   - Model saving and loading utilities

2. **simple_timeseries_example.py** - Minimal example showing core concepts
   - Great for quick start and understanding the workflow
   - ~100 lines of well-commented code

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

Or if you have the requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Run the Simple Example

```python
python simple_timeseries_example.py
```

This will train a model on synthetic data and show you the complete workflow.

### 3. Adapt to Your Data

Replace the data loading section in either script:

```python
# Load your data
df = pd.read_csv('your_data.csv')
X_raw = df[['var1', 'var2', 'var3', ..., 'var10']].values  # 10 input variables
y_raw = df[['target']].values  # 1 target variable
```

## Key Features

### Ratio Transformation (t/t-1)
The code implements proper scaling using ratios:
```python
ratio(t) = value(t) / value(t-1)
```

This approach:
- ✅ Makes data stationary
- ✅ Removes scale differences between variables
- ✅ Captures relative changes (returns) rather than absolute values
- ✅ Works well with financial and economic time series

### Model Architecture

**LSTM-based (default):**
```
Input (sequence_length, 10 features)
    ↓
LSTM Layer (64 units)
    ↓
LSTM Layer (32 units)
    ↓
Dense Layers (32 → 16)
    ↓
Output (1 value)
```

**GRU alternative** (faster, similar performance):
- Just uncomment the GRU model builder in the code

## Hyperparameters to Tune

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequence_length` | 20 | How many past time steps to use |
| `lstm_units` | [64, 32] | Size of LSTM layers |
| `dropout_rate` | 0.2 | Regularization strength (0-0.5) |
| `epochs` | 100 | Maximum training iterations |
| `batch_size` | 32 | Training batch size |

### Tuning Guidelines

- **sequence_length**: Start with 20-30. Increase for long-term patterns, decrease for short-term
- **lstm_units**: Increase for complex patterns, decrease if overfitting
- **dropout_rate**: Increase (0.3-0.5) if overfitting, decrease if underfitting
- **epochs**: Use early stopping (included in full version)

## Expected Outputs

When you run the complete script, it produces:

1. **best_model.keras** - Best model during training
2. **final_model.keras** - Final trained model
3. **preprocessors.pkl** - Saved preprocessing transformations
4. **training_results.png** - Visualization showing:
   - Training/validation loss curves
   - Predictions vs actual values
   - Scatter plot of prediction quality

## Usage Examples

### Training on Your Data

```python
import pandas as pd
from timeseries_prtediction.timeseries_deep_learning import *

# Load data
df = pd.read_csv('mydata.csv')
X_raw = df[['feature1', 'feature2', ..., 'feature10']].values
y_raw = df[['target']].values

# Preprocess
X_preprocessor = TimeSeriesRatioPreprocessor()
y_preprocessor = TimeSeriesRatioPreprocessor()

X_scaled = X_preprocessor.fit_transform(X_raw)
y_scaled = y_preprocessor.fit_transform(y_raw)

# Create sequences
X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length=20)

# Train model
model = build_lstm_model(sequence_length=20, n_features=10)
model.fit(X_seq, y_seq, epochs=50, validation_split=0.2)
```

### Making Predictions

```python
# Load saved model and preprocessors
predictions = load_model_and_predict(
    new_data_X,
    model_path='final_model.keras',
    preprocessor_path='preprocessors.pkl'
)
```

## Model Performance

The code tracks multiple metrics:
- **MSE (Mean Squared Error)**: Overall prediction error
- **MAE (Mean Absolute Error)**: Average absolute error
- **RMSE**: Square root of MSE (same units as target)

Lower values indicate better performance.

## Common Issues and Solutions

### 1. NaN or Inf values
**Problem**: Division by zero in ratio calculation  
**Solution**: Already handled with `np.where()` checks in the code

### 2. Poor predictions
**Try**:
- Increase sequence_length
- Add more LSTM units
- Train for more epochs
- Check if data has strong temporal patterns

### 3. Overfitting (good training, poor validation)
**Try**:
- Increase dropout_rate (0.3-0.5)
- Reduce model complexity (fewer LSTM units)
- Use more training data
- Early stopping (already included)

### 4. Slow training
**Try**:
- Use GRU instead of LSTM (uncomment in code)
- Reduce batch_size
- Reduce LSTM units
- Use GPU if available

## Data Requirements

- **Format**: Time series data sorted by time
- **Input variables**: 10 numeric features
- **Target variable**: 1 numeric value
- **Minimum samples**: 200+ recommended (more is better)
- **Missing values**: Handle before using this code

## Advanced Customization

### Change Network Architecture

```python
# Deeper network
model = build_lstm_model(
    sequence_length=20,
    n_features=10,
    lstm_units=[128, 64, 32],  # 3 LSTM layers
    dropout_rate=0.3
)
```

### Add Bidirectional LSTM

```python
from tensorflow.keras.layers import Bidirectional

model.add(Bidirectional(LSTM(64, return_sequences=True)))
```

### Multi-step Prediction

Modify the `create_sequences()` function to output multiple future steps.

## Further Reading

- LSTM networks: Understanding temporal dependencies
- Time series stationarity: Why ratios/differences matter
- Hyperparameter tuning: Grid search and random search
- Cross-validation for time series: Walk-forward validation

## Support

For TensorFlow installation issues, see: https://www.tensorflow.org/install

For questions about the code, review the detailed comments in the scripts.

---
**Note**: This is a template. Always validate your model on out-of-sample data before deploying in production!
