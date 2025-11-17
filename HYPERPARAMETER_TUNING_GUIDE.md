# Hyperparameter Tuning Guide

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | Email: ajsinha@gmail.com**

---

## Overview

This guide provides comprehensive strategies for tuning hyperparameters across all time series models in the package. Proper tuning can improve model performance by 20-50%.

## Table of Contents

1. [Deep Learning Models](#deep-learning-models)
2. [Statistical Models](#statistical-models)
3. [Tuning Strategies](#tuning-strategies)
4. [AutoML Approaches](#automl-approaches)
5. [Best Practices](#best-practices)

---

## Deep Learning Models

### LSTM Hyperparameters

#### Critical Parameters (Tune First)

**1. Sequence Length**
- **Range:** 5-100
- **Default:** 20
- **Impact:** Very High
- **Tuning Guide:**
  ```python
  # Try these values
  sequence_lengths = [10, 20, 30, 50]
  
  # Rule of thumb:
  # - Financial data: 20-50
  # - Hourly data: 24, 48, 168 (day, 2 days, week)
  # - Daily data: 7, 14, 30 (week, 2 weeks, month)
  ```
- **Signs of wrong value:**
  - Too short: Model misses long-term patterns
  - Too long: Training becomes slow, may overfit

**2. Hidden Size**
- **Range:** 16-256
- **Default:** 64
- **Impact:** High
- **Tuning Guide:**
  ```python
  hidden_sizes = [32, 64, 128, 256]
  
  # Guidelines:
  # - Small data (< 1000): 32-64
  # - Medium data (1000-5000): 64-128
  # - Large data (> 5000): 128-256
  # - Complex patterns: Larger
  # - Simple patterns: Smaller
  ```
- **Signs of wrong value:**
  - Too small: Underfitting, poor train accuracy
  - Too large: Overfitting, slow training

**3. Number of Layers**
- **Range:** 1-4
- **Default:** 2
- **Impact:** Medium
- **Tuning Guide:**
  ```python
  num_layers_options = [1, 2, 3]
  
  # Guidelines:
  # - Start with 2
  # - Add layer if underfitting persists
  # - Remove layer if overfitting occurs
  # - Rarely need more than 3
  ```
- **Signs of wrong value:**
  - Too few: Can't learn complex patterns
  - Too many: Overfitting, diminishing returns

**4. Learning Rate**
- **Range:** 0.0001-0.01
- **Default:** 0.001
- **Impact:** Very High
- **Tuning Guide:**
  ```python
  learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
  
  # Adaptive strategy:
  # Start with 0.001
  # If loss plateaus early: increase to 0.005
  # If loss unstable/diverges: decrease to 0.0005
  # Use learning rate scheduling for best results
  ```
- **Signs of wrong value:**
  - Too high: Loss oscillates or increases
  - Too low: Very slow training, early plateau

#### Secondary Parameters (Tune After Primary)

**5. Dropout**
- **Range:** 0.0-0.5
- **Default:** 0.2
- **Impact:** Medium
- **Purpose:** Prevent overfitting
- **Tuning Guide:**
  ```python
  dropouts = [0.0, 0.1, 0.2, 0.3, 0.5]
  
  # Guidelines:
  # - No overfitting: 0.0-0.1
  # - Mild overfitting: 0.2
  # - Severe overfitting: 0.3-0.5
  # - More layers: Higher dropout
  ```

**6. Batch Size**
- **Range:** 8-256
- **Default:** 32
- **Impact:** Medium (speed and convergence)
- **Tuning Guide:**
  ```python
  batch_sizes = [16, 32, 64, 128]
  
  # Guidelines:
  # - Small data: 16-32
  # - Large data: 64-128
  # - GPU memory limited: Smaller
  # - Want speed: Larger (if GPU allows)
  ```

**7. Epochs**
- **Range:** 50-500
- **Default:** 100
- **Impact:** Medium
- **Tuning Guide:**
  ```python
  # Use early stopping instead of fixed epochs
  from timeseries.utils import EarlyStopping
  
  early_stop = EarlyStopping(
      patience=10,  # Stop after 10 epochs without improvement
      min_delta=0.001  # Minimum improvement threshold
  )
  
  # Or manually: monitor validation loss
  # Stop when it starts increasing
  ```

### Complete LSTM Tuning Example

```python
from timeseries.deeplearning.models.lstm import LSTMModel
from sklearn.model_selection import ParameterGrid

# Define parameter grid
param_grid = {
    'sequence_length': [20, 30, 50],
    'hidden_size': [64, 128],
    'num_layers': [2, 3],
    'learning_rate': [0.001, 0.005],
    'dropout': [0.2, 0.3]
}

best_rmse = float('inf')
best_params = None

# Grid search
for params in ParameterGrid(param_grid):
    config = {
        'sequence_length': params['sequence_length'],
        'hidden_size': params['hidden_size'],
        'num_layers': params['num_layers'],
        'learning_rate': params['learning_rate'],
        'dropout': params['dropout'],
        'epochs': 100,
        'batch_size': 32
    }
    
    model = LSTMModel(config=config)
    model.fit(X_train, y_train, validation_split=0.15, verbose=False)
    
    metrics = model.evaluate(X_val, y_val)
    
    if metrics['rmse'] < best_rmse:
        best_rmse = metrics['rmse']
        best_params = params
        print(f"New best: RMSE={best_rmse:.4f}, Params={params}")

print(f"\nBest configuration: {best_params}")
```

### GRU Hyperparameters

**Same as LSTM with these differences:**
- Can use slightly larger hidden_size (GRU more efficient)
- Often needs fewer layers than LSTM
- Faster training allows more epochs

### TCN Hyperparameters

**Key Parameters:**

**1. Number of Channels**
- **Format:** List of channel sizes per level
- **Default:** [64, 64, 64]
- **Tuning Guide:**
  ```python
  num_channels_options = [
      [32, 32],           # Small, fast
      [64, 64, 64],       # Medium (default)
      [128, 128, 64],     # Large
      [128, 128, 128, 64] # Very large
  ]
  
  # Guidelines:
  # - More levels: Larger receptive field
  # - Decreasing size: Good practice
  # - Start with 3 levels
  ```

**2. Kernel Size**
- **Range:** 2-7
- **Default:** 3
- **Impact:** Controls receptive field growth
- **Tuning Guide:**
  ```python
  kernel_sizes = [2, 3, 5, 7]
  
  # Guidelines:
  # - Larger: Bigger receptive field, more parameters
  # - Smaller: More efficient
  # - Usually 3 or 5 works well
  ```

### Transformer Hyperparameters

**Key Parameters:**

**1. d_model (Model Dimension)**
- **Range:** 32-512
- **Default:** 64
- **Impact:** Very High
- **Must be divisible by nhead**

**2. nhead (Number of Attention Heads)**
- **Range:** 2-16
- **Default:** 4
- **Impact:** High
- **Must divide d_model evenly**
- **Common pairs:**
  - d_model=64, nhead=4
  - d_model=128, nhead=8
  - d_model=256, nhead=8

**3. num_layers**
- **Range:** 2-6
- **Default:** 2
- **Impact:** High
- **More than 6 rarely helpful**

**Tuning Example:**
```python
transformer_configs = [
    {'d_model': 64, 'nhead': 4, 'num_layers': 2},
    {'d_model': 128, 'nhead': 8, 'num_layers': 2},
    {'d_model': 128, 'nhead': 8, 'num_layers': 3},
    {'d_model': 256, 'nhead': 8, 'num_layers': 3},
]
```

---

## Statistical Models

### ARIMA Hyperparameters

**Order Selection: (p, d, q)**

**1. Differencing (d)**
- **Range:** 0-2
- **Method:** Statistical tests
```python
from statsmodels.tsa.stattools import adfuller

# Test for stationarity
def find_d(series, max_d=2):
    for d in range(max_d + 1):
        if d == 0:
            diff_series = series
        else:
            diff_series = series.diff(d).dropna()
        
        result = adfuller(diff_series)
        if result[1] <= 0.05:  # p-value threshold
            print(f"Series is stationary with d={d}")
            return d
    return max_d

d = find_d(y_train)
```

**2. AR Order (p)**
- **Range:** 0-5
- **Method:** PACF plot
```python
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

plot_pacf(y_train.diff(d).dropna(), lags=20)
plt.show()

# Look for sharp cutoff at lag k
# That's your p value
```

**3. MA Order (q)**
- **Range:** 0-5
- **Method:** ACF plot
```python
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(y_train.diff(d).dropna(), lags=20)
plt.show()

# Look for sharp cutoff at lag k
# That's your q value
```

**Grid Search for ARIMA:**
```python
from timeseries.stat.models.statistical import ARIMAModel
from itertools import product

p_range = range(0, 3)
d_range = range(0, 2)
q_range = range(0, 3)

best_aic = float('inf')
best_order = None

for p, d, q in product(p_range, d_range, q_range):
    try:
        model = ARIMAModel(config={'order': (p, d, q)})
        model.fit(X_train, y_train)
        
        params = model.get_params()
        aic = params['aic']
        
        if aic < best_aic:
            best_aic = aic
            best_order = (p, d, q)
            print(f"New best: ARIMA{best_order}, AIC={best_aic:.2f}")
    except:
        continue

print(f"\nBest ARIMA order: {best_order}")
```

### SARIMA Hyperparameters

**Additional Seasonal Parameters: (P, D, Q, s)**

```python
# Seasonal order selection
seasonal_configs = [
    (0, 0, 0, 12),  # No seasonality
    (1, 0, 0, 12),  # Seasonal AR
    (0, 0, 1, 12),  # Seasonal MA
    (1, 0, 1, 12),  # Seasonal ARMA
    (1, 1, 1, 12),  # Full seasonal
]

# For hourly data
seasonal_configs_hourly = [
    (1, 0, 1, 24),   # Daily seasonality
    (1, 0, 1, 168),  # Weekly seasonality
]
```

### Prophet Hyperparameters

**Key Parameters:**

**1. Changepoint Prior Scale**
- **Range:** 0.001-0.5
- **Default:** 0.05
- **Impact:** Trend flexibility
```python
changepoint_scales = [0.001, 0.01, 0.05, 0.1, 0.5]

for scale in changepoint_scales:
    model = ProphetModel(config={
        'changepoint_prior_scale': scale
    })
    # Test and compare
```

**2. Seasonality Prior Scale**
- **Range:** 0.01-10
- **Default:** 10
- **Impact:** Seasonality strength
```python
seasonality_scales = [0.01, 0.1, 1.0, 10.0]
```

**3. Seasonality Mode**
- **Options:** 'additive', 'multiplicative'
- **Default:** 'additive'
- **Guideline:**
  - Additive: Seasonal variations constant over time
  - Multiplicative: Seasonal variations proportional to level

---

## Tuning Strategies

### 1. Sequential Tuning (Recommended for Beginners)

**Step 1: Tune Critical Parameters**
```python
# Focus on sequence_length and hidden_size first
configs = []
for seq_len in [20, 30, 50]:
    for hidden in [64, 128]:
        configs.append({
            'sequence_length': seq_len,
            'hidden_size': hidden,
            'num_layers': 2,  # Fixed
            'learning_rate': 0.001  # Fixed
        })
```

**Step 2: Fix Best, Tune Next**
```python
# Use best seq_len and hidden_size
# Now tune num_layers and learning_rate
best_seq_len = 30
best_hidden = 128

configs = []
for num_layers in [2, 3]:
    for lr in [0.001, 0.005]:
        configs.append({
            'sequence_length': best_seq_len,
            'hidden_size': best_hidden,
            'num_layers': num_layers,
            'learning_rate': lr
        })
```

**Step 3: Fine-tune Remaining**
```python
# Tune dropout, batch_size, etc.
```

### 2. Random Search (Faster, Good Results)

```python
import numpy as np

def random_config(n_samples=20):
    configs = []
    for _ in range(n_samples):
        config = {
            'sequence_length': np.random.choice([20, 30, 50]),
            'hidden_size': np.random.choice([64, 128, 256]),
            'num_layers': np.random.choice([2, 3]),
            'learning_rate': 10 ** np.random.uniform(-4, -2),
            'dropout': np.random.uniform(0.1, 0.4),
            'batch_size': np.random.choice([32, 64, 128])
        }
        configs.append(config)
    return configs

# Test random configurations
for config in random_config(20):
    model = LSTMModel(config=config)
    # Train and evaluate
```

### 3. Bayesian Optimization (Best, More Complex)

```python
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

# Define search space
space = [
    Integer(10, 50, name='sequence_length'),
    Integer(32, 256, name='hidden_size'),
    Integer(1, 3, name='num_layers'),
    Real(0.0001, 0.01, name='learning_rate', prior='log-uniform'),
    Real(0.0, 0.5, name='dropout')
]

# Define objective function
def objective(params):
    config = {
        'sequence_length': params[0],
        'hidden_size': params[1],
        'num_layers': params[2],
        'learning_rate': params[3],
        'dropout': params[4],
        'epochs': 50,
        'batch_size': 32
    }
    
    model = LSTMModel(config=config)
    model.fit(X_train, y_train, validation_split=0.15, verbose=False)
    metrics = model.evaluate(X_val, y_val)
    
    return metrics['rmse']  # Minimize RMSE

# Run optimization
result = gp_minimize(
    objective,
    space,
    n_calls=50,
    random_state=42
)

print(f"Best parameters: {result.x}")
print(f"Best RMSE: {result.fun}")
```

---

## AutoML Approaches

### Using Auto-ARIMA

```python
from timeseries.stat.models.advanced_statistical import AutoARIMAModel

# Automatic ARIMA order selection
auto_arima = AutoARIMAModel(config={
    'seasonal': True,
    'm': 12,  # Seasonal period
    'max_p': 5,
    'max_q': 5,
    'max_d': 2,
    'information_criterion': 'aic',
    'stepwise': True  # Faster search
})

auto_arima.fit(X_train, y_train)

# Get selected order
params = auto_arima.get_params()
print(f"Selected order: {params['order']}")
```

### Learning Rate Scheduling

```python
import torch.optim as optim

# In your training loop
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Reduce LR on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# After each epoch
scheduler.step(val_loss)
```

---

## Best Practices

### 1. Always Use Validation Set

```python
# Split chronologically
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)

train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]
```

### 2. Cross-Validation for Time Series

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

scores = []
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_val, y_val)
    scores.append(metrics['rmse'])

print(f"Average RMSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

### 3. Early Stopping

```python
best_val_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(max_epochs):
    # Training
    train_loss = train_epoch()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        model.save('best_model.pkl')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### 4. Track Experiments

```python
import json

experiments = []

for config in configs:
    model = LSTMModel(config=config)
    model.fit(X_train, y_train, validation_split=0.15, verbose=False)
    
    metrics = model.evaluate(X_test, y_test)
    
    experiment = {
        'config': config,
        'metrics': metrics,
        'train_time': model.train_time
    }
    experiments.append(experiment)

# Save results
with open('experiments.json', 'w') as f:
    json.dump(experiments, f, indent=2)
```

### 5. Hyperparameter Importance

**Priority Order:**
1. Sequence length
2. Learning rate
3. Hidden size
4. Number of layers
5. Dropout
6. Batch size

**Tuning Time Allocation:**
- 40% on sequence length and learning rate
- 30% on architecture (hidden size, layers)
- 20% on regularization (dropout)
- 10% on other parameters

---

## Quick Reference

### LSTM Quick Tune

```python
# Conservative (safe defaults)
config = {
    'sequence_length': 20,
    'hidden_size': 64,
    'num_layers': 2,
    'learning_rate': 0.001,
    'dropout': 0.2
}

# Aggressive (more capacity)
config = {
    'sequence_length': 50,
    'hidden_size': 128,
    'num_layers': 3,
    'learning_rate': 0.001,
    'dropout': 0.3
}
```

### ARIMA Quick Tune

```python
# Try these common orders
orders = [
    (1, 1, 1),  # Simple ARIMA
    (2, 1, 2),  # More complex
    (1, 1, 2),  # MA focused
    (2, 1, 1),  # AR focused
]
```

---

## Contact

For questions or support:
- Email: ajsinha@gmail.com

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | Email: ajsinha@gmail.com**
