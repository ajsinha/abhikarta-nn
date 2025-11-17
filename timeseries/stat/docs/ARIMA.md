# ARIMA Model Documentation

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | Email: ajsinha@gmail.com**

---

## Overview

ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical model for analyzing and forecasting time series data. It combines three components: autoregression (AR), differencing (I for integration), and moving average (MA) to model various temporal patterns.

## Model Components

### 1. AutoRegressive (AR) Component - p
Models the relationship between an observation and a number of lagged observations.
```
y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t
```

### 2. Integrated (I) Component - d
Degree of differencing needed to make the series stationary.
```
Δy_t = y_t - y_{t-1}  (first difference)
Δ²y_t = Δy_t - Δy_{t-1}  (second difference)
```

### 3. Moving Average (MA) Component - q
Models the relationship between an observation and residual errors from past forecasts.
```
y_t = μ + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θₑε_{t-q} + ε_t
```

### Combined ARIMA(p,d,q)
```
Δᵈy_t = c + φ₁Δᵈy_{t-1} + ... + φₚΔᵈy_{t-p} + θ₁ε_{t-1} + ... + θₑε_{t-q} + ε_t
```

## When to Use ARIMA

### Ideal Use Cases
- Univariate time series
- Data with trend and/or seasonality
- Medium-term forecasting (days to months)
- Economic and financial data
- Sales forecasting
- Demand planning
- Stationary or near-stationary data

### Not Recommended For
- Multivariate time series (use VAR instead)
- Non-linear patterns (use neural networks)
- Long-term forecasting with structural breaks
- Very short time series (< 50 observations)
- Data with multiple seasonal patterns (use SARIMA)

## Model Selection

### Determining Order (p, d, q)

#### 1. Differencing (d)
**Check for stationarity:**
- Visual inspection of time series plot
- Augmented Dickey-Fuller (ADF) test
- KPSS test

**Guidelines:**
- d=0: Series is already stationary
- d=1: First difference makes it stationary (most common)
- d=2: Second difference needed (rare)
- d>2: Rarely needed, consider other models

#### 2. AR Order (p)
**Use Partial Autocorrelation Function (PACF):**
- Sharp cutoff at lag k → AR(p=k)
- Gradual decay → High AR order or MA component

**Guidelines:**
- p=0: No autoregressive component
- p=1-2: Most common for business/economic data
- p>5: Usually indicates overfitting

#### 3. MA Order (q)
**Use Autocorrelation Function (ACF):**
- Sharp cutoff at lag k → MA(q=k)
- Gradual decay → Low MA order or AR component

**Guidelines:**
- q=0: No moving average component
- q=1-2: Most common
- q>5: Usually indicates overfitting

### Common ARIMA Models

```
ARIMA(0,0,0) - White noise
ARIMA(1,0,0) - AR(1) - Random walk with drift
ARIMA(0,1,0) - Random walk
ARIMA(0,1,1) - Exponential smoothing
ARIMA(1,1,0) - Differenced AR(1)
ARIMA(0,1,2) - Double exponential smoothing
ARIMA(1,1,1) - Balanced model (good starting point)
ARIMA(2,1,2) - More complex patterns
```

## Configuration Parameters

```python
config = {
    'order': (p, d, q),     # ARIMA order
    'seasonal': False,      # Use SARIMA instead if True
    'trend': 'c'           # 'n': no trend, 'c': constant, 't': linear, 'ct': both
}
```

### Trend Options
- **'n'**: No constant or trend
- **'c'**: Constant only (default)
- **'t'**: Linear trend only
- **'ct'**: Constant and linear trend

## Usage Examples

### Basic Usage

```python
from timeseries.stat.models.statistical import ARIMAModel
import pandas as pd

# Prepare data
X_train, y_train = ...  # Features and target
X_test, y_test = ...

# Configure ARIMA
config = {
    'order': (1, 1, 1),  # ARIMA(1,1,1)
    'trend': 'c'
}

# Train model
model = ARIMAModel(config=config)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Forecast future values
forecast = model.forecast(steps=10)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Advanced Usage with Model Selection

```python
from timeseries.stat.models.statistical import ARIMAModel
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt

# Check stationarity
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Series is stationary")
        return 0
    else:
        print("Series is not stationary")
        return 1

# Determine d
d = check_stationarity(y_train)

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
acf_vals = acf(y_train.diff(d).dropna(), nlags=20)
pacf_vals = pacf(y_train.diff(d).dropna(), nlags=20)

axes[0].stem(acf_vals)
axes[0].set_title('ACF')
axes[1].stem(pacf_vals)
axes[1].set_title('PACF')

# Try multiple configurations
orders = [
    (1, d, 1),
    (2, d, 1),
    (1, d, 2),
    (2, d, 2)
]

best_aic = float('inf')
best_model = None
best_order = None

for order in orders:
    try:
        model = ARIMAModel(config={'order': order})
        model.fit(X_train, y_train)
        
        params = model.get_params()
        print(f"ARIMA{order} - AIC: {params['aic']:.2f}")
        
        if params['aic'] < best_aic:
            best_aic = params['aic']
            best_model = model
            best_order = order
    except:
        continue

print(f"\nBest model: ARIMA{best_order} with AIC={best_aic:.2f}")
```

### Forecasting with Confidence Intervals

```python
# Train model
model = ARIMAModel(config={'order': (1, 1, 1)})
model.fit(X_train, y_train)

# Get forecast with confidence intervals
# (Note: This requires accessing the underlying statsmodels object)
forecast_obj = model.model_fit.get_forecast(steps=10)
forecast_mean = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_train.values, label='Training Data')
plt.plot(range(len(y_train), len(y_train)+10), forecast_mean, 
         label='Forecast', color='red')
plt.fill_between(range(len(y_train), len(y_train)+10),
                conf_int[:, 0], conf_int[:, 1], 
                alpha=0.3, color='red')
plt.legend()
```

## Model Diagnostics

### 1. Residual Analysis

Good ARIMA model should have residuals that are:
- Normally distributed
- Zero mean
- Constant variance
- No autocorrelation

```python
# After fitting
residuals = model.model_fit.resid

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time series plot
axes[0, 0].plot(residuals)
axes[0, 0].set_title('Residuals over Time')

# Histogram
axes[0, 1].hist(residuals, bins=30)
axes[0, 1].set_title('Residual Distribution')

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, ax=axes[1, 0])

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
```

### 2. Information Criteria

Lower values indicate better model:

```python
params = model.get_params()
print(f"AIC: {params['aic']:.2f}")
print(f"BIC: {params['bic']:.2f}")
```

- **AIC (Akaike Information Criterion)**: Penalizes complexity
- **BIC (Bayesian Information Criterion)**: Penalizes complexity more heavily

## Common Issues and Solutions

### Issue: Non-Stationary Series
**Symptoms**: Poor forecasts, trending residuals

**Solutions**:
- Increase d (differencing)
- Apply log transformation
- Remove trend before modeling
- Use SARIMA for seasonal data

### Issue: Overfitting
**Symptoms**: Great fit but poor forecasts

**Solutions**:
- Reduce p and q orders
- Use information criteria (AIC/BIC)
- Implement cross-validation
- Simplify to lower-order model

### Issue: Seasonal Patterns Not Captured
**Symptoms**: Residuals show seasonal pattern

**Solutions**:
- Use SARIMA instead
- Add seasonal dummy variables
- Pre-process with seasonal decomposition
- Use seasonal differencing

### Issue: Structural Breaks
**Symptoms**: Model works on subset but fails overall

**Solutions**:
- Split data at break points
- Use intervention analysis
- Consider regime-switching models
- Use rolling window approach

## Comparison with Other Models

### ARIMA vs SARIMA
- **ARIMA**: No seasonal component
- **SARIMA**: Handles seasonality explicitly
- **Use ARIMA when**: No clear seasonal patterns

### ARIMA vs ETS
- **ARIMA**: More flexible, handles various patterns
- **ETS**: Simpler, good for seasonal data
- **Use ARIMA when**: Complex patterns, need flexibility

### ARIMA vs VAR
- **ARIMA**: Univariate only
- **VAR**: Multivariate time series
- **Use ARIMA when**: Single variable forecasting

### ARIMA vs Neural Networks
- **ARIMA**: Interpretable, less data needed
- **Neural Networks**: Better for complex non-linear patterns
- **Use ARIMA when**: Limited data, need interpretability

## Best Practices

1. **Check stationarity** before modeling
2. **Difference if needed** but don't over-difference
3. **Start simple** with ARIMA(1,1,1)
4. **Use ACF/PACF** for order selection
5. **Compare models** using AIC/BIC
6. **Validate residuals** for white noise
7. **Test on hold-out set** before deployment
8. **Update regularly** with new data
9. **Document assumptions** about data
10. **Monitor performance** in production

## Performance Optimization

### Speed
- ARIMA is generally fast for small to medium datasets
- Use seasonal differencing instead of high d
- Limit p and q to reasonable values (< 5)
- Pre-compute differences for repeated fitting

### Accuracy
- Ensure stationarity through proper differencing
- Select appropriate p and q using ACF/PACF
- Use exogenous variables when available
- Combine with other models in ensemble
- Regular re-training with new data

## References

1. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.
2. Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts.
3. Brockwell, P. J., & Davis, R. A. (2016). Introduction to time series and forecasting. Springer.

## Example Results

### Stock Returns Forecasting
- **Data**: Daily returns, 1000 observations
- **Model**: ARIMA(1,0,1)
- **Results**: RMSE=0.012, MAPE=8.5%

### Sales Forecasting
- **Data**: Monthly sales, 60 observations  
- **Model**: ARIMA(2,1,2)
- **Results**: RMSE=45.3 units, R²=0.76

---

For more information or support, contact: ajsinha@gmail.com
