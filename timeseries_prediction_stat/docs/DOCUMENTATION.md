# Statistical Time Series Prediction - Complete Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Model Overview](#model-overview)
5. [Theoretical Background](#theoretical-background)
6. [Model Selection Guide](#model-selection-guide)
7. [Detailed Examples](#detailed-examples)
8. [API Reference](#api-reference)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

This package provides comprehensive statistical methods for multi-output time series prediction. Unlike neural network approaches, statistical models offer:

- **Interpretability**: Understand relationships between variables
- **Theoretical Guarantees**: Well-established statistical properties
- **Efficiency**: Work well with smaller datasets (50-200 observations)
- **Uncertainty Quantification**: Built-in confidence intervals
- **Causality Testing**: Granger causality and impulse response analysis

### Key Features

✅ **Multi-Output Prediction**: All models predict multiple variables simultaneously  
✅ **10+ Model Types**: VAR, VECM, VARMA, Linear Regression variants, Factor Models  
✅ **Unified Interface**: Same API across all models  
✅ **Comprehensive Diagnostics**: AIC, BIC, residual analysis, Granger causality  
✅ **Production Ready**: Save/load models, evaluation metrics, visualization

---

## 2. Installation

### Requirements

```
Python >= 3.8
numpy >= 1.19.0
pandas >= 1.2.0
statsmodels >= 0.13.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
yfinance >= 0.1.63 (for examples)
```

### Install Dependencies

```bash
pip install numpy pandas statsmodels scikit-learn matplotlib yfinance
```

### Install Package

```bash
cd timeseries_prediction_stat
python setup.py install
```

Or for development:

```bash
pip install -e .
```

---

## 3. Quick Start

### Example: Multi-Output Stock Prediction with VAR

```python
from timeseries_prediction_stat.models.var_models import VARModel
import yfinance as yf

# Download data
data = yf.download(['BMO', 'JPM'], start='2020-01-01', end='2024-01-01')['Close']
returns = data.pct_change().dropna()

# Split data
train_size = int(len(returns) * 0.8)
train_data = returns.iloc[:train_size]
test_data = returns.iloc[train_size:]

# Fit model
model = VARModel()
model.fit(train_data, maxlags=5)

# Forecast
forecast = model.forecast(steps=10)

# Evaluate
metrics = model.evaluate(test_data[:10], forecast)
print(f"RMSE: {metrics['rmse']:.6f}")
print(f"R²: {metrics['r2']:.4f}")

# Granger causality
gc_results = model.granger_causality('BMO', 'JPM')
print(gc_results)
```

---

## 4. Model Overview

### Vector Autoregression (VAR) Models

#### VARModel
**Best for**: 2-10 related time series, understanding interdependencies

```python
from timeseries_prediction_stat.models.var_models import VARModel

model = VARModel()
model.fit(data, maxlags=5, ic='aic')
forecast = model.forecast(steps=10)
```

**Theory**:
```
y_t = c + A_1*y_{t-1} + A_2*y_{t-2} + ... + A_p*y_{t-p} + ε_t
```

**Features**:
- Automatic lag selection (AIC, BIC, HQIC)
- Granger causality testing
- Impulse response functions
- Forecast error variance decomposition

#### VECMModel
**Best for**: Cointegrated time series with long-run equilibrium

```python
from timeseries_prediction_stat.models.var_models import VECMModel

model = VECMModel()
model.fit(data, coint_rank=1, k_ar_diff=2)
forecast = model.forecast(steps=10)
```

**Theory**: Error correction form of VAR for cointegrated series
```
Δy_t = Πy_{t-1} + Γ_1Δy_{t-1} + ... + ε_t
```

**Use when**: Variables have long-run relationship (e.g., spot/futures prices)

#### VARMAModel
**Best for**: Complex temporal dependencies

```python
from timeseries_prediction_stat.models.var_models import VARMAModel

model = VARMAModel()
model.fit(data, order=(2, 1))  # VARMA(2,1)
forecast = model.forecast(steps=10)
```

**Theory**: Adds moving average components to VAR
```
y_t = c + A_1*y_{t-1} + ... + B_1*ε_{t-1} + ... + ε_t
```

---

### Linear Regression Models

#### MultiOutputLinearRegression
**Best for**: Linear relationships, baseline comparisons

```python
from timeseries_prediction_stat.models.regression_models import MultiOutputLinearRegression

model = MultiOutputLinearRegression(lags=10, normalize=True)
model.fit(data)
forecast = model.forecast(steps=10)
```

**Features**:
- Uses lagged observations as features
- Fast training
- Interpretable coefficients

#### RidgeTimeSeriesRegression
**Best for**: Correlated features, preventing overfitting

```python
from timeseries_prediction_stat.models.regression_models import RidgeTimeSeriesRegression

model = RidgeTimeSeriesRegression(lags=20, alpha=1.0)
model.fit(data)
```

**Theory**: L2 regularization
```
min ||y - Xβ||² + α||β||²
```

#### LassoTimeSeriesRegression
**Best for**: Feature selection, identifying important lags

```python
from timeseries_prediction_stat.models.regression_models import LassoTimeSeriesRegression

model = LassoTimeSeriesRegression(lags=20, alpha=0.1)
model.fit(data)

# See which lags were selected
selected = model.get_nonzero_coefficients()
```

**Theory**: L1 regularization (drives some coefficients to zero)
```
min ||y - Xβ||² + α||β||₁
```

#### ElasticNetTimeSeriesRegression
**Best for**: Balance between Ridge and Lasso

```python
from timeseries_prediction_stat.models.regression_models import ElasticNetTimeSeriesRegression

model = ElasticNetTimeSeriesRegression(lags=15, alpha=0.5, l1_ratio=0.5)
```

**Theory**: L1 + L2 regularization
```
min ||y - Xβ||² + α₁||β||₁ + α₂||β||²
```

#### BayesianLinearRegression
**Best for**: Uncertainty quantification

```python
from timeseries_prediction_stat.models.regression_models import BayesianLinearRegression

model = BayesianLinearRegression(lags=5)
model.fit(data)

# Get predictions with uncertainty
forecast_mean, forecast_std = model.forecast_with_uncertainty(steps=10)
```

**Features**:
- Probabilistic predictions
- Confidence intervals
- Automatic regularization

#### RobustLinearRegression
**Best for**: Data with outliers

```python
from timeseries_prediction_stat.models.regression_models import RobustLinearRegression

model = RobustLinearRegression(lags=5, epsilon=1.35)
```

**Theory**: Uses Huber loss (less sensitive to outliers)

---

### Dynamic Factor Models

#### DynamicFactorModel
**Best for**: Many time series (30-100+), dimension reduction

```python
from timeseries_prediction_stat.models.factor_models import DynamicFactorModel

model = DynamicFactorModel(k_factors=5, factor_order=2)
model.fit(data)  # data with 30 variables

# Extract factors
factors = model.get_factors()
loadings = model.get_factor_loadings()

# Forecast all 30 variables
forecast = model.forecast(steps=10)
```

**Theory**: Latent factor model
```
y_t = Λf_t + ε_t    (observation equation)
f_t = Af_{t-1} + η_t (state equation)
```

**Features**:
- Reduces dimensionality
- Extracts common factors
- Variance explained analysis
- Factor loadings interpretation

#### PCABasedForecaster
**Best for**: Quick dimension reduction, exploratory analysis

```python
from timeseries_prediction_stat.models.factor_models import PCABasedForecaster

model = PCABasedForecaster(n_components=5, ar_order=2)
model.fit(data)

# Get components
components = model.get_components()
loadings = model.get_loadings()
```

**Simpler alternative** to DFM using PCA + AR models

---

## 5. Theoretical Background

### Why Statistical Models?

| Aspect | Statistical | Neural Networks |
|--------|------------|-----------------|
| **Data needed** | 50-200 samples | 1000+ samples |
| **Training time** | Seconds | Minutes-Hours |
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Theory** | Well-established | Empirical |
| **Uncertainty** | Built-in | Requires special methods |
| **Causality** | Can test | Cannot test |

### Vector Autoregression (VAR)

VAR is the foundation for multivariate time series analysis:

**Model**:
```
y_t = c + A_1*y_{t-1} + ... + A_p*y_{t-p} + ε_t
```

Where:
- `y_t`: Vector of k variables at time t
- `A_i`: k×k coefficient matrices
- `ε_t`: White noise error

**Key Properties**:
1. **Stationarity**: Requires stationary data (use differencing if needed)
2. **Lag Selection**: Use information criteria (AIC, BIC)
3. **Residuals**: Should be white noise

**Granger Causality**: X Granger-causes Y if past values of X help predict Y

### Cointegration (VECM)

When variables share common stochastic trends:

**Johansen Test**: Tests number of cointegrating relationships

**VECM Form**:
```
Δy_t = Πy_{t-1} + Γ_1Δy_{t-1} + ... + ε_t
```

Where Π = αβ' contains cointegrating relationships

**Use case**: Pairs trading in finance

### Dynamic Factor Models

Extract common factors from many series:

**Two-step process**:
1. Observation equation: Links observed data to latent factors
2. State equation: Dynamics of latent factors

**Advantages**:
- Reduces curse of dimensionality
- Identifies common drivers
- More efficient forecasting

### Regularized Regression

**Ridge** (L2): Shrinks all coefficients
- Good when all features matter
- Handles multicollinearity

**Lasso** (L1): Sets some coefficients to zero
- Performs feature selection
- Sparse models

**Elastic Net**: Combines both
- Balanced approach
- Groups correlated features

---

## 6. Model Selection Guide

### Decision Tree

```
├─ Do you have cointegrated series?
│  └─ YES → Use VECM
│  └─ NO  → Continue
│
├─ How many variables?
│  ├─ 2-10 variables
│  │  ├─ Need interpretability? → VAR
│  │  ├─ Complex dependencies? → VARMA
│  │  └─ Simple baseline? → Linear Regression
│  │
│  └─ 30-100+ variables → Dynamic Factor Model
│
├─ Do you have outliers?
│  └─ YES → Robust Regression
│
├─ Need feature selection?
│  └─ YES → Lasso
│
├─ Need uncertainty estimates?
│  └─ YES → Bayesian Ridge
│
└─ Want to test causality?
   └─ YES → VAR (with Granger tests)
```

### By Use Case

**Stock Portfolio Forecasting**:
- Primary: VAR or DFM (if many stocks)
- Alternative: Ridge Regression
- Why: Captures cross-correlations

**Pairs Trading**:
- Primary: VECM
- Why: Exploits cointegration

**Economic Indicators**:
- Primary: VAR with Granger tests
- Alternative: DFM
- Why: Test relationships, identify factors

**Risk Management**:
- Primary: Bayesian Ridge
- Why: Uncertainty quantification

**High-dimensional data**:
- Primary: DFM or Lasso
- Why: Dimension reduction / feature selection

---

## 7. Detailed Examples

### Example 1: VAR for Stock Prediction

```python
import yfinance as yf
from timeseries_prediction_stat.models.var_models import VARModel

# Download data
tickers = ['BMO', 'JPM', 'GS']
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Close']
returns = data.pct_change().dropna()

# Split
train = returns.iloc[:int(len(returns)*0.8)]
test = returns.iloc[int(len(returns)*0.8):]

# Fit VAR
model = VARModel()
model.fit(train, maxlags=10, ic='aic')

print(f"Selected lag order: {model.maxlags}")

# Diagnostics
diag = model.get_diagnostics()
print(f"AIC: {diag['aic']:.2f}")
print(f"BIC: {diag['bic']:.2f}")

# Forecast
forecast = model.forecast(steps=20)

# Evaluate
metrics = model.evaluate(test[:20], forecast)
print(f"RMSE: {metrics['rmse']:.6f}")
print(f"R²: {metrics['r2']:.4f}")

# Granger causality
gc = model.granger_causality('BMO', 'JPM', maxlag=5)
for lag, result in gc.items():
    print(f"{lag}: p={result['p_value']:.4f}, significant={result['significant']}")

# Impulse response
irf = model.impulse_response(periods=10, impulse='JPM', response='BMO')
print(irf)
```

### Example 2: Dimension Reduction with DFM

```python
from timeseries_prediction_stat.models.factor_models import DynamicFactorModel

# DOW30 data (30 stocks)
data = download_dow30_returns()  # 30 columns

# Reduce to 5 factors
model = DynamicFactorModel(k_factors=5, factor_order=2)
model.fit(data)

# Extract factors
factors = model.get_factors()  # 5 common market factors
loadings = model.get_factor_loadings()  # How each stock loads on factors

# Top stocks for Factor 1
top_stocks = loadings['factor_1'].abs().nlargest(5)
print("Stocks most correlated with Factor 1:")
print(top_stocks)

# Variance explained
var_exp = model.explained_variance_ratio()
print(f"Factor 1 explains {var_exp[0]*100:.1f}% of variance")

# Forecast all 30 stocks
forecast = model.forecast(steps=10)  # Still 30 columns
```

### Example 3: Feature Selection with Lasso

```python
from timeseries_prediction_stat.models.regression_models import LassoTimeSeriesRegression

# Use many lags
model = LassoTimeSeriesRegression(lags=20, alpha=0.01)
model.fit(data)

# Which lags are important?
selected = model.get_nonzero_coefficients()

for var, lags in selected.items():
    print(f"{var}: {len(lags)} important lags")
    print(f"   Selected lags: {lags[:5]}...")  # First 5
```

### Example 4: Uncertainty Quantification

```python
from timeseries_prediction_stat.models.regression_models import BayesianLinearRegression

model = BayesianLinearRegression(lags=5)
model.fit(train_data)

# Get predictions with uncertainty
forecast_mean, forecast_std = model.forecast_with_uncertainty(steps=10)

# 95% confidence intervals
for i in range(len(forecast_mean)):
    for col in forecast_mean.columns:
        mean = forecast_mean[col].iloc[i]
        std = forecast_std[f'{col}_std'].iloc[i]
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        print(f"{col} step {i+1}: [{lower:.4f}, {upper:.4f}]")
```

---

## 8. API Reference

### Base Class: StatisticalTimeSeriesModel

All models inherit from this base class:

```python
class StatisticalTimeSeriesModel(ABC):
    def fit(self, data, **kwargs) -> self
    def forecast(self, steps: int, **kwargs) -> pd.DataFrame
    def predict(self, steps: int, **kwargs) -> pd.DataFrame  # Alias for forecast
    def evaluate(self, actual, forecast, metrics) -> Dict
    def get_diagnostics() -> Dict
    def get_residuals() -> pd.DataFrame
    def save(filepath: str)
    @classmethod
    def load(filepath: str) -> Model
    def summary()
```

### Common Parameters

**All Models**:
- `name`: Model name (optional)

**VAR Models**:
- `maxlags`: Maximum lag order
- `ic`: Information criterion ('aic', 'bic', 'hqic')
- `trend`: Trend specification

**Regression Models**:
- `lags`: Number of lagged observations
- `normalize`: Whether to normalize features
- `alpha`: Regularization strength

**Factor Models**:
- `k_factors`: Number of latent factors
- `factor_order`: AR order of factors

### Evaluation Metrics

Available metrics:
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `mae`: Mean Absolute Error
- `mape`: Mean Absolute Percentage Error
- `r2`: R-squared score

```python
metrics = model.evaluate(actual, forecast, metrics=['rmse', 'mae', 'r2'])
```

---

## 9. Best Practices

### Data Preparation

1. **Check Stationarity**
```python
from statsmodels.tsa.stattools import adfuller

for col in data.columns:
    result = adfuller(data[col])
    print(f"{col}: p-value = {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"   WARNING: {col} may be non-stationary!")
```

2. **Handle Missing Values**
```python
# Forward fill
data = data.fillna(method='ffill')

# Or interpolate
data = data.interpolate(method='linear')
```

3. **Remove Outliers**
```python
# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(data))
data = data[(z_scores < 3).all(axis=1)]
```

### Model Selection

1. **Start Simple**: Begin with linear regression or VAR
2. **Check Diagnostics**: Look at AIC, BIC, residuals
3. **Test Assumptions**: Residuals should be white noise
4. **Cross-Validate**: Use time-series aware splits

### Hyperparameter Tuning

**Lag Selection** (VAR):
```python
# Try multiple lag orders
best_aic = float('inf')
best_lag = None

for lag in range(1, 11):
    model = VARModel()
    model.fit(data, maxlags=lag, ic='aic')
    if model.diagnostics['aic'] < best_aic:
        best_aic = model.diagnostics['aic']
        best_lag = lag

print(f"Best lag: {best_lag}")
```

**Regularization** (Ridge/Lasso):
```python
from sklearn.model_selection import TimeSeriesSplit

alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
best_alpha = None
best_score = float('inf')

for alpha in alphas:
    model = RidgeTimeSeriesRegression(lags=10, alpha=alpha)
    # Time series cross-validation
    # ... (implement CV loop)
```

### Diagnostics

**Check Residuals**:
```python
residuals = model.get_residuals()

# Should be centered at zero
print(residuals.mean())

# Should have no autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
result = acorr_ljungbox(residuals['BMO'], lags=10)
print(result)
```

---

## 10. Troubleshooting

### Common Issues

**1. Non-Convergence**

*Problem*: Model fit fails to converge

*Solutions*:
- Increase `maxiter` parameter
- Scale/normalize data
- Remove outliers
- Try simpler model (fewer lags)

**2. Non-Stationary Data**

*Problem*: Variables have unit roots

*Solutions*:
```python
# Difference the data
data_diff = data.diff().dropna()

# Or use VECM for cointegrated series
model = VECMModel()
```

**3. Poor Forecasts**

*Problem*: High forecast errors

*Solutions*:
- Check if data is stationary
- Try different lag orders
- Consider regularization (Ridge/Lasso)
- Use ensemble of models
- Check for structural breaks

**4. "Singular Matrix" Error**

*Problem*: Perfect collinearity

*Solutions*:
```python
# Use regularization
model = RidgeTimeSeriesRegression(alpha=1.0)

# Or remove highly correlated variables
corr_matrix = data.corr()
# Drop variables with correlation > 0.95
```

**5. Slow Computation**

*Problem*: DFM takes too long

*Solutions*:
```python
# Reduce number of factors
model = DynamicFactorModel(k_factors=3)  # Instead of 10

# Or use PCA-based approach
model = PCABasedForecaster(n_components=5)
```

### Error Messages

**"Model must be fitted first"**
- Call `model.fit()` before `forecast()`

**"Not enough observations"**
- Need at least 50+ observations for VAR
- Use simpler model or get more data

**"Cointegration rank too large"**
- Reduce `coint_rank` in VECM
- Check if variables are truly cointegrated

---

## Comparison with Neural Networks

| When to Use | Statistical Models | Neural Networks |
|-------------|-------------------|-----------------|
| **Sample size** | 50-500 | 1000+ |
| **Interpretability needed** | ✅ | ❌ |
| **Need causality tests** | ✅ | ❌ |
| **Nonlinear relationships** | ❌ | ✅ |
| **Quick prototyping** | ✅ | ❌ |
| **Production deployment** | ✅ Both work | ✅ |
| **Uncertainty quantification** | ✅ Built-in | Requires special methods |

### Hybrid Approach

Combine both for best results:

```python
# 1. Use VAR for linear component
var_model = VARModel()
var_model.fit(data)
var_forecast = var_model.forecast(steps=10)

# 2. Use NN for nonlinear residuals
residuals = actual - var_forecast
nn_model.fit(X, residuals)

# 3. Combine
final_forecast = var_forecast + nn_model.predict(X_future)
```

---

## Conclusion

This package provides production-ready statistical models for multi-output time series prediction. Key advantages:

✅ **Interpretable**: Understand what drives forecasts  
✅ **Efficient**: Works with 50-200 observations  
✅ **Theoretically Sound**: Well-established methods  
✅ **Comprehensive**: 10+ models with unified API  
✅ **Production Ready**: Save/load, diagnostics, evaluation

For questions or issues, check the examples in the `examples/` directory or refer to the source code documentation.

**Happy forecasting!** 📈
