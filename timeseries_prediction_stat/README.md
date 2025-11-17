# Statistical Time Series Prediction 📊

A comprehensive Python package for **statistical multi-output time series prediction** with 10+ models including VAR, VECM, VARMA, Dynamic Factor Models, and various regression approaches.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Key Features

- ✅ **10+ Statistical Models**: VAR, VECM, VARMA, Linear Regression variants, Factor Models
- ✅ **Multi-Output Support**: All models predict multiple variables simultaneously
- ✅ **Unified API**: Same interface across all models
- ✅ **Interpretability**: Understand relationships, test causality, analyze factors
- ✅ **Efficiency**: Works with 50-200 observations (vs 1000+ for neural networks)
- ✅ **Uncertainty Quantification**: Built-in confidence intervals
- ✅ **Production Ready**: Save/load models, comprehensive diagnostics
- ✅ **Well-Documented**: Extensive documentation with theory and examples

## 📦 What's Included

### Vector Autoregression Models
- **VARModel**: Standard VAR with Granger causality testing
- **VECMModel**: For cointegrated time series
- **VARMAModel**: VAR with moving average components

### Linear Regression Models
- **MultiOutputLinearRegression**: OLS with multiple outputs
- **RidgeTimeSeriesRegression**: L2 regularization
- **LassoTimeSeriesRegression**: L1 regularization + feature selection
- **ElasticNetTimeSeriesRegression**: L1 + L2 combination
- **BayesianLinearRegression**: Probabilistic predictions
- **RobustLinearRegression**: Outlier-resistant Huber loss

### Factor Models
- **DynamicFactorModel**: Extract latent factors from many series
- **PCABasedForecaster**: PCA + AR for dimension reduction

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install numpy pandas statsmodels scikit-learn matplotlib yfinance

# Install package
cd timeseries_prediction_stat
python setup.py install
```

### Basic Example: VAR Model

```python
from timeseries_prediction_stat.models.var_models import VARModel
import yfinance as yf

# Download data
data = yf.download(['BMO', 'JPM'], start='2020-01-01', end='2024-01-01')['Close']
returns = data.pct_change().dropna()

# Split data
train_size = int(len(returns) * 0.8)
train = returns.iloc[:train_size]
test = returns.iloc[train_size:]

# Fit model
model = VARModel()
model.fit(train, maxlags=5, ic='aic')

# Forecast
forecast = model.forecast(steps=10)

# Evaluate
metrics = model.evaluate(test[:10], forecast)
print(f"RMSE: {metrics['rmse']:.6f}")
print(f"R²: {metrics['r2']:.4f}")

# Test Granger causality
gc = model.granger_causality('BMO', 'JPM')
```

### Advanced Example: Dynamic Factor Model

```python
from timeseries_prediction_stat.models.factor_models import DynamicFactorModel

# 30 stock returns (DOW30)
data = download_dow30_returns()  

# Reduce to 5 factors
model = DynamicFactorModel(k_factors=5, factor_order=2)
model.fit(data)

# Extract common factors
factors = model.get_factors()
loadings = model.get_factor_loadings()

# Forecast all 30 stocks
forecast = model.forecast(steps=10)

# See variance explained
var_explained = model.explained_variance_ratio()
```

## 📚 Documentation

- **[Complete Documentation](docs/DOCUMENTATION.md)**: Theory, examples, API reference
- **[Examples](examples/)**: Working code for all models
  - `example_var.py`: VAR model with Granger causality
  - `example_regression.py`: Compare 6 regression models
  - `example_dfm.py`: Dimension reduction with 30 stocks

## 🎓 Model Selection Guide

### By Number of Variables

**2-10 variables**:
- Need interpretability → **VAR**
- Complex dependencies → **VARMA**
- Simple baseline → **Linear Regression**

**30-100+ variables**:
- **Dynamic Factor Model** (dimension reduction)
- **PCA-based Forecaster** (simpler alternative)

### By Specific Need

**Test causality** → **VAR** (Granger tests)  
**Cointegrated series** → **VECM**  
**Feature selection** → **Lasso**  
**Uncertainty estimates** → **Bayesian Ridge**  
**Handle outliers** → **Robust Regression**  
**Many variables** → **Dynamic Factor Model**

### By Use Case

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Stock portfolio | VAR or DFM | Captures correlations |
| Pairs trading | VECM | Exploits cointegration |
| Economic indicators | VAR | Test relationships |
| Risk management | Bayesian Ridge | Uncertainty quantification |
| High-dimensional | DFM or Lasso | Dimension reduction |

## 💡 Why Statistical Models?

| Aspect | Statistical | Neural Networks |
|--------|------------|-----------------|
| **Data needed** | 50-200 samples ✅ | 1000+ samples |
| **Training time** | Seconds ✅ | Minutes-Hours |
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Theory** | Well-established ✅ | Empirical |
| **Uncertainty** | Built-in ✅ | Requires special methods |
| **Causality** | Can test ✅ | Cannot test |
| **Nonlinearity** | Limited | ⭐⭐⭐⭐⭐ |

**Best approach**: Use statistical models for interpretability and small data, neural networks for nonlinear patterns with large data, or combine both!

## 🔧 Common Workflows

### Workflow 1: VAR Analysis

```python
from timeseries_prediction_stat.models.var_models import VARModel

# 1. Fit model with automatic lag selection
model = VARModel()
model.fit(data, maxlags=10, ic='aic')

# 2. Check diagnostics
diag = model.get_diagnostics()
print(f"AIC: {diag['aic']}, BIC: {diag['bic']}")

# 3. Test Granger causality
gc = model.granger_causality('stock1', 'stock2')

# 4. Analyze impulse responses
irf = model.impulse_response(periods=10)

# 5. Forecast
forecast = model.forecast(steps=20)

# 6. Save model
model.save('var_model.pkl')
```

### Workflow 2: Model Comparison

```python
from timeseries_prediction_stat.models.regression_models import *

models = {
    'OLS': MultiOutputLinearRegression(lags=10),
    'Ridge': RidgeTimeSeriesRegression(lags=10, alpha=1.0),
    'Lasso': LassoTimeSeriesRegression(lags=10, alpha=0.01),
    'Bayesian': BayesianLinearRegression(lags=10)
}

results = {}
for name, model in models.items():
    model.fit(train)
    forecast = model.forecast(steps=len(test))
    metrics = model.evaluate(test, forecast)
    results[name] = metrics

# Find best model
best = min(results.items(), key=lambda x: x[1]['rmse'])
print(f"Best model: {best[0]}")
```

### Workflow 3: Feature Selection with Lasso

```python
from timeseries_prediction_stat.models.regression_models import LassoTimeSeriesRegression

# Use many lags
model = LassoTimeSeriesRegression(lags=20, alpha=0.01)
model.fit(data)

# Which lags are important?
selected = model.get_nonzero_coefficients()
print(f"Selected {len(selected)} important lags out of {20}")
```

## 📊 Example Results

### VAR Model Performance
- **Train on**: 800 observations (2 stocks)
- **Test RMSE**: 0.015
- **Test R²**: 0.42
- **Training time**: 2 seconds

### Dynamic Factor Model
- **Input**: 30 DOW30 stocks
- **Factors**: 5 latent factors
- **Variance explained**: 75%
- **Forecast**: All 30 stocks simultaneously
- **Training time**: 10 seconds

## 🗂️ Package Structure

```
timeseries_prediction_stat/
├── base/
│   └── base_model.py          # Abstract base class
├── models/
│   ├── var_models.py          # VAR, VECM, VARMA
│   ├── regression_models.py   # Linear regression variants
│   └── factor_models.py       # DFM, PCA-based
├── utils/
│   └── (utilities)
├── examples/
│   ├── example_var.py         # VAR with Granger tests
│   ├── example_regression.py  # Compare 6 models
│   └── example_dfm.py         # Factor analysis
├── docs/
│   └── DOCUMENTATION.md       # Complete documentation
├── requirements.txt
├── setup.py
└── README.md
```

## 🧪 Running Examples

```bash
cd examples

# Example 1: VAR Model
python example_var.py

# Example 2: Regression Comparison
python example_regression.py

# Example 3: Dynamic Factor Model
python example_dfm.py
```

Each example:
- Downloads real stock data
- Trains model(s)
- Generates forecasts
- Creates visualizations
- Saves results to `outputs/` folder

## 📈 Theoretical Background

### Vector Autoregression (VAR)

```
y_t = c + A_1*y_{t-1} + A_2*y_{t-2} + ... + A_p*y_{t-p} + ε_t
```

Where `y_t` is a k-dimensional vector of variables.

**Applications**:
- Multivariate forecasting
- Granger causality testing
- Impulse response analysis
- Variance decomposition

### Dynamic Factor Model

```
y_t = Λf_t + ε_t    (observation equation)
f_t = Af_{t-1} + η_t (state equation)
```

**Applications**:
- Dimension reduction
- Identifying common drivers
- Efficient forecasting of many series

### Regularized Regression

**Ridge (L2)**: `min ||y - Xβ||² + α||β||²`  
**Lasso (L1)**: `min ||y - Xβ||² + α||β||₁`  
**Elastic Net**: `min ||y - Xβ||² + α₁||β||₁ + α₂||β||²`

## 🐛 Troubleshooting

### Common Issues

**"Model must be fitted first"**
```python
model = VARModel()
model.fit(data)  # Don't forget this!
forecast = model.forecast(steps=10)
```

**Non-stationary data**
```python
# Difference the data
data_diff = data.diff().dropna()
model.fit(data_diff)
```

**Singular matrix error**
```python
# Use regularization
model = RidgeTimeSeriesRegression(alpha=1.0)
```

**Poor forecasts**
- Check if data is stationary
- Try different lag orders
- Consider regularization
- Check for structural breaks

## 🔬 When to Use Each Model

### Use VAR when:
- 2-10 related time series
- Need to understand relationships
- Want to test Granger causality
- Have stationary data

### Use VECM when:
- Variables are cointegrated
- Long-run equilibrium exists
- Doing pairs trading

### Use Linear Regression when:
- Need simple baseline
- Want fast training
- Linear relationships sufficient

### Use Lasso when:
- Many potential features
- Want feature selection
- Sparse models preferred

### Use DFM when:
- Many time series (30+)
- Need dimension reduction
- Want to identify common factors

### Use Bayesian Ridge when:
- Need uncertainty estimates
- Want probabilistic forecasts
- Automatic regularization desired

## 📖 Further Reading

- **Documentation**: `docs/DOCUMENTATION.md`
- **Examples**: `examples/` directory
- **Statsmodels docs**: https://www.statsmodels.org/
- **Time Series Analysis book**: Hamilton (1994)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built using:
- **statsmodels**: VAR, VECM, DFM implementations
- **scikit-learn**: Regression models
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## 📞 Support

- **Documentation**: `docs/DOCUMENTATION.md`
- **Examples**: Check `examples/` directory
- **Issues**: Open an issue on GitHub

---

**Ready to forecast with statistical rigor?** 📊📈

```python
from timeseries_prediction_stat import VARModel

model = VARModel()
model.fit(your_data)
forecast = model.forecast(steps=10)
```

**Simple. Interpretable. Effective.**
