# VAR Model Documentation

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | Email: ajsinha@gmail.com**

## Overview

VAR (Vector AutoRegression) is a multivariate extension of AR models, capturing linear interdependencies among multiple time series.

## Mathematical Formulation

```
y_t = c + A₁y_{t-1} + A₂y_{t-2} + ... + Aₚy_{t-p} + ε_t
```

Where:
- y_t is a vector of k time series
- Aᵢ are k×k coefficient matrices
- p is the lag order

## When to Use

- Multiple related time series
- Linear relationships between series
- Granger causality analysis
- Impulse response analysis
- Forecast error variance decomposition

## Configuration

```python
config = {
    'maxlags': 5,    # Maximum lag order
    'ic': 'aic'      # Information criterion
}
```

## Model Selection

The optimal lag order is selected using:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- HQ (Hannan-Quinn Criterion)

## Best Practices

1. Ensure all series are stationary
2. Check for cointegration
3. Start with smaller lag orders
4. Validate Granger causality
5. Check residuals for autocorrelation

---
For more information: ajsinha@gmail.com
