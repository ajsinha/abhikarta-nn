"""
Dynamic Factor Model Example
=============================

Demonstrates dimension reduction and forecasting using Dynamic Factor Models.

Shows how to:
- Extract common factors from many time series
- Reduce dimensionality while preserving information
- Forecast all variables using fewer factors
- Interpret factor loadings

Uses DOW30 stocks to extract market factors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from timeseries_prediction_stat.models.factor_models import DynamicFactorModel, PCABasedForecaster

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance")
    import yfinance as yf


def get_dow30_tickers():
    """Get DOW30 ticker list."""
    return [
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT', 'WBA'
    ]


def main():
    print("=" * 80)
    print("Dynamic Factor Model Example: Dimension Reduction for DOW30")
    print("=" * 80)
    
    # 1. Download DOW30 data
    print("\n[1] Downloading DOW30 stock data...")
    tickers = get_dow30_tickers()
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01', progress=False)['Close']
    
    print(f"   Downloaded {data.shape[1]} stocks, {len(data)} days")
    
    # 2. Calculate returns
    print("\n[2] Calculating returns...")
    returns = data.pct_change().dropna()
    
    # Handle inf and missing values properly
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Check for missing values
    print(f"   Missing values before cleaning: {returns.isna().sum().sum()}")
    
    # Drop columns with too many missing values (>10%)
    missing_pct = returns.isna().sum() / len(returns)
    good_cols = missing_pct[missing_pct < 0.1].index
    returns = returns[good_cols]
    
    print(f"   Kept {len(good_cols)} stocks with <10% missing data")
    
    # Fill remaining missing values
    returns = returns.ffill().bfill().fillna(0)
    
    # Ensure no missing values remain
    assert returns.isna().sum().sum() == 0, "Missing values still present!"
    
    # Set frequency for the date index
    returns.index.freq = 'B'  # Business day frequency
    
    print(f"   Returns shape: {returns.shape}")
    print(f"   {returns.shape[0]} observations × {returns.shape[1]} variables")
    print(f"   Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    # 3. Split data
    train_size = int(len(returns) * 0.8)
    train_data = returns.iloc[:train_size]
    test_data = returns.iloc[train_size:]
    
    print(f"   Train: {len(train_data)} observations")
    print(f"   Test: {len(test_data)} observations")
    
    # 4. Fit Dynamic Factor Model
    print("\n[3] Fitting Dynamic Factor Model...")
    
    # Determine optimal number of factors
    # Maximum factors = min(n_observations, n_variables) - 1
    max_factors = min(len(train_data), len(train_data.columns)) - 1
    n_factors = min(5, max_factors)
    
    print(f"   Data dimensions: {len(train_data)} obs × {len(train_data.columns)} vars")
    print(f"   Maximum possible factors: {max_factors}")
    print(f"   Using {n_factors} factors")
    
    dfm = DynamicFactorModel(k_factors=n_factors, factor_order=2)
    
    try:
        dfm.fit(train_data)
    except Exception as e:
        print(f"   Warning: DFM fitting failed with {n_factors} factors: {e}")
        print(f"   Trying with {n_factors-1} factors...")
        n_factors -= 1
        dfm = DynamicFactorModel(k_factors=n_factors, factor_order=2)
        dfm.fit(train_data)
    
    dfm.summary()
    
    # 5. Extract factors
    print("\n[4] Extracting latent factors...")
    factors = dfm.get_factors()
    print(f"   Factor shape: {factors.shape}")
    print(f"\n   Factor statistics:")
    print(factors.describe())
    
    # 6. Get factor loadings
    print("\n[5] Analyzing factor loadings...")
    loadings = dfm.get_factor_loadings()
    
    # Find stocks most correlated with each factor
    print(f"\n   Top 5 stocks for each of the {n_factors} factors:")
    for factor_col in loadings.columns:
        top_stocks = loadings[factor_col].abs().nlargest(5)
        print(f"\n   {factor_col}:")
        for stock, loading in top_stocks.items():
            print(f"      {stock}: {loading:.4f}")
    
    # 7. Variance explained
    print("\n[6] Variance explained by factors...")
    var_explained = dfm.explained_variance_ratio()
    cumulative_var = np.cumsum(var_explained)
    
    print("   Factor | Variance | Cumulative")
    print("   " + "-" * 35)
    for i, (var, cum) in enumerate(zip(var_explained, cumulative_var)):
        print(f"   Factor {i+1} | {var*100:>6.2f}% | {cum*100:>6.2f}%")
    
    # 8. Forecast
    print("\n[7] Generating forecasts...")
    forecast_steps = 20
    forecast = dfm.forecast(steps=forecast_steps)
    
    print(f"   Forecast shape: {forecast.shape}")
    
    # 9. Evaluate
    print("\n[8] Evaluating forecasts...")
    actual = test_data.iloc[:forecast_steps]
    metrics = dfm.evaluate(actual, forecast)
    
    print(f"   Overall RMSE: {metrics['rmse']:.6f}")
    print(f"   Overall R²:   {metrics['r2']:.4f}")
    
    # Per-stock metrics for selected stocks
    selected_stocks = ['AAPL', 'JPM', 'MSFT', 'GS', 'DIS']
    print(f"\n   Selected stocks performance:")
    for stock in selected_stocks:
        if stock in actual.columns and stock in forecast.columns:
            stock_actual = actual[stock].values
            stock_forecast = forecast[stock].values
            
            rmse = np.sqrt(np.mean((stock_actual - stock_forecast) ** 2))
            print(f"      {stock}: RMSE = {rmse:.6f}")
    
    # 10. Compare with PCA-based forecaster
    print("\n[9] Comparing with PCA-based forecaster...")
    pca_model = PCABasedForecaster(n_components=n_factors, ar_order=2)
    pca_model.fit(train_data)
    pca_forecast = pca_model.forecast(steps=forecast_steps)
    
    pca_metrics = pca_model.evaluate(actual, pca_forecast)
    
    print(f"\n   DFM  RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.4f}")
    print(f"   PCA  RMSE: {pca_metrics['rmse']:.6f}, R²: {pca_metrics['r2']:.4f}")
    
    # 11. Visualize
    print("\n[10] Creating visualizations...")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Dynamic Factor Model: DOW30 Dimension Reduction', fontsize=16, fontweight='bold')
    
    # Plot 1: Extracted factors
    ax = fig.add_subplot(gs[0, :])
    for i in range(n_factors):
        ax.plot(factors.index, factors.iloc[:, i], label=f'Factor {i+1}', alpha=0.7)
    ax.set_title(f'Extracted {n_factors} Latent Factors')
    ax.set_xlabel('Date')
    ax.set_ylabel('Factor Value')
    ax.legend(loc='best', ncol=min(5, n_factors))
    ax.grid(True, alpha=0.3)
    
    # Plot 2-4: Predictions for selected stocks
    for i, stock in enumerate(['AAPL', 'JPM', 'MSFT']):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(actual.index, actual[stock], 'k-', label='Actual', linewidth=2, alpha=0.8)
        ax.plot(actual.index, forecast[stock], 'b--', label='DFM', linewidth=2, alpha=0.8)
        ax.plot(actual.index, pca_forecast[stock], 'r:', label='PCA', linewidth=2, alpha=0.8)
        ax.set_title(f'{stock} Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Variance explained
    ax = fig.add_subplot(gs[2, 0])
    factors_range = range(1, n_factors + 1)
    ax.bar(factors_range, var_explained * 100, color='skyblue', edgecolor='navy')
    ax.plot(factors_range, cumulative_var * 100, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Factor')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Variance Explained by Each Factor')
    ax.set_xticks(factors_range)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(['Cumulative', 'Individual'])
    
    # Plot 6: Factor loadings heatmap
    ax = fig.add_subplot(gs[2, 1:])
    im = ax.imshow(loadings.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_yticks(range(n_factors))
    ax.set_yticklabels([f'Factor {i+1}' for i in range(n_factors)])
    ax.set_xticks(range(len(loadings.index)))
    ax.set_xticklabels(loadings.index, rotation=90, fontsize=7)
    ax.set_title('Factor Loadings Heatmap')
    plt.colorbar(im, ax=ax, label='Loading')
    
    # Plot 7-8: Model comparison
    ax = fig.add_subplot(gs[3, 0])
    models = ['DFM', 'PCA']
    rmse_vals = [metrics['rmse'], pca_metrics['rmse']]
    bars = ax.bar(models, rmse_vals, color=['#1f77b4', '#ff7f0e'], edgecolor='black')
    ax.set_ylabel('RMSE')
    ax.set_title('Model Comparison: RMSE')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, rmse_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom')
    
    ax = fig.add_subplot(gs[3, 1])
    r2_vals = [metrics['r2'], pca_metrics['r2']]
    bars = ax.bar(models, r2_vals, color=['#1f77b4', '#ff7f0e'], edgecolor='black')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Comparison: R²')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, r2_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 9: Top factor loadings
    ax = fig.add_subplot(gs[3, 2])
    top_loadings = loadings['factor_1'].abs().nlargest(10)
    ax.barh(range(len(top_loadings)), top_loadings.values, color='steelblue')
    ax.set_yticks(range(len(top_loadings)))
    ax.set_yticklabels(top_loadings.index)
    ax.set_xlabel('|Loading|')
    ax.set_title('Top 10 Loadings on Factor 1')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'dfm_example.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   Plot saved to: {plot_path}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("Dynamic Factor Model Example Complete!")
    print("=" * 80)
    print(f"\nKey Results:")
    print(f"  ✓ Reduced {len(train_data.columns)} variables to {n_factors} factors")
    print(f"  ✓ Variance explained: {cumulative_var[-1]*100:.2f}%")
    print(f"  ✓ DFM Test RMSE: {metrics['rmse']:.6f}")
    print(f"  ✓ DFM Test R²: {metrics['r2']:.4f}")
    print(f"  ✓ PCA Test RMSE: {pca_metrics['rmse']:.6f}")
    print(f"  ✓ PCA Test R²: {pca_metrics['r2']:.4f}")


if __name__ == '__main__':
    main()
