"""
VAR Model Example
=================

Demonstrates Vector Autoregression for multi-output time series prediction.

Predicts BMO and JPM stock returns using historical data and Granger causality testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from timeseries_prediction_stat.models.var_models import VARModel

# For data download
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system("pip install yfinance")
    import yfinance as yf


def main():
    print("=" * 80)
    print("VAR Model Example: Multi-Output Stock Prediction")
    print("=" * 80)
    
    # 1. Download stock data
    print("\n[1] Downloading stock data...")
    tickers = ['BMO', 'JPM', 'GS', 'WFC', 'BAC']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01', progress=False)['Close']
    
    print(f"   Downloaded {len(data)} days of data for {len(tickers)} stocks")
    
    # 2. Calculate returns
    print("\n[2] Calculating returns...")
    returns = data.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Set business day frequency
    returns.index.freq = 'B'
    
    print(f"   Returns shape: {returns.shape}")
    print(f"   Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    # 3. Split data
    print("\n[3] Splitting data (80% train, 20% test)...")
    train_size = int(len(returns) * 0.8)
    train_data = returns.iloc[:train_size]
    test_data = returns.iloc[train_size:]
    
    print(f"   Train: {len(train_data)} observations")
    print(f"   Test: {len(test_data)} observations")
    
    # 4. Fit VAR model
    print("\n[4] Fitting VAR model...")
    model = VARModel()
    model.fit(train_data, maxlags=5, ic='aic')
    
    print(f"   Selected lag order: {model.maxlags}")
    model.summary()
    
    # 5. Granger Causality Tests
    print("\n[5] Testing Granger Causality...")
    print("   Does JPM Granger-cause BMO?")
    gc_results = model.granger_causality('BMO', 'JPM', maxlag=5)
    for lag, result in gc_results.items():
        sig = "✓" if result['significant'] else "✗"
        print(f"      {lag}: p-value={result['p_value']:.4f} {sig}")
    
    # 6. Generate forecasts
    print("\n[6] Generating forecasts...")
    forecast_steps = 20
    forecast = model.forecast(steps=forecast_steps)
    
    print(f"   Forecast shape: {forecast.shape}")
    print(f"\n   First 5 forecasts:")
    print(forecast.head())
    
    # 7. Evaluate on test set
    print("\n[7] Evaluating model...")
    actual = test_data.iloc[:forecast_steps]
    
    # Overall metrics
    metrics = model.evaluate(actual, forecast)
    print(f"\n   Overall Metrics:")
    print(f"      RMSE: {metrics['rmse']:.6f}")
    print(f"      MAE:  {metrics['mae']:.6f}")
    print(f"      R²:   {metrics['r2']:.4f}")
    
    # Per-variable metrics
    per_var_metrics = model.evaluate_per_variable(actual, forecast)
    print(f"\n   Per-Stock Metrics:")
    for stock, stock_metrics in per_var_metrics.items():
        print(f"      {stock}:")
        print(f"         RMSE: {stock_metrics['rmse']:.6f}")
        print(f"         R²:   {stock_metrics['r2']:.4f}")
    
    # 8. Visualize results
    print("\n[8] Creating visualizations...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('VAR Model: Multi-Output Stock Prediction', fontsize=16, fontweight='bold')
    
    # Plot each stock
    for i, stock in enumerate(['BMO', 'JPM']):
        ax = axes[i, 0]
        ax.plot(actual.index, actual[stock], label='Actual', linewidth=2, alpha=0.8)
        ax.plot(actual.index, forecast[stock], label='Forecast', linewidth=2, alpha=0.8)
        ax.set_title(f'{stock} Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scatter plot
        ax = axes[i, 1]
        ax.scatter(actual[stock], forecast[stock], alpha=0.6)
        ax.plot([actual[stock].min(), actual[stock].max()],
                [actual[stock].min(), actual[stock].max()],
                'r--', linewidth=2, label='Perfect Prediction')
        ax.set_title(f'{stock} Actual vs Predicted')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Residuals plot
    ax = axes[2, 0]
    residuals = model.get_residuals()
    residuals['BMO'].plot(ax=ax, label='BMO', alpha=0.7)
    residuals['JPM'].plot(ax=ax, label='JPM', alpha=0.7)
    ax.set_title('Model Residuals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    
    # Diagnostics
    ax = axes[2, 1]
    diagnostics = model.get_diagnostics()
    metrics_text = f"AIC: {diagnostics['aic']:.2f}\n"
    metrics_text += f"BIC: {diagnostics['bic']:.2f}\n"
    metrics_text += f"HQIC: {diagnostics['hqic']:.2f}\n"
    metrics_text += f"Log-Likelihood: {diagnostics['log_likelihood']:.2f}\n\n"
    metrics_text += f"Test RMSE: {metrics['rmse']:.6f}\n"
    metrics_text += f"Test R²: {metrics['r2']:.4f}"
    ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    ax.set_title('Model Diagnostics')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'var_example.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   Plot saved to: {plot_path}")
    
    plt.show()
    
    # 9. Save model
    print("\n[9] Saving model...")
    model_path = os.path.join(output_dir, 'var_model.pkl')
    model.save(model_path)
    print(f"   Model saved to: {model_path}")
    
    print("\n" + "=" * 80)
    print("VAR Model Example Complete!")
    print("=" * 80)
    print(f"\nKey Results:")
    print(f"  ✓ Trained VAR({model.maxlags}) on {len(tickers)} stocks")
    print(f"  ✓ Test RMSE: {metrics['rmse']:.6f}")
    print(f"  ✓ Test R²: {metrics['r2']:.4f}")
    print(f"  ✓ BMO R²: {per_var_metrics['BMO']['r2']:.4f}")
    print(f"  ✓ JPM R²: {per_var_metrics['JPM']['r2']:.4f}")


if __name__ == '__main__':
    main()
