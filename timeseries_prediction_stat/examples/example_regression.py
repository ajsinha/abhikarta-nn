"""
Regression Models Comparison Example
=====================================

Compares different linear regression approaches for time series:
- OLS (Ordinary Least Squares)
- Ridge Regression
- Lasso Regression
- Elastic Net
- Bayesian Ridge
- Robust Regression

Demonstrates feature selection, regularization, and uncertainty quantification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from timeseries_prediction_stat.models.regression_models import (
    MultiOutputLinearRegression,
    RidgeTimeSeriesRegression,
    LassoTimeSeriesRegression,
    ElasticNetTimeSeriesRegression,
    BayesianLinearRegression,
    RobustLinearRegression
)

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance")
    import yfinance as yf


def main():
    print("=" * 80)
    print("Regression Models Comparison for Time Series")
    print("=" * 80)
    
    # 1. Download data
    print("\n[1] Downloading stock data...")
    tickers = ['BMO', 'JPM']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01', progress=False)['Close']
    
    returns = data.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Set business day frequency
    returns.index.freq = 'B'
    
    print(f"   Data shape: {returns.shape}")
    
    # 2. Split data
    train_size = int(len(returns) * 0.8)
    train_data = returns.iloc[:train_size]
    test_data = returns.iloc[train_size:]
    
    print(f"   Train: {len(train_data)} observations")
    print(f"   Test: {len(test_data)} observations")
    
    # 3. Train multiple models
    print("\n[2] Training multiple regression models...")
    
    models = {
        'OLS': MultiOutputLinearRegression(lags=10),
        'Ridge': RidgeTimeSeriesRegression(lags=10, alpha=1.0),
        'Lasso': LassoTimeSeriesRegression(lags=10, alpha=0.01),
        'ElasticNet': ElasticNetTimeSeriesRegression(lags=10, alpha=0.1, l1_ratio=0.5),
        'Bayesian': BayesianLinearRegression(lags=10),
        'Robust': RobustLinearRegression(lags=10, epsilon=1.35)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        model.fit(train_data)
        
        # Forecast
        forecast = model.forecast(steps=len(test_data))
        
        # Evaluate
        metrics = model.evaluate(test_data.values, forecast.values)
        results[name] = {
            'model': model,
            'forecast': forecast,
            'metrics': metrics
        }
        
        print(f"      RMSE: {metrics['rmse']:.6f}")
        print(f"      R²:   {metrics['r2']:.4f}")
    
    # 4. Feature selection with Lasso
    print("\n[3] Lasso Feature Selection...")
    lasso_model = results['Lasso']['model']
    selected_features = lasso_model.get_nonzero_coefficients()
    
    print("   Selected features (non-zero coefficients):")
    for var, lags in selected_features.items():
        print(f"      {var}: {len(lags)} selected lags")
    
    # 5. Bayesian uncertainty
    print("\n[4] Bayesian Uncertainty Quantification...")
    bayesian_model = models['Bayesian']
    forecast_mean, forecast_std = bayesian_model.forecast_with_uncertainty(steps=len(test_data))
    
    print("   95% Confidence Intervals (first 5 forecasts):")
    for i in range(min(5, len(forecast_mean))):
        for col in forecast_mean.columns:
            mean = forecast_mean[col].iloc[i]
            std = forecast_std[f'{col}_std'].iloc[i]
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            print(f"      {col} step {i+1}: [{lower:.6f}, {upper:.6f}]")
    
    # 6. Model comparison
    print("\n[5] Model Comparison Summary:")
    print(f"\n   {'Model':<15} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
    print("   " + "-" * 45)
    
    best_rmse = min(r['metrics']['rmse'] for r in results.values())
    best_r2 = max(r['metrics']['r2'] for r in results.values())
    
    for name, result in results.items():
        m = result['metrics']
        rmse_mark = " ⭐" if m['rmse'] == best_rmse else ""
        r2_mark = " ⭐" if m['r2'] == best_r2 else ""
        print(f"   {name:<15} {m['rmse']:>10.6f}{rmse_mark} {m['mae']:>10.6f} {m['r2']:>8.4f}{r2_mark}")
    
    # 7. Visualize
    print("\n[6] Creating visualizations...")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Regression Models Comparison for Time Series', fontsize=16, fontweight='bold')
    
    # Plot 1-2: Predictions for each stock
    for i, stock in enumerate(tickers):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(test_data.index, test_data[stock], 'k-', linewidth=2, label='Actual', alpha=0.8)
        
        for name, result in results.items():
            forecast = result['forecast']
            ax.plot(test_data.index, forecast[stock], label=name, linewidth=1.5, alpha=0.7)
        
        ax.set_title(f'{stock} Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Model comparison bar chart
    ax = fig.add_subplot(gs[0, 2])
    model_names = list(results.keys())
    rmse_values = [results[m]['metrics']['rmse'] for m in model_names]
    bars = ax.barh(model_names, rmse_values, color='skyblue', edgecolor='navy')
    ax.set_xlabel('RMSE')
    ax.set_title('Model Comparison (RMSE)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Highlight best
    best_idx = rmse_values.index(min(rmse_values))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    
    # Plot 4-5: Bayesian uncertainty for each stock
    for i, stock in enumerate(tickers):
        ax = fig.add_subplot(gs[1, i])
        
        mean = forecast_mean[stock].values
        std = forecast_std[f'{stock}_std'].values
        x = range(len(mean))
        
        ax.plot(test_data.index, test_data[stock], 'k-', linewidth=2, label='Actual', alpha=0.8)
        ax.plot(test_data.index, mean, 'b-', linewidth=2, label='Prediction', alpha=0.8)
        ax.fill_between(test_data.index,
                        mean - 1.96 * std,
                        mean + 1.96 * std,
                        alpha=0.3, label='95% CI')
        ax.set_title(f'{stock} with Uncertainty (Bayesian)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Plot 6: R² comparison
    ax = fig.add_subplot(gs[1, 2])
    r2_values = [results[m]['metrics']['r2'] for m in model_names]
    bars = ax.barh(model_names, r2_values, color='lightgreen', edgecolor='darkgreen')
    ax.set_xlabel('R² Score')
    ax.set_title('Model Comparison (R²)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Highlight best
    best_idx = r2_values.index(max(r2_values))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    
    # Plot 7-8: Residual histograms
    for i, stock in enumerate(['BMO', 'JPM']):
        ax = fig.add_subplot(gs[2, i])
        
        for name, result in results.items():
            forecast = result['forecast']
            residuals = test_data[stock].values - forecast[stock].values[:len(test_data)]
            ax.hist(residuals, bins=20, alpha=0.5, label=name)
        
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{stock} Residual Distribution')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    
    # Plot 9: Summary table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    table_data = []
    for name, result in results.items():
        m = result['metrics']
        table_data.append([name, f"{m['rmse']:.6f}", f"{m['r2']:.4f}"])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'RMSE', 'R²'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best
    best_rmse_idx = rmse_values.index(min(rmse_values)) + 1
    best_r2_idx = r2_values.index(max(r2_values)) + 1
    table[(best_rmse_idx, 1)].set_facecolor('gold')
    table[(best_r2_idx, 2)].set_facecolor('gold')
    
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'regression_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   Plot saved to: {plot_path}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("Regression Models Comparison Complete!")
    print("=" * 80)
    
    # Find best model
    best_model_name = min(results.items(), key=lambda x: x[1]['metrics']['rmse'])[0]
    print(f"\nBest Model: {best_model_name}")
    print(f"  RMSE: {results[best_model_name]['metrics']['rmse']:.6f}")
    print(f"  R²:   {results[best_model_name]['metrics']['r2']:.4f}")


if __name__ == '__main__':
    main()
