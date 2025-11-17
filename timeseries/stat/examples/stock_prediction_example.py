"""
Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com

Legal Notice: This module and the associated software architecture are proprietary 
and confidential. Unauthorized copying, distribution, modification, or use is 
strictly prohibited without explicit written permission from the copyright holder.

Patent Pending: Certain architectural patterns and implementations described in 
this module may be subject to patent applications.

Statistical Time Series Example
================================
This example demonstrates using statistical models (ARIMA, SARIMA, VAR, ETS, Prophet)
to predict stock prices. Uses 10 stocks as features to predict BMO and C stocks.
"""

import sys
sys.path.append('/home/claude/timeseries_package')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from timeseries.normalization import DataNormalizer
from timeseries.stat.models.statistical import (
    ARIMAModel, SARIMAModel, VARModel, 
    ExponentialSmoothingModel, ProphetModel
)
from timeseries.stat.ensemble.ensemble import StatisticalEnsemble

# Configuration
FEATURE_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'WFC']
TARGET_STOCKS = ['BMO', 'C']
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
TEST_SIZE = 0.2

def download_stock_data(tickers, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    print(f"Downloading data for {len(tickers)} stocks...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Extract closing prices
    if len(tickers) > 1:
        prices = data['Close']
    else:
        prices = pd.DataFrame(data['Close'])
        prices.columns = tickers
    
    # Handle missing values
    prices = prices.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Downloaded {len(prices)} days of data")
    return prices

def prepare_data(feature_stocks, target_stocks, start_date, end_date):
    """Prepare feature and target datasets."""
    # Download data
    all_stocks = feature_stocks + target_stocks
    prices = download_stock_data(all_stocks, start_date, end_date)
    
    # Split into features and targets
    X = prices[feature_stocks]
    y = prices[target_stocks]
    
    return X, y

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train a model and evaluate its performance."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    try:
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        try:
            metrics = model.evaluate(X_test, y_test)
            
            print(f"\n{model_name} Performance:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  R²: {metrics['r2']:.4f}")
            
            # Make predictions
            predictions = model.predict(X_test)
            
            return model, predictions, metrics
        except Exception as e:
            print(f"  Error during evaluation: {str(e)}")
            return None, None, None
            
    except Exception as e:
        print(f"  Error during training: {str(e)}")
        return None, None, None

def plot_predictions(y_test, predictions, model_name, target_name):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    
    # Handle different array shapes
    if predictions is not None:
        if len(predictions.shape) > 1 and predictions.shape[1] > 0:
            pred_values = predictions[:, 0] if predictions.shape[1] == 1 else predictions[:len(y_test), 0]
        else:
            pred_values = predictions[:len(y_test)]
    else:
        return None
    
    plt.plot(y_test.values[:len(pred_values)], label='Actual', alpha=0.7)
    plt.plot(pred_values, label='Predicted', alpha=0.7)
    plt.title(f'{model_name} - {target_name} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def main():
    """Main execution function."""
    print("="*60)
    print("Statistical Stock Price Prediction Example")
    print("="*60)
    
    # Prepare data
    print("\n1. Preparing data...")
    X, y = prepare_data(FEATURE_STOCKS, TARGET_STOCKS, START_DATE, END_DATE)
    
    # For statistical models, we may want to use log returns or keep original scale
    print("\n2. Preparing data for statistical models...")
    
    # Split into train/test
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    # Initialize models
    print("\n3. Initializing statistical models...")
    models = {
        'ARIMA': ARIMAModel(config={'order': (2, 1, 2)}),
        'SARIMA': SARIMAModel(config={'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 5)}),
        'VAR': VARModel(config={'maxlags': 5}),
        'ETS': ExponentialSmoothingModel(config={'trend': 'add', 'seasonal': None}),
    }
    
    # Note: Prophet requires installation separately
    try:
        models['Prophet'] = ProphetModel(config={})
    except Exception as e:
        print(f"  Prophet not available: {str(e)}")
    
    # Train and evaluate each model
    print("\n4. Training and evaluating models...")
    results = {}
    trained_models = []
    
    for name, model in models.items():
        trained_model, predictions, metrics = train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, name
        )
        
        if trained_model is not None and predictions is not None:
            results[name] = {
                'model': trained_model,
                'predictions': predictions,
                'metrics': metrics
            }
            trained_models.append(trained_model)
    
    # Create ensemble
    print("\n5. Creating statistical ensemble...")
    if len(trained_models) >= 2:
        try:
            ensemble = StatisticalEnsemble(trained_models, method='average')
            ensemble.is_fitted = True
            
            ensemble_predictions = ensemble.predict(X_test)
            ensemble_metrics = ensemble.evaluate(X_test, y_test)
            
            print(f"\nEnsemble Performance:")
            print(f"  RMSE: {ensemble_metrics['rmse']:.4f}")
            print(f"  MAE: {ensemble_metrics['mae']:.4f}")
            print(f"  MAPE: {ensemble_metrics['mape']:.2f}%")
            print(f"  R²: {ensemble_metrics['r2']:.4f}")
            
            results['Ensemble'] = {
                'model': ensemble,
                'predictions': ensemble_predictions,
                'metrics': ensemble_metrics
            }
        except Exception as e:
            print(f"  Error creating ensemble: {str(e)}")
    
    # Summary
    if results:
        print("\n" + "="*60)
        print("SUMMARY - All Models Performance")
        print("="*60)
        print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'R²':<10}")
        print("-"*60)
        
        for name, result in results.items():
            if result['metrics'] is not None:
                metrics = result['metrics']
                print(f"{name:<15} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} "
                      f"{metrics['mape']:<10.2f} {metrics['r2']:<10.4f}")
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if v['metrics'] is not None}
        if valid_results:
            best_model_name = min(valid_results.keys(), 
                                key=lambda k: valid_results[k]['metrics']['rmse'])
            print(f"\nBest Model (by RMSE): {best_model_name}")
        
        # Generate visualizations
        print("\n6. Generating visualizations...")
        
        for name, result in results.items():
            if result['predictions'] is not None:
                for i, target in enumerate(TARGET_STOCKS):
                    try:
                        plot = plot_predictions(
                            y_test.iloc[:, i],
                            result['predictions'],
                            name,
                            target
                        )
                        if plot:
                            filename = f"/home/claude/timeseries_package/timeseries/stat/examples/{name}_{target}_prediction.png"
                            plot.savefig(filename)
                            plt.close()
                            print(f"  Saved: {filename}")
                    except Exception as e:
                        print(f"  Error plotting {name}-{target}: {str(e)}")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)
    
    # Forecasting example
    print("\n7. Forecasting next 10 days...")
    for name, result in results.items():
        if result['model'] is not None:
            try:
                forecast = result['model'].forecast(steps=10)
                print(f"\n{name} 10-day forecast (first target):")
                print(forecast[:, 0] if len(forecast.shape) > 1 else forecast)
            except Exception as e:
                print(f"  {name} forecast error: {str(e)}")

if __name__ == "__main__":
    main()
