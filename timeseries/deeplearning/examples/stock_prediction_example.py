"""
Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com

Legal Notice: This module and the associated software architecture are proprietary 
and confidential. Unauthorized copying, distribution, modification, or use is 
strictly prohibited without explicit written permission from the copyright holder.

Patent Pending: Certain architectural patterns and implementations described in 
this module may be subject to patent applications.

Deep Learning Time Series Example
==================================
This example demonstrates using deep learning models (LSTM, GRU, BiLSTM, CNN-LSTM, Transformer)
to predict stock prices. Uses 10 stocks as features to predict BMO and C stocks.
"""

import sys


import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from timeseries.normalization import DataNormalizer
from timeseries.deeplearning.models.lstm import LSTMModel
from timeseries.deeplearning.models.gru import GRUModel
from timeseries.deeplearning.models.advanced import BiLSTMModel, CNNLSTMModel, TransformerModel
from timeseries.deeplearning.ensemble.ensemble import EnsembleModel

# Configuration
FEATURE_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'WFC']
TARGET_STOCKS = ['BMO', 'C']
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
TEST_SIZE = 0.2
SEQUENCE_LENGTH = 20
EPOCHS = 50

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
    
    # Handle missing values (forward fill then backward fill)
    prices = prices.ffill().bfill()
    
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
    
    # Train model
    model.fit(X_train, y_train, validation_split=0.1, verbose=False)
    
    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test)
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²: {metrics['r2']:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    return model, predictions, metrics

def plot_predictions(y_test, predictions, model_name, target_name):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    
    # Ensure same length by taking the minimum
    min_len = min(len(y_test), len(predictions))
    y_test_plot = y_test.values[-min_len:] if hasattr(y_test, 'values') else y_test[-min_len:]
    pred_plot = predictions[-min_len:] if len(predictions) > min_len else predictions
    
    plt.plot(y_test_plot, label='Actual', alpha=0.7)
    plt.plot(pred_plot, label='Predicted', alpha=0.7)
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
    print("Deep Learning Stock Price Prediction Example")
    print("="*60)
    
    # Prepare data
    print("\n1. Preparing data...")
    X, y = prepare_data(FEATURE_STOCKS, TARGET_STOCKS, START_DATE, END_DATE)
    
    # Normalize data
    print("\n2. Normalizing data...")
    normalizer_config = {'strategy': 'zscore'}
    normalizer_X = DataNormalizer(normalizer_config)
    normalizer_y = DataNormalizer(normalizer_config)
    
    X_normalized = normalizer_X.fit_transform(X)
    y_normalized = normalizer_y.fit_transform(y)
    
    # Split into train/test
    split_idx = int(len(X_normalized) * (1 - TEST_SIZE))
    X_train = X_normalized.iloc[:split_idx]
    X_test = X_normalized.iloc[split_idx:]
    y_train = y_normalized.iloc[:split_idx]
    y_test = y_normalized.iloc[split_idx:]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    # Configure models
    model_config = {
        'sequence_length': SEQUENCE_LENGTH,
        'epochs': EPOCHS,
        'batch_size': 32,
        'learning_rate': 0.001,
        'hidden_size': 64,
        'num_layers': 2
    }
    
    # Initialize models
    print("\n3. Initializing models...")
    models = {
        'LSTM': LSTMModel(config=model_config.copy()),
        'GRU': GRUModel(config=model_config.copy()),
        'BiLSTM': BiLSTMModel(config=model_config.copy()),
        'CNN-LSTM': CNNLSTMModel(config=model_config.copy()),
        'Transformer': TransformerModel(config={**model_config, 'd_model': 64, 'nhead': 4})
    }
    
    # Train and evaluate each model
    print("\n4. Training and evaluating models...")
    results = {}
    trained_models = []
    
    for name, model in models.items():
        try:
            trained_model, predictions, metrics = train_and_evaluate_model(
                model, X_train, y_train, X_test, y_test, name
            )
            results[name] = {
                'model': trained_model,
                'predictions': predictions,
                'metrics': metrics
            }
            trained_models.append(trained_model)
        except Exception as e:
            print(f"\nError training {name}: {str(e)}")
    
    # Create ensemble
    print("\n5. Creating ensemble model...")
    if len(trained_models) >= 2:
        try:
            ensemble = EnsembleModel(trained_models, method='average')
            ensemble.is_fitted = True  # Models are already fitted
            
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
            print(f"\nError creating ensemble: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - All Models Performance")
    print("="*60)
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'R²':<10}")
    print("-"*60)
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"{name:<15} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} "
              f"{metrics['mape']:<10.2f} {metrics['r2']:<10.4f}")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
    print(f"\nBest Model (by RMSE): {best_model_name}")
    
    # Inverse transform predictions for visualization
    print("\n6. Generating visualizations...")
    
    for name, result in results.items():
        pred_array = result['predictions']
        
        # Create dataframe with proper shape
        if len(pred_array.shape) == 1:
            pred_array = pred_array.reshape(-1, len(TARGET_STOCKS))
        
        pred_df = pd.DataFrame(pred_array, columns=TARGET_STOCKS)
        pred_original = normalizer_y.inverse_transform(pred_df)
        
        # Adjust y_test to match prediction length
        y_test_aligned = y_test.iloc[-len(pred_df):]
        y_test_original = normalizer_y.inverse_transform(y_test_aligned)
        
        for i, target in enumerate(TARGET_STOCKS):
            try:
                plot = plot_predictions(
                    y_test_original.iloc[:, i],
                    pred_original.iloc[:, i].values,
                    name,
                    target
                )
                filename = f" ./{name}_{target}_prediction.png"
                plot.savefig(filename)
                plt.close()
                print(f"  Saved: {filename}")
            except Exception as e:
                print(f"  Warning: Could not plot {name}-{target}: {str(e)}")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
