"""
Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com

Multi-Step Forecasting Example
===============================
This example demonstrates multi-step ahead forecasting using various
strategies: direct, recursive, and multi-output approaches.
"""

import sys


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from timeseries.normalization import DataNormalizer
from timeseries.deeplearning.models.lstm import LSTMModel
from timeseries.deeplearning.models.attention import EncoderDecoderModel
from timeseries.stat.models.statistical import ARIMAModel, SARIMAModel

def generate_time_series(n_points=1000):
    """Generate synthetic time series with trend and seasonality."""
    t = np.arange(n_points)
    
    # Trend
    trend = 0.05 * t
    
    # Seasonality (period=50)
    seasonality = 10 * np.sin(2 * np.pi * t / 50)
    
    # Noise
    noise = np.random.normal(0, 2, n_points)
    
    # Combine
    series = 100 + trend + seasonality + noise
    
    return pd.DataFrame({
        'value': series,
        'time': pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    })

def recursive_forecast(model, last_sequence, steps, normalizer):
    """
    Recursive multi-step forecasting.
    Use previous predictions as input for next prediction.
    """
    forecasts = []
    current_seq = last_sequence.copy()
    
    for _ in range(steps):
        # Predict next step
        pred = model.predict(pd.DataFrame(current_seq[-1:]))
        forecasts.append(pred[0, 0])
        
        # Update sequence: remove oldest, add newest prediction
        current_seq = np.vstack([current_seq[1:], pred])
    
    return np.array(forecasts)

def direct_forecast_ensemble(models, X_test, steps):
    """
    Direct multi-step forecasting.
    Train separate model for each horizon (ensemble of models).
    """
    # Simplified: use average of model predictions
    all_forecasts = []
    
    for model in models:
        try:
            forecast = model.forecast(steps=steps)
            all_forecasts.append(forecast)
        except:
            pass
    
    if all_forecasts:
        return np.mean(all_forecasts, axis=0)
    else:
        return np.zeros((steps, 1))

def plot_multistep_forecast(actual, forecasts_dict, steps, save_path):
    """Plot multi-step forecasts from different methods."""
    plt.figure(figsize=(15, 8))
    
    # Plot actual values
    plt.plot(range(len(actual)), actual, 
            label='Actual', linewidth=2, color='black', alpha=0.7)
    
    # Plot forecasts
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    start_idx = len(actual) - steps
    
    for i, (name, forecast) in enumerate(forecasts_dict.items()):
        plt.plot(range(start_idx, len(actual)), forecast[:steps],
                label=f'{name} Forecast', linewidth=2, 
                color=colors[i % len(colors)], linestyle='--', alpha=0.7)
    
    plt.axvline(x=start_idx, color='gray', linestyle=':', alpha=0.5, 
               label='Forecast Start')
    
    plt.title(f'Multi-Step Forecasting Comparison ({steps} steps ahead)', fontsize=14)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()

def calculate_horizon_metrics(actual, forecast):
    """Calculate metrics at each forecast horizon."""
    horizons = []
    rmse_values = []
    mae_values = []
    
    for h in range(1, len(forecast) + 1):
        if h <= len(actual):
            error = actual[h-1] - forecast[h-1]
            horizons.append(h)
            rmse_values.append(np.sqrt(error ** 2))
            mae_values.append(np.abs(error))
    
    return horizons, rmse_values, mae_values

def main():
    """Main execution function."""
    print("="*70)
    print("Multi-Step Forecasting Example")
    print("="*70)
    
    # Generate data
    print("\n1. Generating time series data...")
    df = generate_time_series(n_points=1000)
    print(f"   Generated {len(df)} observations")
    
    # Prepare data
    print("\n2. Preparing data...")
    train_size = 800
    test_size = 200
    forecast_horizon = 50  # Forecast 50 steps ahead
    
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:train_size+test_size]
    
    # Create features (lags)
    sequence_length = 30
    X_train, y_train = [], []
    
    for i in range(sequence_length, len(train_data)):
        X_train.append(train_data['value'].iloc[i-sequence_length:i].values)
        y_train.append(train_data['value'].iloc[i])
    
    X_train = np.array(X_train).reshape(-1, sequence_length)
    y_train = np.array(y_train).reshape(-1, 1)
    
    # Same for test
    X_test, y_test = [], []
    for i in range(sequence_length, len(test_data)):
        X_test.append(test_data['value'].iloc[i-sequence_length:i].values)
        y_test.append(test_data['value'].iloc[i])
    
    X_test = np.array(X_test).reshape(-1, sequence_length)
    y_test = np.array(y_test).reshape(-1, 1)
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train)
    y_train_df = pd.DataFrame(y_train, columns=['value'])
    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(y_test, columns=['value'])
    
    print(f"   Training samples: {len(X_train_df)}")
    print(f"   Testing samples: {len(X_test_df)}")
    print(f"   Forecast horizon: {forecast_horizon} steps")
    
    # Normalize
    print("\n3. Normalizing data...")
    normalizer = DataNormalizer({'strategy': 'zscore'})
    X_train_norm = normalizer.fit_transform(X_train_df)
    X_test_norm = normalizer.transform(X_test_df)
    
    # Train models
    print("\n4. Training models for multi-step forecasting...")
    models = {}
    forecasts = {}
    
    # LSTM with recursive forecasting
    print("\n   a) LSTM (Recursive Strategy)...")
    try:
        lstm = LSTMModel(config={
            'sequence_length': sequence_length,
            'hidden_size': 64,
            'num_layers': 2,
            'epochs': 30,
            'batch_size': 32
        })
        lstm.fit(X_train_norm, y_train_df, validation_split=0.1, verbose=False)
        
        # Recursive forecast
        last_sequence = X_test_norm.iloc[-1:].values.reshape(-1, sequence_length)
        forecast = recursive_forecast(lstm, last_sequence, forecast_horizon, normalizer)
        
        forecasts['LSTM-Recursive'] = forecast
        print(f"      Generated {len(forecast)} step forecast")
    except Exception as e:
        print(f"      Error: {str(e)}")
    
    # Encoder-Decoder for direct multi-output
    print("\n   b) Encoder-Decoder (Multi-Output Strategy)...")
    try:
        enc_dec = EncoderDecoderModel(config={
            'sequence_length': sequence_length,
            'hidden_size': 64,
            'num_layers': 2,
            'output_seq_length': forecast_horizon,
            'epochs': 30,
            'batch_size': 32
        })
        # Note: Would need multi-output training data for proper implementation
        print("      Encoder-Decoder requires multi-output training (simplified here)")
    except Exception as e:
        print(f"      Error: {str(e)}")
    
    # ARIMA statistical model
    print("\n   c) ARIMA (Direct Forecast)...")
    try:
        # Use un-normalized data for ARIMA
        arima = ARIMAModel(config={'order': (2, 1, 2)})
        arima.fit(pd.DataFrame(), y_train_df)
        
        forecast = arima.forecast(steps=forecast_horizon)
        forecasts['ARIMA'] = forecast.flatten()
        print(f"      Generated {len(forecast)} step forecast")
    except Exception as e:
        print(f"      Error: {str(e)}")
    
    # Naive baseline (persistence model)
    print("\n   d) Naive Baseline (Persistence)...")
    try:
        last_value = test_data['value'].iloc[sequence_length-1]
        naive_forecast = np.full(forecast_horizon, last_value)
        forecasts['Naive-Persistence'] = naive_forecast
        print(f"      Generated {len(naive_forecast)} step forecast")
    except Exception as e:
        print(f"      Error: {str(e)}")
    
    # Seasonal naive
    print("\n   e) Seasonal Naive (Period=50)...")
    try:
        period = 50
        seasonal_naive = []
        for h in range(forecast_horizon):
            if sequence_length - period + h >= 0:
                seasonal_naive.append(
                    test_data['value'].iloc[sequence_length - period + (h % period)]
                )
            else:
                seasonal_naive.append(test_data['value'].iloc[sequence_length-1])
        forecasts['Seasonal-Naive'] = np.array(seasonal_naive)
        print(f"      Generated {len(seasonal_naive)} step forecast")
    except Exception as e:
        print(f"      Error: {str(e)}")
    
    # Evaluate forecasts
    print("\n5. Evaluating forecasts...")
    print("\n" + "="*70)
    print("Model Performance at Different Horizons")
    print("="*70)
    
    actual_values = test_data['value'].iloc[sequence_length:sequence_length+forecast_horizon].values
    
    for name, forecast in forecasts.items():
        if len(forecast) >= forecast_horizon:
            forecast_trimmed = forecast[:forecast_horizon]
            
            # Overall metrics
            rmse = np.sqrt(np.mean((actual_values - forecast_trimmed) ** 2))
            mae = np.mean(np.abs(actual_values - forecast_trimmed))
            
            print(f"\n{name}:")
            print(f"  Overall RMSE: {rmse:.4f}")
            print(f"  Overall MAE: {mae:.4f}")
            
            # Metrics at specific horizons
            for h in [1, 5, 10, 25, 50]:
                if h <= len(actual_values):
                    error = actual_values[h-1] - forecast_trimmed[h-1]
                    print(f"  Horizon {h:2d}: Error = {error:.4f}")
    
    # Plot results
    print("\n6. Generating visualizations...")
    
    # Plot 1: Multi-step forecasts
    plot_multistep_forecast(
        test_data['value'].iloc[sequence_length:sequence_length+forecast_horizon].values,
        forecasts,
        forecast_horizon,
        " ./multistep_forecast_comparison.png"
    )
    print("   Saved: multistep_forecast_comparison.png")
    
    # Plot 2: Error by horizon
    plt.figure(figsize=(12, 6))
    for name, forecast in forecasts.items():
        if len(forecast) >= forecast_horizon:
            horizons, rmse_vals, _ = calculate_horizon_metrics(
                actual_values, forecast[:forecast_horizon]
            )
            plt.plot(horizons, rmse_vals, label=name, marker='o', markersize=3)
    
    plt.xlabel('Forecast Horizon')
    plt.ylabel('RMSE')
    plt.title('Forecast Error vs Horizon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(" ./forecast_error_by_horizon.png")
    plt.close()
    print("   Saved: forecast_error_by_horizon.png")
    
    print("\n" + "="*70)
    print("Multi-step forecasting example completed!")
    print("="*70)
    
    # Key insights
    print("\nKey Insights:")
    print("1. Recursive strategy accumulates errors over horizons")
    print("2. Direct multi-output models can be more stable")
    print("3. Simple baselines (naive, seasonal naive) are hard to beat for short horizons")
    print("4. Complex models shine at longer horizons with sufficient data")
    print("5. Ensemble of multiple strategies often performs best")

if __name__ == "__main__":
    main()
