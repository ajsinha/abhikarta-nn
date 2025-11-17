"""
Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com

Energy Demand Forecasting Example
==================================
This example demonstrates forecasting hourly energy demand using both
deep learning and statistical models with real-world considerations.
"""

import sys
sys.path.append('/home/claude/timeseries_package')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from timeseries.normalization import DataNormalizer
from timeseries.deeplearning.models.lstm import LSTMModel
from timeseries.deeplearning.models.tcn import TCNModel
from timeseries.deeplearning.models.attention import AttentionLSTMModel
from timeseries.stat.models.statistical import SARIMAModel, ProphetModel
from timeseries.stat.models.advanced_statistical import AutoARIMAModel
from timeseries.deeplearning.ensemble.ensemble import EnsembleModel

def generate_synthetic_energy_data(n_days=365*2, hourly=True):
    """
    Generate synthetic energy demand data with realistic patterns:
    - Daily seasonality (higher during day, lower at night)
    - Weekly seasonality (lower on weekends)
    - Annual seasonality (higher in summer/winter for AC/heating)
    - Temperature dependence
    - Random variations
    """
    if hourly:
        n_points = n_days * 24
        timestamps = pd.date_range(start='2022-01-01', periods=n_points, freq='H')
    else:
        n_points = n_days
        timestamps = pd.date_range(start='2022-01-01', periods=n_points, freq='D')
    
    # Time features
    hour_of_day = timestamps.hour if hourly else np.zeros(n_points)
    day_of_week = timestamps.dayofweek
    day_of_year = timestamps.dayofyear
    
    # Base load (MW)
    base_load = 1000
    
    # Daily pattern (higher during day 8am-8pm)
    if hourly:
        daily_pattern = 300 * (np.sin((hour_of_day - 6) * np.pi / 12) + 1) / 2
        # Peak hours (8am-10am, 6pm-8pm)
        daily_pattern += 150 * (
            ((hour_of_day >= 8) & (hour_of_day <= 10)) |
            ((hour_of_day >= 18) & (hour_of_day <= 20))
        )
    else:
        daily_pattern = 0
    
    # Weekly pattern (lower on weekends)
    weekly_pattern = -100 * ((day_of_week == 5) | (day_of_week == 6))
    
    # Annual pattern (higher in summer and winter)
    annual_pattern = 200 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
    
    # Temperature (simulated)
    temperature = 15 + 15 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
    if hourly:
        # Add daily temperature variation
        temperature = temperature + 5 * np.sin((hour_of_day - 6) * np.pi / 12)
    
    # Temperature effect on demand (U-shaped: high demand at extreme temps)
    temp_effect = 100 * ((temperature - 20) / 10) ** 2
    
    # Random noise
    noise = np.random.normal(0, 50, n_points)
    
    # Combine all components
    demand = (base_load + daily_pattern + weekly_pattern + 
              annual_pattern + temp_effect + noise)
    
    # Ensure non-negative
    demand = np.maximum(demand, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'demand_MW': demand,
        'temperature': temperature,
        'hour': hour_of_day,
        'day_of_week': day_of_week,
        'is_weekend': (day_of_week >= 5).astype(int),
        'month': timestamps.month,
        'day_of_year': day_of_year
    })
    
    return df

def create_features(df):
    """Create additional features for modeling."""
    df = df.copy()
    
    # Lag features
    for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
        df[f'demand_lag_{lag}'] = df['demand_MW'].shift(lag)
    
    # Rolling statistics
    for window in [24, 168]:  # 1 day, 1 week
        df[f'demand_rolling_mean_{window}'] = df['demand_MW'].rolling(window).mean()
        df[f'demand_rolling_std_{window}'] = df['demand_MW'].rolling(window).std()
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df

def prepare_data(df, target_col='demand_MW', test_size=0.2):
    """Prepare features and target for modeling."""
    # Select features (exclude timestamp and target)
    feature_cols = [col for col in df.columns 
                   if col not in ['timestamp', target_col]]
    
    X = df[feature_cols]
    y = df[[target_col]]
    
    # Train/test split
    split_idx = int(len(df) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    timestamps_test = df['timestamp'].iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, timestamps_test

def plot_results(timestamps, actual, predictions, model_name, save_path):
    """Plot actual vs predicted energy demand."""
    plt.figure(figsize=(15, 6))
    
    # Plot first week for clarity
    n_points = min(24*7, len(timestamps))
    
    plt.plot(timestamps[:n_points], actual[:n_points], 
            label='Actual', alpha=0.7, linewidth=2)
    plt.plot(timestamps[:n_points], predictions[:n_points], 
            label='Predicted', alpha=0.7, linewidth=2)
    
    plt.title(f'{model_name} - Energy Demand Forecast (First Week)', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()

def main():
    """Main execution function."""
    print("="*70)
    print("Energy Demand Forecasting Example")
    print("="*70)
    
    # Generate data
    print("\n1. Generating synthetic energy demand data...")
    df = generate_synthetic_energy_data(n_days=365*2, hourly=True)
    print(f"   Generated {len(df)} hourly observations")
    
    # Create features
    print("\n2. Engineering features...")
    df = create_features(df)
    print(f"   Created {len(df.columns)-2} features")
    
    # Prepare data
    print("\n3. Preparing train/test sets...")
    X_train, X_test, y_train, y_test, timestamps_test = prepare_data(df)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Normalize
    print("\n4. Normalizing data...")
    normalizer_X = DataNormalizer({'strategy': 'zscore'})
    normalizer_y = DataNormalizer({'strategy': 'zscore'})
    
    X_train_norm = normalizer_X.fit_transform(X_train)
    X_test_norm = normalizer_X.transform(X_test)
    y_train_norm = normalizer_y.fit_transform(y_train)
    y_test_norm = normalizer_y.transform(y_test)
    
    # Configure models
    dl_config = {
        'sequence_length': 24,  # Use last 24 hours
        'hidden_size': 128,
        'num_layers': 3,
        'epochs': 30,
        'batch_size': 64,
        'learning_rate': 0.001
    }
    
    # Initialize models
    print("\n5. Training models...")
    models = {}
    results = {}
    
    # Deep Learning Models
    print("\n   a) Training LSTM...")
    try:
        lstm = LSTMModel(config=dl_config.copy())
        lstm.fit(X_train_norm, y_train_norm, validation_split=0.1, verbose=False)
        
        predictions_norm = lstm.predict(X_test_norm)
        predictions = normalizer_y.inverse_transform(
            pd.DataFrame(predictions_norm, columns=['demand_MW'])
        )
        
        metrics = lstm.evaluate(X_test_norm, y_test_norm)
        models['LSTM'] = lstm
        results['LSTM'] = {'predictions': predictions, 'metrics': metrics}
        
        print(f"      RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.4f}")
    except Exception as e:
        print(f"      Error: {str(e)}")
    
    print("\n   b) Training TCN...")
    try:
        tcn_config = dl_config.copy()
        tcn_config['num_channels'] = [128, 128, 64]
        tcn = TCNModel(config=tcn_config)
        tcn.fit(X_train_norm, y_train_norm, validation_split=0.1, verbose=False)
        
        predictions_norm = tcn.predict(X_test_norm)
        predictions = normalizer_y.inverse_transform(
            pd.DataFrame(predictions_norm, columns=['demand_MW'])
        )
        
        metrics = tcn.evaluate(X_test_norm, y_test_norm)
        models['TCN'] = tcn
        results['TCN'] = {'predictions': predictions, 'metrics': metrics}
        
        print(f"      RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.4f}")
    except Exception as e:
        print(f"      Error: {str(e)}")
    
    print("\n   c) Training Attention LSTM...")
    try:
        attn_lstm = AttentionLSTMModel(config=dl_config.copy())
        attn_lstm.fit(X_train_norm, y_train_norm, validation_split=0.1, verbose=False)
        
        predictions_norm = attn_lstm.predict(X_test_norm)
        predictions = normalizer_y.inverse_transform(
            pd.DataFrame(predictions_norm, columns=['demand_MW'])
        )
        
        metrics = attn_lstm.evaluate(X_test_norm, y_test_norm)
        models['Attention-LSTM'] = attn_lstm
        results['Attention-LSTM'] = {'predictions': predictions, 'metrics': metrics}
        
        print(f"      RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.4f}")
    except Exception as e:
        print(f"      Error: {str(e)}")
    
    # Statistical Models
    print("\n   d) Training SARIMA...")
    try:
        sarima = SARIMAModel(config={
            'order': (1, 0, 1),
            'seasonal_order': (1, 0, 1, 24)  # 24-hour seasonality
        })
        sarima.fit(X_train, y_train)
        
        predictions = sarima.predict(X_test)
        metrics = sarima.evaluate(X_test, y_test)
        models['SARIMA'] = sarima
        results['SARIMA'] = {'predictions': predictions, 'metrics': metrics}
        
        print(f"      RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.4f}")
    except Exception as e:
        print(f"      Error: {str(e)}")
    
    # Create ensemble
    if len(list(models.values())) >= 2:
        print("\n   e) Creating ensemble...")
        try:
            ensemble = EnsembleModel(
                models=list(models.values())[:3],  # Use top 3 models
                method='average'
            )
            ensemble.is_fitted = True
            
            predictions_norm = ensemble.predict(X_test_norm)
            predictions = normalizer_y.inverse_transform(
                pd.DataFrame(predictions_norm, columns=['demand_MW'])
            )
            
            metrics = ensemble.evaluate(X_test_norm, y_test_norm)
            results['Ensemble'] = {'predictions': predictions, 'metrics': metrics}
            
            print(f"      RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.4f}")
        except Exception as e:
            print(f"      Error: {str(e)}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Model Performance")
    print("="*70)
    print(f"{'Model':<20} {'RMSE (MW)':<15} {'MAE (MW)':<15} {'R²':<10}")
    print("-"*70)
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"{name:<20} {metrics['rmse']:<15.2f} {metrics['mae']:<15.2f} {metrics['r2']:<10.4f}")
    
    # Best model
    best_model = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
    print(f"\nBest Model (by RMSE): {best_model}")
    
    # Plot results
    print("\n6. Generating visualizations...")
    for name, result in results.items():
        save_path = f"/home/claude/timeseries_package/timeseries/deeplearning/examples/{name}_energy_forecast.png"
        plot_results(
            timestamps_test.values,
            y_test.values.flatten(),
            result['predictions'].values.flatten(),
            name,
            save_path
        )
        print(f"   Saved: {save_path}")
    
    print("\n" + "="*70)
    print("Energy forecasting example completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
