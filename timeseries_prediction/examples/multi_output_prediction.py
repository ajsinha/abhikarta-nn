"""
Multi-Output Prediction Example
================================

This example demonstrates predicting multiple stocks (BMO and JPM) simultaneously
using DOW30 stocks as features.

Features: DOW30 daily returns (30 features)
Targets: BMO and JPM daily returns (2 outputs)

This showcases the multi-output capability of the TimeSeriesModel base class.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from timeseries_prediction.models.rnn_models import LSTMModel, GRUModel
from timeseries_prediction.models.transformer_models import TransformerModel
from timeseries_prediction.utils.data_utils import (
    download_stock_data, get_dow30_tickers, calculate_returns,
    create_sequences, train_val_test_split, normalize_data
)


def main():
    print("=" * 80)
    print("Multi-Output Stock Prediction: DOW30 → BMO + JPM")
    print("=" * 80)
    
    # Configuration
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    LOOKBACK = 20  # Use 20 days of history
    TARGET_TICKERS = ['BMO', 'JPM']  # Predict these two stocks
    
    # Step 1: Download DOW30 data
    print("\n[1] Downloading DOW30 stock data...")
    dow30_tickers = get_dow30_tickers()
    dow30_prices = download_stock_data(dow30_tickers, START_DATE, END_DATE)
    
    # Step 2: Download target stocks (BMO and JPM)
    print("\n[2] Downloading target stock data (BMO and JPM)...")
    target_prices = download_stock_data(TARGET_TICKERS, START_DATE, END_DATE)
    
    # Step 3: Calculate returns (robust to division by zero)
    print("\n[3] Calculating daily returns...")
    print("   Using robust percentage change calculation with zero handling")
    dow30_returns = calculate_returns(dow30_prices, method='pct_change', handle_zeros=True)
    target_returns = calculate_returns(target_prices, method='pct_change', handle_zeros=True)
    
    # Align dates (inner join)
    common_dates = dow30_returns.index.intersection(target_returns.index)
    dow30_returns = dow30_returns.loc[common_dates]
    target_returns = target_returns.loc[common_dates]
    
    print(f"   Data shape: {len(common_dates)} trading days")
    print(f"   Features: {dow30_returns.shape[1]} (DOW30 stocks)")
    print(f"   Targets: {target_returns.shape[1]} (BMO and JPM)")
    
    # Step 4: Create sequences
    print(f"\n[4] Creating sequences with lookback={LOOKBACK}...")
    X, y = create_sequences(dow30_returns.values, LOOKBACK, horizon=1)
    
    # For targets, create sequences aligned with X
    _, y_targets = create_sequences(target_returns.values, LOOKBACK, horizon=1)
    
    print(f"   X shape: {X.shape} (samples, lookback, features)")
    print(f"   y shape: {y_targets.shape} (samples, num_targets)")
    
    # Step 5: Split data (time-series split)
    print("\n[5] Splitting data (70% train, 15% val, 15% test)...")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y_targets, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Step 6: Normalize data
    print("\n[6] Normalizing data...")
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_val_2d = X_val.reshape(-1, X_val.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_norm, X_val_norm, X_test_norm, stats = normalize_data(
        X_train_2d, X_val_2d, X_test_2d, method='standardize'
    )
    
    X_train = X_train_norm.reshape(X_train.shape)
    X_val = X_val_norm.reshape(X_val.shape)
    X_test = X_test_norm.reshape(X_test.shape)
    
    # Step 7: Build and train model
    print("\n[7] Building LSTM model for multi-output prediction...")
    
    INPUT_SIZE = dow30_returns.shape[1]  # 30 (DOW30 features)
    OUTPUT_SIZE = len(TARGET_TICKERS)  # 2 (BMO and JPM)
    
    model = LSTMModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,  # Multi-output: predicts 2 stocks simultaneously
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    model.summary()
    
    print("\n[8] Training model...")
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=15,
        verbose=True
    )
    
    # Step 8: Evaluate model
    print("\n[9] Evaluating model on test set...")
    metrics = model.evaluate(X_test, y_test, per_output=True)
    
    print("\nOverall Metrics:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   MAE: {metrics['mae']:.6f}")
    print(f"   R²: {metrics['r2']:.4f}")
    
    print(f"\nPer-Stock Metrics:")
    for i, ticker in enumerate(TARGET_TICKERS):
        print(f"\n   {ticker}:")
        print(f"      MSE: {metrics[f'mse_output_{i}']:.6f}")
        print(f"      RMSE: {metrics[f'rmse_output_{i}']:.6f}")
        print(f"      MAE: {metrics[f'mae_output_{i}']:.6f}")
        print(f"      R²: {metrics[f'r2_output_{i}']:.4f}")
    
    # Step 9: Make predictions
    print("\n[10] Generating predictions...")
    predictions = model.predict(X_test)
    
    # Step 10: Visualize results
    print("\n[11] Visualizing results...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Output Prediction: DOW30 → BMO + JPM', fontsize=16)
    
    # Training history
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training History')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # BMO predictions
    axes[0, 1].plot(y_test[:, 0], label='Actual BMO', alpha=0.7, linewidth=2)
    axes[0, 1].plot(predictions[:, 0], label='Predicted BMO', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Returns')
    axes[0, 1].set_title(f'BMO Predictions (R² = {metrics["r2_output_0"]:.4f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # JPM predictions
    axes[1, 0].plot(y_test[:, 1], label='Actual JPM', alpha=0.7, linewidth=2)
    axes[1, 0].plot(predictions[:, 1], label='Predicted JPM', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Returns')
    axes[1, 0].set_title(f'JPM Predictions (R² = {metrics["r2_output_1"]:.4f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot: Actual vs Predicted (both stocks)
    axes[1, 1].scatter(y_test[:, 0], predictions[:, 0], alpha=0.5, label='BMO', s=30)
    axes[1, 1].scatter(y_test[:, 1], predictions[:, 1], alpha=0.5, label='JPM', s=30)
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                     'k--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_xlabel('Actual Returns')
    axes[1, 1].set_ylabel('Predicted Returns')
    axes[1, 1].set_title('Actual vs Predicted (Both Stocks)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'multi_output_prediction.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n   Plot saved to: {plot_path}")
    
    plt.show()
    
    # Step 11: Save model
    print("\n[12] Saving model...")
    model_path = os.path.join(output_dir, 'lstm_multi_output_model.pth')
    model.save(model_path)
    print(f"   Model saved to: {model_path}")
    
    print("\n" + "=" * 80)
    print("Multi-Output Prediction Complete!")
    print("=" * 80)
    print(f"\nKey Achievements:")
    print(f"  ✓ Successfully predicted 2 stocks simultaneously (BMO and JPM)")
    print(f"  ✓ Used 30 DOW30 stocks as input features")
    print(f"  ✓ Handled zero-division robustly in return calculations")
    print(f"  ✓ Achieved R² scores: BMO={metrics['r2_output_0']:.4f}, JPM={metrics['r2_output_1']:.4f}")
    print(f"  ✓ Model parameters: {model.get_num_parameters():,}")
    

if __name__ == '__main__':
    main()
