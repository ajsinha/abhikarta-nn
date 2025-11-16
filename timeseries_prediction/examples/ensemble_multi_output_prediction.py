"""
Ensemble Multi-Output Prediction Example
=========================================

This example demonstrates using ensemble methods to predict multiple stocks
(BMO and JPM) simultaneously using DOW30 stocks as features.

Ensemble combines:
- LSTM Model
- GRU Model  
- Transformer Model

Features: DOW30 daily returns (30 features)
Targets: BMO and JPM daily returns (2 outputs)
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
from timeseries_prediction.models.ensemble_models import EnsembleModel
from timeseries_prediction.utils.data_utils import (
    download_stock_data, get_dow30_tickers, calculate_returns,
    create_sequences, train_val_test_split, normalize_data
)


def main():
    print("=" * 80)
    print("Ensemble Multi-Output Prediction: DOW30 → BMO + JPM")
    print("=" * 80)
    
    # Configuration
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    LOOKBACK = 20
    TARGET_TICKERS = ['BMO', 'JPM']
    
    # Step 1: Download and prepare data
    print("\n[1] Downloading DOW30 stock data...")
    dow30_tickers = get_dow30_tickers()
    dow30_prices = download_stock_data(dow30_tickers, START_DATE, END_DATE)
    
    print("\n[2] Downloading target stock data (BMO and JPM)...")
    target_prices = download_stock_data(TARGET_TICKERS, START_DATE, END_DATE)
    
    print("\n[3] Calculating robust daily returns...")
    dow30_returns = calculate_returns(dow30_prices, method='pct_change', handle_zeros=True)
    target_returns = calculate_returns(target_prices, method='pct_change', handle_zeros=True)
    
    # Align dates
    common_dates = dow30_returns.index.intersection(target_returns.index)
    dow30_returns = dow30_returns.loc[common_dates]
    target_returns = target_returns.loc[common_dates]
    
    print(f"\n   Data: {len(common_dates)} days, {dow30_returns.shape[1]} features, {target_returns.shape[1]} targets")
    
    # Step 2: Create sequences and split
    print(f"\n[4] Creating sequences (lookback={LOOKBACK})...")
    X, _ = create_sequences(dow30_returns.values, LOOKBACK, horizon=1)
    _, y_targets = create_sequences(target_returns.values, LOOKBACK, horizon=1)
    
    print(f"   X: {X.shape}, y: {y_targets.shape}")
    
    print("\n[5] Splitting data (70/15/15)...")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y_targets, train_ratio=0.7, val_ratio=0.15
    )
    
    # Step 3: Normalize
    print("\n[6] Normalizing features...")
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_val_2d = X_val.reshape(-1, X_val.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_norm, X_val_norm, X_test_norm, stats = normalize_data(
        X_train_2d, X_val_2d, X_test_2d, method='standardize'
    )
    
    X_train = X_train_norm.reshape(X_train.shape)
    X_val = X_val_norm.reshape(X_val.shape)
    X_test = X_test_norm.reshape(X_test.shape)
    
    # Step 4: Build individual models
    INPUT_SIZE = dow30_returns.shape[1]
    OUTPUT_SIZE = len(TARGET_TICKERS)
    
    print("\n[7] Building ensemble components...")
    print("   Creating LSTM model...")
    lstm_model = LSTMModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    print("   Creating GRU model...")
    gru_model = GRUModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    print("   Creating Transformer model...")
    transformer_model = TransformerModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.2
    )
    
    # Step 5: Create ensemble
    print("\n[8] Creating ensemble model...")
    ensemble = EnsembleModel(
        models=[lstm_model, gru_model, transformer_model],
        model_names=['LSTM', 'GRU', 'Transformer'],
        ensemble_method='mean'  # or 'median', 'weighted'
    )
    
    print(f"\n   Ensemble contains {len(ensemble.models)} models:")
    for name in ensemble.model_names:
        print(f"      - {name}")
    
    # Step 6: Train ensemble (trains all models)
    print("\n[9] Training ensemble models...")
    print("   This will train LSTM, GRU, and Transformer separately...")
    
    histories = ensemble.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=80,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=12,
        verbose=True
    )
    
    # Step 7: Evaluate ensemble
    print("\n[10] Evaluating ensemble on test set...")
    metrics = ensemble.evaluate(X_test, y_test, per_output=True)
    
    print("\nEnsemble Overall Metrics:")
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
    
    # Step 8: Evaluate individual models for comparison
    print("\n[11] Evaluating individual models...")
    individual_metrics = {}
    
    for i, (model, name) in enumerate(zip(ensemble.models, ensemble.model_names)):
        model_metrics = model.evaluate(X_test, y_test)
        individual_metrics[name] = model_metrics
        print(f"\n   {name}:")
        print(f"      RMSE: {model_metrics['rmse']:.6f}")
        print(f"      R²: {model_metrics['r2']:.4f}")
    
    # Step 9: Make predictions
    print("\n[12] Generating predictions...")
    ensemble_predictions = ensemble.predict(X_test)
    
    # Get individual predictions for comparison
    individual_predictions = {}
    for model, name in zip(ensemble.models, ensemble.model_names):
        individual_predictions[name] = model.predict(X_test)
    
    # Step 10: Visualize results
    print("\n[13] Visualizing results...")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Ensemble Multi-Output Prediction: DOW30 → BMO + JPM', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Training histories for each model
    for i, (name, history) in enumerate(histories.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(history['train_loss'], label='Train', linewidth=2, alpha=0.8)
        ax.plot(history['val_loss'], label='Val', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title(f'{name} Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 2: BMO predictions (ensemble vs individual)
    ax_bmo = fig.add_subplot(gs[1, :])
    ax_bmo.plot(y_test[:, 0], label='Actual BMO', linewidth=3, alpha=0.9, color='black')
    ax_bmo.plot(ensemble_predictions[:, 0], label='Ensemble BMO', 
                linewidth=2.5, alpha=0.9, color='red', linestyle='--')
    
    colors = ['blue', 'green', 'purple']
    for i, (name, pred) in enumerate(individual_predictions.items()):
        ax_bmo.plot(pred[:, 0], label=f'{name} BMO', 
                   linewidth=1.5, alpha=0.6, color=colors[i], linestyle=':')
    
    ax_bmo.set_xlabel('Time Steps')
    ax_bmo.set_ylabel('Returns')
    ax_bmo.set_title(f'BMO Predictions (Ensemble R² = {metrics["r2_output_0"]:.4f})')
    ax_bmo.legend(loc='best', ncol=5)
    ax_bmo.grid(True, alpha=0.3)
    
    # Row 3: JPM predictions (ensemble vs individual)
    ax_jpm = fig.add_subplot(gs[2, :])
    ax_jpm.plot(y_test[:, 1], label='Actual JPM', linewidth=3, alpha=0.9, color='black')
    ax_jpm.plot(ensemble_predictions[:, 1], label='Ensemble JPM', 
                linewidth=2.5, alpha=0.9, color='red', linestyle='--')
    
    for i, (name, pred) in enumerate(individual_predictions.items()):
        ax_jpm.plot(pred[:, 1], label=f'{name} JPM', 
                   linewidth=1.5, alpha=0.6, color=colors[i], linestyle=':')
    
    ax_jpm.set_xlabel('Time Steps')
    ax_jpm.set_ylabel('Returns')
    ax_jpm.set_title(f'JPM Predictions (Ensemble R² = {metrics["r2_output_1"]:.4f})')
    ax_jpm.legend(loc='best', ncol=5)
    ax_jpm.grid(True, alpha=0.3)
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'ensemble_multi_output_prediction.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n   Plot saved to: {plot_path}")
    
    plt.show()
    
    # Step 11: Model comparison
    print("\n[14] Model Performance Comparison:")
    print("\n   " + "-" * 60)
    print(f"   {'Model':<15} {'RMSE':>10} {'R²':>10} {'MAE':>10}")
    print("   " + "-" * 60)
    
    for name, m in individual_metrics.items():
        print(f"   {name:<15} {m['rmse']:>10.6f} {m['r2']:>10.4f} {m['mae']:>10.6f}")
    
    print("   " + "-" * 60)
    print(f"   {'ENSEMBLE':<15} {metrics['rmse']:>10.6f} {metrics['r2']:>10.4f} {metrics['mae']:>10.6f}")
    print("   " + "-" * 60)
    
    # Step 12: Save ensemble
    print("\n[15] Saving ensemble model...")
    model_path = os.path.join(output_dir, 'ensemble_multi_output_model.pth')
    ensemble.save(model_path)
    print(f"   Ensemble saved to: {model_path}")
    
    print("\n" + "=" * 80)
    print("Ensemble Multi-Output Prediction Complete!")
    print("=" * 80)
    print(f"\nKey Achievements:")
    print(f"  ✓ Ensemble of 3 models (LSTM, GRU, Transformer)")
    print(f"  ✓ Predicted 2 stocks simultaneously (BMO and JPM)")
    print(f"  ✓ Used 30 DOW30 stocks as features")
    print(f"  ✓ Robust return calculation with zero-division handling")
    print(f"  ✓ Ensemble R² scores: BMO={metrics['r2_output_0']:.4f}, JPM={metrics['r2_output_1']:.4f}")
    
    # Show if ensemble improved over individual models
    best_individual_r2 = max([m['r2'] for m in individual_metrics.values()])
    improvement = metrics['r2'] - best_individual_r2
    print(f"  ✓ Ensemble improvement over best individual: {improvement:+.4f} R²")
    

if __name__ == '__main__':
    main()
