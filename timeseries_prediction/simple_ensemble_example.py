"""
ENSEMBLE METHODS EXAMPLE
========================

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha
Email: ajsinha@gmail.com

Legal Notice:
This document and the associated software architecture are proprietary and confidential. 
Unauthorized copying, distribution, modification, or use of this document or the software 
system it describes is strictly prohibited without explicit written permission from the 
copyright holder. This document is provided "as is" without warranty of any kind, either 
expressed or implied. The copyright holder shall not be liable for any damages arising 
from the use of this document or the software system it describes.

Patent Pending: Certain architectural patterns and implementations described in this 
document may be subject to patent applications.

================================================================================

This example demonstrates all 5 ensemble methods:
1. Simple Averaging Ensemble
2. Weighted Averaging Ensemble
3. Stacking Ensemble
4. Bagging Ensemble
5. Diversity Ensemble
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from ensemble_methods import (
    AveragingEnsemble,
    WeightedEnsemble,
    StackingEnsemble,
    LSTMModel,
    GRUModel,
    train_single_model,
    evaluate_ensemble
)

print("="*70)
print("ENSEMBLE METHODS - COMPLETE EXAMPLE")
print("="*70)
print("\nCopyright © 2025-2030, All Rights Reserved")
print("Ashutosh Sinha (ajsinha@gmail.com)")
print("="*70)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# ============================================================================
# STEP 1: GENERATE SAMPLE DATA
# ============================================================================

print("\n" + "="*70)
print("STEP 1: GENERATING SAMPLE DATA")
print("="*70)

n_samples = 1000
n_features = 10
sequence_length = 20

# Generate time series with trend and seasonality
time = np.arange(n_samples)
X_raw = np.zeros((n_samples, n_features))

for i in range(n_features):
    trend = 100 + 0.1 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 50 + i)
    noise = np.random.randn(n_samples) * 2
    X_raw[:, i] = trend + seasonality + noise

# Create target variable
y_raw = (0.3 * X_raw[:, 0] + 0.2 * X_raw[:, 1] + 
         0.15 * X_raw[:, 2] + np.random.randn(n_samples) * 3).reshape(-1, 1)

print(f"Generated data:")
print(f"  X shape: {X_raw.shape}")
print(f"  y shape: {y_raw.shape}")
print(f"  X range: [{X_raw.min():.2f}, {X_raw.max():.2f}]")
print(f"  y range: [{y_raw.min():.2f}, {y_raw.max():.2f}]")

# ============================================================================
# STEP 2: PREPROCESS DATA
# ============================================================================

print("\n" + "="*70)
print("STEP 2: PREPROCESSING DATA")
print("="*70)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_raw).astype(np.float32)
y_scaled = scaler_y.fit_transform(y_raw).astype(np.float32)

print(f"After scaling:")
print(f"  X mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")
print(f"  y mean: {y_scaled.mean():.4f}, std: {y_scaled.std():.4f}")

# ============================================================================
# STEP 3: CREATE DATASETS
# ============================================================================

print("\n" + "="*70)
print("STEP 3: CREATING DATASETS")
print("="*70)

# Split data: 60% train, 20% val, 20% test
train_size = int(0.6 * len(X_scaled))
val_size = int(0.2 * len(X_scaled))

X_train = X_scaled[:train_size]
y_train = y_scaled[:train_size]
X_val = X_scaled[train_size:train_size + val_size]
y_val = y_scaled[train_size:train_size + val_size]
X_test = X_scaled[train_size + val_size:]
y_test = y_scaled[train_size + val_size:]

# Create sequences
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return torch.FloatTensor(np.array(X_seq)), torch.FloatTensor(np.array(y_seq))

X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

# Create data loaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), 
                         batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_seq, y_val_seq), 
                       batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_seq, y_test_seq), 
                        batch_size=batch_size, shuffle=False)

print(f"Dataset sizes:")
print(f"  Train: {len(X_train_seq)} sequences")
print(f"  Val:   {len(X_val_seq)} sequences")
print(f"  Test:  {len(X_test_seq)} sequences")

# ============================================================================
# STEP 4: TRAIN BASE MODELS
# ============================================================================

print("\n" + "="*70)
print("STEP 4: TRAINING BASE MODELS")
print("="*70)

base_models = []
n_base_models = 5

print(f"\nTraining {n_base_models} LSTM models...")

for i in range(n_base_models):
    print(f"\nTraining base model {i+1}/{n_base_models}...")
    
    # Create model with slight variations
    hidden_sizes = [[64, 32], [64, 48], [80, 40], [64, 32], [72, 36]][i]
    model = LSTMModel(n_features, hidden_sizes=hidden_sizes, dropout=0.2).to(device)
    
    # Train
    model = train_single_model(
        model, train_loader, val_loader, device,
        epochs=30, lr=0.001, patience=5
    )
    
    base_models.append(model)
    print(f"✓ Model {i+1} trained successfully")

print(f"\n✓ All {n_base_models} base models trained!")

# ============================================================================
# STEP 5: EVALUATE SINGLE MODELS
# ============================================================================

print("\n" + "="*70)
print("STEP 5: EVALUATING INDIVIDUAL MODELS")
print("="*70)

actuals = y_test_seq.numpy()

print(f"\n{'Model':<15} {'RMSE':<12} {'MAE':<12} {'MSE':<12}")
print("-"*70)

single_model_results = []
for i, model in enumerate(base_models):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
    
    predictions = np.array(predictions)
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    
    single_model_results.append({'rmse': rmse, 'mae': mae, 'mse': mse})
    print(f"Model {i+1:<9} {rmse:<12.4f} {mae:<12.4f} {mse:<12.4f}")

best_single_rmse = min(r['rmse'] for r in single_model_results)
print(f"\n🏆 Best single model RMSE: {best_single_rmse:.4f}")

# ============================================================================
# STEP 6: ENSEMBLE METHOD 1 - SIMPLE AVERAGING
# ============================================================================

print("\n" + "="*70)
print("STEP 6: ENSEMBLE METHOD 1 - SIMPLE AVERAGING")
print("="*70)

print("\nCreating averaging ensemble...")
averaging_ensemble = AveragingEnsemble(base_models)

print("Making predictions...")
metrics, predictions = evaluate_ensemble(averaging_ensemble, test_loader, actuals, device)

print(f"\nAveraging Ensemble Results:")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  MSE:  {metrics['mse']:.4f}")

improvement = ((best_single_rmse - metrics['rmse']) / best_single_rmse) * 100
print(f"\n📈 Improvement over best single model: {improvement:.2f}%")

averaging_results = {'metrics': metrics, 'predictions': predictions}

# ============================================================================
# STEP 7: ENSEMBLE METHOD 2 - WEIGHTED AVERAGING
# ============================================================================

print("\n" + "="*70)
print("STEP 7: ENSEMBLE METHOD 2 - WEIGHTED AVERAGING")
print("="*70)

print("\nCreating weighted ensemble based on validation performance...")
weighted_ensemble = WeightedEnsemble.from_validation_performance(
    base_models, val_loader, device, nn.MSELoss()
)

print(f"\nModel weights:")
for i, weight in enumerate(weighted_ensemble.weights):
    print(f"  Model {i+1}: {weight:.4f}")

print("\nMaking predictions...")
metrics, predictions = evaluate_ensemble(weighted_ensemble, test_loader, actuals, device)

print(f"\nWeighted Ensemble Results:")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  MSE:  {metrics['mse']:.4f}")

improvement = ((best_single_rmse - metrics['rmse']) / best_single_rmse) * 100
print(f"\n📈 Improvement over best single model: {improvement:.2f}%")

weighted_results = {'metrics': metrics, 'predictions': predictions}

# ============================================================================
# STEP 8: ENSEMBLE METHOD 3 - STACKING
# ============================================================================

print("\n" + "="*70)
print("STEP 8: ENSEMBLE METHOD 3 - STACKING ENSEMBLE")
print("="*70)

print("\nCreating stacking ensemble...")
stacking_ensemble = StackingEnsemble(base_models)

print("Training meta-learner...")
stacking_ensemble.train_meta_learner(val_loader, device, epochs=20, lr=0.001)

print("\nMaking predictions...")
metrics, predictions = evaluate_ensemble(stacking_ensemble, test_loader, actuals, device)

print(f"\nStacking Ensemble Results:")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  MSE:  {metrics['mse']:.4f}")

improvement = ((best_single_rmse - metrics['rmse']) / best_single_rmse) * 100
print(f"\n📈 Improvement over best single model: {improvement:.2f}%")

stacking_results = {'metrics': metrics, 'predictions': predictions}

# ============================================================================
# STEP 9: ENSEMBLE METHOD 4 - DIVERSITY (MULTIPLE ARCHITECTURES)
# ============================================================================

print("\n" + "="*70)
print("STEP 9: ENSEMBLE METHOD 4 - DIVERSITY ENSEMBLE")
print("="*70)

print("\nCreating diversity ensemble with different architectures...")

diversity_models = []

# Train 2 LSTM models
print("\nTraining LSTM models...")
for i in range(2):
    model = LSTMModel(n_features, hidden_sizes=[64, 32], dropout=0.2).to(device)
    model = train_single_model(model, train_loader, val_loader, device, epochs=20, patience=5)
    diversity_models.append(model)
    print(f"✓ LSTM model {i+1} trained")

# Train 2 GRU models
print("\nTraining GRU models...")
for i in range(2):
    model = GRUModel(n_features, hidden_sizes=[64, 32], dropout=0.2).to(device)
    model = train_single_model(model, train_loader, val_loader, device, epochs=20, patience=5)
    diversity_models.append(model)
    print(f"✓ GRU model {i+1} trained")

print("\nCreating averaging ensemble from diverse models...")
diversity_ensemble = AveragingEnsemble(diversity_models)

print("Making predictions...")
metrics, predictions = evaluate_ensemble(diversity_ensemble, test_loader, actuals, device)

print(f"\nDiversity Ensemble Results:")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  MSE:  {metrics['mse']:.4f}")

improvement = ((best_single_rmse - metrics['rmse']) / best_single_rmse) * 100
print(f"\n📈 Improvement over best single model: {improvement:.2f}%")

diversity_results = {'metrics': metrics, 'predictions': predictions}

# ============================================================================
# STEP 10: COMPARE ALL METHODS
# ============================================================================

print("\n" + "="*70)
print("STEP 10: FINAL COMPARISON - ALL ENSEMBLE METHODS")
print("="*70)

results = {
    'Best Single Model': {'metrics': {'rmse': best_single_rmse, 
                                      'mae': min(r['mae'] for r in single_model_results),
                                      'mse': min(r['mse'] for r in single_model_results)}},
    'Averaging Ensemble': averaging_results,
    'Weighted Ensemble': weighted_results,
    'Stacking Ensemble': stacking_results,
    'Diversity Ensemble': diversity_results
}

print(f"\n{'Method':<25} {'RMSE':<12} {'MAE':<12} {'MSE':<12} {'Improvement':<12}")
print("-"*80)

for method_name, result in results.items():
    m = result['metrics']
    if method_name == 'Best Single Model':
        improvement = 0.0
    else:
        improvement = ((best_single_rmse - m['rmse']) / best_single_rmse) * 100
    
    print(f"{method_name:<25} {m['rmse']:<12.4f} {m['mae']:<12.4f} "
          f"{m['mse']:<12.4f} {improvement:<12.2f}%")

# Find best ensemble
ensemble_results = {k: v for k, v in results.items() if k != 'Best Single Model'}
best_ensemble = min(ensemble_results.items(), key=lambda x: x[1]['metrics']['rmse'])

print(f"\n🏆 Best Ensemble Method: {best_ensemble[0]}")
print(f"   RMSE: {best_ensemble[1]['metrics']['rmse']:.4f}")
improvement = ((best_single_rmse - best_ensemble[1]['metrics']['rmse']) / best_single_rmse) * 100
print(f"   Improvement: {improvement:.2f}%")

# ============================================================================
# STEP 11: VISUALIZE RESULTS
# ============================================================================

print("\n" + "="*70)
print("STEP 11: CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: RMSE Comparison
ax = axes[0, 0]
methods = list(results.keys())
rmse_values = [results[m]['metrics']['rmse'] for m in methods]
colors = ['red'] + ['green'] * (len(methods) - 1)

bars = ax.bar(range(len(methods)), rmse_values, color=colors, alpha=0.7)
bars[0].set_color('lightcoral')
best_idx = rmse_values.index(min(rmse_values[1:]), 1)
bars[best_idx].set_color('darkgreen')

ax.set_xlabel('Method', fontsize=10)
ax.set_ylabel('RMSE', fontsize=10)
ax.set_title('RMSE Comparison Across All Methods', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=best_single_rmse, color='red', linestyle='--', alpha=0.5, label='Single Model Baseline')
ax.legend()

# Plot 2: Improvement Percentage
ax = axes[0, 1]
improvements = [0.0] + [((best_single_rmse - results[m]['metrics']['rmse']) / best_single_rmse) * 100 
                        for m in methods[1:]]

bars = ax.bar(range(len(methods)), improvements, alpha=0.7, color='skyblue')
bars[0].set_color('lightcoral')
best_idx = improvements.index(max(improvements))
bars[best_idx].set_color('darkgreen')

ax.set_xlabel('Method', fontsize=10)
ax.set_ylabel('Improvement (%)', fontsize=10)
ax.set_title('Improvement Over Best Single Model', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.5)

# Plot 3: Predictions - Best Ensemble
ax = axes[1, 0]
samples_to_plot = 100
best_predictions = best_ensemble[1]['predictions'][:samples_to_plot]
actual_sample = actuals[:samples_to_plot]

ax.plot(actual_sample, label='Actual', marker='o', markersize=3, alpha=0.7, linewidth=1.5)
ax.plot(best_predictions, label='Predicted', marker='x', markersize=3, alpha=0.7, linewidth=1.5)
ax.set_xlabel('Sample', fontsize=10)
ax.set_ylabel('Value', fontsize=10)
ax.set_title(f'Best Ensemble: {best_ensemble[0]}\n(RMSE: {best_ensemble[1]["metrics"]["rmse"]:.4f})', 
            fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Error Distribution
ax = axes[1, 1]
for method in list(results.keys())[1:3]:  # Plot first 2 ensemble methods
    predictions = results[method]['predictions']
    errors = predictions.flatten() - actuals.flatten()
    ax.hist(errors, bins=30, alpha=0.5, label=method)

ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero Error')
ax.set_xlabel('Prediction Error', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ensemble_methods_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization to 'ensemble_methods_comparison.png'")

# ============================================================================
# STEP 12: SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("STEP 12: SAVING RESULTS")
print("="*70)

# Save best ensemble
torch.save({
    'ensemble_type': best_ensemble[0],
    'base_models': [m.state_dict() for m in base_models],
    'metrics': best_ensemble[1]['metrics'],
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'config': {
        'n_features': n_features,
        'sequence_length': sequence_length,
        'hidden_sizes': [64, 32]
    }
}, 'best_ensemble.pth')

print("✓ Saved best ensemble to 'best_ensemble.pth'")

# Save all results
import pickle
with open('ensemble_comparison_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✓ Saved all results to 'ensemble_comparison_results.pkl'")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nBase Models Trained: {n_base_models}")
print(f"Ensemble Methods Tested: 4")
print(f"Best Single Model RMSE: {best_single_rmse:.4f}")
print(f"\nBest Ensemble: {best_ensemble[0]}")
print(f"Best Ensemble RMSE: {best_ensemble[1]['metrics']['rmse']:.4f}")
print(f"Improvement: {improvement:.2f}%")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

print("\n1. Simple Averaging:")
print("   - Easy to implement")
print(f"   - RMSE: {averaging_results['metrics']['rmse']:.4f}")

print("\n2. Weighted Averaging:")
print("   - Performance-based weights")
print(f"   - RMSE: {weighted_results['metrics']['rmse']:.4f}")

print("\n3. Stacking:")
print("   - Meta-learner combines predictions")
print(f"   - RMSE: {stacking_results['metrics']['rmse']:.4f}")

print("\n4. Diversity:")
print("   - Multiple architectures (LSTM + GRU)")
print(f"   - RMSE: {diversity_results['metrics']['rmse']:.4f}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("\n✓ Use Averaging for: Quick implementation, stable results")
print("✓ Use Weighted for: When models have different quality")
print("✓ Use Stacking for: Maximum performance, have validation data")
print("✓ Use Diversity for: Robust predictions, capture different patterns")

print("\n" + "="*70)
print("FILES CREATED")
print("="*70)

print("\n✓ ensemble_methods_comparison.png - Visualization")
print("✓ best_ensemble.pth - Best ensemble model")
print("✓ ensemble_comparison_results.pkl - All results")

print("\n" + "="*70)
print("✅ ENSEMBLE METHODS EXAMPLE COMPLETED SUCCESSFULLY!")
print("="*70)

print("\nCopyright © 2025-2030, All Rights Reserved")
print("Ashutosh Sinha | ajsinha@gmail.com")
print("="*70)
