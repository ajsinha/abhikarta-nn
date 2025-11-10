"""
SIMPLE EXAMPLE - Configurable Transformation Methods
====================================================

This example demonstrates how to use the enhanced preprocessing
with configurable transformation methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from timeseries_enhanced_config import (
    TransformConfig,
    TransformMethod,
    EnhancedTimeSeriesPreprocessor,
    TimeSeriesDataset,
    LSTMModel,
    train_model,
    evaluate_model
)

print("=" * 70)
print("SIMPLE EXAMPLE: Configurable Transformation Methods")
print("=" * 70)

# ============================================================================
# STEP 1: GENERATE SAMPLE DATA
# ============================================================================

print("\nStep 1: Generating sample data...")
n_samples = 1000
n_features = 10

# Generate time series with trend and seasonality
time = np.arange(n_samples)
X_raw = np.zeros((n_samples, n_features))

for i in range(n_features):
    trend = 100 + 0.1 * time + np.random.randn(n_samples) * 5
    seasonality = 10 * np.sin(2 * np.pi * time / 50 + i)
    X_raw[:, i] = trend + seasonality + np.random.randn(n_samples) * 2

y_raw = (0.3 * X_raw[:, 0] + 0.2 * X_raw[:, 1] + 
         0.15 * X_raw[:, 2] + np.random.randn(n_samples) * 3).reshape(-1, 1)

print(f"Generated data: X shape={X_raw.shape}, y shape={y_raw.shape}")
print(f"X range: [{X_raw.min():.2f}, {X_raw.max():.2f}]")
print(f"y range: [{y_raw.min():.2f}, {y_raw.max():.2f}]")

# ============================================================================
# STEP 2: CONFIGURE TRANSFORMATION METHOD
# ============================================================================

print("\n" + "=" * 70)
print("Step 2: Choose your transformation method")
print("=" * 70)

print("\nAvailable methods:")
print("1. RATIO: value(t) / value(t-1)")
print("2. FRACTIONAL_CHANGE: (value(t) - value(t-1)) / value(t-1)")
print("3. PERCENTAGE_CHANGE: 100 * (value(t) - value(t-1)) / value(t-1)")

# Option 1: Ratio (multiplicative changes)
config_ratio = TransformConfig(
    method=TransformMethod.RATIO,
    log_transform=True,
    clip_values=False
)

# Option 2: Fractional Change (additive returns) - RECOMMENDED for most cases
config_fractional = TransformConfig(
    method=TransformMethod.FRACTIONAL_CHANGE,
    log_transform=True,
    clip_values=False
)

# Option 3: Percentage Change (human-readable)
config_percentage = TransformConfig(
    method=TransformMethod.PERCENTAGE_CHANGE,
    log_transform=True,
    clip_values=False
)

# Choose one (we'll use fractional change as default)
config = config_fractional
print(f"\nUsing: {config}")

# ============================================================================
# STEP 3: PREPROCESS DATA
# ============================================================================

print("\nStep 3: Preprocessing data...")

# Create preprocessors
X_preprocessor = EnhancedTimeSeriesPreprocessor(config)
y_preprocessor = EnhancedTimeSeriesPreprocessor(config)

# Transform data
X_scaled = X_preprocessor.fit_transform(X_raw)
y_scaled = y_preprocessor.fit_transform(y_raw)

print(f"After transformation:")
print(f"X_scaled: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")
print(f"y_scaled: mean={y_scaled.mean():.4f}, std={y_scaled.std():.4f}")

# Get transformation info
info = X_preprocessor.get_transformation_info()
print(f"\nTransformation details:")
print(f"  Method: {info['method']}")
print(f"  Log transform: {info['log_transform']}")
print(f"  Clip values: {info['clip_values']}")

# ============================================================================
# STEP 4: CREATE DATASETS
# ============================================================================

print("\nStep 4: Creating datasets...")

sequence_length = 20
batch_size = 32

# Split data: 70% train, 15% val, 15% test
train_size = int(0.7 * len(X_scaled))
val_size = int(0.15 * len(X_scaled))

X_train = X_scaled[:train_size]
y_train = y_scaled[:train_size]
X_val = X_scaled[train_size:train_size + val_size]
y_val = y_scaled[train_size:train_size + val_size]
X_test = X_scaled[train_size + val_size:]
y_test = y_scaled[train_size + val_size:]

# Create PyTorch datasets
train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================================================
# STEP 5: CREATE AND TRAIN MODEL
# ============================================================================

print("\nStep 5: Creating and training model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=n_features, hidden_sizes=[64, 32], dropout=0.2)
model = model.to(device)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
history, best_state = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=50,
    device=device,
    patience=10,
    model_name=f"LSTM-{config.method.value}"
)

# Load best model
model.load_state_dict(best_state)

# ============================================================================
# STEP 6: EVALUATE
# ============================================================================

print("\nStep 6: Evaluating on test set...")

predictions, actuals, metrics = evaluate_model(model, test_loader, device)

print(f"\nTest Results:")
print(f"  MSE:  {metrics['mse']:.4f}")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  RMSE: {metrics['rmse']:.4f}")

# ============================================================================
# STEP 7: VISUALIZE
# ============================================================================

print("\nStep 7: Creating visualizations...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Training history
ax = axes[0, 0]
ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training History')
ax.legend()
ax.grid(True)

# Plot 2: Predictions vs Actual
ax = axes[0, 1]
samples_to_plot = min(100, len(predictions))
ax.plot(actuals[:samples_to_plot], label='Actual', marker='o', 
        markersize=3, alpha=0.7, linewidth=1.5)
ax.plot(predictions[:samples_to_plot], label='Predicted', marker='x', 
        markersize=3, alpha=0.7, linewidth=1.5)
ax.set_xlabel('Sample')
ax.set_ylabel('Scaled Value')
ax.set_title(f'Predictions (Method: {config.method.value})')
ax.legend()
ax.grid(True)

# Plot 3: Scatter plot
ax = axes[1, 0]
ax.scatter(actuals, predictions, alpha=0.5, s=10)
ax.plot([actuals.min(), actuals.max()], 
        [actuals.min(), actuals.max()], 
        'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Prediction Quality')
ax.legend()
ax.grid(True)

# Plot 4: Error distribution
ax = axes[1, 1]
errors = predictions.flatten() - actuals.flatten()
ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
ax.set_xlabel('Prediction Error')
ax.set_ylabel('Frequency')
ax.set_title(f'Error Distribution (Mean: {errors.mean():.4f})')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('simple_example_results.png', dpi=300, bbox_inches='tight')
print("Saved visualization to 'simple_example_results.png'")

# ============================================================================
# STEP 8: SAVE MODEL WITH CONFIGURATION
# ============================================================================

print("\nStep 8: Saving model with configuration...")

import pickle

# Save everything
save_dict = {
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_size': n_features,
        'hidden_sizes': [64, 32],
        'dropout': 0.2
    },
    'transform_config': config,
    'X_preprocessor': X_preprocessor,
    'y_preprocessor': y_preprocessor,
    'sequence_length': sequence_length,
    'metrics': metrics,
    'transformation_info': info
}

torch.save(save_dict, 'simple_example_model.pth')
print("Model saved to 'simple_example_model.pth'")

# ============================================================================
# STEP 9: DEMONSTRATE LOADING AND PREDICTION
# ============================================================================

print("\nStep 9: Demonstrating model loading and prediction...")

# Load model
checkpoint = torch.load('simple_example_model.pth')

# Recreate model
loaded_model = LSTMModel(
    input_size=checkpoint['model_config']['input_size'],
    hidden_sizes=checkpoint['model_config']['hidden_sizes'],
    dropout=checkpoint['model_config']['dropout']
)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model = loaded_model.to(device)
loaded_model.eval()

# Get preprocessor and config
loaded_X_preprocessor = checkpoint['X_preprocessor']
loaded_config = checkpoint['transform_config']
loaded_seq_len = checkpoint['sequence_length']

print(f"Loaded model with transformation: {loaded_config.method.value}")

# Make prediction on new data (last 20 samples)
new_data = X_raw[-loaded_seq_len:]
new_data_scaled = loaded_X_preprocessor.transform(new_data)
new_data_tensor = torch.FloatTensor(new_data_scaled).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = loaded_model(new_data_tensor)

print(f"New data prediction: {prediction.item():.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nTransformation Method: {config.method.value}")
print(f"Model: LSTM with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"\nPerformance:")
print(f"  MSE:  {metrics['mse']:.4f}")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"\nFiles created:")
print(f"  - simple_example_model.pth (model + config)")
print(f"  - simple_example_results.png (visualizations)")

print("\n" + "=" * 70)
print("TO TRY DIFFERENT TRANSFORMATION METHODS:")
print("=" * 70)
print("1. Change 'config' variable to config_ratio or config_percentage")
print("2. Re-run the script")
print("3. Compare results!")
print("\nOR use compare_transformation_methods() to test all automatically")
print("=" * 70)

print("\n✅ Example completed successfully!")
