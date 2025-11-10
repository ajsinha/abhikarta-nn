# Practical Usage Guide - Time Series Models

## 📋 Quick Navigation
- [Installation & Setup](#installation--setup)
- [Basic Workflow](#basic-workflow)
- [Model-Specific Examples](#model-specific-examples)
- [Advanced Techniques](#advanced-techniques)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

---

## Installation & Setup

### 1. Install Dependencies

```bash
# Basic installation
pip install numpy pandas matplotlib scikit-learn torch torchvision

# Or use requirements file
pip install -r requirements_pytorch.txt

# For GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify Installation

```python
import torch
import numpy as np
import pandas as pd

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

### 3. Import the Module

```python
from timeseries_prtediction.timeseries_all_models import (
    TimeSeriesRatioPreprocessor,
    TimeSeriesDataset,
    create_model,
    train_model,
    evaluate_model,
    LSTMModel,
    GRUModel,
    # ... other models
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
```

---

## Basic Workflow

### Step 1: Load Your Data

```python
import pandas as pd
import numpy as np

# Option 1: From CSV
df = pd.read_csv('your_data.csv')
X_raw = df[['feature1', 'feature2', ..., 'feature10']].values
y_raw = df[['target']].values

# Option 2: From NumPy arrays
# X_raw shape: (n_samples, 10)
# y_raw shape: (n_samples, 1)

print(f"Data shape: X={X_raw.shape}, y={y_raw.shape}")
```

### Step 2: Preprocess with Ratio Transformation

```python
from timeseries_prtediction.timeseries_all_models import TimeSeriesRatioPreprocessor

# Create preprocessors
X_preprocessor = TimeSeriesRatioPreprocessor()
y_preprocessor = TimeSeriesRatioPreprocessor()

# Fit and transform
X_scaled = X_preprocessor.fit_transform(X_raw)
y_scaled = y_preprocessor.fit_transform(y_raw)

print(f"Preprocessed shape: X={X_scaled.shape}, y={y_scaled.shape}")
```

### Step 3: Create Datasets

```python
from timeseries_prtediction.timeseries_all_models import TimeSeriesDataset
from torch.utils.data import DataLoader

# Split data temporally
train_size = int(0.7 * len(X_scaled))
val_size = int(0.15 * len(X_scaled))

X_train = X_scaled[:train_size]
y_train = y_scaled[:train_size]
X_val = X_scaled[train_size:train_size + val_size]
y_val = y_scaled[train_size:train_size + val_size]
X_test = X_scaled[train_size + val_size:]
y_test = y_scaled[train_size + val_size:]

# Create PyTorch datasets
sequence_length = 20
batch_size = 32

train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
```

### Step 4: Create and Train Model

```python
from timeseries_prtediction.timeseries_all_models import create_model, train_model

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('lstm', input_size=10, hidden_sizes=[64, 32], dropout=0.2)
model = model.to(device)

# Setup training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
history, best_state = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=100,
    device=device,
    patience=15,
    model_name="LSTM"
)

# Load best model
model.load_state_dict(best_state)
```

### Step 5: Evaluate

```python
from timeseries_prtediction.timeseries_all_models import evaluate_model

predictions, actuals, metrics = evaluate_model(model, test_loader, device)

print(f"\nTest Results:")
print(f"MSE: {metrics['mse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Step 6: Visualize Results

```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
plt.plot(actuals[:100], label='Actual', marker='o', markersize=3)
plt.plot(predictions[:100], label='Predicted', marker='x', markersize=3)
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.title('Predictions vs Actual')

plt.tight_layout()
plt.savefig('results.png')
plt.show()
```

### Step 7: Save Model

```python
import pickle

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'model_type': 'lstm',
    'config': {
        'input_size': 10,
        'hidden_sizes': [64, 32],
        'dropout': 0.2,
        'sequence_length': sequence_length
    }
}, 'trained_model.pth')

# Save preprocessors
with open('preprocessors.pkl', 'wb') as f:
    pickle.dump({
        'X_preprocessor': X_preprocessor,
        'y_preprocessor': y_preprocessor,
        'sequence_length': sequence_length
    }, f)

print("Model and preprocessors saved!")
```

---

## Model-Specific Examples

### Example 1: LSTM for General Time Series

```python
# Best for: Most time series problems
# Good balance of performance and speed

model = create_model(
    'lstm',
    input_size=10,
    hidden_sizes=[64, 32],  # Two LSTM layers
    dropout=0.2
)

# Training tips:
# - Start with learning_rate=0.001
# - Use sequence_length=20-30
# - Monitor validation loss for early stopping
```

### Example 2: GRU for Fast Training

```python
# Best for: Large datasets, when speed matters
# 30-40% faster than LSTM with similar performance

model = create_model(
    'gru',
    input_size=10,
    hidden_sizes=[64, 32],
    dropout=0.2
)

# Use larger batch size for speed
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training will be faster with minimal accuracy loss
```

### Example 3: BiLSTM for Pattern Recognition

```python
# Best for: Post-hoc analysis (not real-time)
# WARNING: Needs full sequence, cannot predict incrementally

model = create_model(
    'bilstm',
    input_size=10,
    hidden_sizes=[64, 32],
    dropout=0.2
)

# Good for:
# - Anomaly detection in historical data
# - Pattern classification
# - Filling in missing values

# NOT good for:
# - Real-time prediction
# - Live trading systems
```

### Example 4: CNN-LSTM for High-Frequency Data

```python
# Best for: Stock prices, sensor data, high-frequency signals
# Extracts local features before temporal modeling

model = create_model(
    'cnn_lstm',
    input_size=10,
    cnn_filters=64,      # More filters = more feature extraction
    lstm_hidden=64,      # LSTM capacity
    dropout=0.3          # Higher dropout for noisy data
)

# Recommended settings:
# - Use shorter sequences (10-20)
# - Higher dropout (0.3-0.4)
# - Batch size 32-64
```

### Example 5: TCN for Long Sequences

```python
# Best for: Long sequences (50-200 steps)
# Very fast and efficient

model = create_model(
    'tcn',
    input_size=10,
    num_channels=[64, 64, 32],  # Each level doubles receptive field
    kernel_size=3,
    dropout=0.2
)

# Receptive field calculation:
# RF = (kernel_size - 1) * (2^num_layers - 1) + 1
# With kernel_size=3, 3 layers: (3-1) * (2^3-1) + 1 = 15

# Good for:
# - Sequences >50 time steps
# - Real-time systems
# - When RNN is too slow
```

### Example 6: Transformer for Complex Patterns

```python
# Best for: Large datasets (1000+ samples), complex patterns
# State-of-the-art but needs more data

model = create_model(
    'transformer',
    input_size=10,
    d_model=64,          # Model dimension (try 64, 128, 256)
    nhead=4,             # Number of attention heads (4, 8)
    num_layers=2,        # Transformer layers (2-4)
    dropout=0.1
)

# IMPORTANT:
# - Needs lots of data (1000+ samples)
# - Use learning rate scheduler
# - May need warmup

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# In training loop:
# scheduler.step(val_loss)
```

### Example 7: Attention-LSTM for Interpretability

```python
# Best for: When you need to understand predictions
# Shows which time steps are most important

model = create_model(
    'attention_lstm',
    input_size=10,
    hidden_size=64,
    dropout=0.2
)

# After training, extract attention weights:
model.eval()
with torch.no_grad():
    sample = test_dataset[0][0].unsqueeze(0).to(device)
    
    # Forward pass through LSTM
    lstm_out, _ = model.lstm(sample)
    
    # Get attention weights
    context, attention_weights = model.attention(lstm_out)
    
    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(attention_weights[0])), 
            attention_weights[0].cpu().numpy())
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.title('Which Time Steps Matter Most?')
    plt.show()
```

### Example 8: MLP as Baseline

```python
# Always start with this as baseline!
# If MLP performs well, your problem might be simple

model = create_model(
    'mlp',
    input_size=10,
    sequence_length=20,
    hidden_sizes=[128, 64, 32],
    dropout=0.2
)

# If MLP performs well:
# - Your problem might be simpler than expected
# - Consider using MLP for speed
# - Still try LSTM/GRU to see if they improve

# If MLP performs poorly:
# - Temporal patterns are important
# - Use LSTM/GRU/TCN
```

---

## Advanced Techniques

### 1. Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Option 1: Reduce on plateau
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Minimize loss
    factor=0.5,      # Multiply lr by 0.5
    patience=5,      # After 5 epochs without improvement
    verbose=True
)

# Option 2: Cosine annealing
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=50,        # Number of epochs
    eta_min=1e-6     # Minimum learning rate
)

# In training loop:
for epoch in range(num_epochs):
    # ... training ...
    val_loss = validate()
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    # scheduler.step()  # For CosineAnnealingLR
```

### 2. Gradient Clipping

```python
# Prevents exploding gradients
max_grad_norm = 1.0

for batch_X, batch_y in train_loader:
    optimizer.zero_grad()
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    optimizer.step()
```

### 3. Mixed Precision Training (Faster on GPU)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_X, batch_y in train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    
    optimizer.zero_grad()
    
    # Use autocast for forward pass
    with autocast():
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
    
    # Scale loss and backward
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4. Ensemble Methods

```python
# Train multiple models and average predictions

models = []
for i in range(5):
    model = create_model('lstm', input_size=10)
    # ... train model with different random seed ...
    models.append(model)

# Make ensemble predictions
def ensemble_predict(models, test_loader, device):
    all_predictions = []
    
    for model in models:
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions.extend(outputs.cpu().numpy())
        all_predictions.append(predictions)
    
    # Average predictions
    ensemble_pred = np.mean(all_predictions, axis=0)
    return ensemble_pred
```

### 5. Cross-Validation for Time Series

```python
# Walk-forward validation
def walk_forward_validation(X, y, n_splits=5):
    results = []
    split_size = len(X) // (n_splits + 1)
    
    for i in range(n_splits):
        # Expanding window
        train_end = split_size * (i + 2)
        test_start = train_end
        test_end = train_end + split_size
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Train and evaluate
        # ... training code ...
        
        results.append(metrics)
    
    return results
```

### 6. Hyperparameter Tuning with Optuna

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    # Create model
    model = create_model('lstm', input_size=10, 
                        hidden_sizes=[hidden_size, hidden_size//2],
                        dropout=dropout)
    model = model.to(device)
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history, best_state = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, num_epochs=30, device=device
    )
    
    # Return validation loss
    return min(history['val_loss'])

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f"Best parameters: {study.best_params}")
print(f"Best validation loss: {study.best_value}")
```

---

## Troubleshooting

### Problem 1: Loss Not Decreasing

**Symptoms:**
- Loss stays flat or decreases very slowly
- Training loss and validation loss both high

**Solutions:**
```python
# 1. Increase learning rate
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Was 0.001

# 2. Use larger model
model = create_model('lstm', input_size=10, hidden_sizes=[128, 64, 32])

# 3. Decrease dropout
model = create_model('lstm', input_size=10, dropout=0.1)  # Was 0.2

# 4. Check data preprocessing
# Make sure ratios are calculated correctly
# Verify data doesn't have NaN or Inf values

# 5. Try different model
model = create_model('transformer', input_size=10)  # More capacity
```

### Problem 2: Overfitting

**Symptoms:**
- Training loss low, validation loss high
- Gap between train and val loss increasing

**Solutions:**
```python
# 1. Increase dropout
model = create_model('lstm', input_size=10, dropout=0.4)  # Was 0.2

# 2. Use smaller model
model = create_model('lstm', input_size=10, hidden_sizes=[32, 16])

# 3. Early stopping (already included)
# Adjust patience
history, best_state = train_model(..., patience=10)  # Was 15

# 4. Add L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 5. Get more training data
# No code solution - need more data!
```

### Problem 3: Slow Training

**Symptoms:**
- Training takes too long
- Each epoch is slow

**Solutions:**
```python
# 1. Use GRU instead of LSTM (30-40% faster)
model = create_model('gru', input_size=10)

# 2. Increase batch size
train_loader = DataLoader(train_dataset, batch_size=64)  # Was 32

# 3. Reduce sequence length
sequence_length = 15  # Was 20

# 4. Use smaller model
model = create_model('lstm', input_size=10, hidden_sizes=[32])

# 5. Use GPU
device = torch.device('cuda')  # Make sure CUDA is available

# 6. Use mixed precision (if on GPU)
from torch.cuda.amp import autocast, GradScaler
# See Advanced Techniques section
```

### Problem 4: CUDA Out of Memory

**Symptoms:**
- Error: "CUDA out of memory"

**Solutions:**
```python
# 1. Reduce batch size
train_loader = DataLoader(train_dataset, batch_size=16)  # Was 32

# 2. Reduce sequence length
sequence_length = 10  # Was 20

# 3. Use smaller model
model = create_model('lstm', input_size=10, hidden_sizes=[32, 16])

# 4. Clear cache
torch.cuda.empty_cache()

# 5. Use gradient accumulation
accumulation_steps = 4
for i, (batch_X, batch_y) in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Problem 5: NaN Loss

**Symptoms:**
- Loss becomes NaN during training

**Solutions:**
```python
# 1. Check data for NaN/Inf
assert not np.isnan(X_raw).any(), "Data contains NaN"
assert not np.isinf(X_raw).any(), "Data contains Inf"

# 2. Reduce learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Was 0.001

# 3. Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Check for division by zero in preprocessing
# Already handled in TimeSeriesRatioPreprocessor

# 5. Initialize model weights carefully
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
```

---

## Production Deployment

### 1. Save Model for Production

```python
# Save complete model info
torch.save({
    'model_state_dict': model.state_dict(),
    'model_type': 'lstm',
    'model_config': {
        'input_size': 10,
        'hidden_sizes': [64, 32],
        'dropout': 0.2
    },
    'sequence_length': 20,
    'train_metrics': metrics,
    'pytorch_version': torch.__version__,
    'timestamp': pd.Timestamp.now().isoformat()
}, 'production_model.pth')

# Save preprocessors
with open('production_preprocessors.pkl', 'wb') as f:
    pickle.dump({
        'X_preprocessor': X_preprocessor,
        'y_preprocessor': y_preprocessor
    }, f)
```

### 2. Load Model for Inference

```python
import torch
import pickle

def load_production_model(model_path, preprocessor_path, device='cpu'):
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model
    model = create_model(
        checkpoint['model_type'],
        input_size=checkpoint['model_config']['input_size'],
        **{k: v for k, v in checkpoint['model_config'].items() if k != 'input_size'}
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load preprocessors
    with open(preprocessor_path, 'rb') as f:
        preprocessors = pickle.load(f)
    
    return model, preprocessors, checkpoint['sequence_length']

# Usage
model, preprocessors, seq_len = load_production_model(
    'production_model.pth',
    'production_preprocessors.pkl',
    device='cpu'
)
```

### 3. Make Predictions on New Data

```python
def predict_next_value(model, preprocessors, new_data, sequence_length, device='cpu'):
    """
    Make prediction on new data
    
    Args:
        model: Trained model
        preprocessors: Dict with X_preprocessor
        new_data: NumPy array of shape (n_samples, n_features)
        sequence_length: Length of input sequences
        device: 'cpu' or 'cuda'
    
    Returns:
        predictions: Array of predictions
    """
    # Preprocess
    X_scaled = preprocessors['X_preprocessor'].transform(new_data)
    
    # Need at least sequence_length samples
    if len(X_scaled) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} samples")
    
    # Take last sequence_length samples
    X_seq = X_scaled[-sequence_length:]
    X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model(X_tensor)
    
    return prediction.cpu().numpy()[0]

# Example usage
new_data = np.random.randn(30, 10)  # New 30 samples
prediction = predict_next_value(model, preprocessors, new_data, seq_len)
print(f"Next predicted value: {prediction}")
```

### 4. Batch Predictions

```python
def batch_predict(model, preprocessors, data, sequence_length, device='cpu', batch_size=32):
    """Make predictions for multiple sequences"""
    
    X_scaled = preprocessors['X_preprocessor'].transform(data)
    
    # Create dataset
    dummy_y = np.zeros((len(X_scaled), 1))
    dataset = TimeSeriesDataset(X_scaled, dummy_y, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Predict
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)
```

### 5. Real-Time Inference Server (Flask Example)

```python
from flask import Flask, request, jsonify
import torch
import numpy as np

app = Flask(__name__)

# Load model at startup
model, preprocessors, seq_len = load_production_model(
    'production_model.pth',
    'production_preprocessors.pkl'
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json['data']  # List of lists
        data = np.array(data)
        
        # Make prediction
        prediction = predict_next_value(model, preprocessors, data, seq_len)
        
        return jsonify({
            'prediction': float(prediction[0]),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 6. Monitoring in Production

```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='model_predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def monitored_predict(model, data):
    start_time = datetime.now()
    
    try:
        prediction = predict_next_value(model, preprocessors, data, seq_len)
        
        # Log successful prediction
        logging.info(f"Prediction: {prediction[0]:.4f}, "
                    f"Time: {(datetime.now() - start_time).total_seconds():.3f}s")
        
        return prediction
    
    except Exception as e:
        # Log error
        logging.error(f"Error: {str(e)}")
        raise
```

---

## Complete Example: End-to-End

```python
"""
Complete example: Load data → Preprocess → Train → Evaluate → Save → Deploy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle

from timeseries_prtediction.timeseries_all_models import (
    TimeSeriesRatioPreprocessor,
    TimeSeriesDataset,
    create_model,
    train_model,
    evaluate_model
)

# 1. LOAD DATA
print("Step 1: Loading data...")
df = pd.read_csv('your_data.csv')
X_raw = df[['var1', 'var2', 'var3', 'var4', 'var5',
            'var6', 'var7', 'var8', 'var9', 'var10']].values
y_raw = df[['target']].values

# 2. PREPROCESS
print("Step 2: Preprocessing...")
X_preprocessor = TimeSeriesRatioPreprocessor()
y_preprocessor = TimeSeriesRatioPreprocessor()

X_scaled = X_preprocessor.fit_transform(X_raw)
y_scaled = y_preprocessor.fit_transform(y_raw)

# 3. CREATE DATASETS
print("Step 3: Creating datasets...")
sequence_length = 20
batch_size = 32

# Split
train_size = int(0.7 * len(X_scaled))
val_size = int(0.15 * len(X_scaled))

X_train = X_scaled[:train_size]
y_train = y_scaled[:train_size]
X_val = X_scaled[train_size:train_size + val_size]
y_val = y_scaled[train_size:train_size + val_size]
X_test = X_scaled[train_size + val_size:]
y_test = y_scaled[train_size + val_size:]

# Datasets
train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. CREATE MODEL
print("Step 4: Creating model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('lstm', input_size=10, hidden_sizes=[64, 32], dropout=0.2)
model = model.to(device)

# 5. TRAIN
print("Step 5: Training...")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history, best_state = train_model(
    model, train_loader, val_loader,
    criterion, optimizer, num_epochs=100, device=device,
    patience=15, model_name="LSTM"
)

model.load_state_dict(best_state)

# 6. EVALUATE
print("Step 6: Evaluating...")
predictions, actuals, metrics = evaluate_model(model, test_loader, device)

print(f"\nTest Results:")
print(f"MSE: {metrics['mse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")

# 7. VISUALIZE
print("Step 7: Visualizing...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
plt.plot(actuals[:100], label='Actual')
plt.plot(predictions[:100], label='Predicted')
plt.legend()
plt.title('Predictions')

plt.tight_layout()
plt.savefig('final_results.png')

# 8. SAVE
print("Step 8: Saving model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'model_type': 'lstm',
    'model_config': {'input_size': 10, 'hidden_sizes': [64, 32], 'dropout': 0.2},
    'sequence_length': sequence_length,
    'metrics': metrics
}, 'final_model.pth')

with open('final_preprocessors.pkl', 'wb') as f:
    pickle.dump({
        'X_preprocessor': X_preprocessor,
        'y_preprocessor': y_preprocessor
    }, f)

print("\n✅ Complete! Model ready for production.")
```

---

**Happy Modeling! 🚀**
