# Complete Guide to Time Series Models

## 📚 Table of Contents
1. [Model Overview](#model-overview)
2. [Detailed Model Descriptions](#detailed-model-descriptions)
3. [When to Use Each Model](#when-to-use-each-model)
4. [Performance Comparison](#performance-comparison)
5. [Usage Examples](#usage-examples)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Best Practices](#best-practices)

---

## Model Overview

This package includes **8 different neural network architectures** for time series prediction:

| Model | Type | Complexity | Speed | Best For |
|-------|------|------------|-------|----------|
| **LSTM** | Recurrent | Medium | Medium | General-purpose, long dependencies |
| **GRU** | Recurrent | Medium | Fast | Faster alternative to LSTM |
| **BiLSTM** | Recurrent | High | Slow | Pattern recognition, no real-time |
| **CNN-LSTM** | Hybrid | High | Medium | Multi-scale patterns |
| **TCN** | Convolutional | Medium | Fast | Long sequences, parallel processing |
| **Transformer** | Attention | High | Medium | Complex patterns, long-range |
| **Attention-LSTM** | Recurrent + Attention | High | Slow | Interpretable predictions |
| **MLP** | Feedforward | Low | Very Fast | Baseline comparison |

---

## Detailed Model Descriptions

### 1. LSTM (Long Short-Term Memory)

**Architecture:**
```
Input → LSTM Layer 1 → Dropout → LSTM Layer 2 → Dropout → Dense → Output
```

**How it works:**
- Uses "gates" (input, forget, output) to control information flow
- Maintains a "cell state" to remember long-term dependencies
- Can learn which information to keep and which to forget

**Key Features:**
- ✅ Handles vanishing gradient problem
- ✅ Captures long-term dependencies
- ✅ Standard choice for time series
- ❌ Slower than GRU
- ❌ More parameters to train

**Mathematical Formulation:**
```
forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
input gate:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
cell state:  c_t = f_t * c_{t-1} + i_t * tanh(W_c · [h_{t-1}, x_t] + b_c)
output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
hidden:      h_t = o_t * tanh(c_t)
```

**Hyperparameters:**
```python
LSTMModel(
    input_size=10,           # Number of features
    hidden_sizes=[64, 32],   # Size of each LSTM layer
    dropout=0.2              # Dropout rate (0.0-0.5)
)
```

**When to Use:**
- ✓ Default choice for most time series problems
- ✓ When you need to capture long-term patterns
- ✓ Sequential dependencies are important
- ✓ You have sufficient training data (500+ samples)

**When NOT to Use:**
- ✗ When speed is critical (use GRU or TCN)
- ✗ Very short sequences (use MLP)
- ✗ Real-time constraints (consider TCN)

---

### 2. GRU (Gated Recurrent Unit)

**Architecture:**
```
Input → GRU Layer 1 → Dropout → GRU Layer 2 → Dropout → Dense → Output
```

**How it works:**
- Simplified version of LSTM with only 2 gates (reset, update)
- Combines cell state and hidden state into one
- Generally faster than LSTM with similar performance

**Key Features:**
- ✅ Faster training than LSTM (30-40% speedup)
- ✅ Fewer parameters
- ✅ Often performs as well as LSTM
- ❌ May struggle with very long dependencies

**Mathematical Formulation:**
```
reset gate:  r_t = σ(W_r · [h_{t-1}, x_t])
update gate: z_t = σ(W_z · [h_{t-1}, x_t])
candidate:   h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
hidden:      h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

**Hyperparameters:**
```python
GRUModel(
    input_size=10,
    hidden_sizes=[64, 32],
    dropout=0.2
)
```

**When to Use:**
- ✓ When training speed matters
- ✓ Large datasets where LSTM is too slow
- ✓ Memory constraints (fewer parameters)
- ✓ As a faster alternative to LSTM

**When NOT to Use:**
- ✗ When you need absolute best performance
- ✗ Very long-term dependencies (>100 time steps)

---

### 3. Bidirectional LSTM (BiLSTM)

**Architecture:**
```
Input → BiLSTM (forward + backward) → LSTM → Dropout → Dense → Output
```

**How it works:**
- Processes sequence in both forward and backward directions
- Combines information from past and future
- Better feature extraction at each time step

**Key Features:**
- ✅ Captures patterns from both directions
- ✅ Better feature representation
- ✅ Often improves accuracy
- ❌ Cannot be used for real-time prediction
- ❌ Slower (processes sequence twice)

**When to Use:**
- ✓ Post-hoc analysis (not real-time)
- ✓ Pattern recognition in historical data
- ✓ Classification tasks
- ✓ When you have the full sequence available

**When NOT to Use:**
- ✗ Real-time predictions (needs future data)
- ✗ Online learning scenarios
- ✗ Stream processing

---

### 4. CNN-LSTM Hybrid

**Architecture:**
```
Input → Conv1D → MaxPool → Conv1D → LSTM → Dense → Output
```

**How it works:**
- CNN layers extract local features and patterns
- Pooling reduces dimensionality
- LSTM captures temporal dependencies in extracted features
- Combines spatial and temporal processing

**Key Features:**
- ✅ Efficient local feature extraction
- ✅ Good for high-frequency data
- ✅ Multi-scale pattern recognition
- ❌ More complex to tune
- ❌ Requires understanding of both CNN and LSTM

**Hyperparameters:**
```python
CNNLSTMModel(
    input_size=10,
    cnn_filters=64,      # Number of CNN filters
    lstm_hidden=64,      # LSTM hidden size
    dropout=0.2
)
```

**When to Use:**
- ✓ High-frequency time series (stock prices, sensor data)
- ✓ Both local and global patterns are important
- ✓ Data has multi-scale structure
- ✓ Images or spectrograms as input

**When NOT to Use:**
- ✗ Low-frequency data
- ✗ Very short sequences
- ✗ When interpretability is critical

---

### 5. Temporal Convolutional Network (TCN)

**Architecture:**
```
Input → Temporal Block 1 → Temporal Block 2 → ... → Temporal Block N → Dense → Output
```

**Each Temporal Block:**
```
Input → Causal Conv1D → ReLU → Dropout → Causal Conv1D → ReLU → Dropout → (+Residual)
```

**How it works:**
- Uses dilated causal convolutions
- Each layer has exponentially increasing dilation
- Achieves large receptive field efficiently
- Parallel processing unlike RNNs

**Key Features:**
- ✅ Very fast training (fully parallelizable)
- ✅ Long receptive field with fewer layers
- ✅ Stable gradients (no vanishing gradient)
- ✅ Deterministic predictions
- ❌ May need many layers for very long sequences
- ❌ Less intuitive than LSTM

**Hyperparameters:**
```python
TCNModel(
    input_size=10,
    num_channels=[64, 64, 32],  # Channels in each temporal block
    kernel_size=3,               # Convolution kernel size
    dropout=0.2
)
```

**Receptive Field:**
```
receptive_field = 2^n * (kernel_size - 1) + 1
# With 3 layers, kernel_size=3: 2^3 * 2 + 1 = 17
```

**When to Use:**
- ✓ Long sequences (100+ time steps)
- ✓ Need fast training/inference
- ✓ Real-time applications
- ✓ When you want stable training

**When NOT to Use:**
- ✗ Very short sequences
- ✗ When RNN structure is proven better for your domain

---

### 6. Transformer

**Architecture:**
```
Input → Linear Projection → Positional Encoding → 
Transformer Encoder Blocks → Dense → Output
```

**Each Encoder Block:**
```
Input → Multi-Head Self-Attention → Add & Norm → 
Feed Forward → Add & Norm → Output
```

**How it works:**
- Uses self-attention to weigh importance of different time steps
- Positional encoding adds sequence order information
- Parallel processing of entire sequence
- Can capture complex, long-range dependencies

**Key Features:**
- ✅ Captures complex patterns
- ✅ Handles very long sequences
- ✅ Highly parallelizable
- ✅ State-of-the-art in many domains
- ❌ Requires lots of data (1000+ samples)
- ❌ Computationally expensive
- ❌ Many hyperparameters to tune

**Hyperparameters:**
```python
TransformerModel(
    input_size=10,
    d_model=64,          # Model dimension
    nhead=4,             # Number of attention heads
    num_layers=2,        # Number of encoder layers
    dropout=0.2
)
```

**Attention Mechanism:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**When to Use:**
- ✓ Large datasets (1000+ samples)
- ✓ Complex temporal patterns
- ✓ Long-range dependencies crucial
- ✓ State-of-the-art performance needed
- ✓ Parallel processing available

**When NOT to Use:**
- ✗ Small datasets (<500 samples)
- ✗ Limited computational resources
- ✗ Simple patterns
- ✗ Need fast prototyping

---

### 7. Attention-LSTM

**Architecture:**
```
Input → LSTM Layers → Attention Mechanism → Dense → Output
```

**How it works:**
- LSTM processes sequence and outputs hidden states for all time steps
- Attention mechanism learns which time steps are most important
- Weighted combination of all time steps used for prediction
- More interpretable than standard LSTM

**Key Features:**
- ✅ Interpretable (can visualize attention weights)
- ✅ Often better performance than vanilla LSTM
- ✅ Focuses on important time steps
- ❌ Slightly slower than LSTM
- ❌ More parameters

**Attention Weights:**
```
α_t = exp(score(h_t)) / Σ exp(score(h_i))
context = Σ α_t * h_t
```

**Hyperparameters:**
```python
AttentionLSTMModel(
    input_size=10,
    hidden_size=64,
    dropout=0.2
)
```

**When to Use:**
- ✓ Need to understand which time steps matter
- ✓ Interpretability is important
- ✓ Variable-length sequences
- ✓ When some time steps are more informative

**When NOT to Use:**
- ✗ Speed is critical
- ✗ Simple patterns
- ✗ Very short sequences

---

### 8. MLP (Multi-Layer Perceptron)

**Architecture:**
```
Input (flattened) → Dense Layer 1 → ReLU → Dropout → 
Dense Layer 2 → ReLU → Dropout → Dense Layer 3 → Output
```

**How it works:**
- Flattens the sequence into a single vector
- Processes with standard feedforward layers
- No explicit temporal modeling
- Very simple and fast

**Key Features:**
- ✅ Very fast training and inference
- ✅ Simple to understand and debug
- ✅ Good baseline for comparison
- ❌ Doesn't capture temporal structure
- ❌ Fixed sequence length required
- ❌ Limited capacity for complex patterns

**Hyperparameters:**
```python
MLPModel(
    input_size=10,
    sequence_length=20,
    hidden_sizes=[128, 64, 32],
    dropout=0.2
)
```

**When to Use:**
- ✓ As a baseline for comparison
- ✓ Very simple patterns
- ✓ When speed is critical
- ✓ Debugging other models

**When NOT to Use:**
- ✗ Complex temporal dependencies
- ✗ Long sequences
- ✗ When temporal structure matters
- ✗ Production systems (usually outperformed)

---

## When to Use Each Model

### Decision Tree

```
Do you need real-time predictions?
│
├─ YES → Do you have long sequences?
│         ├─ YES → Use TCN
│         └─ NO  → Use GRU or LSTM
│
└─ NO  → Do you have lots of data (1000+ samples)?
          ├─ YES → Is performance critical?
          │        ├─ YES → Try Transformer or Attention-LSTM
          │        └─ NO  → Start with LSTM or BiLSTM
          │
          └─ NO  → Do you need fast training?
                   ├─ YES → Use GRU
                   └─ NO  → Use LSTM
```

### By Use Case

**Financial Time Series (Stock Prices, Forex):**
1st choice: GRU or LSTM
2nd choice: TCN
3rd choice: Attention-LSTM

**High-Frequency Sensor Data:**
1st choice: CNN-LSTM
2nd choice: TCN
3rd choice: GRU

**Long-Term Forecasting (>100 steps ahead):**
1st choice: Transformer
2nd choice: TCN
3rd choice: LSTM

**Resource-Constrained (Edge Devices):**
1st choice: GRU
2nd choice: MLP
3rd choice: TCN

**Research/Experimentation:**
1st choice: Transformer
2nd choice: Attention-LSTM
3rd choice: BiLSTM

**Quick Prototyping:**
1st choice: LSTM
2nd choice: GRU
3rd choice: MLP

---

## Performance Comparison

### Typical Performance on Standard Datasets

| Model | Training Time | Inference Speed | Accuracy | Memory Usage |
|-------|--------------|-----------------|----------|--------------|
| LSTM | Medium | Medium | ★★★★☆ | Medium |
| GRU | Fast | Fast | ★★★★☆ | Low |
| BiLSTM | Slow | Slow | ★★★★★ | High |
| CNN-LSTM | Medium | Medium | ★★★★☆ | Medium |
| TCN | Fast | Very Fast | ★★★★☆ | Medium |
| Transformer | Slow | Medium | ★★★★★ | High |
| Attention-LSTM | Slow | Slow | ★★★★★ | High |
| MLP | Very Fast | Very Fast | ★★★☆☆ | Low |

### Scalability

**Small Datasets (<500 samples):**
1. LSTM / GRU
2. MLP
3. CNN-LSTM

**Medium Datasets (500-2000 samples):**
1. LSTM / GRU
2. CNN-LSTM
3. TCN
4. Attention-LSTM

**Large Datasets (>2000 samples):**
1. Transformer
2. TCN
3. CNN-LSTM
4. Attention-LSTM

### Sequence Length

**Short Sequences (10-30 steps):**
- LSTM, GRU, MLP all work well
- Transformer may be overkill

**Medium Sequences (30-100 steps):**
- LSTM, GRU, TCN recommended
- CNN-LSTM for high-frequency data

**Long Sequences (>100 steps):**
- TCN (most efficient)
- Transformer (best accuracy with enough data)
- LSTM (may struggle with very long sequences)

---

## Usage Examples

### Example 1: Quick Start with LSTM

```python
from timeseries_prtediction.timeseries_all_models import *

# Create model
model = create_model('lstm', input_size=10, hidden_sizes=[64, 32])
model = model.to(device)

# Train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history, best_state = train_model(
    model, train_loader, val_loader,
    criterion, optimizer, num_epochs=50, device=device
)

# Evaluate
predictions, actuals, metrics = evaluate_model(model, test_loader, device)
print(f"Test RMSE: {metrics['rmse']:.4f}")
```

### Example 2: Compare Multiple Models

```python
models_to_test = ['lstm', 'gru', 'tcn', 'transformer']
results = {}

for model_type in models_to_test:
    model = create_model(model_type, input_size=10)
    model = model.to(device)
    
    # Train and evaluate
    # ... (training code)
    
    results[model_type] = metrics

# Find best model
best = min(results.items(), key=lambda x: x[1]['rmse'])
print(f"Best model: {best[0]} with RMSE: {best[1]['rmse']:.4f}")
```

### Example 3: Using CNN-LSTM for High-Frequency Data

```python
# For data with local patterns
model = CNNLSTMModel(
    input_size=10,
    cnn_filters=64,      # Adjust based on data complexity
    lstm_hidden=64,
    dropout=0.3          # Higher dropout for high-frequency noise
)

# Use larger batch size for stability
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

### Example 4: Transformer for Long Sequences

```python
# Prepare longer sequences
sequence_length = 100

model = TransformerModel(
    input_size=10,
    d_model=128,         # Larger model for complex patterns
    nhead=8,            # More attention heads
    num_layers=3,       # Deeper network
    dropout=0.1
)

# Use learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

### Example 5: Attention-LSTM with Visualization

```python
model = AttentionLSTMModel(input_size=10, hidden_size=64)

# After training, visualize attention weights
model.eval()
with torch.no_grad():
    sample_X = test_dataset[0][0].unsqueeze(0).to(device)
    output = model(sample_X)
    
    # Get attention weights from the model
    lstm_out, _ = model.lstm(sample_X)
    context, attention_weights = model.attention(lstm_out)
    
    # Plot attention weights
    plt.figure(figsize=(10, 4))
    plt.plot(attention_weights[0].cpu().numpy())
    plt.title('Attention Weights Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.show()
```

---

## Hyperparameter Tuning

### General Guidelines

**Learning Rate:**
- Start: 0.001
- Range: 0.0001 to 0.01
- Use scheduler for longer training

**Dropout:**
- Low noise: 0.1-0.2
- High noise: 0.3-0.5
- Prevent overfitting: increase dropout

**Batch Size:**
- Small datasets: 16-32
- Large datasets: 64-128
- GPU memory limited: reduce batch size

**Sequence Length:**
- Start: 20-30
- Financial data: 20-50
- Seasonal data: match seasonality period
- Longer = more context but slower

### Model-Specific Tuning

**LSTM/GRU:**
```python
# Start simple
hidden_sizes=[32]

# If underfitting
hidden_sizes=[64, 32]

# If still underfitting
hidden_sizes=[128, 64, 32]
```

**TCN:**
```python
# Adjust receptive field
num_channels=[32, 32, 32]  # Moderate receptive field
num_channels=[64, 64, 64, 64]  # Larger receptive field

# Kernel size affects receptive field exponentially
kernel_size=3  # Standard
kernel_size=5  # Larger patterns
```

**Transformer:**
```python
# Balance between model capacity and overfitting
d_model=64, nhead=4, num_layers=2  # Small dataset
d_model=128, nhead=8, num_layers=3  # Medium dataset
d_model=256, nhead=8, num_layers=4  # Large dataset
```

---

## Best Practices

### 1. Data Preparation
- ✅ Remove outliers carefully
- ✅ Check for missing values
- ✅ Ensure chronological order
- ✅ Use ratio transformation for stationarity
- ✅ Split data temporally, not randomly

### 2. Model Selection
- ✅ Start simple (LSTM/GRU)
- ✅ Try multiple models
- ✅ Use MLP as baseline
- ✅ Consider computational constraints
- ✅ Match model to problem complexity

### 3. Training
- ✅ Use early stopping
- ✅ Monitor both train and validation loss
- ✅ Save best model
- ✅ Use learning rate scheduling
- ✅ Try different random seeds

### 4. Evaluation
- ✅ Use multiple metrics (MSE, MAE, RMSE)
- ✅ Walk-forward validation for time series
- ✅ Visualize predictions
- ✅ Check residuals for patterns
- ✅ Test on out-of-sample data

### 5. Production
- ✅ Version control models
- ✅ Monitor performance drift
- ✅ Plan for retraining
- ✅ Document assumptions
- ✅ A/B test new models

---

## Common Pitfalls

### ❌ Using Random Train/Test Split
**Problem:** Leaks future information
**Solution:** Use temporal split

### ❌ Not Checking for Data Leakage
**Problem:** Model sees future in features
**Solution:** Ensure all features use only past data

### ❌ Over-Complicating Early
**Problem:** Complex model without baseline
**Solution:** Start with LSTM, compare to MLP baseline

### ❌ Ignoring Validation Loss
**Problem:** Overfitting goes unnoticed
**Solution:** Always monitor validation metrics

### ❌ Using Too Long Sequences
**Problem:** Slower training, diminishing returns
**Solution:** Experiment with different lengths

---

## Quick Reference

### Model Selection Cheat Sheet

| Requirement | Recommended Model |
|-------------|------------------|
| Fast training | GRU, MLP |
| Best accuracy (large data) | Transformer, Attention-LSTM |
| Interpretability | Attention-LSTM |
| Real-time prediction | GRU, TCN |
| Long sequences | TCN, Transformer |
| Limited data | LSTM, GRU |
| High-frequency data | CNN-LSTM, TCN |
| Baseline | MLP |

### Quick Commands

```bash
# Run all models comparison
python timeseries_all_models.py

# Train specific model
model = create_model('lstm', input_size=10)

# Available models
['lstm', 'gru', 'bilstm', 'cnn_lstm', 'tcn', 'transformer', 'attention_lstm', 'mlp']
```

---

**Remember:** The best model depends on your specific data and requirements. Always experiment with multiple approaches!
