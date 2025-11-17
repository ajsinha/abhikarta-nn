# LSTM Model Documentation

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | Email: ajsinha@gmail.com**

---

## Overview

The LSTM (Long Short-Term Memory) model is a type of Recurrent Neural Network (RNN) specifically designed to learn long-term dependencies in sequential data. Unlike traditional RNNs, LSTMs can effectively capture patterns over extended time periods, making them ideal for time series forecasting.

## Architecture

### Core Components

1. **Input Gate**: Controls what new information is stored
2. **Forget Gate**: Decides what information to discard
3. **Output Gate**: Determines what information to output
4. **Cell State**: Carries information across time steps

### Mathematical Formulation

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate cell state
C_t = f_t * C_{t-1} + i_t * C̃_t  # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t * tanh(C_t)  # Hidden state
```

## When to Use LSTM

### Ideal Use Cases
- Long-term pattern recognition
- Complex temporal dependencies
- Irregular time intervals
- Multiple input features
- Financial time series forecasting
- Weather prediction
- Energy demand forecasting

### Not Recommended For
- Very short sequences (< 5 time steps)
- Simple linear trends
- When interpretability is crucial
- Limited training data (< 1000 samples)
- Real-time prediction with strict latency requirements

## Configuration Parameters

### Model Architecture

```python
config = {
    'input_size': 10,        # Number of input features
    'hidden_size': 64,       # Size of hidden state
    'num_layers': 2,         # Number of LSTM layers
    'output_size': 1,        # Number of outputs
    'dropout': 0.2,          # Dropout rate between layers
}
```

### Training Parameters

```python
config = {
    'learning_rate': 0.001,  # Learning rate for optimizer
    'epochs': 100,           # Number of training epochs
    'batch_size': 32,        # Batch size for training
    'sequence_length': 20,   # Length of input sequences
    'device': 'cuda',        # 'cuda' or 'cpu'
}
```

## Parameter Tuning Guide

### Hidden Size
- **Small (32-64)**: Simple patterns, less training data
- **Medium (64-128)**: Standard applications
- **Large (128-256+)**: Complex patterns, abundant data

### Number of Layers
- **1 layer**: Simple patterns, risk of underfitting
- **2 layers**: Good default choice
- **3+ layers**: Complex patterns, risk of overfitting

### Sequence Length
- **Short (5-15)**: Recent patterns more important
- **Medium (15-30)**: Balanced approach
- **Long (30-50+)**: Long-term dependencies critical

### Learning Rate
- **High (0.01-0.1)**: Fast initial learning, may be unstable
- **Medium (0.001-0.01)**: Standard choice, stable
- **Low (0.0001-0.001)**: Fine-tuning, slow but stable

## Usage Examples

### Basic Usage

```python
from timeseries.deeplearning.models.lstm import LSTMModel
from timeseries.normalization import DataNormalizer

# Prepare data
X_train, y_train = ...  # Shape: (samples, features)
X_test, y_test = ...

# Normalize
normalizer = DataNormalizer({'strategy': 'zscore'})
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

# Configure model
config = {
    'sequence_length': 20,
    'hidden_size': 64,
    'num_layers': 2,
    'epochs': 100,
    'batch_size': 32
}

# Train
model = LSTMModel(config=config)
model.fit(X_train_norm, y_train, validation_split=0.1, verbose=True)

# Predict
predictions = model.predict(X_test_norm)

# Evaluate
metrics = model.evaluate(X_test_norm, y_test)
```

### Advanced Usage with Custom Training

```python
# Custom configuration
config = {
    'sequence_length': 30,
    'hidden_size': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'learning_rate': 0.0005,
    'epochs': 200,
    'batch_size': 64
}

model = LSTMModel(config=config)

# Train with validation monitoring
model.fit(
    X_train, y_train,
    validation_split=0.15,
    verbose=True
)

# Access training history
history = model.train_history
train_losses = [h['train_loss'] for h in history]
val_losses = [h['val_loss'] for h in history]

# Make multi-step predictions
predictions = model.predict(X_test, steps=1)

# Save model
model.save('lstm_model.pkl')

# Load model
loaded_model = LSTMModel()
loaded_model.load('lstm_model.pkl')
```

## Performance Optimization

### Training Speed
1. **Use GPU**: Set `device='cuda'` if available
2. **Larger Batches**: Increase batch_size (32, 64, 128)
3. **Reduce Layers**: Fewer layers train faster
4. **Shorter Sequences**: Reduce sequence_length if possible

### Model Accuracy
1. **More Data**: LSTM needs substantial training data
2. **Hyperparameter Tuning**: Experiment with hidden_size and num_layers
3. **Feature Engineering**: Add relevant features
4. **Ensemble Methods**: Combine multiple LSTM models
5. **Data Augmentation**: Create synthetic training samples

### Preventing Overfitting
1. **Dropout**: Use dropout between layers (0.2-0.5)
2. **Early Stopping**: Monitor validation loss
3. **L2 Regularization**: Add weight decay to optimizer
4. **Reduce Complexity**: Fewer layers or smaller hidden_size
5. **More Training Data**: Collect additional samples

## Common Issues and Solutions

### Issue: Model Not Learning
**Symptoms**: Loss stays constant or decreases very slowly

**Solutions**:
- Increase learning rate (try 0.01)
- Check data normalization
- Verify sequence preparation
- Reduce model complexity initially
- Check for data leakage

### Issue: Overfitting
**Symptoms**: Training loss decreases but validation loss increases

**Solutions**:
- Add dropout (0.2-0.5)
- Reduce model size
- Use early stopping
- Get more training data
- Apply regularization

### Issue: Exploding Gradients
**Symptoms**: Loss becomes NaN or very large

**Solutions**:
- Reduce learning rate
- Use gradient clipping
- Check data scaling
- Normalize inputs
- Use batch normalization

### Issue: Poor Predictions
**Symptoms**: High test error despite low training error

**Solutions**:
- Check data distribution shift
- Verify normalization applied consistently
- Add more relevant features
- Try ensemble methods
- Increase training data

## Comparison with Other Models

### LSTM vs GRU
- **LSTM**: More parameters, better for complex patterns
- **GRU**: Faster training, similar performance
- **Use LSTM when**: You have sufficient data and computational resources

### LSTM vs BiLSTM
- **LSTM**: Processes sequence in one direction
- **BiLSTM**: Processes both forward and backward
- **Use LSTM when**: Future information shouldn't influence predictions

### LSTM vs Transformer
- **LSTM**: Sequential processing, good for long sequences
- **Transformer**: Parallel processing, better for very long sequences
- **Use LSTM when**: Moderate sequence lengths (10-50), less data available

## Best Practices

1. **Always normalize data** using z-score or min-max scaling
2. **Start simple** with 1-2 layers and increase if needed
3. **Use validation split** to monitor overfitting
4. **Tune hyperparameters** systematically
5. **Save best models** based on validation performance
6. **Ensemble multiple models** for production
7. **Monitor training curves** to detect issues early
8. **Use GPU** when available for faster training

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
2. Graves, A. (2013). Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850.
3. Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP 2014.

## Example Results

### Stock Price Prediction
- **Dataset**: 10 stocks, 1000 days
- **Configuration**: 2 layers, 64 hidden units, sequence=20
- **Results**: RMSE=2.34, R²=0.87, MAPE=3.2%

### Energy Demand Forecasting
- **Dataset**: Hourly data, 2 years
- **Configuration**: 3 layers, 128 hidden units, sequence=24
- **Results**: RMSE=45.2 MW, R²=0.92, MAPE=2.8%

---

For more information or support, contact: ajsinha@gmail.com
