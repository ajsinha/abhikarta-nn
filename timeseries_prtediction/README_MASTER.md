# Time Series Neural Network - Complete Package Overview

## 📦 What You Have

This package contains **complete implementations in both PyTorch and TensorFlow/Keras** for time series prediction with ratio-based scaling (t/t-1). Choose the framework that best suits your needs!

## 🎯 Quick Start Guide

### Option 1: PyTorch (More Control) 🔥
```bash
# Install dependencies
pip install -r requirements_pytorch.txt

# Run simple example
python simple_pytorch_example.py

# Or run complete version
python timeseries_pytorch.py
```

### Option 2: TensorFlow/Keras (Faster Prototyping) ⚡
```bash
# Install dependencies
pip install -r requirements.txt

# Run simple example
python simple_timeseries_example.py

# Or run complete version
python timeseries_deep_learning.py
```

## 📁 File Structure

### PyTorch Implementation
| File | Lines | Description |
|------|-------|-------------|
| `timeseries_pytorch.py` | ~800 | Complete production-ready implementation |
| `simple_pytorch_example.py` | ~250 | Minimal example for quick start |
| `README_PYTORCH.md` | - | Comprehensive PyTorch documentation |
| `requirements_pytorch.txt` | - | PyTorch dependencies |

### TensorFlow/Keras Implementation
| File | Lines | Description |
|------|-------|-------------|
| `timeseries_deep_learning.py` | ~500 | Complete production-ready implementation |
| `simple_timeseries_example.py` | ~150 | Minimal example for quick start |
| `README.md` | - | Comprehensive Keras documentation |
| `requirements.txt` | - | TensorFlow dependencies |

### Additional Resources
| File | Description |
|------|-------------|
| `COMPARISON_GUIDE.md` | Detailed comparison to help you choose |
| `README_MASTER.md` | This file - overview of everything |

## 🚀 Features (Both Implementations)

### ✅ Ratio-Based Scaling
- Automatically calculates ratios: `value(t) / value(t-1)`
- Applies log transformation for stability
- Standardizes data for neural network training

### ✅ Time Series Preprocessing
- Custom preprocessor class handles all transformations
- Sequence generation for LSTM/GRU input
- Train/validation/test splitting

### ✅ Neural Network Models
- **LSTM** - Long Short-Term Memory (default)
- **GRU** - Gated Recurrent Unit (faster alternative)
- Configurable architecture (layers, units, dropout)

### ✅ Training Features
- Early stopping to prevent overfitting
- Model checkpointing (saves best model)
- Learning rate optimization
- GPU acceleration support

### ✅ Evaluation & Visualization
- Multiple metrics (MSE, MAE, RMSE)
- Training history plots
- Prediction vs actual comparison
- Scatter plots for prediction quality

### ✅ Model Persistence
- Save trained models
- Save preprocessing transformations
- Load and make predictions on new data

## 🎓 Learning Path

### Beginner Path (Recommended)
1. Read `COMPARISON_GUIDE.md` to choose framework
2. Run the simple example for your chosen framework
3. Modify simple example with your own data
4. Review the README for your framework
5. Graduate to the complete implementation

### Advanced Path
1. Run complete implementation immediately
2. Study the code structure and classes
3. Customize model architecture
4. Experiment with hyperparameters
5. Deploy to production

## 💡 Key Concepts

### Ratio Transformation (t/t-1)
**Why?** Makes data stationary and properly scaled
```python
# Raw data
[100, 102, 105, 103, 107]

# Ratios (t/t-1)
[1.0, 1.02, 1.029, 0.981, 1.039]

# Log ratios
[0.0, 0.0198, 0.0286, -0.0192, 0.0383]

# Standardized
[0.0, 0.182, 0.264, -0.177, 0.353]
```

### Sequence Creation
**Why?** LSTM needs historical context
```python
# Input features at time t
X[t] = [feature1[t], feature2[t], ..., feature10[t]]

# Sequence for prediction at time t
X_seq[t] = [X[t-20], X[t-19], ..., X[t-1], X[t]]

# Target for time t+1
y[t+1] = target[t+1]
```

### Model Architecture
```
Input: (batch_size, sequence_length, n_features)
       ↓
LSTM/GRU Layers (capture temporal patterns)
       ↓
Dense Layers (non-linear transformations)
       ↓
Output: (batch_size, 1)
```

## 🔧 Common Modifications

### Change Input Variables
```python
# Current: 10 input variables
n_features = 10

# Change to 5 variables
n_features = 5
X_raw = df[['var1', 'var2', 'var3', 'var4', 'var5']].values
```

### Adjust Lookback Window
```python
# Current: Look back 20 time steps
sequence_length = 20

# Change to 30 time steps
sequence_length = 30
```

### Modify Model Complexity
```python
# PyTorch
model = LSTMModel(
    input_size=10,
    hidden_sizes=[128, 64, 32],  # Deeper network
    dropout=0.3  # More regularization
)

# Keras
model = build_lstm_model(
    sequence_length=20,
    n_features=10,
    lstm_units=[128, 64, 32],  # Deeper network
    dropout_rate=0.3  # More regularization
)
```

### Multi-Step Prediction
```python
# Modify create_sequences to return multiple future steps
def create_sequences(X, y, seq_len, forecast_horizon=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len - forecast_horizon):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len:i + seq_len + forecast_horizon])
    return np.array(X_seq), np.array(y_seq)

# Update output layer
# PyTorch: self.fc_out = nn.Linear(16, forecast_horizon)
# Keras: Dense(forecast_horizon)
```

## 📊 Expected Performance

### Synthetic Data (Demo)
- Training Time: 1-5 minutes (CPU) / 10-30 seconds (GPU)
- Test RMSE: 0.3-0.5 (scaled values)
- R² Score: 0.6-0.8

### Real Financial Data (Typical)
- Training Time: 5-20 minutes (depends on data size)
- Test RMSE: Varies widely by asset/market
- R² Score: 0.3-0.6 (challenging domain)

**Note**: Time series prediction is inherently difficult. Focus on:
- Consistent preprocessing
- Proper validation (walk-forward)
- Multiple evaluation metrics
- Understanding the domain

## 🐛 Troubleshooting

### "Model not learning"
- ✓ Check data has temporal patterns
- ✓ Increase model capacity
- ✓ Adjust learning rate
- ✓ Train longer (more epochs)

### "High training error"
- ✓ Increase sequence_length
- ✓ Add more features
- ✓ Use deeper network
- ✓ Check data quality

### "Good training, poor test"
- ✓ Overfitting - increase dropout
- ✓ Use early stopping (already included)
- ✓ Reduce model complexity
- ✓ Get more training data

### "Training too slow"
- ✓ Use GPU if available
- ✓ Increase batch size
- ✓ Use GRU instead of LSTM
- ✓ Reduce sequence length

### "Out of memory"
- ✓ Reduce batch size
- ✓ Reduce sequence length
- ✓ Use smaller model
- ✓ Process data in chunks

## 🎯 Next Steps

### Improve Model Performance
1. **Feature Engineering**
   - Add technical indicators (moving averages, RSI, MACD)
   - Include lagged features
   - Add time-based features (day of week, month, etc.)

2. **Hyperparameter Tuning**
   - Grid search or random search
   - Bayesian optimization
   - Use validation set for selection

3. **Ensemble Methods**
   - Train multiple models
   - Average predictions
   - Use different architectures

4. **Advanced Architectures**
   - Attention mechanisms
   - Transformer models
   - Hybrid CNN-LSTM
   - Bidirectional LSTM

### Production Deployment
1. **Model Serving**
   - PyTorch: TorchServe
   - TensorFlow: TF Serving
   - ONNX for framework-agnostic

2. **Monitoring**
   - Track prediction accuracy
   - Monitor data drift
   - Alert on anomalies

3. **Retraining**
   - Schedule regular retraining
   - Detect when retraining needed
   - Version control models

## 📚 Learning Resources

### Understanding LSTM/GRU
- Original LSTM paper: Hochreiter & Schmidhuber (1997)
- Understanding LSTM Networks: colah.github.io/posts/2015-08-Understanding-LSTMs
- GRU paper: Cho et al. (2014)

### Time Series Forecasting
- Forecasting: Principles and Practice (Hyndman & Athanasopoulos)
- Deep Learning for Time Series Forecasting (Brownlee)

### Framework Documentation
- PyTorch: pytorch.org/docs
- TensorFlow: tensorflow.org/guide
- Keras: keras.io/guides

## 🏆 Best Practices

### Data Preparation
- ✓ Remove outliers carefully (time series sensitive)
- ✓ Handle missing values appropriately
- ✓ Ensure data is sorted chronologically
- ✓ Split data temporally (not randomly)

### Model Development
- ✓ Start simple, then increase complexity
- ✓ Use validation set for hyperparameter tuning
- ✓ Never use test set until final evaluation
- ✓ Save preprocessing parameters with model

### Evaluation
- ✓ Use walk-forward validation
- ✓ Report multiple metrics (MAE, RMSE, MAPE)
- ✓ Visualize predictions vs actual
- ✓ Check residuals for patterns

### Production
- ✓ Monitor model performance continuously
- ✓ Version control code and models
- ✓ Document assumptions and limitations
- ✓ Plan for model retraining

## 🤝 Getting Help

### Framework-Specific Issues
- PyTorch: discuss.pytorch.org
- TensorFlow: stackoverflow.com/questions/tagged/tensorflow
- Keras: github.com/keras-team/keras/issues

### Time Series Questions
- Cross Validated: stats.stackexchange.com
- r/MachineLearning
- r/datascience

## 📝 Citation

If you use this code in your research, please cite:
```
@misc{timeseries_neural_network,
  title={Time Series Neural Network with Ratio Scaling},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  journal={GitHub Repository}
}
```

---

## 🎉 You're Ready!

You now have everything you need to build, train, and deploy time series prediction models with neural networks. Start with the simple examples, experiment, and have fun!

**Remember**: The best model is one that works for YOUR specific problem. Don't be afraid to experiment and adapt the code to your needs.

Good luck with your time series forecasting! 🚀
