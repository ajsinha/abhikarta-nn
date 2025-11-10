"""
SIMPLIFIED TIME SERIES NEURAL NETWORK WITH RATIO SCALING
==========================================================

This is a minimal example showing the core concepts for quick implementation.
For a complete, production-ready version, see timeseries_deep_learning.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ============================================================================
# STEP 1: LOAD YOUR DATA
# ============================================================================

# Replace this with your actual data
# df = pd.read_csv('your_data.csv')
# X_raw = df[['var1', 'var2', ..., 'var10']].values  # 10 input variables
# y_raw = df[['target']].values  # 1 target variable

# For demo purposes, generate sample data
n_samples = 1000
X_raw = np.random.randn(n_samples, 10) * 10 + 100  # 10 input variables
y_raw = np.random.randn(n_samples, 1) * 5 + 50     # 1 target variable

print(f"Original data shape: X={X_raw.shape}, y={y_raw.shape}")

# ============================================================================
# STEP 2: APPLY RATIO TRANSFORMATION (t / t-1)
# ============================================================================

def calculate_ratios(data):
    """Convert time series to ratios: value(t) / value(t-1)"""
    ratios = np.ones_like(data)  # First row: ratio = 1.0
    
    for i in range(1, len(data)):
        # Calculate ratio, avoiding division by zero
        ratios[i] = np.where(data[i-1] != 0, data[i] / data[i-1], 1.0)
    
    # Apply log transformation for stability
    ratios = np.log(ratios + 1e-10)
    
    return ratios

# Calculate ratios for input and target
X_ratios = calculate_ratios(X_raw)
y_ratios = calculate_ratios(y_raw)

# Standardize the ratios
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X_ratios)
y_scaled = y_scaler.fit_transform(y_ratios)

print(f"After ratio transformation: X={X_scaled.shape}, y={y_scaled.shape}")

# ============================================================================
# STEP 3: CREATE SEQUENCES FOR TIME SERIES
# ============================================================================

sequence_length = 20  # Use past 20 time steps to predict next value

def create_sequences(X, y, seq_len):
    """Create overlapping sequences for training"""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])  # Past seq_len time steps
        y_seq.append(y[i + seq_len])     # Target at next time step
    
    return np.array(X_seq), np.array(y_seq)

X_sequences, y_sequences = create_sequences(X_scaled, y_scaled, sequence_length)

print(f"Sequence data shape: X_seq={X_sequences.shape}, y_seq={y_sequences.shape}")
# Expected: X_seq=(980, 20, 10), y_seq=(980, 1)
#           (samples, time_steps, features)

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================

split_idx = int(0.8 * len(X_sequences))

X_train = X_sequences[:split_idx]
X_test = X_sequences[split_idx:]
y_train = y_sequences[:split_idx]
y_test = y_sequences[split_idx:]

print(f"\nTrain: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# ============================================================================
# STEP 5: BUILD LSTM MODEL
# ============================================================================

model = Sequential([
    # First LSTM layer
    LSTM(64, return_sequences=True, input_shape=(sequence_length, 10)),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Dense layers
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    
    # Output layer
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\nModel Architecture:")
model.summary()

# ============================================================================
# STEP 6: TRAIN MODEL
# ============================================================================

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# ============================================================================
# STEP 7: EVALUATE AND PREDICT
# ============================================================================

# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Make predictions
predictions = model.predict(X_test, verbose=0)

# Show first 5 predictions vs actual
print("\nFirst 5 predictions vs actual:")
for i in range(5):
    print(f"Predicted: {predictions[i][0]:.4f}, Actual: {y_test[i][0]:.4f}")

# ============================================================================
# STEP 8: SAVE MODEL (Optional)
# ============================================================================

model.save('simple_model.keras')
print("\nModel saved to 'simple_model.keras'")

"""
KEY POINTS:
-----------

1. RATIO TRANSFORMATION:
   - Converts value(t) / value(t-1) to make data stationary
   - Log transformation stabilizes the ratios
   - StandardScaler normalizes for neural network training

2. SEQUENCE CREATION:
   - LSTM needs sequences of past values to predict future
   - sequence_length=20 means use past 20 time steps
   - Each sequence predicts one future value

3. MODEL ARCHITECTURE:
   - 2 LSTM layers (64 and 32 units) capture temporal patterns
   - Dropout (0.2) prevents overfitting
   - Dense layers for final processing
   - Single output neuron for regression

4. HYPERPARAMETERS TO TUNE:
   - sequence_length: How far back to look
   - LSTM units: Larger = more capacity, but risk overfitting
   - dropout_rate: Higher = more regularization
   - epochs & batch_size: Training duration and speed

5. TO USE WITH YOUR DATA:
   - Replace X_raw and y_raw with your actual data
   - Ensure X has 10 columns and y has 1 column
   - Make sure data is sorted by time
   - Scale your data if values vary widely across variables
"""
