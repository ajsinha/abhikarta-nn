import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class TimeSeriesRatioPreprocessor:
    """
    Preprocessor for time series data that converts values to ratios (t/t-1)
    This ensures proper scaling and makes the data stationary
    """
    
    def __init__(self):
        self.first_values = None
        self.scaler = StandardScaler()
        
    def fit_transform(self, data):
        """
        Transform data to ratios and fit the scaler
        
        Args:
            data: numpy array or DataFrame with shape (n_samples, n_features)
        
        Returns:
            ratio_data: transformed data with ratios
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        # Store first values for potential inverse transform
        self.first_values = data[0].copy()
        
        # Calculate ratios: value(t) / value(t-1)
        # For the first row, we'll use ratio of 1.0 (no change)
        ratio_data = np.zeros_like(data)
        ratio_data[0] = 1.0  # First row: no previous value, so ratio = 1
        
        # Calculate ratios for subsequent rows
        for i in range(1, len(data)):
            # Avoid division by zero
            ratio_data[i] = np.where(data[i-1] != 0, 
                                     data[i] / data[i-1], 
                                     1.0)
        
        # Optional: Apply log transform to ratios for better stability
        # This converts multiplicative relationships to additive
        ratio_data = np.log(ratio_data + 1e-10)  # Add small constant to avoid log(0)
        
        # Standardize the ratio data
        ratio_data_scaled = self.scaler.fit_transform(ratio_data)
        
        return ratio_data_scaled
    
    def transform(self, data):
        """Transform new data using fitted scaler"""
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        ratio_data = np.zeros_like(data)
        ratio_data[0] = 1.0
        
        for i in range(1, len(data)):
            ratio_data[i] = np.where(data[i-1] != 0, 
                                     data[i] / data[i-1], 
                                     1.0)
        
        ratio_data = np.log(ratio_data + 1e-10)
        ratio_data_scaled = self.scaler.transform(ratio_data)
        
        return ratio_data_scaled


def create_sequences(X, y, sequence_length):
    """
    Create sequences for time series prediction
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target variable (n_samples, 1)
        sequence_length: Number of time steps to look back
    
    Returns:
        X_seq: Sequences of shape (n_sequences, sequence_length, n_features)
        y_seq: Targets of shape (n_sequences, 1)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(sequence_length, n_features, lstm_units=[64, 32], dropout_rate=0.2):
    """
    Build LSTM neural network model
    
    Args:
        sequence_length: Number of time steps in input sequences
        n_features: Number of input features
        lstm_units: List of LSTM layer sizes
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential()
    
    # First LSTM layer
    model.add(layers.LSTM(lstm_units[0], 
                          return_sequences=True if len(lstm_units) > 1 else False,
                          input_shape=(sequence_length, n_features)))
    model.add(layers.Dropout(dropout_rate))
    
    # Additional LSTM layers
    for i, units in enumerate(lstm_units[1:]):
        return_seq = i < len(lstm_units) - 2
        model.add(layers.LSTM(units, return_sequences=return_seq))
        model.add(layers.Dropout(dropout_rate))
    
    # Dense layers
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(16, activation='relu'))
    
    # Output layer
    model.add(layers.Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', 
                  loss='mse', 
                  metrics=['mae', 'mse'])
    
    return model


def build_gru_model(sequence_length, n_features, gru_units=[64, 32], dropout_rate=0.2):
    """
    Build GRU neural network model (alternative to LSTM)
    
    GRU is often faster than LSTM with similar performance
    """
    model = models.Sequential()
    
    # First GRU layer
    model.add(layers.GRU(gru_units[0], 
                         return_sequences=True if len(gru_units) > 1 else False,
                         input_shape=(sequence_length, n_features)))
    model.add(layers.Dropout(dropout_rate))
    
    # Additional GRU layers
    for i, units in enumerate(gru_units[1:]):
        return_seq = i < len(gru_units) - 2
        model.add(layers.GRU(units, return_sequences=return_seq))
        model.add(layers.Dropout(dropout_rate))
    
    # Dense layers
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(16, activation='relu'))
    
    # Output layer
    model.add(layers.Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', 
                  loss='mse', 
                  metrics=['mae', 'mse'])
    
    return model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # 1. GENERATE SAMPLE DATA (Replace this with your actual data)
    # ========================================================================
    
    print("Generating sample time series data...")
    n_samples = 1000
    n_features = 10  # 10 input variables
    
    # Generate synthetic time series data with trends and seasonality
    time = np.arange(n_samples)
    
    # Create 10 input variables with different patterns
    X_raw = np.zeros((n_samples, n_features))
    for i in range(n_features):
        trend = 100 + 0.1 * time + np.random.randn(n_samples) * 5
        seasonality = 10 * np.sin(2 * np.pi * time / 50 + i)
        X_raw[:, i] = trend + seasonality + np.random.randn(n_samples) * 2
    
    # Create target variable (influenced by input variables)
    y_raw = (0.3 * X_raw[:, 0] + 0.2 * X_raw[:, 1] + 0.15 * X_raw[:, 2] + 
             np.random.randn(n_samples) * 3).reshape(-1, 1)
    
    # ========================================================================
    # 2. PREPROCESS DATA WITH RATIO TRANSFORMATION
    # ========================================================================
    
    print("\nPreprocessing data with ratio transformation...")
    
    # Initialize preprocessors for X and y separately
    X_preprocessor = TimeSeriesRatioPreprocessor()
    y_preprocessor = TimeSeriesRatioPreprocessor()
    
    # Transform data to ratios
    X_scaled = X_preprocessor.fit_transform(X_raw)
    y_scaled = y_preprocessor.fit_transform(y_raw)
    
    print(f"X shape after preprocessing: {X_scaled.shape}")
    print(f"y shape after preprocessing: {y_scaled.shape}")
    
    # ========================================================================
    # 3. CREATE SEQUENCES
    # ========================================================================
    
    sequence_length = 20  # Look back 20 time steps
    
    print(f"\nCreating sequences with length {sequence_length}...")
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    
    print(f"X_seq shape: {X_seq.shape}")
    print(f"y_seq shape: {y_seq.shape}")
    
    # ========================================================================
    # 4. TRAIN-TEST SPLIT
    # ========================================================================
    
    # Use 80% for training, 20% for testing
    split_idx = int(0.8 * len(X_seq))
    
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ========================================================================
    # 5. BUILD AND TRAIN MODEL
    # ========================================================================
    
    print("\nBuilding LSTM model...")
    model = build_lstm_model(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=[64, 32],
        dropout_rate=0.2
    )
    
    # Alternative: Use GRU model
    # model = build_gru_model(
    #     sequence_length=sequence_length,
    #     n_features=n_features,
    #     gru_units=[64, 32],
    #     dropout_rate=0.2
    # )
    
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # ========================================================================
    # 6. EVALUATE MODEL
    # ========================================================================
    
    print("\nEvaluating model on test set...")
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {np.sqrt(test_mse):.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # ========================================================================
    # 7. VISUALIZE RESULTS
    # ========================================================================
    
    print("\nGenerating visualizations...")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Plot predictions vs actual
    plt.subplot(1, 3, 2)
    plt.plot(y_test[:100], label='Actual', marker='o', markersize=3)
    plt.plot(y_pred[:100], label='Predicted', marker='x', markersize=3)
    plt.title('Predictions vs Actual (First 100 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.grid(True)
    
    # Plot scatter: predicted vs actual
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to 'training_results.png'")
    
    # ========================================================================
    # 8. SAVE MODEL AND PREPROCESSORS
    # ========================================================================
    
    print("\nSaving model and preprocessors...")
    model.save('final_model.keras')
    
    # Save preprocessors using pickle
    import pickle
    with open('preprocessors.pkl', 'wb') as f:
        pickle.dump({
            'X_preprocessor': X_preprocessor,
            'y_preprocessor': y_preprocessor,
            'sequence_length': sequence_length
        }, f)
    
    print("Model saved to 'final_model.keras'")
    print("Preprocessors saved to 'preprocessors.pkl'")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


# ============================================================================
# HELPER FUNCTIONS FOR MAKING PREDICTIONS ON NEW DATA
# ============================================================================

def load_model_and_predict(new_data_X, model_path='final_model.keras', 
                          preprocessor_path='preprocessors.pkl'):
    """
    Load saved model and make predictions on new data
    
    Args:
        new_data_X: New input data (numpy array or DataFrame)
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessors
    
    Returns:
        predictions: Predicted values
    """
    import pickle
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load preprocessors
    with open(preprocessor_path, 'rb') as f:
        saved_data = pickle.load(f)
        X_preprocessor = saved_data['X_preprocessor']
        y_preprocessor = saved_data['y_preprocessor']
        sequence_length = saved_data['sequence_length']
    
    # Preprocess new data
    X_scaled = X_preprocessor.transform(new_data_X)
    
    # Create sequences
    X_seq, _ = create_sequences(X_scaled, 
                                np.zeros((len(X_scaled), 1)), 
                                sequence_length)
    
    # Make predictions
    predictions_scaled = model.predict(X_seq, verbose=0)
    
    # Inverse transform predictions (if needed)
    # Note: This requires storing additional information about the original scale
    
    return predictions_scaled


"""
USAGE NOTES:
============

1. Replace the sample data generation section with your actual data loading:
   
   # Load your data
   df = pd.read_csv('your_data.csv')
   X_raw = df[['feature1', 'feature2', ..., 'feature10']].values
   y_raw = df[['target_variable']].values

2. The ratio transformation (t/t-1) is applied in the TimeSeriesRatioPreprocessor class
   This makes the data stationary and properly scaled

3. You can adjust hyperparameters:
   - sequence_length: How many time steps to look back
   - lstm_units: Size of LSTM layers
   - dropout_rate: Regularization strength
   - epochs, batch_size: Training parameters

4. To use GRU instead of LSTM, uncomment the build_gru_model() call

5. The model automatically saves:
   - best_model.keras: Best model during training
   - final_model.keras: Final trained model
   - preprocessors.pkl: Preprocessing transformations
   - training_results.png: Visualization of results

6. To make predictions on new data, use the load_model_and_predict() function
"""
