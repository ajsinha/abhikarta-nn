import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class TimeSeriesRatioPredictor:
    """
    Neural Network for time series prediction using ratio-based scaling.
    Converts raw values to ratios (t/t-1) for better scaling.
    """
    
    def __init__(self, n_features=10, lookback=10, model_type='lstm'):
        """
        Initialize the predictor.
        
        Parameters:
        -----------
        n_features : int
            Number of input variables (default: 10)
        lookback : int
            Number of time steps to look back (sequence length)
        model_type : str
            Type of model: 'lstm', 'gru', or 'dense'
        """
        self.n_features = n_features
        self.lookback = lookback
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
    def create_ratio_features(self, data):
        """
        Convert raw time series to ratios (t/t-1).
        
        Parameters:
        -----------
        data : numpy array or pandas DataFrame
            Raw time series data of shape (n_samples, n_features)
            
        Returns:
        --------
        ratios : numpy array
            Ratio features, first row is removed due to division
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Calculate ratios: current / previous
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        ratios = data[1:] / (data[:-1] + epsilon)
        
        # Replace inf and nan values with 1 (no change)
        ratios = np.nan_to_num(ratios, nan=1.0, posinf=1.0, neginf=1.0)
        
        return ratios
    
    def create_sequences(self, X, y, lookback):
        """
        Create sequences for time series prediction.
        
        Parameters:
        -----------
        X : numpy array
            Input features of shape (n_samples, n_features)
        y : numpy array
            Target variable of shape (n_samples,)
        lookback : int
            Number of time steps in each sequence
            
        Returns:
        --------
        X_seq : numpy array
            Sequences of shape (n_sequences, lookback, n_features)
        y_seq : numpy array
            Targets of shape (n_sequences,)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - lookback):
            X_seq.append(X[i:i+lookback])
            y_seq.append(y[i+lookback])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self):
        """
        Build the neural network model.
        """
        model = keras.Sequential()
        
        if self.model_type == 'lstm':
            # LSTM-based architecture
            model.add(layers.LSTM(128, activation='tanh', 
                                 return_sequences=True,
                                 input_shape=(self.lookback, self.n_features)))
            model.add(layers.Dropout(0.2))
            model.add(layers.LSTM(64, activation='tanh'))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(1))
            
        elif self.model_type == 'gru':
            # GRU-based architecture
            model.add(layers.GRU(128, activation='tanh',
                                return_sequences=True,
                                input_shape=(self.lookback, self.n_features)))
            model.add(layers.Dropout(0.2))
            model.add(layers.GRU(64, activation='tanh'))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(1))
            
        else:  # dense
            # Fully connected architecture
            model.add(layers.Flatten(input_shape=(self.lookback, self.n_features)))
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(1))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, X, y):
        """
        Prepare data with ratio scaling and sequence creation.
        
        Parameters:
        -----------
        X : numpy array or pandas DataFrame
            Input features of shape (n_samples, n_features)
        y : numpy array or pandas Series
            Target variable of shape (n_samples,)
            
        Returns:
        --------
        X_seq : numpy array
            Prepared sequences
        y_seq : numpy array
            Prepared targets
        """
        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.flatten()
        
        # Create ratio features
        X_ratios = self.create_ratio_features(X)
        
        # Also create ratios for target (since we lost one row)
        y_ratios = y[1:] / (y[:-1] + 1e-8)
        y_ratios = np.nan_to_num(y_ratios, nan=1.0, posinf=1.0, neginf=1.0)
        
        # Standardize the ratio features
        X_scaled = self.scaler.fit_transform(X_ratios)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_ratios, self.lookback)
        
        return X_seq, y_seq
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        """
        Train the model.
        
        Parameters:
        -----------
        X : numpy array or pandas DataFrame
            Input features
        y : numpy array or pandas Series
            Target variable
        validation_split : float
            Fraction of data to use for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level
            
        Returns:
        --------
        history : keras History object
            Training history
        """
        # Prepare data
        X_seq, y_seq = self.prepare_data(X, y)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        return history
    
    def predict(self, X, y_previous=None):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : numpy array or pandas DataFrame
            Input features for prediction
        y_previous : numpy array, optional
            Previous values of target to convert ratios back to actual values
            
        Returns:
        --------
        predictions : numpy array
            Predicted values (as ratios if y_previous not provided)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Create ratio features
        X_ratios = self.create_ratio_features(X)
        
        # Standardize
        X_scaled = self.scaler.transform(X_ratios)
        
        # Create sequences (no y needed for prediction)
        X_seq = []
        for i in range(len(X_scaled) - self.lookback + 1):
            X_seq.append(X_scaled[i:i+self.lookback])
        X_seq = np.array(X_seq)
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0).flatten()
        
        # Convert ratios back to actual values if previous values provided
        if y_previous is not None:
            # Align with predictions (account for lookback and ratio calculation)
            y_prev_aligned = y_previous[self.lookback:]
            predictions = predictions * y_prev_aligned
        
        return predictions
    
    def plot_training_history(self, history):
        """
        Plot training history.
        
        Parameters:
        -----------
        history : keras History object
            Training history
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot
        axes[1].plot(history.history['mae'], label='Training MAE')
        axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Model MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for demonstration
    print("Generating synthetic time series data...")
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 10
    
    # Create synthetic time series with trend and seasonality
    time = np.arange(n_samples)
    X = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        trend = 0.01 * time
        seasonality = 10 * np.sin(2 * np.pi * time / 50 + i)
        noise = np.random.normal(0, 1, n_samples)
        X[:, i] = 100 + trend + seasonality + noise
    
    # Create target variable (dependent on features)
    y = (X[:, 0] + X[:, 1] + X[:, 2]) / 3 + np.random.normal(0, 2, n_samples)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Initialize predictor
    predictor = TimeSeriesRatioPredictor(
        n_features=n_features,
        lookback=20,
        model_type='lstm'  # Options: 'lstm', 'gru', 'dense'
    )
    
    # Train model
    print("\nTraining model...")
    history = predictor.train(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = predictor.predict(X_test, y_previous=y_test)
    
    # Calculate metrics on test set
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Align test values (account for lookback)
    y_test_aligned = y_test[predictor.lookback:]
    
    mse = mean_squared_error(y_test_aligned, y_pred)
    mae = mean_absolute_error(y_test_aligned, y_pred)
    r2 = r2_score(y_test_aligned, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot results
    fig_history = predictor.plot_training_history(history)
    plt.savefig('/mnt/user-data/outputs/training_history.png', dpi=150, bbox_inches='tight')
    print("\nTraining history plot saved to outputs/training_history.png")
    
    # Plot predictions vs actual
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(y_test_aligned, label='Actual', alpha=0.7)
    ax.plot(y_pred, label='Predicted', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Prediction: Actual vs Predicted')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/predictions.png', dpi=150, bbox_inches='tight')
    print("Predictions plot saved to outputs/predictions.png")
    
    print("\nModel training complete!")
