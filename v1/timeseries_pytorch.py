import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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
        ratio_data = np.zeros_like(data, dtype=np.float32)
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
        
        return ratio_data_scaled.astype(np.float32)
    
    def transform(self, data):
        """Transform new data using fitted scaler"""
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        ratio_data = np.zeros_like(data, dtype=np.float32)
        ratio_data[0] = 1.0
        
        for i in range(1, len(data)):
            ratio_data[i] = np.where(data[i-1] != 0, 
                                     data[i] / data[i-1], 
                                     1.0)
        
        ratio_data = np.log(ratio_data + 1e-10)
        ratio_data_scaled = self.scaler.transform(ratio_data)
        
        return ratio_data_scaled.astype(np.float32)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series sequences
    """
    
    def __init__(self, X, y, sequence_length):
        """
        Args:
            X: Input features (n_samples, n_features)
            y: Target variable (n_samples, 1)
            sequence_length: Number of time steps to look back
        """
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        """
        Returns a sequence and its target
        """
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length]
        
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_target)


class LSTMModel(nn.Module):
    """
    LSTM-based neural network for time series prediction
    """
    
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.2, num_layers_per_lstm=1):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes for each LSTM
            dropout: Dropout rate for regularization
            num_layers_per_lstm: Number of stacked layers in each LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_layers_per_lstm = num_layers_per_lstm
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=num_layers_per_lstm,
            batch_first=True,
            dropout=dropout if num_layers_per_lstm > 1 else 0
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Additional LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(1, len(hidden_sizes)):
            lstm = nn.LSTM(
                input_size=hidden_sizes[i-1],
                hidden_size=hidden_sizes[i],
                num_layers=num_layers_per_lstm,
                batch_first=True,
                dropout=dropout if num_layers_per_lstm > 1 else 0
            )
            self.lstm_layers.append(lstm)
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_sizes[-1], 32)
        self.relu1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(dropout)
        
        # Output layer
        self.fc_out = nn.Linear(16, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # First LSTM
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        
        # Additional LSTM layers
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = dropout(lstm_out)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Dense layers
        out = self.fc1(lstm_out)
        out = self.relu1(out)
        out = self.dropout_fc1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout_fc2(out)
        
        # Output
        out = self.fc_out(out)
        
        return out


class GRUModel(nn.Module):
    """
    GRU-based neural network for time series prediction
    GRU is often faster than LSTM with similar performance
    """
    
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.2, num_layers_per_gru=1):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes for each GRU
            dropout: Dropout rate for regularization
            num_layers_per_gru: Number of stacked layers in each GRU
        """
        super(GRUModel, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_layers_per_gru = num_layers_per_gru
        
        # First GRU layer
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=num_layers_per_gru,
            batch_first=True,
            dropout=dropout if num_layers_per_gru > 1 else 0
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Additional GRU layers
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(1, len(hidden_sizes)):
            gru = nn.GRU(
                input_size=hidden_sizes[i-1],
                hidden_size=hidden_sizes[i],
                num_layers=num_layers_per_gru,
                batch_first=True,
                dropout=dropout if num_layers_per_gru > 1 else 0
            )
            self.gru_layers.append(gru)
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_sizes[-1], 32)
        self.relu1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(dropout)
        
        # Output layer
        self.fc_out = nn.Linear(16, 1)
        
    def forward(self, x):
        """Forward pass"""
        # First GRU
        gru_out, _ = self.gru1(x)
        gru_out = self.dropout1(gru_out)
        
        # Additional GRU layers
        for gru, dropout in zip(self.gru_layers, self.dropout_layers):
            gru_out, _ = gru(gru_out)
            gru_out = dropout(gru_out)
        
        # Take the output from the last time step
        gru_out = gru_out[:, -1, :]
        
        # Dense layers
        out = self.fc1(gru_out)
        out = self.relu1(out)
        out = self.dropout_fc1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout_fc2(out)
        
        # Output
        out = self.fc_out(out)
        
        return out


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, patience=15):
    """
    Train the model with early stopping
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        device: Device to train on
        patience: Early stopping patience
    
    Returns:
        history: Dictionary containing training history
        best_model_state: State dict of the best model
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - batch_y)).item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(outputs - batch_y)).item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after epoch {epoch+1}')
            print(f'Best validation loss: {best_val_loss:.4f}')
            break
    
    return history, best_model_state


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data
    
    Returns:
        predictions: Array of predictions
        actuals: Array of actual values
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }
    
    return predictions, actuals, metrics


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
    # 3. CREATE DATASETS AND DATALOADERS
    # ========================================================================
    
    sequence_length = 20  # Look back 20 time steps
    batch_size = 32
    
    print(f"\nCreating datasets with sequence length {sequence_length}...")
    
    # Split data: 70% train, 15% validation, 15% test
    train_size = int(0.7 * len(X_scaled))
    val_size = int(0.15 * len(X_scaled))
    
    X_train = X_scaled[:train_size]
    y_train = y_scaled[:train_size]
    
    X_val = X_scaled[train_size:train_size + val_size]
    y_val = y_scaled[train_size:train_size + val_size]
    
    X_test = X_scaled[train_size + val_size:]
    y_test = y_scaled[train_size + val_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
    val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # ========================================================================
    # 4. BUILD MODEL
    # ========================================================================
    
    print("\nBuilding LSTM model...")
    model = LSTMModel(
        input_size=n_features,
        hidden_sizes=[64, 32],
        dropout=0.2,
        num_layers_per_lstm=1
    ).to(device)
    
    # Alternative: Use GRU model
    # model = GRUModel(
    #     input_size=n_features,
    #     hidden_sizes=[64, 32],
    #     dropout=0.2,
    #     num_layers_per_gru=1
    # ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========================================================================
    # 5. TRAIN MODEL
    # ========================================================================
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining model...")
    history, best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        device=device,
        patience=15
    )
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # ========================================================================
    # 6. EVALUATE MODEL
    # ========================================================================
    
    print("\nEvaluating model on test set...")
    predictions, actuals, metrics = evaluate_model(model, test_loader, device)
    
    print(f"\nTest Metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    # ========================================================================
    # 7. VISUALIZE RESULTS
    # ========================================================================
    
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot training history
    axes[0].plot(history['train_loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss During Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot predictions vs actual
    plot_samples = min(100, len(predictions))
    axes[1].plot(actuals[:plot_samples], label='Actual', marker='o', markersize=3)
    axes[1].plot(predictions[:plot_samples], label='Predicted', marker='x', markersize=3)
    axes[1].set_title(f'Predictions vs Actual (First {plot_samples} samples)')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Scaled Value')
    axes[1].legend()
    axes[1].grid(True)
    
    # Scatter plot
    axes[2].scatter(actuals, predictions, alpha=0.5)
    axes[2].plot([actuals.min(), actuals.max()], 
                 [actuals.min(), actuals.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[2].set_title('Predicted vs Actual Values')
    axes[2].set_xlabel('Actual Values')
    axes[2].set_ylabel('Predicted Values')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('pytorch_training_results.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to 'pytorch_training_results.png'")
    
    # ========================================================================
    # 8. SAVE MODEL AND PREPROCESSORS
    # ========================================================================
    
    print("\nSaving model and preprocessors...")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': n_features,
            'hidden_sizes': [64, 32],
            'dropout': 0.2,
            'sequence_length': sequence_length
        }
    }, 'pytorch_model.pth')
    
    # Save preprocessors
    with open('pytorch_preprocessors.pkl', 'wb') as f:
        pickle.dump({
            'X_preprocessor': X_preprocessor,
            'y_preprocessor': y_preprocessor,
            'sequence_length': sequence_length
        }, f)
    
    print("Model saved to 'pytorch_model.pth'")
    print("Preprocessors saved to 'pytorch_preprocessors.pkl'")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


# ============================================================================
# HELPER FUNCTIONS FOR MAKING PREDICTIONS ON NEW DATA
# ============================================================================

def load_model_and_predict(new_data_X, model_path='pytorch_model.pth',
                          preprocessor_path='pytorch_preprocessors.pkl',
                          device='cpu'):
    """
    Load saved model and make predictions on new data
    
    Args:
        new_data_X: New input data (numpy array or DataFrame)
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessors
        device: Device to run predictions on
    
    Returns:
        predictions: Predicted values
    """
    # Load preprocessors
    with open(preprocessor_path, 'rb') as f:
        saved_data = pickle.load(f)
        X_preprocessor = saved_data['X_preprocessor']
        sequence_length = saved_data['sequence_length']
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = LSTMModel(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        dropout=model_config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Preprocess new data
    X_scaled = X_preprocessor.transform(new_data_X)
    
    # Create dataset
    dummy_y = np.zeros((len(X_scaled), 1))
    dataset = TimeSeriesDataset(X_scaled, dummy_y, sequence_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)


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

3. PyTorch advantages over Keras:
   - More control over training loop
   - Easier debugging with standard Python
   - Better for research and custom architectures
   - More efficient for large-scale training

4. You can adjust hyperparameters:
   - sequence_length: How many time steps to look back
   - hidden_sizes: Size of LSTM/GRU layers
   - dropout: Regularization strength
   - learning_rate: In the optimizer (currently 0.001)
   - batch_size: Training batch size

5. To use GRU instead of LSTM, uncomment the GRUModel() initialization

6. The model automatically saves:
   - pytorch_model.pth: Model weights and configuration
   - pytorch_preprocessors.pkl: Preprocessing transformations
   - pytorch_training_results.png: Visualization of results

7. GPU support is automatic if CUDA is available
"""
