"""
COMPREHENSIVE TIME SERIES MODELS - PYTORCH IMPLEMENTATION
==========================================================

This module includes multiple neural network architectures for time series:
1. LSTM (Long Short-Term Memory)
2. GRU (Gated Recurrent Unit)
3. Bidirectional LSTM
4. CNN-LSTM Hybrid
5. Temporal Convolutional Network (TCN)
6. Transformer
7. Attention-LSTM
8. Simple MLP Baseline

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import math

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class TimeSeriesRatioPreprocessor:
    """
    Preprocessor for time series data that converts values to ratios (t/t-1)
    This ensures proper scaling and makes the data stationary
    """
    
    def __init__(self):
        self.first_values = None
        self.scaler = StandardScaler()
        
    def fit_transform(self, data):
        """Transform data to ratios and fit the scaler"""
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        self.first_values = data[0].copy()
        
        ratio_data = np.zeros_like(data, dtype=np.float32)
        ratio_data[0] = 1.0
        
        for i in range(1, len(data)):
            ratio_data[i] = np.where(data[i-1] != 0, data[i] / data[i-1], 1.0)
        
        ratio_data = np.log(ratio_data + 1e-10)
        ratio_data_scaled = self.scaler.fit_transform(ratio_data)
        
        return ratio_data_scaled.astype(np.float32)
    
    def transform(self, data):
        """Transform new data using fitted scaler"""
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        ratio_data = np.zeros_like(data, dtype=np.float32)
        ratio_data[0] = 1.0
        
        for i in range(1, len(data)):
            ratio_data[i] = np.where(data[i-1] != 0, data[i] / data[i-1], 1.0)
        
        ratio_data = np.log(ratio_data + 1e-10)
        ratio_data_scaled = self.scaler.transform(ratio_data)
        
        return ratio_data_scaled.astype(np.float32)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences"""
    
    def __init__(self, X, y, sequence_length):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length]
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_target)


# ============================================================================
# MODEL 1: LSTM (Long Short-Term Memory)
# ============================================================================

class LSTMModel(nn.Module):
    """
    Standard LSTM model for time series prediction.
    
    Best for: Long-term dependencies, general-purpose time series
    Pros: Good at capturing long-term patterns, handles vanishing gradients
    Cons: Slower than GRU, more parameters
    """
    
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(
                nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True)
            )
            self.dropout_layers.append(nn.Dropout(dropout))
        
        self.fc1 = nn.Linear(hidden_sizes[-1], 32)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = dropout(lstm_out)
        
        lstm_out = lstm_out[:, -1, :]
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc_out(out)
        return out


# ============================================================================
# MODEL 2: GRU (Gated Recurrent Unit)
# ============================================================================

class GRUModel(nn.Module):
    """
    GRU model - simpler and faster alternative to LSTM.
    
    Best for: When speed is important, large datasets
    Pros: Faster training, fewer parameters, similar performance to LSTM
    Cons: May struggle with very long-term dependencies
    """
    
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.gru1 = nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(1, len(hidden_sizes)):
            self.gru_layers.append(
                nn.GRU(hidden_sizes[i-1], hidden_sizes[i], batch_first=True)
            )
            self.dropout_layers.append(nn.Dropout(dropout))
        
        self.fc1 = nn.Linear(hidden_sizes[-1], 32)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, 1)
        
    def forward(self, x):
        gru_out, _ = self.gru1(x)
        gru_out = self.dropout1(gru_out)
        
        for gru, dropout in zip(self.gru_layers, self.dropout_layers):
            gru_out, _ = gru(gru_out)
            gru_out = dropout(gru_out)
        
        gru_out = gru_out[:, -1, :]
        out = self.fc1(gru_out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc_out(out)
        return out


# ============================================================================
# MODEL 3: Bidirectional LSTM
# ============================================================================

class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM - processes sequences in both directions.
    
    Best for: When future context is available, pattern recognition
    Pros: Captures patterns from both directions, better feature extraction
    Cons: Cannot be used for real-time prediction (needs future data)
    """
    
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.2):
        super(BiLSTMModel, self).__init__()
        
        self.bilstm1 = nn.LSTM(
            input_size, hidden_sizes[0], 
            batch_first=True, bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # After bidirectional, output size is hidden_size * 2
        self.lstm2 = nn.LSTM(
            hidden_sizes[0] * 2, hidden_sizes[1], 
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_sizes[1], 32)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, _ = self.bilstm1(x)
        lstm_out = self.dropout1(lstm_out)
        
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        
        lstm_out = lstm_out[:, -1, :]
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc_out(out)
        return out


# ============================================================================
# MODEL 4: CNN-LSTM Hybrid
# ============================================================================

class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM Hybrid - CNN extracts local features, LSTM captures temporal dependencies.
    
    Best for: Multi-scale patterns, data with local and global dependencies
    Pros: Efficient feature extraction, good for high-frequency data
    Cons: More complex, requires tuning of both CNN and LSTM components
    """
    
    def __init__(self, input_size, cnn_filters=64, lstm_hidden=64, dropout=0.2):
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers for local feature extraction
        self.conv1 = nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters//2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # LSTM layers for temporal dependencies
        self.lstm = nn.LSTM(cnn_filters//2, lstm_hidden, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.relu3 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Transpose back for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout3(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        
        # Dense layers
        out = self.fc1(lstm_out)
        out = self.relu3(out)
        out = self.dropout4(out)
        out = self.fc_out(out)
        return out


# ============================================================================
# MODEL 5: Temporal Convolutional Network (TCN)
# ============================================================================

class TemporalBlock(nn.Module):
    """Building block for TCN with causal convolutions"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Causal convolution: remove future information
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network - uses dilated causal convolutions.
    
    Best for: Long sequences, parallel processing, real-time applications
    Pros: Faster than RNN, better parallelization, long receptive field
    Cons: May need many layers for very long dependencies
    """
    
    def __init__(self, input_size, num_channels=[64, 64, 32], kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, 
                    dilation, dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.network(x)
        # Take last time step
        x = x[:, :, -1]
        return self.fc(x)


# ============================================================================
# MODEL 6: Transformer
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer model using self-attention mechanism.
    
    Best for: Complex patterns, long-range dependencies, parallel processing
    Pros: Captures long-range dependencies well, highly parallelizable
    Cons: Requires more data, computationally expensive, needs careful tuning
    """
    
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, 1)
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Take last time step
        x = x[:, -1, :]
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


# ============================================================================
# MODEL 7: Attention-LSTM
# ============================================================================

class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum of all time steps
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class AttentionLSTMModel(nn.Module):
    """
    LSTM with attention mechanism - focuses on important time steps.
    
    Best for: When some time steps are more important than others
    Pros: Interpretable (can see which time steps matter), often better performance
    Cons: Slightly slower than regular LSTM, more parameters
    """
    
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super(AttentionLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, 
            num_layers=2, batch_first=True, dropout=dropout
        )
        
        self.attention = AttentionLayer(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, 1)
        
    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # Dense layers
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        
        return out


# ============================================================================
# MODEL 8: Simple MLP Baseline
# ============================================================================

class MLPModel(nn.Module):
    """
    Simple Multi-Layer Perceptron - flatten sequence and use dense layers.
    
    Best for: Baseline comparison, very simple patterns
    Pros: Fast, simple, easy to understand
    Cons: Doesn't capture temporal structure well, limited capacity
    """
    
    def __init__(self, input_size, sequence_length, hidden_sizes=[128, 64, 32], dropout=0.2):
        super(MLPModel, self).__init__()
        
        flatten_size = input_size * sequence_length
        
        layers = []
        prev_size = flatten_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten sequence
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, patience=15, model_name="model"):
    """Train model with early stopping"""
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - batch_y)).item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Validation
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
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return history, best_model_state


def evaluate_model(model, test_loader, device):
    """Evaluate model on test data"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    
    return predictions, actuals, {'mse': mse, 'mae': mae, 'rmse': rmse}


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type, input_size, sequence_length=20, **kwargs):
    """
    Factory function to create different model types.
    
    Args:
        model_type: One of ['lstm', 'gru', 'bilstm', 'cnn_lstm', 'tcn', 
                            'transformer', 'attention_lstm', 'mlp']
        input_size: Number of input features
        sequence_length: Length of input sequences
        **kwargs: Additional model-specific parameters
    
    Returns:
        model: PyTorch model
    """
    
    models = {
        'lstm': lambda: LSTMModel(input_size, **kwargs),
        'gru': lambda: GRUModel(input_size, **kwargs),
        'bilstm': lambda: BiLSTMModel(input_size, **kwargs),
        'cnn_lstm': lambda: CNNLSTMModel(input_size, **kwargs),
        'tcn': lambda: TCNModel(input_size, **kwargs),
        'transformer': lambda: TransformerModel(input_size, **kwargs),
        'attention_lstm': lambda: AttentionLSTMModel(input_size, **kwargs),
        'mlp': lambda: MLPModel(input_size, sequence_length, **kwargs)
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type]()


# ============================================================================
# MAIN EXECUTION - COMPARE ALL MODELS
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("TIME SERIES PREDICTION - MULTIPLE MODEL COMPARISON")
    print("="*70)
    
    # Generate sample data
    print("\nGenerating sample data...")
    n_samples = 1000
    n_features = 10
    time = np.arange(n_samples)
    
    X_raw = np.zeros((n_samples, n_features))
    for i in range(n_features):
        trend = 100 + 0.1 * time + np.random.randn(n_samples) * 5
        seasonality = 10 * np.sin(2 * np.pi * time / 50 + i)
        X_raw[:, i] = trend + seasonality + np.random.randn(n_samples) * 2
    
    y_raw = (0.3 * X_raw[:, 0] + 0.2 * X_raw[:, 1] + 0.15 * X_raw[:, 2] + 
             np.random.randn(n_samples) * 3).reshape(-1, 1)
    
    # Preprocess
    print("Preprocessing with ratio transformation...")
    X_preprocessor = TimeSeriesRatioPreprocessor()
    y_preprocessor = TimeSeriesRatioPreprocessor()
    
    X_scaled = X_preprocessor.fit_transform(X_raw)
    y_scaled = y_preprocessor.fit_transform(y_raw)
    
    # Create datasets
    sequence_length = 20
    batch_size = 32
    
    train_size = int(0.7 * len(X_scaled))
    val_size = int(0.15 * len(X_scaled))
    
    X_train = X_scaled[:train_size]
    y_train = y_scaled[:train_size]
    X_val = X_scaled[train_size:train_size + val_size]
    y_val = y_scaled[train_size:train_size + val_size]
    X_test = X_scaled[train_size + val_size:]
    y_test = y_scaled[train_size + val_size:]
    
    train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
    val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define models to compare
    models_to_train = {
        'LSTM': ('lstm', {'hidden_sizes': [64, 32], 'dropout': 0.2}),
        'GRU': ('gru', {'hidden_sizes': [64, 32], 'dropout': 0.2}),
        'BiLSTM': ('bilstm', {'hidden_sizes': [64, 32], 'dropout': 0.2}),
        'CNN-LSTM': ('cnn_lstm', {'cnn_filters': 64, 'lstm_hidden': 64, 'dropout': 0.2}),
        'TCN': ('tcn', {'num_channels': [64, 64, 32], 'dropout': 0.2}),
        'Transformer': ('transformer', {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2}),
        'Attention-LSTM': ('attention_lstm', {'hidden_size': 64, 'dropout': 0.2}),
        'MLP': ('mlp', {'hidden_sizes': [128, 64, 32], 'dropout': 0.2})
    }
    
    # Train and evaluate all models
    results = {}
    
    for model_name, (model_type, kwargs) in models_to_train.items():
        print(f"\n{'='*70}")
        print(f"Training {model_name}")
        print(f"{'='*70}")
        
        # Create model
        model = create_model(model_type, n_features, sequence_length, **kwargs)
        model = model.to(device)
        
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        history, best_state = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=50, device=device, patience=10, model_name=model_name
        )
        
        # Load best model and evaluate
        model.load_state_dict(best_state)
        predictions, actuals, metrics = evaluate_model(model, test_loader, device)
        
        results[model_name] = {
            'metrics': metrics,
            'history': history,
            'predictions': predictions,
            'actuals': actuals
        }
        
        print(f"\nTest Results for {model_name}:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
    
    # Compare all models
    print(f"\n{'='*70}")
    print("FINAL COMPARISON - ALL MODELS")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
    print("-"*70)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name:<20} {metrics['mse']:<12.4f} {metrics['mae']:<12.4f} {metrics['rmse']:<12.4f}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['metrics']['rmse'])
    print(f"\n🏆 Best Model: {best_model[0]} (RMSE: {best_model[1]['metrics']['rmse']:.4f})")
    
    # Save results
    with open('all_models_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ All results saved to 'all_models_results.pkl'")
    print(f"{'='*70}")
