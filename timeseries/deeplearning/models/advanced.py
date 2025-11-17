"""
Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com

Legal Notice: This module and the associated software architecture are proprietary 
and confidential. Unauthorized copying, distribution, modification, or use is 
strictly prohibited without explicit written permission from the copyright holder.

Patent Pending: Certain architectural patterns and implementations described in 
this module may be subject to patent applications.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import sys
sys.path.append('/home/claude/timeseries_package')
from timeseries.model import TimeSeriesModel
import math


class BiLSTMNetwork(nn.Module):
    """Bidirectional LSTM network."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(BiLSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class CNNLSTMNetwork(nn.Module):
    """CNN-LSTM hybrid network."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, num_filters: int = 64, kernel_size: int = 3, 
                 dropout: float = 0.2):
        super(CNNLSTMNetwork, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class TransformerNetwork(nn.Module):
    """Transformer network for time series."""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, 
                 output_size: int, dropout: float = 0.1):
        super(TransformerNetwork, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (seq, batch, features)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq, features)
        out = self.fc(x[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)


class BiLSTMModel(TimeSeriesModel):
    """Bidirectional LSTM model for time series prediction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        default_config = {
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'sequence_length': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        default_config.update(self.config)
        self.config = default_config
        
        self.device = torch.device(self.config['device'])
        self.model = None
    
    def _build_model(self):
        self.model = BiLSTMNetwork(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout=self.config['dropout']
        ).to(self.device)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'BiLSTMModel':
        self.feature_names = X.columns.tolist()
        self.target_names = y.columns.tolist()
        
        self.config['input_size'] = len(self.feature_names)
        self.config['output_size'] = len(self.target_names)
        
        self._build_model()
        
        X_seq, y_seq = self._prepare_sequences(X.values, y.values)
        
        val_split = kwargs.get('validation_split', 0.1)
        split_idx = int(len(X_seq) * (1 - val_split))
        
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        self.train_history = []
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        verbose = kwargs.get('verbose', True)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= (len(X_train) // batch_size)
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        X_seq = self._prepare_prediction_sequences(X.values)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.zeros((steps, self.config['output_size']))
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray):
        seq_length = self.config['sequence_length']
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _prepare_prediction_sequences(self, X: np.ndarray):
        seq_length = self.config['sequence_length']
        X_seq = []
        
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i+seq_length])
        
        return np.array(X_seq)
    
    def _get_model_state(self) -> Dict:
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': self.config
        }
    
    def _set_model_state(self, state: Dict) -> None:
        if state.get('model_state_dict'):
            self._build_model()
            self.model.load_state_dict(state['model_state_dict'])


class CNNLSTMModel(TimeSeriesModel):
    """CNN-LSTM hybrid model for time series prediction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        default_config = {
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'num_filters': 64,
            'kernel_size': 3,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'sequence_length': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        default_config.update(self.config)
        self.config = default_config
        
        self.device = torch.device(self.config['device'])
        self.model = None
    
    def _build_model(self):
        self.model = CNNLSTMNetwork(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            num_filters=self.config['num_filters'],
            kernel_size=self.config['kernel_size'],
            dropout=self.config['dropout']
        ).to(self.device)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'CNNLSTMModel':
        self.feature_names = X.columns.tolist()
        self.target_names = y.columns.tolist()
        
        self.config['input_size'] = len(self.feature_names)
        self.config['output_size'] = len(self.target_names)
        
        self._build_model()
        
        X_seq, y_seq = self._prepare_sequences(X.values, y.values)
        
        val_split = kwargs.get('validation_split', 0.1)
        split_idx = int(len(X_seq) * (1 - val_split))
        
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        self.train_history = []
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        verbose = kwargs.get('verbose', True)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= max(1, len(X_train) // batch_size)
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        X_seq = self._prepare_prediction_sequences(X.values)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.zeros((steps, self.config['output_size']))
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray):
        seq_length = self.config['sequence_length']
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _prepare_prediction_sequences(self, X: np.ndarray):
        seq_length = self.config['sequence_length']
        X_seq = []
        
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i+seq_length])
        
        return np.array(X_seq)
    
    def _get_model_state(self) -> Dict:
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': self.config
        }
    
    def _set_model_state(self, state: Dict) -> None:
        if state.get('model_state_dict'):
            self._build_model()
            self.model.load_state_dict(state['model_state_dict'])


class TransformerModel(TimeSeriesModel):
    """Transformer model for time series prediction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        default_config = {
            'input_size': 1,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'sequence_length': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        default_config.update(self.config)
        self.config = default_config
        
        self.device = torch.device(self.config['device'])
        self.model = None
    
    def _build_model(self):
        self.model = TransformerNetwork(
            input_size=self.config['input_size'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout=self.config['dropout']
        ).to(self.device)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'TransformerModel':
        self.feature_names = X.columns.tolist()
        self.target_names = y.columns.tolist()
        
        self.config['input_size'] = len(self.feature_names)
        self.config['output_size'] = len(self.target_names)
        
        self._build_model()
        
        X_seq, y_seq = self._prepare_sequences(X.values, y.values)
        
        val_split = kwargs.get('validation_split', 0.1)
        split_idx = int(len(X_seq) * (1 - val_split))
        
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        self.train_history = []
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        verbose = kwargs.get('verbose', True)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= max(1, len(X_train) // batch_size)
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        X_seq = self._prepare_prediction_sequences(X.values)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.zeros((steps, self.config['output_size']))
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray):
        seq_length = self.config['sequence_length']
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _prepare_prediction_sequences(self, X: np.ndarray):
        seq_length = self.config['sequence_length']
        X_seq = []
        
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i+seq_length])
        
        return np.array(X_seq)
    
    def _get_model_state(self) -> Dict:
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': self.config
        }
    
    def _set_model_state(self, state: Dict) -> None:
        if state.get('model_state_dict'):
            self._build_model()
            self.model.load_state_dict(state['model_state_dict'])
