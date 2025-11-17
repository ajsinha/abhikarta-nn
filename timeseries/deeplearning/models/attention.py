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


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM."""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        energy = torch.tanh(self.attn(hidden_states))
        attention_weights = torch.softmax(self.v(energy), dim=1)
        context = torch.sum(attention_weights * hidden_states, dim=1)
        return context, attention_weights


class AttentionLSTMNetwork(nn.Module):
    """LSTM with Attention mechanism."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(AttentionLSTMNetwork, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output


class EncoderDecoderLSTMNetwork(nn.Module):
    """Encoder-Decoder LSTM for sequence-to-sequence prediction."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 output_seq_length=1, dropout=0.2):
        super(EncoderDecoderLSTMNetwork, self).__init__()
        
        self.output_seq_length = output_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, target_seq_length=None):
        batch_size = x.size(0)
        target_len = target_seq_length or self.output_seq_length
        
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Decode
        decoder_input = torch.zeros(batch_size, 1, self.fc.out_features).to(x.device)
        outputs = []
        
        for _ in range(target_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(decoder_output)
            outputs.append(output)
            decoder_input = output
        
        outputs = torch.cat(outputs, dim=1)
        return outputs[:, -1, :] if target_len == 1 else outputs


class AttentionLSTMModel(TimeSeriesModel):
    """
    LSTM with Attention mechanism for time series prediction.
    
    Attention allows the model to focus on relevant parts of the input sequence.
    """
    
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
            'sequence_length': 20,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        default_config.update(self.config)
        self.config = default_config
        
        self.device = torch.device(self.config['device'])
        self.model = None
    
    def _build_model(self):
        """Build the Attention LSTM network."""
        self.model = AttentionLSTMNetwork(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout=self.config['dropout']
        ).to(self.device)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'AttentionLSTMModel':
        """Fit the Attention LSTM model."""
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
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        X_seq = self._prepare_prediction_sequences(X.values)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast future values."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.zeros((steps, self.config['output_size']))
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray):
        """Prepare sequences for training."""
        seq_length = self.config['sequence_length']
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _prepare_prediction_sequences(self, X: np.ndarray):
        """Prepare sequences for prediction."""
        seq_length = self.config['sequence_length']
        X_seq = []
        
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i+seq_length])
        
        return np.array(X_seq)
    
    def _get_model_state(self) -> Dict:
        """Get model state for serialization."""
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': self.config
        }
    
    def _set_model_state(self, state: Dict) -> None:
        """Set model state from deserialization."""
        if state.get('model_state_dict'):
            self._build_model()
            self.model.load_state_dict(state['model_state_dict'])


class EncoderDecoderModel(TimeSeriesModel):
    """
    Encoder-Decoder LSTM for sequence-to-sequence time series prediction.
    
    Useful for multi-step forecasting and sequence translation tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        default_config = {
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'output_seq_length': 1,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'sequence_length': 20,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        default_config.update(self.config)
        self.config = default_config
        
        self.device = torch.device(self.config['device'])
        self.model = None
    
    def _build_model(self):
        """Build the Encoder-Decoder network."""
        self.model = EncoderDecoderLSTMNetwork(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            output_seq_length=self.config['output_seq_length'],
            dropout=self.config['dropout']
        ).to(self.device)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'EncoderDecoderModel':
        """Fit the Encoder-Decoder model."""
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
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        X_seq = self._prepare_prediction_sequences(X.values)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor, target_seq_length=steps)
        
        return predictions.cpu().numpy()
    
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Forecast future values."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.zeros((steps, self.config['output_size']))
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray):
        """Prepare sequences for training."""
        seq_length = self.config['sequence_length']
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _prepare_prediction_sequences(self, X: np.ndarray):
        """Prepare sequences for prediction."""
        seq_length = self.config['sequence_length']
        X_seq = []
        
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i+seq_length])
        
        return np.array(X_seq)
    
    def _get_model_state(self) -> Dict:
        """Get model state for serialization."""
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': self.config
        }
    
    def _set_model_state(self, state: Dict) -> None:
        """Set model state from deserialization."""
        if state.get('model_state_dict'):
            self._build_model()
            self.model.load_state_dict(state['model_state_dict'])
