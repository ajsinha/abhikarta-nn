"""
RNN-based Time Series Models

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import TimeSeriesModel


class LSTMModel(TimeSeriesModel):
    """
    Long Short-Term Memory (LSTM) model for time series prediction.
    
    LSTM networks are designed to learn long-term dependencies in sequential data
    through gating mechanisms that control information flow.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        batch_first: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            output_size: Number of output predictions
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            batch_first: Whether batch dimension is first
            device: Device to run on
        """
        super(LSTMModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        
        # Calculate output size based on bidirectionality
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        self.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional hidden state tuple (h_0, c_0)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last time step output
        if self.batch_first:
            last_output = lstm_out[:, -1, :]
        else:
            last_output = lstm_out[-1, :, :]
        
        # Fully connected layers
        output = self.fc(last_output)
        
        return output


class GRUModel(TimeSeriesModel):
    """
    Gated Recurrent Unit (GRU) model for time series prediction.
    
    GRU is a simplified version of LSTM with fewer parameters, often achieving
    comparable performance with faster training.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        batch_first: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize GRU model."""
        super(GRUModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        
        # Calculate output size
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        self.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through GRU."""
        # GRU forward
        gru_out, hidden = self.gru(x, hidden)
        
        # Take the last time step output
        if self.batch_first:
            last_output = gru_out[:, -1, :]
        else:
            last_output = gru_out[-1, :, :]
        
        # Fully connected layers
        output = self.fc(last_output)
        
        return output


class BiLSTMModel(TimeSeriesModel):
    """
    Bidirectional LSTM model for time series prediction.
    
    BiLSTM processes sequences in both forward and backward directions,
    capturing dependencies from both past and future contexts.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        batch_first: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize BiLSTM model."""
        super(BiLSTMModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.batch_first = batch_first
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=batch_first
        )
        
        # Attention mechanism for BiLSTM
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        self.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass through BiLSTM with attention."""
        # BiLSTM forward
        lstm_out, hidden = self.bilstm(x, hidden)
        
        # Apply attention
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1
        )
        
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(
            attention_weights.unsqueeze(-1) * lstm_out, dim=1
        )
        
        # Fully connected layers
        output = self.fc(context_vector)
        
        return output


class StackedLSTMModel(TimeSeriesModel):
    """
    Stacked LSTM with residual connections for deep architectures.
    
    Uses residual connections between layers to facilitate gradient flow
    in deeper networks.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_residual: bool = True,
        batch_first: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize Stacked LSTM model."""
        super(StackedLSTMModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.use_residual = use_residual
        self.batch_first = batch_first
        
        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=batch_first
                )
            )
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacked LSTM."""
        for i, (lstm_layer, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            out, _ = lstm_layer(x)
            out = dropout(out)
            
            # Add residual connection if not the first layer
            if self.use_residual and i > 0 and x.size(-1) == out.size(-1):
                out = out + x
            
            x = out
        
        # Take last time step
        if self.batch_first:
            last_output = x[:, -1, :]
        else:
            last_output = x[-1, :, :]
        
        # Output projection
        output = self.fc(last_output)
        
        return output
