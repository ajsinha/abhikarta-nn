"""
CNN and TCN based Time Series Models

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .base import TimeSeriesModel


class CNNModel(TimeSeriesModel):
    """
    1D Convolutional Neural Network for time series prediction.
    
    Uses multiple convolutional layers to extract temporal patterns at
    different scales.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 3,
        kernel_sizes: List[int] = None,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """
        Initialize CNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of filters in conv layers
            output_size: Number of output predictions
            num_layers: Number of convolutional layers
            kernel_sizes: List of kernel sizes for each layer
            dropout: Dropout rate
            device: Device to run on
        """
        super(CNNModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        
        self.kernel_sizes = kernel_sizes[:num_layers]
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = input_size if i == 0 else hidden_size
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=self.kernel_sizes[i] if i < len(self.kernel_sizes) else 3,
                    padding='same'
                )
            )
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Transpose for Conv1d: (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layers
        output = self.fc(x)
        
        return output


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) block with dilated convolutions.
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2
    ):
        """Initialize TCN block."""
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through TCN."""
        return self.network(x)


class TemporalBlock(nn.Module):
    """
    Temporal block with dilated causal convolutions and residual connections.
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        """Initialize temporal block."""
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass through temporal block."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Removes trailing elements to maintain causality."""
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TCNModel(TimeSeriesModel):
    """
    Temporal Convolutional Network for time series prediction.
    
    TCN uses dilated causal convolutions to capture long-range dependencies
    with parallel processing advantages over RNNs.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize TCN model."""
        super(TCNModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        # Create channel sizes (increasing with depth)
        num_channels = [hidden_size] * num_layers
        
        # TCN layers
        self.tcn = TemporalConvNet(
            input_size,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Transpose for Conv1d: (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # TCN layers
        y = self.tcn(x)
        
        # Take the last time step
        y = y[:, :, -1]
        
        # Output projection
        output = self.fc(y)
        
        return output


class CNNLSTMModel(TimeSeriesModel):
    """
    Hybrid CNN-LSTM model combining spatial and temporal feature extraction.
    
    CNN layers extract local patterns, LSTM layers capture long-term dependencies.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_cnn_layers: int = 2,
        num_lstm_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize CNN-LSTM model."""
        super(CNNLSTMModel, self).__init__(
            input_size, hidden_size, output_size, num_lstm_layers, dropout, device, **kwargs
        )
        
        self.num_cnn_layers = num_cnn_layers
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_cnn_layers):
            in_channels = input_size if i == 0 else hidden_size
            self.conv_layers.append(
                nn.Conv1d(in_channels, hidden_size, kernel_size, padding='same')
            )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(num_cnn_layers)
        ])
        
        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN-LSTM."""
        # CNN feature extraction
        x = x.transpose(1, 2)
        
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
        
        # Back to (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_output = lstm_out[:, -1, :]
        
        # Output projection
        output = self.fc(last_output)
        
        return output
