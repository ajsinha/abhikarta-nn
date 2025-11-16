"""
Transformer and Attention-based Time Series Models

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .base import TimeSeriesModel


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    
    Adds sinusoidal positional information to input embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(TimeSeriesModel):
    """
    Transformer model for time series prediction.
    
    Uses self-attention mechanisms to capture dependencies across all time steps
    in parallel, enabling efficient long-range dependency modeling.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """
        Initialize Transformer model.
        
        Args:
            input_size: Number of input features
            hidden_size: Dimension of model (d_model)
            output_size: Number of output predictions
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            device: Device to run on
        """
        super(TransformerModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, mask=mask)
        
        # Take last time step or global pooling
        x = x[:, -1, :]  # Using last time step
        
        # Output projection
        output = self.fc(x)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, Q, K, V, mask=None):
        """Forward pass through multi-head attention."""
        batch_size = Q.size(0)
        
        # Linear projections and split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection
        output = self.W_o(x)
        
        return output, attention_weights


class AttentionModel(TimeSeriesModel):
    """
    Attention-based model for time series prediction.
    
    Uses self-attention to learn which time steps are most relevant for prediction.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize attention model."""
        super(AttentionModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.num_heads = num_heads
        
        # Input embedding
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention model."""
        # Input embedding
        x = self.embedding(x)
        
        # Attention layers
        for attn, ffn, ln1, ln2 in zip(
            self.attention_layers,
            self.ffn_layers,
            self.layer_norms_1,
            self.layer_norms_2
        ):
            # Multi-head attention with residual
            attn_out, _ = attn(x, x, x)
            x = ln1(x + attn_out)
            
            # Feed-forward with residual
            ffn_out = ffn(x)
            x = ln2(x + ffn_out)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output projection
        output = self.fc(x)
        
        return output


class TemporalFusionTransformer(TimeSeriesModel):
    """
    Temporal Fusion Transformer for multi-horizon forecasting.
    
    A sophisticated architecture combining variable selection networks,
    temporal processing, and multi-horizon attention for interpretable predictions.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize Temporal Fusion Transformer."""
        super(TemporalFusionTransformer, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        # Variable selection network
        self.variable_selection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1)
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        
        # Gated residual network
        self.grn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Output network
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TFT."""
        # Variable selection
        importance_weights = self.variable_selection(x)
        x_selected = x * importance_weights
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_selected)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Gated residual network
        grn_out = self.grn(attn_out)
        gate_out = self.gate(attn_out)
        x_gated = attn_out + gate_out * grn_out
        
        # Take last time step
        x_final = x_gated[:, -1, :]
        
        # Output
        output = self.output_layer(x_final)
        
        return output
