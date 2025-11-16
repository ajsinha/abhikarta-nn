"""
Hierarchical Time Series Models

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from .base import TimeSeriesModel


class HierarchicalModel(TimeSeriesModel):
    """
    Hierarchical time series model for multi-scale predictions.
    
    Decomposes time series into multiple time scales (e.g., trend, seasonal, residual)
    and models each component separately.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        scales: List[int] = None,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """
        Initialize hierarchical model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension
            output_size: Output dimension
            scales: List of time scales to model
            num_layers: Number of layers per scale
            dropout: Dropout rate
            device: Device to run on
        """
        super(HierarchicalModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        if scales is None:
            scales = [1, 5, 10]  # Short, medium, long term
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Create a model for each time scale
        self.scale_models = nn.ModuleList()
        for scale in scales:
            scale_model = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.scale_models.append(scale_model)
        
        # Attention to combine scales
        self.scale_attention = nn.Linear(hidden_size * self.num_scales, self.num_scales)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_scales, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor
        """
        scale_outputs = []
        
        # Process each time scale
        for scale, model in zip(self.scales, self.scale_models):
            # Downsample for this scale
            if scale > 1:
                # Average pooling to downsample
                x_scaled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(1, 2)
            else:
                x_scaled = x
            
            # Process with LSTM
            out, (h_n, _) = model(x_scaled)
            scale_outputs.append(h_n[-1])
        
        # Concatenate scale outputs
        combined = torch.cat(scale_outputs, dim=-1)
        
        # Compute attention weights for scales
        attention_weights = F.softmax(self.scale_attention(combined), dim=-1)
        
        # Weighted combination of scales
        weighted_features = []
        for i, feat in enumerate(scale_outputs):
            weighted_features.append(attention_weights[:, i:i+1] * feat)
        
        # Output
        output = self.fc(combined)
        
        return output


class MultiResolutionModel(TimeSeriesModel):
    """
    Multi-resolution model using wavelet-like decomposition.
    
    Processes time series at multiple resolutions simultaneously.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_resolutions: int = 3,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize multi-resolution model."""
        super(MultiResolutionModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.num_resolutions = num_resolutions
        
        # Decomposition layers (learnable filters)
        self.decomposition_layers = nn.ModuleList([
            nn.Conv1d(
                input_size,
                hidden_size,
                kernel_size=2**(i+1),
                stride=2**i,
                padding=2**i
            )
            for i in range(num_resolutions)
        ])
        
        # Processing for each resolution
        self.resolution_processors = nn.ModuleList([
            nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            for _ in range(num_resolutions)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * num_resolutions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-resolution model."""
        # Transpose for Conv1d
        x_transposed = x.transpose(1, 2)
        
        resolution_features = []
        
        for decomp, processor in zip(self.decomposition_layers, self.resolution_processors):
            # Decompose at this resolution
            decomposed = decomp(x_transposed)
            decomposed = decomposed.transpose(1, 2)
            
            # Process with LSTM
            _, (h_n, _) = processor(decomposed)
            resolution_features.append(h_n[-1])
        
        # Fuse all resolutions
        fused = torch.cat(resolution_features, dim=-1)
        output = self.fusion(fused)
        
        return output


class DeepHierarchicalModel(TimeSeriesModel):
    """
    Deep hierarchical model with multiple levels of abstraction.
    
    Builds a hierarchy of features from low-level to high-level patterns.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = None,
        output_size: int = 1,
        num_layers_per_level: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize deep hierarchical model."""
        if hidden_sizes is None:
            hidden_sizes = [64, 128, 256]
        
        super(DeepHierarchicalModel, self).__init__(
            input_size,
            hidden_sizes[0],
            output_size,
            len(hidden_sizes),
            dropout,
            device,
            **kwargs
        )
        
        self.hidden_sizes = hidden_sizes
        self.num_levels = len(hidden_sizes)
        
        # Create hierarchical levels
        self.levels = nn.ModuleList()
        
        for i, hidden_size in enumerate(hidden_sizes):
            level_input_size = input_size if i == 0 else hidden_sizes[i-1]
            
            level = nn.LSTM(
                level_input_size,
                hidden_size,
                num_layers_per_level,
                batch_first=True,
                dropout=dropout if num_layers_per_level > 1 else 0
            )
            self.levels.append(level)
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_sizes[i], hidden_sizes[-1])
            for i in range(len(hidden_sizes) - 1)
        ])
        
        # Output layer
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through deep hierarchical model."""
        skip_features = []
        
        # Process through hierarchical levels
        for i, level in enumerate(self.levels):
            out, (h_n, _) = level(x)
            
            # Store skip connection (except last level)
            if i < len(self.levels) - 1:
                skip_feat = self.skip_connections[i](h_n[-1])
                skip_features.append(skip_feat)
                # Prepare input for next level
                x = out
            else:
                final_feat = h_n[-1]
        
        # Combine skip connections with final features
        if skip_features:
            combined = final_feat + sum(skip_features)
        else:
            combined = final_feat
        
        # Output
        output = self.fc(combined)
        
        return output


class TemporalHierarchyModel(TimeSeriesModel):
    """
    Temporal hierarchy model for different forecast horizons.
    
    Separately models short-term, medium-term, and long-term predictions.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_sizes: List[int] = None,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize temporal hierarchy model."""
        if output_sizes is None:
            output_sizes = [1, 5, 10]  # Short, medium, long horizon
        
        super(TemporalHierarchyModel, self).__init__(
            input_size,
            hidden_size,
            sum(output_sizes),
            num_layers,
            dropout,
            device,
            **kwargs
        )
        
        self.output_sizes = output_sizes
        
        # Shared encoder
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Separate decoders for each horizon
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, out_size)
            )
            for out_size in output_sizes
        ])
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning predictions for all horizons.
        
        Returns:
            Dictionary with predictions for each horizon
        """
        # Shared encoding
        _, (h_n, _) = self.encoder(x)
        encoded = h_n[-1]
        
        # Decode for each horizon
        outputs = {}
        for i, decoder in enumerate(self.decoders):
            horizon_name = f"horizon_{i+1}"
            outputs[horizon_name] = decoder(encoded)
        
        # Concatenate all outputs for compatibility
        combined = torch.cat([outputs[k] for k in sorted(outputs.keys())], dim=-1)
        
        return combined
