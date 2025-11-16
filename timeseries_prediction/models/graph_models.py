"""
Graph Neural Network Models for Interrelated Time Series

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import TimeSeriesModel


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer.
    
    Implements spatial graph convolution for processing graph-structured data.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Input features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Output features (batch_size, num_nodes, out_features)
        """
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GNNModel(TimeSeriesModel):
    """
    Graph Neural Network for multivariate time series with known relationships.
    
    Models dependencies between time series using graph structure,
    where nodes represent variables and edges represent relationships.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_nodes: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """
        Initialize GNN model.
        
        Args:
            input_size: Number of features per node per timestep
            hidden_size: Hidden dimension
            output_size: Output dimension
            num_nodes: Number of nodes in graph
            num_layers: Number of GNN layers
            dropout: Dropout rate
            device: Device to run on
        """
        super(GNNModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.num_nodes = num_nodes
        
        # Temporal encoding (LSTM or GRU for each node)
        self.temporal_encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Graph convolution layers
        self.gc_layers = nn.ModuleList([
            GraphConvolution(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * num_nodes, output_size)
        
        # Learnable adjacency matrix (if not provided)
        self.adj_matrix = nn.Parameter(
            torch.FloatTensor(num_nodes, num_nodes)
        )
        nn.init.xavier_uniform_(self.adj_matrix)
        
        self.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_nodes * input_size)
            adj: Optional adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Reshape to (batch_size, seq_len, num_nodes, input_size)
        x = x.view(batch_size, seq_len, self.num_nodes, -1)
        
        # Temporal encoding for each node
        node_features = []
        for i in range(self.num_nodes):
            node_x = x[:, :, i, :]
            _, (h_n, _) = self.temporal_encoder(node_x)
            node_features.append(h_n[-1])
        
        # Stack node features: (batch_size, num_nodes, hidden_size)
        node_features = torch.stack(node_features, dim=1)
        
        # Use learnable adjacency if not provided
        if adj is None:
            adj = F.softmax(self.adj_matrix, dim=1)
        
        # Graph convolutions
        for gc in self.gc_layers:
            node_features = F.relu(gc(node_features, adj))
            node_features = self.dropout(node_features)
        
        # Flatten node features
        x_flat = node_features.view(batch_size, -1)
        
        # Output projection
        output = self.fc(x_flat)
        
        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    
    Implements attention mechanism for graph convolutions.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h: Input features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Output features (batch_size, num_nodes, out_features)
        """
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # (batch_size, num_nodes, out_features)
        
        # Attention mechanism
        batch_size, num_nodes, _ = Wh.size()
        
        # Concatenate features for attention
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # Mask attention based on adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """Prepare input for attention mechanism."""
        batch_size, num_nodes, out_features = Wh.size()
        
        # Repeat for all pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)
        
        # Concatenate
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating],
            dim=2
        )
        
        return all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * out_features)


class GATModel(TimeSeriesModel):
    """
    Graph Attention Network for time series.
    
    Uses attention mechanisms to learn edge importance in graph-structured data.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_nodes: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize GAT model."""
        super(GATModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        
        # Temporal encoder
        self.temporal_encoder = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True
        )
        
        # Multi-head graph attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_heads = []
            for _ in range(num_heads):
                layer_heads.append(
                    GraphAttentionLayer(
                        hidden_size if i == 0 else hidden_size * num_heads,
                        hidden_size,
                        dropout=dropout,
                        concat=True
                    )
                )
            self.gat_layers.append(nn.ModuleList(layer_heads))
        
        # Output layer
        self.fc = nn.Linear(hidden_size * num_heads * num_nodes, output_size)
        
        # Learnable adjacency
        self.adj_matrix = nn.Parameter(torch.ones(num_nodes, num_nodes))
        
        self.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through GAT."""
        batch_size, seq_len, _ = x.size()
        
        # Reshape for nodes
        x = x.view(batch_size, seq_len, self.num_nodes, -1)
        
        # Temporal encoding
        node_features = []
        for i in range(self.num_nodes):
            node_x = x[:, :, i, :]
            _, (h_n, _) = self.temporal_encoder(node_x)
            node_features.append(h_n[-1])
        
        node_features = torch.stack(node_features, dim=1)
        
        # Use learnable adjacency if not provided
        if adj is None:
            adj = torch.sigmoid(self.adj_matrix)
        
        # Multi-head graph attention
        for layer_heads in self.gat_layers:
            head_outputs = []
            for attention_head in layer_heads:
                head_outputs.append(attention_head(node_features, adj))
            node_features = torch.cat(head_outputs, dim=-1)
        
        # Flatten
        x_flat = node_features.view(batch_size, -1)
        
        # Output
        output = self.fc(x_flat)
        
        return output
