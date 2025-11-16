"""
Time Series Models Package

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

from .base import TimeSeriesModel

# RNN Models
from .rnn_models import (
    LSTMModel,
    GRUModel,
    BiLSTMModel,
    StackedLSTMModel
)

# CNN Models
from .cnn_models import (
    CNNModel,
    TCNModel,
    CNNLSTMModel
)

# Transformer Models
from .transformer_models import (
    TransformerModel,
    AttentionModel,
    TemporalFusionTransformer
)

# Graph Models
from .graph_models import (
    GNNModel,
    GATModel
)

# Ensemble Models
from .ensemble_models import (
    EnsembleModel,
    BaggingModel,
    BoostingModel,
    VotingModel
)

# Hierarchical Models
from .hierarchical_models import (
    HierarchicalModel,
    MultiResolutionModel,
    DeepHierarchicalModel,
    TemporalHierarchyModel
)

# Probabilistic Models
from .probabilistic_models import (
    DeepARModel,
    VAETimeSeriesModel,
    QuantileRegressionModel
)

__all__ = [
    'TimeSeriesModel',
    # RNN
    'LSTMModel',
    'GRUModel',
    'BiLSTMModel',
    'StackedLSTMModel',
    # CNN
    'CNNModel',
    'TCNModel',
    'CNNLSTMModel',
    # Transformer
    'TransformerModel',
    'AttentionModel',
    'TemporalFusionTransformer',
    # Graph
    'GNNModel',
    'GATModel',
    # Ensemble
    'EnsembleModel',
    'BaggingModel',
    'BoostingModel',
    'VotingModel',
    # Hierarchical
    'HierarchicalModel',
    'MultiResolutionModel',
    'DeepHierarchicalModel',
    'TemporalHierarchyModel',
    # Probabilistic
    'DeepARModel',
    'VAETimeSeriesModel',
    'QuantileRegressionModel',
]
