"""
Model Factory - Factory Pattern Implementation

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

from typing import Dict, Any, Type
from ..models.base import TimeSeriesModel
from ..models import (
    LSTMModel, GRUModel, BiLSTMModel, StackedLSTMModel,
    CNNModel, TCNModel, CNNLSTMModel,
    TransformerModel, AttentionModel, TemporalFusionTransformer,
    GNNModel, GATModel,
    EnsembleModel, BaggingModel, BoostingModel, VotingModel,
    HierarchicalModel, MultiResolutionModel, DeepHierarchicalModel,
    TemporalHierarchyModel,
    DeepARModel, VAETimeSeriesModel, QuantileRegressionModel
)


class ModelFactory:
    """
    Factory class for creating time series models.
    
    Implements the Factory pattern to provide a centralized way to create
    different model instances with consistent interfaces.
    
    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create_model(
        ...     'lstm',
        ...     input_size=10,
        ...     hidden_size=64,
        ...     output_size=1
        ... )
    """
    
    # Registry of available models
    _model_registry: Dict[str, Type[TimeSeriesModel]] = {
        # RNN Models
        'lstm': LSTMModel,
        'gru': GRUModel,
        'bilstm': BiLSTMModel,
        'stacked_lstm': StackedLSTMModel,
        
        # CNN Models
        'cnn': CNNModel,
        'tcn': TCNModel,
        'cnn_lstm': CNNLSTMModel,
        
        # Transformer Models
        'transformer': TransformerModel,
        'attention': AttentionModel,
        'tft': TemporalFusionTransformer,
        
        # Graph Models
        'gnn': GNNModel,
        'gat': GATModel,
        
        # Ensemble Models
        'ensemble': EnsembleModel,
        'bagging': BaggingModel,
        'boosting': BoostingModel,
        'voting': VotingModel,
        
        # Hierarchical Models
        'hierarchical': HierarchicalModel,
        'multiresolution': MultiResolutionModel,
        'deep_hierarchical': DeepHierarchicalModel,
        'temporal_hierarchy': TemporalHierarchyModel,
        
        # Probabilistic Models
        'deepar': DeepARModel,
        'vae': VAETimeSeriesModel,
        'quantile': QuantileRegressionModel,
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        **kwargs
    ) -> TimeSeriesModel:
        """
        Create a time series model instance.
        
        Args:
            model_type: Type of model to create (e.g., 'lstm', 'transformer')
            **kwargs: Model-specific parameters
            
        Returns:
            Instance of the requested model
            
        Raises:
            ValueError: If model_type is not registered
            
        Example:
            >>> model = ModelFactory.create_model(
            ...     'lstm',
            ...     input_size=10,
            ...     hidden_size=64,
            ...     output_size=1,
            ...     num_layers=2
            ... )
        """
        model_type = model_type.lower()
        
        if model_type not in cls._model_registry:
            available = ', '.join(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available models: {available}"
            )
        
        model_class = cls._model_registry[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def register_model(
        cls,
        name: str,
        model_class: Type[TimeSeriesModel]
    ) -> None:
        """
        Register a new model type.
        
        Allows users to register custom model classes.
        
        Args:
            name: Name to register the model under
            model_class: Model class to register
            
        Example:
            >>> class MyCustomModel(TimeSeriesModel):
            ...     pass
            >>> ModelFactory.register_model('custom', MyCustomModel)
        """
        if not issubclass(model_class, TimeSeriesModel):
            raise TypeError(
                f"{model_class} must be a subclass of TimeSeriesModel"
            )
        
        cls._model_registry[name.lower()] = model_class
    
    @classmethod
    def list_models(cls) -> list:
        """
        List all available model types.
        
        Returns:
            List of registered model names
        """
        return sorted(cls._model_registry.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get information about a specific model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary containing model information
        """
        model_type = model_type.lower()
        
        if model_type not in cls._model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._model_registry[model_type]
        
        return {
            'name': model_type,
            'class': model_class.__name__,
            'docstring': model_class.__doc__,
            'module': model_class.__module__
        }
    
    @classmethod
    def create_ensemble_from_configs(
        cls,
        model_configs: list,
        ensemble_method: str = 'average',
        **ensemble_kwargs
    ) -> EnsembleModel:
        """
        Create an ensemble model from multiple model configurations.
        
        Args:
            model_configs: List of dictionaries with model configurations
                          Each dict should have 'type' and model parameters
            ensemble_method: Method for combining predictions
            **ensemble_kwargs: Additional ensemble parameters
            
        Returns:
            EnsembleModel instance
            
        Example:
            >>> configs = [
            ...     {'type': 'lstm', 'input_size': 10, 'hidden_size': 64, 'output_size': 1},
            ...     {'type': 'gru', 'input_size': 10, 'hidden_size': 64, 'output_size': 1},
            ...     {'type': 'transformer', 'input_size': 10, 'hidden_size': 64, 'output_size': 1}
            ... ]
            >>> ensemble = ModelFactory.create_ensemble_from_configs(
            ...     configs,
            ...     ensemble_method='weighted'
            ... )
        """
        models = []
        
        for config in model_configs:
            config = config.copy()
            model_type = config.pop('type')
            model = cls.create_model(model_type, **config)
            models.append(model)
        
        return EnsembleModel(
            models=models,
            ensemble_method=ensemble_method,
            **ensemble_kwargs
        )
    
    @classmethod
    def get_default_config(cls, model_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with default parameters
        """
        default_configs = {
            'lstm': {
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': False
            },
            'gru': {
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': False
            },
            'bilstm': {
                'num_layers': 2,
                'dropout': 0.2
            },
            'transformer': {
                'num_layers': 3,
                'num_heads': 4,
                'dim_feedforward': 512,
                'dropout': 0.2
            },
            'cnn': {
                'num_layers': 3,
                'kernel_sizes': [3, 3, 3],
                'dropout': 0.2
            },
            'tcn': {
                'num_layers': 3,
                'kernel_size': 3,
                'dropout': 0.2
            },
            'gnn': {
                'num_layers': 2,
                'dropout': 0.2
            },
            'tft': {
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.2
            },
            'deepar': {
                'num_layers': 2,
                'dropout': 0.2,
                'distribution': 'gaussian'
            },
            'vae': {
                'num_layers': 2,
                'dropout': 0.2
            }
        }
        
        model_type = model_type.lower()
        return default_configs.get(model_type, {})


class ModelBuilder:
    """
    Builder pattern for constructing complex models step-by-step.
    
    Provides a fluent interface for model creation with validation.
    """
    
    def __init__(self):
        self._model_type = None
        self._params = {}
    
    def set_type(self, model_type: str) -> 'ModelBuilder':
        """Set the model type."""
        self._model_type = model_type
        return self
    
    def set_input_size(self, size: int) -> 'ModelBuilder':
        """Set input size."""
        self._params['input_size'] = size
        return self
    
    def set_hidden_size(self, size: int) -> 'ModelBuilder':
        """Set hidden size."""
        self._params['hidden_size'] = size
        return self
    
    def set_output_size(self, size: int) -> 'ModelBuilder':
        """Set output size."""
        self._params['output_size'] = size
        return self
    
    def set_num_layers(self, num: int) -> 'ModelBuilder':
        """Set number of layers."""
        self._params['num_layers'] = num
        return self
    
    def set_dropout(self, rate: float) -> 'ModelBuilder':
        """Set dropout rate."""
        self._params['dropout'] = rate
        return self
    
    def set_device(self, device: str) -> 'ModelBuilder':
        """Set device."""
        self._params['device'] = device
        return self
    
    def add_param(self, key: str, value: Any) -> 'ModelBuilder':
        """Add custom parameter."""
        self._params[key] = value
        return self
    
    def build(self) -> TimeSeriesModel:
        """
        Build and return the model.
        
        Returns:
            Configured model instance
        """
        if self._model_type is None:
            raise ValueError("Model type must be set before building")
        
        # Validate required parameters
        required = ['input_size', 'hidden_size', 'output_size']
        missing = [p for p in required if p not in self._params]
        
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        return ModelFactory.create_model(self._model_type, **self._params)
    
    def reset(self) -> 'ModelBuilder':
        """Reset the builder."""
        self._model_type = None
        self._params = {}
        return self
