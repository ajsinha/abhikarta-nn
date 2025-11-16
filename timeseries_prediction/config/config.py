"""
Configuration Management

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    model_type: str
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int = 2
    dropout: float = 0.2
    device: str = 'cuda'
    
    # Model-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss_function: str = 'mse'
    early_stopping_patience: int = 10
    validation_split: float = 0.15
    test_split: float = 0.15
    shuffle: bool = True
    random_seed: int = 42
    
    # Additional parameters
    gradient_clip: Optional[float] = 1.0
    weight_decay: float = 0.0
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    seq_length: int = 30
    prediction_horizon: int = 1
    stride: int = 1
    scaler_method: str = 'standard'
    handle_missing: str = 'forward_fill'
    
    # Feature engineering
    add_time_features: bool = False
    add_lag_features: bool = False
    lag_periods: list = field(default_factory=lambda: [1, 2, 3])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    name: str
    description: str = ""
    
    model_config: ModelConfig = None
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    
    # Paths
    data_path: Optional[str] = None
    save_dir: str = './experiments'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'model_config': self.model_config.to_dict() if self.model_config else None,
            'training_config': self.training_config.to_dict(),
            'data_config': self.data_config.to_dict(),
            'data_path': self.data_path,
            'save_dir': self.save_dir
        }
    
    def save(self, path: Optional[str] = None):
        """Save complete configuration."""
        if path is None:
            path = Path(self.save_dir) / self.name / 'config.json'
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load complete configuration."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested configs
        if config_dict.get('model_config'):
            config_dict['model_config'] = ModelConfig.from_dict(config_dict['model_config'])
        
        config_dict['training_config'] = TrainingConfig.from_dict(config_dict['training_config'])
        config_dict['data_config'] = DataConfig.from_dict(config_dict['data_config'])
        
        return cls(**config_dict)


# Predefined configurations for common use cases
PRESET_CONFIGS = {
    'financial_short_term': {
        'model': ModelConfig(
            model_type='lstm',
            input_size=10,
            hidden_size=128,
            output_size=1,
            num_layers=3,
            dropout=0.3
        ),
        'data': DataConfig(
            seq_length=20,
            prediction_horizon=1,
            scaler_method='robust'
        ),
        'training': TrainingConfig(
            epochs=200,
            batch_size=64,
            learning_rate=0.0001,
            early_stopping_patience=20
        )
    },
    'financial_long_term': {
        'model': ModelConfig(
            model_type='transformer',
            input_size=10,
            hidden_size=256,
            output_size=1,
            num_layers=4,
            dropout=0.2,
            extra_params={'num_heads': 8, 'dim_feedforward': 1024}
        ),
        'data': DataConfig(
            seq_length=60,
            prediction_horizon=10,
            scaler_method='standard'
        ),
        'training': TrainingConfig(
            epochs=300,
            batch_size=32,
            learning_rate=0.00005,
            early_stopping_patience=30
        )
    },
    'multivariate_complex': {
        'model': ModelConfig(
            model_type='tft',
            input_size=20,
            hidden_size=256,
            output_size=5,
            num_layers=3,
            dropout=0.2,
            extra_params={'num_heads': 4}
        ),
        'data': DataConfig(
            seq_length=30,
            prediction_horizon=5,
            scaler_method='standard',
            add_time_features=True
        ),
        'training': TrainingConfig(
            epochs=250,
            batch_size=64,
            learning_rate=0.0001,
            early_stopping_patience=25
        )
    }
}


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """
    Get a preset configuration.
    
    Args:
        preset_name: Name of preset configuration
        
    Returns:
        Dictionary with model, data, and training configs
    """
    if preset_name not in PRESET_CONFIGS:
        available = ', '.join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    return PRESET_CONFIGS[preset_name]


def list_preset_configs() -> list:
    """List available preset configurations."""
    return list(PRESET_CONFIGS.keys())
