"""
Ensemble Time Series Models

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
from .base import TimeSeriesModel


class EnsembleModel(TimeSeriesModel):
    """
    Ensemble model combining multiple base models.
    
    Combines predictions from multiple models using various strategies
    (averaging, weighted averaging, stacking, etc.).
    
    Supports multi-output prediction when base models support it.
    """
    
    def __init__(
        self,
        models: List[TimeSeriesModel],
        model_names: List[str] = None,
        ensemble_method: str = 'mean',
        input_size: int = None,
        hidden_size: int = None,
        output_size: int = None,
        dropout: float = 0.2,
        device: str = None,
        **kwargs
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of base models
            model_names: Names for each model (for tracking)
            ensemble_method: Method to combine predictions 
                           ('mean', 'median', 'weighted', 'stacking')
            input_size: Number of input features
            hidden_size: Hidden size (used for stacking)
            output_size: Number of outputs
            dropout: Dropout rate
            device: Device to run on
        """
        # Get sizes from first model if not provided
        if models and len(models) > 0:
            input_size = input_size or models[0].input_size
            output_size = output_size or models[0].output_size
            hidden_size = hidden_size or models[0].hidden_size
        
        super(EnsembleModel, self).__init__(
            input_size=input_size,
            output_size=output_size, 
            hidden_size=hidden_size, 
            num_layers=len(models),
            dropout=dropout,
            device=device
        )
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.num_models = len(models)
        self.model_names = model_names if model_names else [f'Model_{i}' for i in range(len(models))]
        
        # Weights for weighted averaging
        if ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        
        # Meta-learner for stacking
        if ensemble_method == 'stacking':
            self.meta_learner = nn.Sequential(
                nn.Linear(output_size * self.num_models, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, output_size)
            )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments for base models
            
        Returns:
            Ensemble prediction
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x, **kwargs)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (num_models, batch_size, output_size)
        
        # Combine predictions based on method
        if self.ensemble_method == 'mean':
            output = torch.mean(predictions, dim=0)
        
        elif self.ensemble_method == 'median':
            output = torch.median(predictions, dim=0)[0]
        
        elif self.ensemble_method == 'weighted':
            # Normalize weights using softmax
            import torch.nn.functional as F
            weights = F.softmax(self.weights, dim=0)
            weights = weights.view(-1, 1, 1)  # (num_models, 1, 1)
            output = torch.sum(predictions * weights, dim=0)
        
        elif self.ensemble_method == 'stacking':
            # Concatenate predictions and feed to meta-learner
            batch_size = predictions.shape[1]
            stacked = predictions.permute(1, 0, 2).reshape(batch_size, -1)  # (batch_size, num_models*output_size)
            output = self.meta_learner(stacked)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return output
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        train_base_models: bool = True,
        early_stopping_patience: int = 10,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Train ensemble model.
        
        Args:
            X_train: Training features (numpy array)
            y_train: Training targets (numpy array)
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            train_base_models: Whether to train base models
            early_stopping_patience: Patience for early stopping
            verbose: Print progress
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of training histories for each model
        """
        histories = {}
        
        # Train base models if requested
        if train_base_models:
            if verbose:
                print(f"Training {self.num_models} base models...")
            
            for i, (model, name) in enumerate(zip(self.models, self.model_names)):
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Training {name} ({i + 1}/{self.num_models})")
                    print(f"{'='*60}")
                
                history = model.fit(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    early_stopping_patience=early_stopping_patience,
                    verbose=verbose,
                    **kwargs
                )
                histories[name] = history
        
        # For weighted or stacking methods, train the combiner
        if self.ensemble_method in ['weighted', 'stacking']:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Training ensemble combiner ({self.ensemble_method})")
                print(f"{'='*60}")
            
            # Train only the combiner weights/meta-learner
            # First freeze base models
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False
            
            # Train combiner using parent class fit
            combiner_history = super().fit(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs // 2,  # Use fewer epochs for combiner
                batch_size=batch_size,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                verbose=verbose,
                **kwargs
            )
            histories['combiner'] = combiner_history
            
            # Unfreeze base models
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = True
        
        return histories


class BaggingModel(TimeSeriesModel):
    """
    Bagging ensemble for time series.
    
    Trains multiple models on different bootstrap samples of the data.
    """
    
    def __init__(
        self,
        base_model_class,
        num_models: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        model_kwargs: Dict[str, Any] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize bagging ensemble."""
        super(BaggingModel, self).__init__(
            input_size, hidden_size, output_size, num_models, device=device, **kwargs
        )
        
        if model_kwargs is None:
            model_kwargs = {}
        
        # Create base models
        self.models = nn.ModuleList([
            base_model_class(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                device=device,
                **model_kwargs
            )
            for _ in range(num_models)
        ])
        
        self.num_models = num_models
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - average predictions from all models."""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Average predictions
        output = torch.mean(torch.stack(predictions), dim=0)
        return output
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader = None,
        epochs: int = 100,
        lr: float = 0.001,
        bootstrap_ratio: float = 0.8,
        **kwargs
    ):
        """Train bagging ensemble with bootstrap sampling."""
        print("Training bagging ensemble...")
        
        # Get all data for bootstrap sampling
        all_data = []
        for batch_x, batch_y in train_loader:
            all_data.append((batch_x, batch_y))
        
        # Train each model on bootstrap sample
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i + 1}/{self.num_models}")
            
            # Create bootstrap sample
            # In practice, you'd want to implement proper bootstrap DataLoader
            model.fit(train_loader, val_loader, epochs, lr, **kwargs)
        
        return {'train_losses': [], 'val_losses': []}


class BoostingModel(TimeSeriesModel):
    """
    Gradient Boosting for time series.
    
    Sequentially trains models where each focuses on errors of previous models.
    """
    
    def __init__(
        self,
        base_model_class,
        num_models: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.1,
        model_kwargs: Dict[str, Any] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize boosting ensemble."""
        super(BoostingModel, self).__init__(
            input_size, hidden_size, output_size, num_models, device=device, **kwargs
        )
        
        if model_kwargs is None:
            model_kwargs = {}
        
        self.models = nn.ModuleList([
            base_model_class(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                device=device,
                **model_kwargs
            )
            for _ in range(num_models)
        ])
        
        self.num_models = num_models
        self.learning_rate = learning_rate
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - sum weighted predictions."""
        output = torch.zeros(x.size(0), self.output_size).to(self.device)
        
        for model in self.models:
            pred = model(x)
            output = output + self.learning_rate * pred
        
        return output


class VotingModel(TimeSeriesModel):
    """
    Voting ensemble for classification tasks.
    
    Combines predictions through majority voting (hard) or probability averaging (soft).
    """
    
    def __init__(
        self,
        models: List[TimeSeriesModel],
        voting: str = 'soft',
        input_size: int = None,
        hidden_size: int = None,
        output_size: int = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize voting ensemble."""
        if models and len(models) > 0:
            input_size = input_size or models[0].input_size
            output_size = output_size or models[0].output_size
            hidden_size = hidden_size or models[0].hidden_size
        
        super(VotingModel, self).__init__(
            input_size, hidden_size, output_size, len(models), device=device, **kwargs
        )
        
        self.models = nn.ModuleList(models)
        self.voting = voting  # 'hard' or 'soft'
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through voting ensemble."""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        if self.voting == 'soft':
            # Average predictions
            output = torch.mean(torch.stack(predictions), dim=0)
        else:
            # Hard voting (for classification)
            stacked = torch.stack(predictions)
            output = torch.mode(stacked, dim=0)[0]
        
        return output
