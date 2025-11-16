"""
Base Model Module
=================

Abstract base class for all time series prediction models.
Provides common interface for training, testing, and prediction.

SUPPORTS MULTI-OUTPUT PREDICTION: Set output_size > 1 to predict multiple variables simultaneously.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
from pathlib import Path


class TimeSeriesModel(ABC, nn.Module):
    """
    Abstract base class for all time series models.
    
    MULTI-OUTPUT SUPPORT:
    ---------------------
    This model fully supports predicting multiple variables simultaneously.
    Simply set output_size to the number of target variables you want to predict.
    
    Example:
        - output_size=1: Predict single variable (e.g., one stock price)
        - output_size=2: Predict two variables (e.g., BMO and JPM prices)
        - output_size=30: Predict 30 variables (e.g., all DOW30 stocks)
    
    All models must implement:
    - forward(): Forward pass computation
    - _build_model(): Model architecture construction
    
    Provides common functionality:
    - fit(): Training with validation
    - predict(): Generate predictions
    - evaluate(): Compute metrics
    - save()/load(): Model persistence
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 device: str = None):
        """
        Initialize base time series model.
        
        Args:
            input_size: Number of input features (e.g., 30 for DOW30 stocks)
            output_size: Number of output features to predict (e.g., 2 for BMO+JPM)
            hidden_size: Size of hidden layers
            num_layers: Number of layers
            dropout: Dropout probability
            device: Device to run model on ('cuda' or 'cpu')
        """
        super(TimeSeriesModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
        # Build model (implemented by subclasses)
        self._build_model()
        self.to(self.device)
    
    @abstractmethod
    def _build_model(self):
        """Build the model architecture. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass. Must be implemented by subclasses.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Predictions of shape (batch_size, output_size)
            - If output_size=1: shape is (batch_size, 1)
            - If output_size=2: shape is (batch_size, 2) for two variables
            - etc.
        """
        pass
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            optimizer_name: str = 'adam',
            loss_fn: Optional[nn.Module] = None,
            early_stopping_patience: int = 10,
            verbose: bool = True,
            **kwargs) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training features (samples, seq_length, input_features)
            y_train: Training targets (samples, output_size) - can be multi-output!
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            optimizer_name: Optimizer ('adam', 'sgd', 'adamw')
            loss_fn: Loss function (default: MSELoss)
            early_stopping_patience: Patience for early stopping
            verbose: Print training progress
            **kwargs: Additional model-specific arguments
            
        Returns:
            Dictionary containing training history
        """
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        # Ensure y_train has correct shape for multi-output
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
        
        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            if len(y_val.shape) == 1:
                y_val = y_val.unsqueeze(1)
            use_validation = True
        else:
            use_validation = False
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Loss function
        if loss_fn is None:
            criterion = nn.MSELoss()
        else:
            criterion = loss_fn
        
        # Optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self(batch_X, **kwargs)
                
                # Ensure shapes match
                if len(predictions.shape) == 1:
                    predictions = predictions.unsqueeze(1)
                
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if use_validation:
                self.eval()
                with torch.no_grad():
                    val_predictions = self(X_val, **kwargs)
                    
                    # Ensure shapes match
                    if len(val_predictions.shape) == 1:
                        val_predictions = val_predictions.unsqueeze(1)
                    
                    val_loss = criterion(val_predictions, y_val).item()
                    self.history['val_loss'].append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}")
            
            self.history['epochs'].append(epoch + 1)
        
        # Restore best model if using validation
        if use_validation and best_model_state is not None:
            self.load_state_dict(best_model_state)
        
        return self.history
    
    def predict(self, 
                X: np.ndarray,
                batch_size: int = 32,
                **kwargs) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input features (samples, seq_length, input_features)
            batch_size: Batch size for prediction
            **kwargs: Additional model-specific arguments
            
        Returns:
            Predictions as numpy array (samples, output_size)
            - For multi-output: each row contains predictions for all target variables
        """
        self.eval()
        
        X = torch.FloatTensor(X).to(self.device)
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                pred = self(batch, **kwargs)
                predictions.append(pred.cpu().numpy())
        
        return np.vstack(predictions)
    
    def evaluate(self,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 metrics: Optional[List[str]] = None,
                 per_output: bool = False,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets (can be multi-output)
            metrics: List of metrics to compute 
                    ['mse', 'rmse', 'mae', 'mape', 'r2']
            per_output: If True and multi-output, compute metrics per output variable
            **kwargs: Additional prediction arguments
            
        Returns:
            Dictionary of metric values
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score
        )
        
        if metrics is None:
            metrics = ['mse', 'rmse', 'mae', 'r2']
        
        predictions = self.predict(X_test, **kwargs)
        
        # Ensure y_test is 2D
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
        
        results = {}
        
        if per_output and self.output_size > 1:
            # Compute metrics for each output variable
            for i in range(self.output_size):
                suffix = f"_output_{i}"
                y_true_i = y_test[:, i]
                y_pred_i = predictions[:, i]
                
                if 'mse' in metrics:
                    results[f'mse{suffix}'] = mean_squared_error(y_true_i, y_pred_i)
                
                if 'rmse' in metrics:
                    results[f'rmse{suffix}'] = np.sqrt(mean_squared_error(y_true_i, y_pred_i))
                
                if 'mae' in metrics:
                    results[f'mae{suffix}'] = mean_absolute_error(y_true_i, y_pred_i)
                
                if 'mape' in metrics:
                    mape = np.mean(np.abs((y_true_i - y_pred_i) / (np.abs(y_true_i) + 1e-8))) * 100
                    results[f'mape{suffix}'] = mape
                
                if 'r2' in metrics:
                    results[f'r2{suffix}'] = r2_score(y_true_i, y_pred_i)
        
        # Overall metrics (averaged across all outputs)
        if 'mse' in metrics:
            results['mse'] = mean_squared_error(y_test, predictions)
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
        
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_test, predictions)
        
        if 'mape' in metrics:
            mape = np.mean(np.abs((y_test - predictions) / (np.abs(y_test) + 1e-8))) * 100
            results['mape'] = mape
        
        if 'r2' in metrics:
            results['r2'] = r2_score(y_test, predictions)
        
        return results
    
    def save(self, filepath: str, save_history: bool = True):
        """Save model to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }
        
        if save_history:
            save_dict['history'] = self.history
        
        torch.save(save_dict, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: str = None):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location='cpu')
        model = cls(**checkpoint['model_config'], device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'history' in checkpoint:
            model.history = checkpoint['history']
        
        model.eval()
        return model
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        """Print model summary."""
        print(f"\n{'=' * 70}")
        print(f"{self.__class__.__name__} Summary")
        print(f"{'=' * 70}")
        print(f"Input size: {self.input_size}")
        print(f"Output size: {self.output_size} ({'MULTI-OUTPUT' if self.output_size > 1 else 'SINGLE OUTPUT'})")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Dropout: {self.dropout}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {self.get_num_parameters():,}")
        print(f"{'=' * 70}\n")
