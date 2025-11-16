"""
Base Abstract Class for Time Series Models

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class TimeSeriesModel(ABC, nn.Module):
    """
    Abstract base class for all time series prediction models.
    
    This class defines the interface that all time series models must implement,
    following the Template Method and Strategy patterns for maximum flexibility
    and reusability.
    
    Attributes:
        input_size (int): Number of input features
        hidden_size (int): Size of hidden layers
        output_size (int): Number of output features
        num_layers (int): Number of layers in the model
        device (torch.device): Device to run the model on
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """
        Initialize the base time series model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            output_size: Number of output predictions
            num_layers: Number of layers
            dropout: Dropout rate for regularization
            device: Device to run model on ('cuda' or 'cpu')
            **kwargs: Additional model-specific parameters
        """
        super(TimeSeriesModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device(device)
        self.kwargs = kwargs
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            **kwargs: Additional arguments for specific models
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        pass
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        lr: float = 0.001,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        early_stopping_patience: int = 10,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            lr: Learning rate
            optimizer: Custom optimizer (if None, uses Adam)
            criterion: Loss function (if None, uses MSELoss)
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print training progress
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training and validation losses
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.to(self.device)
        self.train_losses = []
        self.val_losses = []
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                
                # Handle different output shapes
                if outputs.dim() == 3:
                    outputs = outputs[:, -1, :]  # Take last time step
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                avg_val_loss = self.evaluate(val_loader, criterion)
                self.val_losses.append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], "
                          f"Train Loss: {avg_train_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> float:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data
            criterion: Loss function
            
        Returns:
            Average loss on the dataset
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.forward(batch_x)
                
                if outputs.dim() == 3:
                    outputs = outputs[:, -1, :]
                
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def predict(
        self,
        x: torch.Tensor,
        return_numpy: bool = True
    ) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor or numpy array
            return_numpy: Whether to return numpy array or torch tensor
            
        Returns:
            Predictions as numpy array or torch tensor
        """
        self.eval()
        
        # Convert to tensor if numpy
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            predictions = self.forward(x)
            
            if predictions.dim() == 3:
                predictions = predictions[:, -1, :]
        
        if return_numpy:
            return predictions.cpu().numpy()
        return predictions
    
    def predict_sequence(
        self,
        x: torch.Tensor,
        steps: int = 1,
        return_numpy: bool = True
    ) -> np.ndarray:
        """
        Make multi-step ahead predictions.
        
        Args:
            x: Input tensor
            steps: Number of steps to predict ahead
            return_numpy: Whether to return numpy array
            
        Returns:
            Multi-step predictions
        """
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            current_input = x.clone()
            
            for _ in range(steps):
                pred = self.forward(current_input)
                
                if pred.dim() == 3:
                    pred = pred[:, -1:, :]
                else:
                    pred = pred.unsqueeze(1)
                
                predictions.append(pred)
                
                # Update input for next prediction
                current_input = torch.cat([current_input[:, 1:, :], pred], dim=1)
        
        predictions = torch.cat(predictions, dim=1)
        
        if return_numpy:
            return predictions.cpu().numpy()
        return predictions
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                **self.kwargs
            },
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    def get_num_parameters(self) -> int:
        """
        Get the total number of parameters in the model.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """
        Get a summary of the model architecture.
        
        Returns:
            String summary of the model
        """
        summary_str = f"{self.__class__.__name__}\n"
        summary_str += f"{'=' * 50}\n"
        summary_str += f"Input Size: {self.input_size}\n"
        summary_str += f"Hidden Size: {self.hidden_size}\n"
        summary_str += f"Output Size: {self.output_size}\n"
        summary_str += f"Number of Layers: {self.num_layers}\n"
        summary_str += f"Dropout: {self.dropout}\n"
        summary_str += f"Total Parameters: {self.get_num_parameters():,}\n"
        summary_str += f"Device: {self.device}\n"
        summary_str += f"{'=' * 50}\n"
        return summary_str
