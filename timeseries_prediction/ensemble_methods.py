"""
TIME SERIES ENSEMBLE METHODS
=============================

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha
Email: ajsinha@gmail.com

Legal Notice:
This document and the associated software architecture are proprietary and confidential. 
Unauthorized copying, distribution, modification, or use of this document or the software 
system it describes is strictly prohibited without explicit written permission from the 
copyright holder. This document is provided "as is" without warranty of any kind, either 
expressed or implied. The copyright holder shall not be liable for any damages arising 
from the use of this document or the software system it describes.

Patent Pending: Certain architectural patterns and implementations described in this 
document may be subject to patent applications.

================================================================================

This module implements various ensemble methods for time series prediction:
1. Simple Averaging Ensemble
2. Weighted Averaging Ensemble  
3. Stacking Ensemble
4. Bagging Ensemble
5. Diversity Ensemble (Multiple Architectures)

Author: Ashutosh Sinha
Contact: ajsinha@gmail.com
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
from typing import List, Optional
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# BASE MODELS
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout))
        self.fc1 = nn.Linear(hidden_sizes[-1], 32)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = dropout(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        return self.fc_out(out)


class GRUModel(nn.Module):
    """GRU model for time series prediction"""
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(1, len(hidden_sizes)):
            self.gru_layers.append(nn.GRU(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout))
        self.fc1 = nn.Linear(hidden_sizes[-1], 32)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, 1)
        
    def forward(self, x):
        gru_out, _ = self.gru1(x)
        gru_out = self.dropout1(gru_out)
        for gru, dropout in zip(self.gru_layers, self.dropout_layers):
            gru_out, _ = gru(gru_out)
            gru_out = dropout(gru_out)
        gru_out = gru_out[:, -1, :]
        out = self.fc1(gru_out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        return self.fc_out(out)


# ============================================================================
# ENSEMBLE METHODS
# ============================================================================

class AveragingEnsemble:
    """
    Simple Averaging Ensemble
    Averages predictions from multiple models
    
    Best for: When all models have similar performance
    Pros: Simple, reduces variance
    Cons: Treats all models equally
    """
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
        
    def predict(self, dataloader, device):
        """Make predictions by averaging all model outputs"""
        all_predictions = []
        for model in self.models:
            model.eval()
            predictions = []
            with torch.no_grad():
                for batch_X, _ in dataloader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    predictions.extend(outputs.cpu().numpy())
            all_predictions.append(predictions)
        return np.mean(all_predictions, axis=0)


class WeightedEnsemble:
    """
    Weighted Averaging Ensemble
    Uses performance-based weights
    
    Best for: When models have different performance
    Pros: Prioritizes better models
    Cons: Requires validation set
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    @classmethod
    def from_validation_performance(cls, models: List[nn.Module], val_loader, device, criterion):
        """Create ensemble with weights based on validation performance"""
        val_losses = []
        for model in models:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))
        
        # Inverse of loss as weight
        inverse_losses = [1.0 / loss for loss in val_losses]
        weights = [w / sum(inverse_losses) for w in inverse_losses]
        return cls(models, weights)
    
    def predict(self, dataloader, device):
        """Make weighted predictions"""
        all_predictions = []
        for model in self.models:
            model.eval()
            predictions = []
            with torch.no_grad():
                for batch_X, _ in dataloader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    predictions.extend(outputs.cpu().numpy())
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)
        weights = np.array(self.weights).reshape(-1, 1)
        return np.sum(all_predictions * weights, axis=0)


class StackingEnsemble:
    """
    Stacking Ensemble
    Uses meta-learner to combine predictions
    
    Best for: Learning optimal combination
    Pros: Can learn complex patterns
    Cons: Risk of overfitting
    """
    
    def __init__(self, base_models: List[nn.Module], meta_learner=None):
        self.base_models = base_models
        self.meta_learner = meta_learner or self._create_default_meta_learner()
    
    def _create_default_meta_learner(self):
        """Create simple linear meta-learner"""
        n_models = len(self.base_models)
        return nn.Sequential(
            nn.Linear(n_models, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def train_meta_learner(self, val_loader, device, epochs=20, lr=0.001):
        """Train meta-learner on validation set"""
        self.meta_learner = self.meta_learner.to(device)
        optimizer = optim.Adam(self.meta_learner.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        print(f"\nTraining meta-learner...")
        for epoch in range(epochs):
            self.meta_learner.train()
            total_loss = 0.0
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                base_predictions = []
                for model in self.base_models:
                    model.eval()
                    with torch.no_grad():
                        pred = model(batch_X)
                    base_predictions.append(pred)
                meta_input = torch.cat(base_predictions, dim=1)
                optimizer.zero_grad()
                meta_output = self.meta_learner(meta_input)
                loss = criterion(meta_output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(val_loader):.4f}")
    
    def predict(self, dataloader, device):
        """Make predictions using stacking"""
        self.meta_learner.eval()
        ensemble_predictions = []
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(device)
                base_predictions = []
                for model in self.base_models:
                    model.eval()
                    pred = model(batch_X)
                    base_predictions.append(pred)
                meta_input = torch.cat(base_predictions, dim=1)
                meta_output = self.meta_learner(meta_input)
                ensemble_predictions.extend(meta_output.cpu().numpy())
        return np.array(ensemble_predictions)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def train_single_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, patience=10):
    """Train a single model with early stopping"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_state)
    return model


def evaluate_ensemble(ensemble, test_loader, actuals, device):
    """Evaluate ensemble performance"""
    predictions = ensemble.predict(test_loader, device)
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    return {'mse': mse, 'mae': mae, 'rmse': rmse}, predictions


# Save this file as ensemble_methods.py
# Usage example in separate script
