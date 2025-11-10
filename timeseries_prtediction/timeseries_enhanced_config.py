"""
ENHANCED TIME SERIES MODELS - CONFIGURABLE TRANSFORMATION
==========================================================

Supports two transformation methods:
1. RATIO: value(t) / value(t-1)
2. FRACTIONAL_CHANGE: (value(t) - value(t-1)) / value(t-1)

Configuration via TransformConfig class

Author: Ashutosh Sinha
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import math
from enum import Enum
from dataclasses import dataclass

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# CONFIGURATION
# ============================================================================

class TransformMethod(Enum):
    """Enumeration of available transformation methods"""
    RATIO = "ratio"                      # value(t) / value(t-1)
    FRACTIONAL_CHANGE = "fractional"     # (value(t) - value(t-1)) / value(t-1)
    PERCENTAGE_CHANGE = "percentage"     # 100 * (value(t) - value(t-1)) / value(t-1)


@dataclass
class TransformConfig:
    """
    Configuration for data transformation
    
    Attributes:
        method: Transformation method to use
        log_transform: Whether to apply log transformation after ratio/change
        clip_values: Clip extreme values to prevent outliers
        clip_range: Range for clipping (min, max)
    """
    method: TransformMethod = TransformMethod.RATIO
    log_transform: bool = True
    clip_values: bool = False
    clip_range: tuple = (-3, 3)  # After standardization
    
    def __str__(self):
        return (f"TransformConfig(method={self.method.value}, "
                f"log_transform={self.log_transform}, "
                f"clip_values={self.clip_values})")


# ============================================================================
# ENHANCED DATA PREPROCESSING
# ============================================================================

class EnhancedTimeSeriesPreprocessor:
    """
    Enhanced preprocessor with configurable transformation methods
    
    Supports:
    - Ratio: value(t) / value(t-1)
    - Fractional change: (value(t) - value(t-1)) / value(t-1)
    - Percentage change: 100 * (value(t) - value(t-1)) / value(t-1)
    
    With optional log transformation and clipping
    """
    
    def __init__(self, config: TransformConfig = None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: TransformConfig object specifying transformation method
        """
        self.config = config or TransformConfig()
        self.first_values = None
        self.scaler = StandardScaler()
        
    def _calculate_ratio(self, data):
        """Calculate ratio: value(t) / value(t-1)"""
        ratios = np.ones_like(data, dtype=np.float32)
        
        for i in range(1, len(data)):
            ratios[i] = np.where(data[i-1] != 0, data[i] / data[i-1], 1.0)
        
        return ratios
    
    def _calculate_fractional_change(self, data):
        """Calculate fractional change: (value(t) - value(t-1)) / value(t-1)"""
        changes = np.zeros_like(data, dtype=np.float32)
        
        for i in range(1, len(data)):
            # (value(t) - value(t-1)) / value(t-1) = value(t)/value(t-1) - 1
            changes[i] = np.where(data[i-1] != 0, 
                                 (data[i] - data[i-1]) / data[i-1], 
                                 0.0)
        
        return changes
    
    def _calculate_percentage_change(self, data):
        """Calculate percentage change: 100 * (value(t) - value(t-1)) / value(t-1)"""
        changes = self._calculate_fractional_change(data)
        return changes * 100
    
    def _apply_transformation(self, data):
        """Apply the configured transformation method"""
        if self.config.method == TransformMethod.RATIO:
            transformed = self._calculate_ratio(data)
            
            # For ratio, log transform makes sense
            if self.config.log_transform:
                transformed = np.log(transformed + 1e-10)
                
        elif self.config.method == TransformMethod.FRACTIONAL_CHANGE:
            transformed = self._calculate_fractional_change(data)
            
            # For fractional change, log1p is better
            if self.config.log_transform:
                # log1p(x) = log(1+x), works well for small changes
                transformed = np.log1p(transformed)
                
        elif self.config.method == TransformMethod.PERCENTAGE_CHANGE:
            transformed = self._calculate_percentage_change(data)
            
            # For percentage, log1p after dividing by 100
            if self.config.log_transform:
                transformed = np.log1p(transformed / 100) * 100
        
        else:
            raise ValueError(f"Unknown transformation method: {self.config.method}")
        
        return transformed
    
    def fit_transform(self, data):
        """
        Fit the preprocessor and transform data
        
        Args:
            data: numpy array or DataFrame with shape (n_samples, n_features)
        
        Returns:
            transformed_data: Scaled and transformed data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Store first values for potential inverse transform
        self.first_values = data[0].copy()
        
        # Apply transformation
        transformed_data = self._apply_transformation(data)
        
        # Standardize
        scaled_data = self.scaler.fit_transform(transformed_data)
        
        # Optional clipping
        if self.config.clip_values:
            scaled_data = np.clip(scaled_data, 
                                 self.config.clip_range[0], 
                                 self.config.clip_range[1])
        
        return scaled_data.astype(np.float32)
    
    def transform(self, data):
        """Transform new data using fitted scaler"""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Apply transformation
        transformed_data = self._apply_transformation(data)
        
        # Standardize
        scaled_data = self.scaler.transform(transformed_data)
        
        # Optional clipping
        if self.config.clip_values:
            scaled_data = np.clip(scaled_data,
                                 self.config.clip_range[0],
                                 self.config.clip_range[1])
        
        return scaled_data.astype(np.float32)
    
    def get_transformation_info(self):
        """Get information about the transformation"""
        info = {
            'method': self.config.method.value,
            'log_transform': self.config.log_transform,
            'clip_values': self.config.clip_values,
            'clip_range': self.config.clip_range,
            'scaler_mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
            'scaler_std': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None
        }
        return info


# ============================================================================
# DATASET (Same as before)
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences"""
    
    def __init__(self, X, y, sequence_length):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length]
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_target)


# ============================================================================
# MODELS (Import from previous implementation)
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
            self.lstm_layers.append(
                nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True)
            )
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
        out = self.fc_out(out)
        return out


class GRUModel(nn.Module):
    """GRU model for time series prediction"""
    
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.gru1 = nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(1, len(hidden_sizes)):
            self.gru_layers.append(
                nn.GRU(hidden_sizes[i-1], hidden_sizes[i], batch_first=True)
            )
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
        out = self.fc_out(out)
        return out


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, patience=15, model_name="model"):
    """Train model with early stopping"""
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - batch_y)).item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(outputs - batch_y)).item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return history, best_model_state


def evaluate_model(model, test_loader, device):
    """Evaluate model on test data"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    
    return predictions, actuals, {'mse': mse, 'mae': mae, 'rmse': rmse}


# ============================================================================
# COMPARISON FUNCTION
# ============================================================================

def compare_transformation_methods(X_raw, y_raw, sequence_length=20, batch_size=32):
    """
    Compare different transformation methods on the same data
    
    Args:
        X_raw: Raw input data
        y_raw: Raw target data
        sequence_length: Length of sequences
        batch_size: Batch size for training
    
    Returns:
        results: Dictionary with results for each method
    """
    
    methods = [
        TransformMethod.RATIO,
        TransformMethod.FRACTIONAL_CHANGE,
        TransformMethod.PERCENTAGE_CHANGE
    ]
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"Testing Transformation Method: {method.value.upper()}")
        print(f"{'='*70}")
        
        # Create config
        config = TransformConfig(
            method=method,
            log_transform=True,
            clip_values=False
        )
        
        print(f"Config: {config}")
        
        # Preprocess data
        X_preprocessor = EnhancedTimeSeriesPreprocessor(config)
        y_preprocessor = EnhancedTimeSeriesPreprocessor(config)
        
        X_scaled = X_preprocessor.fit_transform(X_raw)
        y_scaled = y_preprocessor.fit_transform(y_raw)
        
        # Split data
        train_size = int(0.7 * len(X_scaled))
        val_size = int(0.15 * len(X_scaled))
        
        X_train = X_scaled[:train_size]
        y_train = y_scaled[:train_size]
        X_val = X_scaled[train_size:train_size + val_size]
        y_val = y_scaled[train_size:train_size + val_size]
        X_test = X_scaled[train_size + val_size:]
        y_test = y_scaled[train_size + val_size:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
        test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create and train model
        model = LSTMModel(input_size=X_raw.shape[1], hidden_sizes=[64, 32], dropout=0.2)
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        history, best_state = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=50, device=device, patience=10, 
            model_name=f"LSTM-{method.value}"
        )
        
        # Load best model and evaluate
        model.load_state_dict(best_state)
        predictions, actuals, metrics = evaluate_model(model, test_loader, device)
        
        # Store results
        results[method.value] = {
            'config': config,
            'metrics': metrics,
            'history': history,
            'predictions': predictions,
            'actuals': actuals,
            'transform_info': X_preprocessor.get_transformation_info()
        }
        
        print(f"\nTest Results for {method.value}:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
    
    return results


# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def visualize_transformation_comparison(results):
    """Visualize comparison of different transformation methods"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training history comparison
    ax = axes[0, 0]
    for method_name, result in results.items():
        history = result['history']
        ax.plot(history['val_loss'], label=f"{method_name}", linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Metrics comparison
    ax = axes[0, 1]
    methods = list(results.keys())
    mse_values = [results[m]['metrics']['mse'] for m in methods]
    mae_values = [results[m]['metrics']['mae'] for m in methods]
    rmse_values = [results[m]['metrics']['rmse'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x - width, mse_values, width, label='MSE', alpha=0.8)
    ax.bar(x, mae_values, width, label='MAE', alpha=0.8)
    ax.bar(x + width, rmse_values, width, label='RMSE', alpha=0.8)
    
    ax.set_xlabel('Transformation Method')
    ax.set_ylabel('Error Metric')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y')
    
    # Plot 3 & 4: Predictions for best and worst methods
    sorted_methods = sorted(results.items(), 
                           key=lambda x: x[1]['metrics']['rmse'])
    
    for idx, (title, (method_name, result)) in enumerate([
        ('Best Method', sorted_methods[0]),
        ('Worst Method', sorted_methods[-1])
    ]):
        ax = axes[1, idx]
        predictions = result['predictions'][:100]
        actuals = result['actuals'][:100]
        
        ax.plot(actuals, label='Actual', marker='o', markersize=3, alpha=0.7)
        ax.plot(predictions, label='Predicted', marker='x', markersize=3, alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.set_title(f'{title}: {method_name} (RMSE: {result["metrics"]["rmse"]:.4f})')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('transformation_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved visualization to 'transformation_comparison.png'")
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("TRANSFORMATION METHOD COMPARISON")
    print("="*70)
    
    # Generate sample data
    print("\nGenerating sample data...")
    n_samples = 1000
    n_features = 10
    time = np.arange(n_samples)
    
    X_raw = np.zeros((n_samples, n_features))
    for i in range(n_features):
        trend = 100 + 0.1 * time + np.random.randn(n_samples) * 5
        seasonality = 10 * np.sin(2 * np.pi * time / 50 + i)
        X_raw[:, i] = trend + seasonality + np.random.randn(n_samples) * 2
    
    y_raw = (0.3 * X_raw[:, 0] + 0.2 * X_raw[:, 1] + 0.15 * X_raw[:, 2] + 
             np.random.randn(n_samples) * 3).reshape(-1, 1)
    
    # Compare transformation methods
    results = compare_transformation_methods(X_raw, y_raw)
    
    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON - ALL TRANSFORMATION METHODS")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
    print("-"*70)
    
    for method_name, result in results.items():
        metrics = result['metrics']
        print(f"{method_name:<25} {metrics['mse']:<12.4f} "
              f"{metrics['mae']:<12.4f} {metrics['rmse']:<12.4f}")
    
    # Find best method
    best_method = min(results.items(), key=lambda x: x[1]['metrics']['rmse'])
    print(f"\n🏆 Best Method: {best_method[0].upper()} "
          f"(RMSE: {best_method[1]['metrics']['rmse']:.4f})")
    
    # Visualize
    visualize_transformation_comparison(results)
    
    # Save results
    with open('transformation_comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ All results saved to 'transformation_comparison_results.pkl'")
    print(f"{'='*70}")
    
    # Detailed explanation
    print("\n" + "="*70)
    print("TRANSFORMATION METHOD DETAILS")
    print("="*70)
    
    print("\n1. RATIO: value(t) / value(t-1)")
    print("   - Multiplicative changes")
    print("   - Good for: Price series, growth rates")
    print("   - Log transform: Makes it additive")
    
    print("\n2. FRACTIONAL_CHANGE: (value(t) - value(t-1)) / value(t-1)")
    print("   - Returns or percentage change (as decimal)")
    print("   - Good for: Financial returns, rate of change")
    print("   - Log1p transform: Handles small changes well")
    
    print("\n3. PERCENTAGE_CHANGE: 100 * (value(t) - value(t-1)) / value(t-1)")
    print("   - Percentage change")
    print("   - Good for: Human-readable changes")
    print("   - Log1p transform: Applied after scaling")
    
    print("\n" + "="*70)
