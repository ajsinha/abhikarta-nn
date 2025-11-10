"""
SIMPLIFIED PYTORCH TIME SERIES NEURAL NETWORK WITH RATIO SCALING
=================================================================

This is a minimal example showing the core concepts for quick implementation.
For a complete version, see timeseries_pytorch.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# STEP 1: LOAD YOUR DATA
# ============================================================================

# Replace this with your actual data
# df = pd.read_csv('your_data.csv')
# X_raw = df[['var1', 'var2', ..., 'var10']].values  # 10 input variables
# y_raw = df[['target']].values  # 1 target variable

# For demo purposes, generate sample data
n_samples = 1000
X_raw = np.random.randn(n_samples, 10) * 10 + 100  # 10 input variables
y_raw = np.random.randn(n_samples, 1) * 5 + 50     # 1 target variable

print(f"Original data shape: X={X_raw.shape}, y={y_raw.shape}")

# ============================================================================
# STEP 2: APPLY RATIO TRANSFORMATION (t / t-1)
# ============================================================================

def calculate_ratios(data):
    """Convert time series to ratios: value(t) / value(t-1)"""
    ratios = np.ones_like(data, dtype=np.float32)  # First row: ratio = 1.0
    
    for i in range(1, len(data)):
        # Calculate ratio, avoiding division by zero
        ratios[i] = np.where(data[i-1] != 0, data[i] / data[i-1], 1.0)
    
    # Apply log transformation for stability
    ratios = np.log(ratios + 1e-10)
    
    return ratios

# Calculate ratios for input and target
X_ratios = calculate_ratios(X_raw)
y_ratios = calculate_ratios(y_raw)

# Standardize the ratios
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X_ratios).astype(np.float32)
y_scaled = y_scaler.fit_transform(y_ratios).astype(np.float32)

print(f"After ratio transformation: X={X_scaled.shape}, y={y_scaled.shape}")

# ============================================================================
# STEP 3: CREATE PYTORCH DATASET
# ============================================================================

sequence_length = 20  # Use past 20 time steps to predict next value

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series"""
    
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_len]  # Past seq_len time steps
        y_target = self.y[idx + self.seq_len]    # Target at next time step
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_target)

# Split data: 70% train, 15% val, 15% test
train_size = int(0.7 * len(X_scaled))
val_size = int(0.15 * len(X_scaled))

X_train = X_scaled[:train_size]
y_train = y_scaled[:train_size]
X_val = X_scaled[train_size:train_size + val_size]
y_val = y_scaled[train_size:train_size + val_size]
X_test = X_scaled[train_size + val_size:]
y_test = y_scaled[train_size + val_size:]

# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"\nTrain: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")

# ============================================================================
# STEP 4: BUILD LSTM MODEL
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM neural network for time series prediction"""
    
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        
        # Output layer
        self.fc_out = nn.Linear(16, 1)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        
        # First LSTM
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        
        # Second LSTM
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        
        # Take output from last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Dense layers
        out = self.fc1(lstm_out)
        out = self.relu1(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.fc_out(out)
        
        return out

# Initialize model
model = LSTMModel(input_size=10, hidden_size1=64, hidden_size2=32, dropout=0.2)
model = model.to(device)

print("\nModel Architecture:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
print(f"\nTraining for {num_epochs} epochs...")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# ============================================================================
# STEP 6: EVALUATE AND PREDICT
# ============================================================================

print("\nEvaluating on test set...")
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(batch_y.cpu().numpy())

predictions = np.array(predictions)
actuals = np.array(actuals)

# Calculate metrics
mse = np.mean((predictions - actuals) ** 2)
mae = np.mean(np.abs(predictions - actuals))
rmse = np.sqrt(mse)

print(f"\nTest Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Show first 5 predictions
print("\nFirst 5 predictions vs actual:")
for i in range(5):
    print(f"Predicted: {predictions[i][0]:.4f}, Actual: {actuals[i][0]:.4f}")

# ============================================================================
# STEP 7: SAVE MODEL
# ============================================================================

torch.save({
    'model_state_dict': model.state_dict(),
    'X_scaler': X_scaler,
    'y_scaler': y_scaler,
    'sequence_length': sequence_length
}, 'simple_pytorch_model.pth')

print("\nModel saved to 'simple_pytorch_model.pth'")

"""
KEY POINTS:
-----------

1. PYTORCH vs KERAS:
   - More explicit control over training loop
   - Easier debugging (it's just Python!)
   - Better for research and custom models
   - Native GPU support with .to(device)

2. RATIO TRANSFORMATION:
   - Converts value(t) / value(t-1) to make data stationary
   - Log transformation stabilizes the ratios
   - StandardScaler normalizes for neural network

3. DATASET & DATALOADER:
   - Dataset class: Defines how to access data
   - DataLoader: Handles batching and shuffling
   - batch_first=True in LSTM for easier processing

4. MODEL TRAINING:
   - optimizer.zero_grad(): Clear gradients
   - loss.backward(): Compute gradients
   - optimizer.step(): Update weights
   - model.train() / model.eval(): Switch modes

5. GPU ACCELERATION:
   - Automatically uses CUDA if available
   - Move model and data to device: .to(device)
   - Much faster training on GPU

6. HYPERPARAMETERS TO TUNE:
   - sequence_length: How far back to look
   - hidden_size1, hidden_size2: LSTM capacity
   - dropout: Regularization (0.1-0.5)
   - learning_rate: Step size (0.0001-0.01)
   - batch_size: Memory vs speed tradeoff

7. TO USE WITH YOUR DATA:
   - Replace X_raw and y_raw with your actual data
   - Ensure X has 10 columns and y has 1 column
   - Data should be sorted by time
   - Remove any NaN or infinite values
"""
