# Quick Start & Installation Guide

---

## Copyright Notice

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha**  
**Email: ajsinha@gmail.com**

This software is proprietary. See COPYRIGHT_NOTICE.txt for full legal terms.  
**Patent Pending** - Certain implementations subject to patent applications.

---

## 📦 Package Information

**Package:** Time Series Neural Networks - Complete  
**Version:** 1.0.0  
**Size:** 75KB compressed, ~550KB uncompressed  
**Files:** 23 files (10 Python scripts, 13 documentation files)

---

## 🚀 Installation

### Step 1: Extract Archive

```bash
tar -xzf timeseries-neural-networks-complete.tar.gz
cd timeseries-neural-networks-complete/
```

### Step 2: Install Dependencies

**For PyTorch (Recommended):**
```bash
pip install -r requirements_pytorch.txt
```

**For TensorFlow/Keras:**
```bash
pip install -r requirements.txt
```

**For Both:**
```bash
pip install -r requirements_pytorch.txt
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

---

## ⚡ 5-Minute Quick Start

### 1. Read the Documentation Index

```bash
# Open in your editor or browser
cat DOCUMENTATION_INDEX.md
```

### 2. Run Your First Example

```bash
# Simple LSTM example
python simple_pytorch_example.py
```

**Expected output:**
```
Generating sample data...
Preprocessing data...
Training model...
Epoch [10/50], Train Loss: 0.3421, Val Loss: 0.3889
...
Test RMSE: 0.2643
✅ Example completed successfully!
```

### 3. Try All 8 Models

```bash
# Compare all model architectures
python timeseries_all_models.py
```

### 4. Test Different Transformations

```bash
# Compare ratio vs fractional change vs percentage
python timeseries_enhanced_config.py
```

### 5. Experiment with Ensembles

```bash
# Compare ensemble methods
python ensemble_methods.py
```

---

## 📚 Documentation Quick Reference

| Read This | When You Want To |
|-----------|------------------|
| **DOCUMENTATION_INDEX.md** | Navigate all documentation |
| **README_PACKAGE.md** | Understand complete package |
| **README_MASTER.md** | Get package overview |
| **MODEL_GUIDE.md** | Learn about 8 model types |
| **USAGE_GUIDE.md** | See practical code examples |
| **TRANSFORMATION_GUIDE.md** | Understand ratio/fractional change |
| **ENSEMBLE_GUIDE.md** | Learn ensemble methods |
| **TECHNICAL_GUIDE.md** | Deep dive into mathematics |

---

## 🎯 Your First Project

### Use Your Own Data

Create a new file `my_project.py`:

```python
import pandas as pd
import numpy as np
from timeseries_prtediction.timeseries_enhanced_config import *
from timeseries_prtediction.ensemble_methods import *
import torch
from torch.utils.data import DataLoader

# 1. Load your data
df = pd.read_csv('your_data.csv')
X_raw = df[['feature1', 'feature2', ..., 'feature10']].values
y_raw = df[['target']].values

# 2. Configure transformation
config = TransformConfig(
    method=TransformMethod.FRACTIONAL_CHANGE,  # or RATIO or PERCENTAGE_CHANGE
    log_transform=True
)

# 3. Preprocess
X_preprocessor = EnhancedTimeSeriesPreprocessor(config)
y_preprocessor = EnhancedTimeSeriesPreprocessor(config)

X_scaled = X_preprocessor.fit_transform(X_raw)
y_scaled = y_preprocessor.fit_transform(y_raw)

# 4. Create datasets
from torch.utils.data import DataLoader

sequence_length = 20
train_size = int(0.7 * len(X_scaled))

X_train = X_scaled[:train_size]
y_train = y_scaled[:train_size]
X_test = X_scaled[train_size:]
y_test = y_scaled[train_size:]

train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. Train model (choose one)

# Option A: Single Model
from timeseries_prtediction.timeseries_all_models import LSTMModel, train_model, evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=10, hidden_sizes=[64, 32])
model = model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

history, best_state = train_model(
    model, train_loader, test_loader, criterion, optimizer,
    num_epochs=50, device=device
)

# Option B: Ensemble
ensemble = create_ensemble('stacking', n_models=5)
ensemble.train(train_loader, test_loader, criterion, torch.optim.Adam,
               num_epochs=50, device=device)

# 6. Evaluate
predictions, actuals, metrics = evaluate_model(model, test_loader, device)
print(f"Test RMSE: {metrics['rmse']:.4f}")

# 7. Save
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'preprocessor': X_preprocessor
}, 'my_model.pth')
```

Run it:
```bash
python my_project.py
```

---

## 🔧 Common Issues & Solutions

### Issue 1: Import Errors

```bash
# Error: No module named 'torch'
pip install torch

# Error: No module named 'tensorflow'
pip install tensorflow
```

### Issue 2: CUDA Out of Memory

```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Or use CPU
device = torch.device('cpu')
```

### Issue 3: Slow Training

```python
# Use GRU instead of LSTM (30-40% faster)
model = GRUModel(input_size=10)

# Reduce epochs
num_epochs = 30  # Instead of 50
```

---

## 📁 Package Structure

```
timeseries-neural-networks-complete/
│
├── COPYRIGHT_NOTICE.txt           ← Legal notice
├── README_PACKAGE.md              ← Package overview
├── QUICKSTART.md                  ← This file
├── PACKAGE_MANIFEST.txt           ← File listing
│
├── Documentation/
│   ├── DOCUMENTATION_INDEX.md     ← Start here!
│   ├── README_MASTER.md
│   ├── MODEL_GUIDE.md
│   ├── USAGE_GUIDE.md
│   ├── TECHNICAL_GUIDE.md
│   ├── TRANSFORMATION_GUIDE.md
│   ├── ENSEMBLE_GUIDE.md          ← Patent pending
│   ├── COMPARISON_GUIDE.md
│   └── ENHANCEMENT_README.md
│
├── Code - Main/
│   ├── timeseries_all_models.py       ← 8 models
│   ├── timeseries_enhanced_config.py  ← Configurable
│   └── ensemble_methods.py            ← Patent pending
│
├── Code - PyTorch/
│   ├── timeseries_pytorch.py
│   ├── simple_pytorch_example.py
│   ├── simple_config_example.py
│   ├── README_PYTORCH.md
│   └── requirements_pytorch.txt
│
└── Code - TensorFlow/
    ├── timeseries_deep_learning.py
    ├── simple_timeseries_example.py
    ├── README.md
    └── requirements.txt
```

---

## ✅ Checklist

### Installation
- [ ] Extract archive
- [ ] Install dependencies
- [ ] Verify installation

### Learning
- [ ] Read DOCUMENTATION_INDEX.md
- [ ] Run simple_pytorch_example.py
- [ ] Read MODEL_GUIDE.md
- [ ] Read USAGE_GUIDE.md

### First Project
- [ ] Prepare your data
- [ ] Choose transformation method
- [ ] Select model architecture
- [ ] Train and evaluate
- [ ] Save model

### Advanced
- [ ] Try all 8 models
- [ ] Compare transformations
- [ ] Experiment with ensembles
- [ ] Deploy to production

---

## 🎓 Learning Paths

### Path 1: Quick Results (2 hours)
1. Run simple_pytorch_example.py (15 min)
2. Read USAGE_GUIDE.md - Basic Workflow (30 min)
3. Modify with your data (60 min)
4. Done! ✅

### Path 2: Comprehensive (1 day)
1. Read DOCUMENTATION_INDEX.md (15 min)
2. Read README_MASTER.md (20 min)
3. Read MODEL_GUIDE.md (90 min)
4. Read USAGE_GUIDE.md (90 min)
5. Experiment with all examples (3 hours)

### Path 3: Expert (1 week)
1. Read all documentation (1 day)
2. Study TECHNICAL_GUIDE.md (4 hours)
3. Understand ensemble_methods.py (4 hours)
4. Implement custom models (2 days)
5. Production deployment (2 days)

---

## 🏆 What You Get

### Neural Networks
- **8 architectures**: LSTM, GRU, BiLSTM, CNN-LSTM, TCN, Transformer, Attention-LSTM, MLP
- **Complete implementations**: Ready to use
- **Documented**: Every model explained

### Transformations
- **3 methods**: Ratio, Fractional Change, Percentage Change
- **Configurable**: Easy to switch
- **Automatic comparison**: Find best for your data

### Ensembles (Patent Pending)
- **5 strategies**: Average, Weighted, Stacking, Bagging, Boosting
- **Automatic optimization**: Performance-based weighting
- **Production-ready**: Save and load functionality

### Documentation
- **144 pages**: Comprehensive guides
- **Step-by-step**: Easy to follow
- **Mathematical**: Theory explained
- **Practical**: Code examples

---

## 📞 Support & Contact

### Documentation
- All questions answered in the guides
- See troubleshooting sections
- Check USAGE_GUIDE.md for examples

### Licensing & Commercial Use
**Email: ajsinha@gmail.com**

For:
- Commercial licensing
- Custom implementations
- Enterprise support
- Consulting services

### Legal
- Copyright © 2025-2030 Ashutosh Sinha
- All rights reserved
- Patent pending
- See COPYRIGHT_NOTICE.txt

---

## 🎉 You're Ready!

**Next Step:** Open DOCUMENTATION_INDEX.md and start exploring!

```bash
# Read the docs
cat DOCUMENTATION_INDEX.md

# Or run your first example
python simple_pytorch_example.py
```

---

## 📊 Expected Results

After running the examples, you should see:

### Simple Example
```
Test RMSE: 0.2643
Test MAE: 0.1987
✅ Example completed successfully!
```

### All Models Comparison
```
Model          RMSE     MAE
-------------------------------
LSTM           0.2557   0.1987
GRU            0.2492   0.1923  ← Fastest
Transformer    0.2427   0.1856  ← Best
...
🏆 Best Model: Transformer
```

### Ensemble Comparison
```
Ensemble Type   RMSE     Improvement
--------------------------------------
Average         0.218    11.0%
Weighted        0.215    12.2%
Stacking        0.208    15.1%  ← Best
...
🏆 Best Ensemble: Stacking
```

---

## ⚡ Quick Commands Reference

```bash
# Extract
tar -xzf timeseries-neural-networks-complete.tar.gz

# Install
pip install -r requirements_pytorch.txt

# Run examples
python simple_pytorch_example.py
python timeseries_all_models.py
python timeseries_enhanced_config.py
python ensemble_methods.py

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

**Copyright © 2025-2030 Ashutosh Sinha**  
**All Rights Reserved | Patent Pending**  
**Contact: ajsinha@gmail.com**
