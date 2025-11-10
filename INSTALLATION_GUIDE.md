# Installation and Usage Guide

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | ajsinha@gmail.com**

---

## 📦 Package Download

**File:** timeseries_neural_network_package.tar.gz  
**Size:** ~361 KB  
**Contents:** Complete time series neural network implementation

---

## 🔓 Extraction

### Linux/Mac:
```bash
tar -xzf timeseries_neural_network_package.tar.gz
cd timeseries_neural_network_package
```

### Windows:
1. Use 7-Zip or WinRAR to extract
2. Or use Windows Subsystem for Linux (WSL)

---

## ⚖️ IMPORTANT: Legal Notice

**BEFORE USING THIS SOFTWARE, YOU MUST:**

1. **Read COPYRIGHT.txt in full**
2. **Understand the licensing terms**
3. **Contact ajsinha@gmail.com for commercial use**

**This software is proprietary. Unauthorized use is prohibited.**

---

## 📋 Prerequisites

### System Requirements:
- Python 3.7 or higher
- 8GB RAM (16GB recommended)
- GPU optional (but recommended for faster training)

### For PyTorch:
```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision
```

### For TensorFlow/Keras:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## 🚀 Installation Steps

### Step 1: Extract Package
```bash
tar -xzf timeseries_neural_network_package.tar.gz
cd extracted_folder
```

### Step 2: Install Dependencies
```bash
# For PyTorch
pip install -r requirements_pytorch.txt

# OR for TensorFlow
pip install -r requirements.txt
```

### Step 3: Verify Installation
```python
import torch
import numpy as np
import pandas as pd

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

### Step 4: Run First Example
```bash
python simple_pytorch_example.py
```

---

## 📚 Learning Path

### Day 1: Setup & Basics
1. Read COPYRIGHT.txt
2. Read README.txt
3. Install dependencies
4. Run simple_pytorch_example.py

### Day 2: Understanding
1. Read DOCUMENTATION_INDEX.md
2. Read README_MASTER.md
3. Review MODEL_GUIDE.md

### Day 3: Experimentation
1. Run timeseries_all_models.py
2. Try simple_config_example.py
3. Test with your own data

### Week 2: Advanced
1. Read TECHNICAL_GUIDE.md
2. Study ensemble_methods.py
3. Customize for your needs

---

## 🎯 Quick Examples

### Example 1: Train LSTM Model
```python
from timeseries_all_models import create_model, train_model

# Create model
model = create_model('lstm', input_size=10)

# Train
history, best_state = train_model(
    model, train_loader, val_loader,
    criterion, optimizer, num_epochs=50, device=device
)
```

### Example 2: Compare Transformations
```python
from timeseries_enhanced_config import compare_transformation_methods

results = compare_transformation_methods(X_raw, y_raw)
# Automatically tests all 3 transformation methods
```

### Example 3: Use Ensemble
```python
from ensemble_methods import AveragingEnsemble

ensemble = AveragingEnsemble([model1, model2, model3])
predictions = ensemble.predict(test_loader, device)
```

---

## 📖 Documentation Structure

### Essential (Read First):
1. **COPYRIGHT.txt** - Legal notice
2. **README.txt** - Package overview
3. **DOCUMENTATION_INDEX.md** - Navigation guide

### Getting Started:
1. **README_MASTER.md** - Complete overview
2. **simple_pytorch_example.py** - Working example
3. **USAGE_GUIDE.md** - Practical guide

### Deep Dive:
1. **MODEL_GUIDE.md** - All 8 models explained
2. **TRANSFORMATION_GUIDE.md** - 3 transformation methods
3. **TECHNICAL_GUIDE.md** - Mathematical theory

### Advanced:
1. **ensemble_methods.py** - 5 ensemble techniques
2. **timeseries_all_models.py** - Complete implementation
3. **ENHANCEMENT_README.md** - Configuration guide

---

## 🔧 Common Issues & Solutions

### Issue 1: Import Errors
```bash
# Solution: Install dependencies
pip install -r requirements_pytorch.txt
```

### Issue 2: CUDA Not Available
```python
# Solution: Use CPU
device = torch.device('cpu')
```

### Issue 3: Out of Memory
```python
# Solution: Reduce batch size
batch_size = 16  # instead of 32
```

### Issue 4: Slow Training
```python
# Solution: Use GRU instead of LSTM
model = create_model('gru', input_size=10)
```

---

## 📞 Support

**For Technical Issues:**
- Check USAGE_GUIDE.md troubleshooting section
- Read relevant documentation

**For Licensing:**
- Contact: ajsinha@gmail.com
- Include: Your use case and organization

**For Commercial Use:**
- Email: ajsinha@gmail.com
- Subject: "Commercial License Inquiry"

---

## ⚠️ Important Reminders

1. **Read COPYRIGHT.txt before any use**
2. **License required for commercial use**
3. **Do not remove copyright notices**
4. **Contact for permissions**

---

## 🎓 Best Practices

### 1. Version Control
```bash
git init
git add .
git commit -m "Initial commit"
```

### 2. Environment Management
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Save Your Work
```python
torch.save({
    'model': model.state_dict(),
    'config': config,
    'preprocessor': preprocessor
}, 'my_model.pth')
```

### 4. Document Changes
- Keep notes on modifications
- Record hyperparameters
- Save experiment results

---

## 📊 Performance Tips

### For Faster Training:
1. Use GPU if available
2. Use GRU instead of LSTM
3. Increase batch size
4. Reduce sequence length

### For Better Accuracy:
1. Try different models
2. Test transformation methods
3. Use ensemble methods
4. Tune hyperparameters

---

## 🎯 Next Steps

After installation:

1. ✅ Read COPYRIGHT.txt
2. ✅ Run simple example
3. ✅ Read DOCUMENTATION_INDEX.md
4. ✅ Try with your data
5. ✅ Explore advanced features
6. ✅ Contact for licensing if needed

---

## 📧 Contact Information

**Copyright Holder:** Ashutosh Sinha  
**Email:** ajsinha@gmail.com

**For:**
- Licensing inquiries
- Commercial use permissions
- Technical partnerships
- Patent information

**Please include:**
- Your name and organization
- Intended use
- Timeframe
- Any specific requirements

---

## 🏆 Package Features Summary

- ✅ 8 neural network architectures
- ✅ 3 transformation methods
- ✅ 5 ensemble techniques
- ✅ 110+ pages documentation
- ✅ Complete working examples
- ✅ Production-ready code
- ✅ Both PyTorch and Keras
- ✅ Comprehensive guides

---

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | ajsinha@gmail.com**

**Always refer to COPYRIGHT.txt for complete legal information.**

---

**🎉 Happy modeling with proper licensing! 🚀**
