# Installation Guide

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | Email: ajsinha@gmail.com**

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- GPU (optional, for faster deep learning training)

### Operating Systems
- Linux (Ubuntu 20.04+, CentOS 7+, etc.)
- macOS (10.14+)
- Windows 10/11

## Installation Methods

### Method 1: Install from Source (Recommended)

1. **Extract the package:**
```bash
tar -xzf timeseries_package.tar.gz
cd timeseries_package
```

2. **Create a virtual environment (recommended):**
```bash
# Using venv
python -m venv timeseries_env
source timeseries_env/bin/activate  # On Windows: timeseries_env\Scripts\activate

# Or using conda
conda create -n timeseries_env python=3.9
conda activate timeseries_env
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package:**
```bash
pip install -e .
```

### Method 2: Direct Dependency Installation

If you don't want to install the package itself:

```bash
cd timeseries_package
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## GPU Support (Optional)

For CUDA-enabled GPU support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Optional Dependencies

### Facebook Prophet

Prophet requires additional compilation steps:

```bash
# On Linux/macOS
pip install prophet

# On Windows, you may need to install from conda:
conda install -c conda-forge prophet
```

### Development Tools

For development and testing:

```bash
pip install pytest pytest-cov black flake8 mypy
```

## Verification

Verify the installation:

```python
python -c "
import timeseries
from timeseries.model import TimeSeriesModel
from timeseries.normalization import DataNormalizer
from timeseries.deeplearning.models.lstm import LSTMModel
from timeseries.stat.models.statistical import ARIMAModel
print('Installation successful!')
"
```

## Running Examples

### Deep Learning Example

```bash
cd timeseries_package
python timeseries/deeplearning/examples/stock_prediction_example.py
```

Or using entry point (if installed):
```bash
timeseries-dl-example
```

### Statistical Models Example

```bash
cd timeseries_package
python timeseries/stat/examples/stock_prediction_example.py
```

Or using entry point (if installed):
```bash
timeseries-stat-example
```

## Troubleshooting

### Issue: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'timeseries'`

**Solution:**
```bash
# If not installed via pip:
export PYTHONPATH="${PYTHONPATH}:/path/to/timeseries_package"

# Or install the package:
pip install -e .
```

### Issue: PyTorch Installation

**Problem:** PyTorch not installing correctly

**Solution:**
```bash
# Uninstall existing torch
pip uninstall torch torchvision torchaudio

# Reinstall with specific version
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

### Issue: Statsmodels Errors

**Problem:** `ImportError: cannot import name 'ARIMA'`

**Solution:**
```bash
pip install --upgrade statsmodels
```

### Issue: Prophet Installation Fails

**Problem:** Prophet compilation errors on Windows

**Solution:**
```bash
# Use conda instead
conda install -c conda-forge prophet

# Or install pre-built wheels from
# https://github.com/facebook/prophet/releases
```

### Issue: Memory Errors

**Problem:** Out of memory when training models

**Solution:**
- Reduce batch size in configuration
- Reduce sequence length
- Use smaller model architecture
- Close other applications
- Use gradient checkpointing (for very large models)

### Issue: CUDA Errors

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Reduce batch size
config['batch_size'] = 16  # Instead of 32

# Or use CPU
config['device'] = 'cpu'
```

## Performance Tips

### For CPU-Only Systems
- Use smaller models (hidden_size=32-64)
- Reduce batch size
- Reduce number of epochs
- Consider using statistical models for faster training

### For GPU Systems
- Increase batch size (64, 128, 256)
- Use larger models if memory permits
- Enable GPU acceleration in configuration
- Monitor GPU usage: `nvidia-smi`

## Configuration

### Environment Variables

```bash
# Set device preference
export TIMESERIES_DEVICE=cuda  # or 'cpu'

# Set default data directory
export TIMESERIES_DATA_DIR=/path/to/data

# Enable verbose logging
export TIMESERIES_VERBOSE=1
```

### Config Files

You can create a config file `~/.timeseries/config.json`:

```json
{
  "default_device": "cuda",
  "default_batch_size": 32,
  "default_epochs": 100,
  "data_directory": "/path/to/data",
  "cache_directory": "/path/to/cache"
}
```

## Updating

To update to a newer version:

```bash
cd timeseries_package
git pull  # If using git
pip install --upgrade -e .
```

## Uninstallation

To uninstall:

```bash
pip uninstall timeseries-analysis
```

To remove all dependencies:

```bash
pip freeze | xargs pip uninstall -y
```

## Support

For installation support or issues:
- Email: ajsinha@gmail.com
- Check README.md for detailed documentation
- Review example scripts in `examples/` directories

## License

This software is proprietary. See LICENSE file for details.

Copyright © 2025-2030, All Rights Reserved  
Ashutosh Sinha | Email: ajsinha@gmail.com
