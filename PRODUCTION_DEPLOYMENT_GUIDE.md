# Production Deployment Guide

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | Email: ajsinha@gmail.com**

---

## Overview

This guide covers deploying time series models to production, including API design, monitoring, scaling, and maintenance strategies.

## Deployment Checklist

- [ ] Model selection and validation complete
- [ ] Hyperparameters tuned on validation set
- [ ] Performance tested on hold-out test set
- [ ] Code reviewed and tested
- [ ] Dependencies documented and locked
- [ ] API designed and documented
- [ ] Monitoring and alerting configured
- [ ] Rollback strategy defined
- [ ] Retraining schedule established
- [ ] Documentation complete

---

## Architecture Options

### 1. REST API (Recommended)

```python
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from timeseries.normalization import DataNormalizer

app = Flask(__name__)

# Load model and normalizer at startup
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('normalizer.pkl', 'rb') as f:
    normalizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        df = pd.DataFrame(data['features'])
        
        # Normalize
        df_norm = normalizer.transform(df)
        
        # Predict
        predictions = model.predict(df_norm)
        
        # Return results
        return jsonify({
            'success': True,
            'predictions': predictions.tolist(),
            'model_version': '1.0.0'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. Batch Prediction Service

```python
import schedule
import time
from datetime import datetime

def batch_predict():
    """Run batch predictions on schedule."""
    print(f"Running batch prediction at {datetime.now()}")
    
    # Load new data
    data = load_new_data()
    
    # Predict
    predictions = model.predict(data)
    
    # Save results
    save_predictions(predictions)
    
    # Log metrics
    log_metrics(predictions)

# Schedule batch job
schedule.every().day.at("00:00").do(batch_predict)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 3. Streaming Predictions

```python
from kafka import KafkaConsumer, KafkaProducer
import json

consumer = KafkaConsumer(
    'input_data',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for message in consumer:
    # Get data
    data = message.value
    
    # Predict
    prediction = model.predict(pd.DataFrame([data]))
    
    # Send result
    producer.send('predictions', {
        'input': data,
        'prediction': prediction.tolist(),
        'timestamp': datetime.now().isoformat()
    })
```

---

## Model Serving Best Practices

### 1. Model Versioning

```python
import os
from datetime import datetime

class ModelRegistry:
    """Manage model versions."""
    
    def __init__(self, base_path='models'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_model(self, model, version=None, metadata=None):
        """Save model with version."""
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        version_path = os.path.join(self.base_path, version)
        os.makedirs(version_path, exist_ok=True)
        
        # Save model
        model.save(os.path.join(version_path, 'model.pkl'))
        
        # Save metadata
        if metadata:
            with open(os.path.join(version_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return version
    
    def load_model(self, version='latest'):
        """Load specific model version."""
        if version == 'latest':
            versions = sorted(os.listdir(self.base_path))
            version = versions[-1]
        
        model_path = os.path.join(self.base_path, version, 'model.pkl')
        # Load and return model
        return model_path

# Usage
registry = ModelRegistry()

# Save new version
version = registry.save_model(
    model,
    metadata={
        'rmse': 2.34,
        'training_date': datetime.now().isoformat(),
        'features': feature_names
    }
)

print(f"Saved model version: {version}")
```

### 2. A/B Testing

```python
import random

class ABTestingRouter:
    """Route requests to different model versions."""
    
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
    
    def predict(self, X):
        """Route to model A or B."""
        if random.random() < self.split_ratio:
            model_used = 'A'
            prediction = self.model_a.predict(X)
        else:
            model_used = 'B'
            prediction = self.model_b.predict(X)
        
        return prediction, model_used

# Usage
router = ABTestingRouter(old_model, new_model, split_ratio=0.1)
prediction, model_used = router.predict(X)
```

### 3. Caching

```python
from functools import lru_cache
import hashlib

class PredictionCache:
    """Cache predictions for repeated inputs."""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def _hash_input(self, X):
        """Create hash of input."""
        return hashlib.md5(X.to_json().encode()).hexdigest()
    
    def get(self, X):
        """Get cached prediction."""
        key = self._hash_input(X)
        return self.cache.get(key)
    
    def set(self, X, prediction):
        """Cache prediction."""
        key = self._hash_input(X)
        if len(self.cache) >= self.max_size:
            # Remove oldest
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = prediction

# Usage
cache = PredictionCache()

def predict_with_cache(X):
    # Check cache
    cached = cache.get(X)
    if cached is not None:
        return cached
    
    # Predict
    prediction = model.predict(X)
    
    # Cache result
    cache.set(X, prediction)
    
    return prediction
```

---

## Monitoring

### 1. Performance Metrics

```python
import logging
from datetime import datetime

class ModelMonitor:
    """Monitor model performance in production."""
    
    def __init__(self, log_file='model_monitor.log'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.predictions = []
        self.actuals = []
    
    def log_prediction(self, prediction, features, request_id):
        """Log prediction."""
        logging.info(f"PREDICTION,{request_id},{prediction},{features}")
    
    def log_actual(self, actual, request_id):
        """Log actual value when available."""
        logging.info(f"ACTUAL,{request_id},{actual}")
    
    def calculate_metrics(self):
        """Calculate current performance metrics."""
        if len(self.actuals) == 0:
            return None
        
        rmse = np.sqrt(np.mean((np.array(self.actuals) - 
                                np.array(self.predictions[:len(self.actuals)])) ** 2))
        
        logging.info(f"METRICS,rmse={rmse}")
        return {'rmse': rmse}

monitor = ModelMonitor()
```

### 2. Data Drift Detection

```python
import numpy as np
from scipy import stats

class DriftDetector:
    """Detect distribution shifts in input data."""
    
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold
    
    def detect_drift(self, new_data):
        """Detect if new data has drifted."""
        drifted_features = []
        
        for col in self.reference_data.columns:
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[col],
                new_data[col]
            )
            
            if p_value < self.threshold:
                drifted_features.append({
                    'feature': col,
                    'p_value': p_value,
                    'statistic': statistic
                })
        
        return {
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features
        }

# Usage
drift_detector = DriftDetector(X_train)

# Check new data
result = drift_detector.detect_drift(X_new)
if result['drift_detected']:
    print("WARNING: Data drift detected!")
    print(f"Drifted features: {result['drifted_features']}")
```

### 3. Alert System

```python
import smtplib
from email.mime.text import MIMEText

class AlertSystem:
    """Send alerts when issues detected."""
    
    def __init__(self, smtp_server, from_email, to_emails):
        self.smtp_server = smtp_server
        self.from_email = from_email
        self.to_emails = to_emails
    
    def send_alert(self, subject, message):
        """Send email alert."""
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        
        with smtplib.SMTP(self.smtp_server) as server:
            server.send_message(msg)
    
    def check_and_alert(self, metrics, thresholds):
        """Check metrics and alert if threshold exceeded."""
        for metric, value in metrics.items():
            if metric in thresholds:
                if value > thresholds[metric]:
                    self.send_alert(
                        f"Alert: {metric} threshold exceeded",
                        f"{metric} = {value} (threshold: {thresholds[metric]})"
                    )

# Usage
alert_system = AlertSystem(
    smtp_server='smtp.gmail.com:587',
    from_email='alerts@company.com',
    to_emails=['team@company.com']
)

# Check metrics
alert_system.check_and_alert(
    metrics={'rmse': 5.2},
    thresholds={'rmse': 3.0}
)
```

---

## Retraining Strategies

### 1. Scheduled Retraining

```python
def retrain_model():
    """Retrain model with latest data."""
    print("Starting model retraining...")
    
    # Load latest data
    data = load_latest_data()
    
    # Split
    X_train, y_train = prepare_data(data)
    
    # Train new model
    new_model = LSTMModel(config=best_config)
    new_model.fit(X_train, y_train)
    
    # Evaluate
    metrics = new_model.evaluate(X_test, y_test)
    
    # Save if better
    if metrics['rmse'] < current_best_rmse:
        new_model.save('models/latest_model.pkl')
        print(f"New model saved: RMSE={metrics['rmse']}")
    else:
        print("New model not better, keeping current")

# Schedule retraining
schedule.every().week.do(retrain_model)
```

### 2. Performance-Based Retraining

```python
class PerformanceMonitor:
    """Monitor performance and trigger retraining."""
    
    def __init__(self, threshold_rmse=3.0, window_size=1000):
        self.threshold_rmse = threshold_rmse
        self.window_size = window_size
        self.errors = []
    
    def add_error(self, error):
        """Add new prediction error."""
        self.errors.append(error)
        if len(self.errors) > self.window_size:
            self.errors.pop(0)
    
    def should_retrain(self):
        """Check if retraining needed."""
        if len(self.errors) < self.window_size:
            return False
        
        current_rmse = np.sqrt(np.mean(np.array(self.errors) ** 2))
        return current_rmse > self.threshold_rmse

monitor = PerformanceMonitor(threshold_rmse=3.0)

# In prediction loop
error = actual - prediction
monitor.add_error(error)

if monitor.should_retrain():
    print("Performance degraded, triggering retraining")
    retrain_model()
```

---

## Scaling

### 1. Horizontal Scaling

```docker
# docker-compose.yml
version: '3'
services:
  model_api:
    image: timeseries_model:latest
    replicas: 5
    ports:
      - "5000-5004:5000"
  
  load_balancer:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### 2. GPU Optimization

```python
# Multi-GPU training
import torch

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

model = model.to('cuda')

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for X, y in dataloader:
    with autocast():
        output = model(X)
        loss = criterion(output, y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Best Practices Summary

1. **Version Everything**
   - Models
   - Data
   - Code
   - Dependencies

2. **Monitor Continuously**
   - Prediction accuracy
   - Latency
   - Data drift
   - System resources

3. **Test Thoroughly**
   - Unit tests
   - Integration tests
   - Performance tests
   - A/B tests

4. **Document Extensively**
   - API documentation
   - Model card
   - Deployment procedures
   - Rollback procedures

5. **Plan for Failure**
   - Fallback models
   - Graceful degradation
   - Circuit breakers
   - Automated rollback

---

## Contact

For questions or support:
- Email: ajsinha@gmail.com

**Copyright © 2025-2030, All Rights Reserved**  
**Ashutosh Sinha | Email: ajsinha@gmail.com**
