# Use Cases: When Temporal Stability Matters

Temporal Optimizer excels in scenarios where model performance needs to remain stable over time. Here are key use cases with specific examples.

## 🏦 Financial Services

### Credit Scoring

**Problem**: Customer behavior and market conditions change over time, causing model performance to degrade.

**Solution**: StableAdam maintains consistent performance as data distribution shifts.

```python
from temporal_optimizer import StableAdam
import torch.nn as nn

# Credit scoring model
class CreditModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 128),  # 50 financial features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)     # Good/Bad credit
        )
    
    def forward(self, x):
        return self.layers(x)

# Use StableAdam for temporal stability
model = CreditModel()
optimizer = StableAdam(
    model.parameters(), 
    lr=0.001,
    temporal_stability=0.02  # Higher for financial stability
)
```

**Why it works**: StableAdam prevents the model from over-adapting to recent data, maintaining consistent decision boundaries.

### Fraud Detection

**Problem**: Fraudsters constantly evolve tactics, requiring models that adapt without losing performance on known patterns.

```python
# High temporal stability for fraud detection
optimizer = StableAdam(
    model.parameters(),
    lr=0.0005,
    temporal_stability=0.05,    # High stability
    energy_conservation=True    # Adaptive learning rates
)
```

## 🏥 Healthcare

### Medical Diagnosis

**Problem**: Medical models must maintain consistent accuracy over time as patient populations and medical practices evolve.

```python
class DiagnosisModel(nn.Module):
    def __init__(self, num_features=200, num_conditions=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_conditions)
        )
    
    def forward(self, x):
        return self.network(x)

# Conservative optimization for medical applications
optimizer = StableAdam(
    model.parameters(),
    lr=0.0001,                 # Low learning rate
    temporal_stability=0.03,   # High stability for safety
    momentum_decay=0.95        # Smooth parameter updates
)
```

### Drug Discovery

**Problem**: Models need to maintain performance as new compounds are discovered and tested.

```python
# Molecular property prediction
optimizer = StableSGD(  # SGD often better for chemistry data
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    temporal_stability=0.02
)
```

## 🛒 E-commerce & Recommendations

### Recommendation Systems

**Problem**: User preferences evolve over time, but recommendations should remain stable to avoid confusing users.

```python
class RecommendationModel(nn.Module):
    def __init__(self, num_users=10000, num_items=50000, embedding_dim=128):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.output = nn.Linear(embedding_dim * 2, 1)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        combined = torch.cat([user_emb, item_emb], dim=1)
        return self.output(combined)

# Balanced stability for recommendations
optimizer = StableAdam(
    model.parameters(),
    lr=0.001,
    temporal_stability=0.01,   # Moderate stability
    energy_conservation=True
)
```

### Dynamic Pricing

**Problem**: Pricing models must adapt to market conditions while maintaining consistency.

```python
# Price prediction with temporal constraints
optimizer = StableAdam(
    model.parameters(),
    lr=0.002,
    temporal_stability=0.025,  # Prevent erratic price changes
    momentum_decay=0.9
)
```

## 📈 Time Series Forecasting

### Financial Forecasting

**Problem**: Financial time series models need to adapt to market regime changes while maintaining long-term consistency.

```python
class TimeSeriesModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.output(lstm_out[:, -1, :])

# StableSGD often works well for sequential data
optimizer = StableSGD(
    model.parameters(),
    lr=0.005,
    momentum=0.95,
    temporal_stability=0.015   # Smooth predictions over time
)
```

### Energy Load Forecasting

**Problem**: Power grid forecasting must be stable to ensure reliable energy distribution.

```python
# Conservative approach for critical infrastructure
optimizer = StableAdam(
    model.parameters(),
    lr=0.0005,
    temporal_stability=0.04,   # High stability for grid reliability
    energy_conservation=True
)
```

## 🚗 Autonomous Systems

### Self-Driving Cars

**Problem**: Autonomous vehicle models must maintain consistent behavior as road conditions and traffic patterns change.

```python
class VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, 256)
        self.decision_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Steering decisions
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.decision_head(features)

# High stability for safety-critical applications
optimizer = StableAdam(
    model.parameters(),
    lr=0.0001,
    temporal_stability=0.05,   # Maximum stability
    momentum_decay=0.98        # Very smooth updates
)
```

## 🔬 Research & Production ML

### A/B Testing Models

**Problem**: Models used in A/B tests need consistent behavior to ensure valid experimental results.

```python
# Consistent model behavior for experiments
optimizer = StableAdam(
    model.parameters(),
    lr=0.001,
    temporal_stability=0.03,   # High consistency
    energy_conservation=False  # Deterministic behavior
)
```

### Continual Learning

**Problem**: Models that learn continuously from streams of data need to avoid catastrophic forgetting.

```python
# Continual learning setup
optimizer = StableSGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    temporal_stability=0.02,   # Prevent forgetting
    momentum_decay=0.95
)

# Training loop with stability regularization
for new_task_data in continuous_data_stream:
    # Store previous parameters for stability
    prev_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Train on new data
    optimizer.zero_grad()
    loss = criterion(model(new_task_data), labels)
    loss.backward()
    optimizer.step()
    
    # Temporal stability automatically applied by optimizer
```

## Configuration Guidelines by Use Case

| Use Case | Optimizer | Learning Rate | Temporal Stability | Energy Conservation |
|----------|-----------|---------------|-------------------|-------------------|
| Credit Scoring | StableAdam | 0.001 | 0.02 | True |
| Medical Diagnosis | StableAdam | 0.0001 | 0.03 | True |
| Recommendations | StableAdam | 0.001 | 0.01 | True |
| Time Series | StableSGD | 0.005 | 0.015 | True |
| Autonomous Vehicles | StableAdam | 0.0001 | 0.05 | True |
| A/B Testing | StableAdam | 0.001 | 0.03 | False |
| High-Frequency Trading | StableSGD | 0.01 | 0.01 | True |

## Warning Signs You Need Temporal Stability

Watch for these indicators that temporal optimizers could help:

### Training Instability
- Loss oscillates significantly during training
- Model performance varies dramatically between epochs
- Gradient norms are highly variable

### Production Degradation
- Model accuracy decreases over time in production
- Performance varies significantly across time periods
- Need frequent retraining to maintain performance

### Regulatory Requirements
- Need consistent decision-making for compliance
- Model outputs must be explainable and stable
- Auditable training processes required

## Next Steps

- Try the [Quick Start Guide](quickstart.md) with your specific use case
- Check the [Benchmarks](benchmarks.md) for performance comparisons
- Review the [API Reference](api_reference.md) for detailed parameter tuning
- Run the [examples](../examples/) to see these use cases in action