# Quick Start Guide

Get up and running with Temporal Optimizer in minutes.

## Installation

```bash
pip install temporal-optimizer
```

## Basic Usage

### 1. Replace Your Optimizer

The only change needed is importing and using our optimizers:

```python
# Before
from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=0.001)

# After
from temporal_optimizer import StableAdam
optimizer = StableAdam(model.parameters(), lr=0.001)
```

### 2. Complete Training Example

Here's a complete example showing the integration:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from temporal_optimizer import StableAdam

# Define your model (any PyTorch model works)
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Create model and optimizer
model = SimpleNet()
optimizer = StableAdam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (exactly the same as before!)
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

### 3. Choose the Right Optimizer

| Use Case | Recommended Optimizer | Why |
|----------|----------------------|-----|
| Computer Vision | `StableAdam` | Good default for most tasks |
| NLP/Transformers | `StableAdam` | Handles large models well |
| Time Series | `StableSGD` | Better for sequential data |
| Small Models | `StableSGD` | More memory efficient |
| Unstable Training | `StableAdam` with high `temporal_stability` | Maximum stability |

## Configuration Options

### StableAdam Parameters

```python
from temporal_optimizer import StableAdam

optimizer = StableAdam(
    model.parameters(),
    lr=0.001,                    # Learning rate (same as Adam)
    betas=(0.9, 0.999),         # Beta parameters (same as Adam)
    eps=1e-8,                   # Numerical stability (same as Adam)
    weight_decay=0,             # L2 regularization (same as Adam)
    
    # New parameters for temporal stability
    temporal_stability=0.01,     # Stability regularization weight
    momentum_decay=0.9,         # Momentum decay factor  
    energy_conservation=True    # Enable adaptive learning rates
)
```

### StableSGD Parameters

```python
from temporal_optimizer import StableSGD

optimizer = StableSGD(
    model.parameters(),
    lr=0.01,                    # Learning rate (same as SGD)
    momentum=0.9,              # Momentum (same as SGD)
    weight_decay=0,            # L2 regularization (same as SGD)
    dampening=0,               # Dampening (same as SGD)
    nesterov=False,            # Nesterov momentum (same as SGD)
    
    # New parameters for temporal stability
    temporal_stability=0.01,    # Stability regularization weight
    momentum_decay=0.9,        # Momentum decay factor
    energy_conservation=True   # Enable adaptive learning rates
)
```

## Parameter Tuning Guide

### Default Settings (Recommended)
Most users should start with defaults:
```python
optimizer = StableAdam(model.parameters(), lr=0.001)
```

### High Stability (For Unstable Training)
If you're experiencing training oscillations:
```python
optimizer = StableAdam(
    model.parameters(), 
    lr=0.001,
    temporal_stability=0.05,    # Higher for more stability
    momentum_decay=0.95        # Higher for smoother updates
)
```

### Performance Focus (Lower Overhead)
For maximum speed with minimal stability benefits:
```python
optimizer = StableAdam(
    model.parameters(), 
    lr=0.001,
    temporal_stability=0.001,   # Minimal stability regularization
    energy_conservation=False   # Disable adaptive learning rates
)
```

## Integration with Popular Frameworks

### PyTorch Lightning

```python
import pytorch_lightning as pl
from temporal_optimizer import StableAdam

class MyLightningModule(pl.LightningModule):
    def configure_optimizers(self):
        return StableAdam(self.parameters(), lr=0.001)
```

### Hugging Face Transformers

```python
from transformers import Trainer, TrainingArguments
from temporal_optimizer import StableAdam
import torch

# Custom trainer with StableAdam
class StableAdamTrainer(Trainer):
    def create_optimizer(self):
        self.optimizer = StableAdam(
            self.model.parameters(),
            lr=self.args.learning_rate
        )
```

### FastAI

```python
from fastai.vision.all import *
from temporal_optimizer import StableAdam

# Custom optimizer function
def stable_adam(params, lr=0.001, **kwargs):
    return StableAdam(params, lr=lr, **kwargs)

# Use with FastAI
learn = vision_learner(dls, resnet34, opt_func=stable_adam)
```

## Troubleshooting

### Common Issues

**Q: My loss is NaN after switching to temporal optimizers**
A: Try reducing the learning rate by 10x. Temporal optimizers can be more aggressive initially.

**Q: Training is slower than before**
A: Set `energy_conservation=False` to disable adaptive learning rates, or reduce `temporal_stability` to 0.001.

**Q: Results are too similar to standard optimizers**
A: Increase `temporal_stability` to 0.05-0.1 for more pronounced effects.

**Q: Import errors**
A: Ensure you're using Python 3.8+ and PyTorch 1.10+. Run `pip install --upgrade temporal-optimizer`.

### Getting Help

- Check the [API Reference](api_reference.md) for detailed parameter descriptions
- See [Use Cases](use_cases.md) for task-specific guidance
- Run [Benchmarks](../benchmarks/) to verify installation
- Open an issue on GitHub for bugs or feature requests

## Next Steps

- Read about [specific use cases](use_cases.md) where temporal optimizers excel
- Review the [complete API documentation](api_reference.md)
- Run the [benchmarks](../benchmarks/) to see performance improvements
- Explore the [theory](theory.md) behind temporal stability (optional)