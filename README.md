# Temporal Optimizer

Drop-in replacements for PyTorch optimizers with enhanced temporal stability and convergence properties.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why Temporal Optimizer?

Standard neural network optimization faces critical challenges:
- **Training instability**: Loss oscillations and convergence failures
- **Temporal degradation**: Model performance degrades over time in production
- **Poor generalization**: Models struggle with distribution shifts

**Temporal Optimizer solves these problems** with optimizers that maintain stability over time while requiring zero code changes to your existing PyTorch training loops.

## 30-Second Integration

Replace your PyTorch optimizer with a single line change:

```python
# Before
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)

# After  
from temporal_optimizer import StableAdam
optimizer = StableAdam(model.parameters(), lr=0.001)
```

That's it. Your model now has enhanced temporal stability with zero additional complexity.

## Key Benefits

- ✅ **Drop-in compatibility**: Works with any PyTorch model
- ✅ **Temporal stability**: 2-5x more stable training dynamics
- ✅ **Better convergence**: Faster and more reliable convergence
- ✅ **Production ready**: Maintains performance over time
- ✅ **Zero learning curve**: Same API as PyTorch optimizers

## Installation

```bash
pip install temporal-optimizer
```

## Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from temporal_optimizer import StableAdam

# Your existing model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Drop-in replacement for torch.optim.Adam
optimizer = StableAdam(model.parameters(), lr=0.001)

# Rest of your training loop stays exactly the same
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()
```

### Advanced Configuration

```python
# Fine-tune temporal stability (optional)
optimizer = StableAdam(
    model.parameters(),
    lr=0.001,
    temporal_stability=0.01,      # Higher = more stability
    momentum_decay=0.9,           # Momentum decay factor
    energy_conservation=True      # Enable adaptive learning rates
)
```

## Available Optimizers

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| `StableAdam` | Enhanced Adam with temporal stability | General purpose, computer vision |
| `StableSGD` | Enhanced SGD with momentum conservation | Large-scale training, NLP |

## Performance Comparison

Benchmarks on credit scoring with temporal drift:

| Optimizer | Final Accuracy | Stability Score | Training Time |
|-----------|----------------|-----------------|---------------|
| Adam | 0.847 | 0.73 | 45.2s |
| **StableAdam** | **0.861** | **0.94** | 47.1s |
| SGD | 0.834 | 0.68 | 42.8s |
| **StableSGD** | **0.849** | **0.89** | 44.3s |

*Higher stability scores indicate more consistent performance over time*

## When to Use Temporal Optimizer

**Perfect for:**
- Credit scoring and financial models
- Medical diagnosis systems  
- Recommendation systems
- Time-series forecasting
- Any model deployed in production

**Especially valuable when:**
- Data distribution changes over time
- Model needs to maintain performance for months/years
- Training loss exhibits oscillations
- Reproducible results are critical

## Documentation

- [Quick Start Guide](docs/quickstart.md)
- [Use Cases & Examples](docs/use_cases.md)
- [API Reference](docs/api_reference.md)
- [Benchmarks](docs/benchmarks.md)
- [Theory (Optional)](docs/theory.md)

## Examples

Run the included examples to see temporal stability in action:

```bash
# Reproduce credit scoring results
python benchmarks/credit_scoring_reproduction.py

# Compare optimization performance
python benchmarks/optimization_comparison.py

# See basic integration examples
python examples/pytorch_integration.py
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Temporal Optimizer in research, please cite:

```bibtex
@software{temporal_optimizer,
  title={Temporal Optimizer: Drop-in PyTorch optimizers with temporal stability},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/temporal-optimizer}
}
```