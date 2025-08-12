# Contributing to Temporal Optimizer

Thank you for your interest in contributing to Temporal Optimizer! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style and Standards](#code-style-and-standards)
4. [Testing Guidelines](#testing-guidelines)
5. [Submitting Changes](#submitting-changes)
6. [Issue Reporting](#issue-reporting)
7. [Performance Considerations](#performance-considerations)
8. [Documentation](#documentation)

## Getting Started

### Ways to Contribute

- **Bug fixes**: Help identify and fix issues
- **New features**: Implement new optimizers or functionality
- **Documentation**: Improve docs, examples, and tutorials
- **Testing**: Add test cases and improve test coverage
- **Benchmarks**: Create performance benchmarks and comparisons
- **Examples**: Add real-world usage examples

### Areas of Focus

We especially welcome contributions in:
- **New optimizer variants**: Extensions of StableAdam/StableSGD
- **Application domains**: Finance, healthcare, autonomous systems
- **Performance optimization**: Memory usage, computation speed
- **Integration examples**: PyTorch Lightning, Transformers, etc.

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/temporal-optimizer.git
cd temporal-optimizer
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,benchmarks,docs,test]"
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run code quality checks before each commit.

### 4. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Run a simple benchmark
python benchmarks/optimization_comparison.py

# Check imports
python -c "from temporal_optimizer import StableAdam, StableSGD; print('✅ Setup complete')"
```

## Code Style and Standards

### Code Formatting

We use several tools to maintain code quality:

```bash
# Format code
black temporal_optimizer tests
isort temporal_optimizer tests

# Lint code
flake8 temporal_optimizer tests

# Type checking
mypy temporal_optimizer

# Security check
bandit -r temporal_optimizer
```

All of these run automatically via pre-commit hooks.

### Python Style Guide

- **Line length**: Maximum 100 characters
- **Docstrings**: Use numpy-style docstrings
- **Type hints**: Required for all public functions
- **Imports**: Organize with isort, separate standard/third-party/local
- **Naming**: Use descriptive names, follow PEP 8 conventions

### Example Code Style

```python
from typing import Optional, Dict, Any
import torch
from torch.optim import Optimizer


class NewOptimizer(Optimizer):
    """
    Brief description of the optimizer.
    
    Longer description explaining the algorithm, when to use it,
    and key benefits over standard optimizers.
    
    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    lr : float, optional
        Learning rate (default: 0.001)
    stability_weight : float, optional  
        Weight for stability regularization (default: 0.01)
    
    Examples
    --------
    >>> model = torch.nn.Linear(10, 1)
    >>> optimizer = NewOptimizer(model.parameters(), lr=0.001)
    >>> 
    >>> # Training loop
    >>> for batch in dataloader:
    ...     optimizer.zero_grad()
    ...     loss = criterion(model(batch.x), batch.y)
    ...     loss.backward() 
    ...     optimizer.step()
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        stability_weight: float = 0.01,
        **kwargs
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr, stability_weight=stability_weight, **kwargs)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        """Perform single optimization step."""
        # Implementation here
        pass
```

## Testing Guidelines

### Test Structure

```
tests/
├── test_optimizers/         # Core optimizer tests
│   ├── test_stable_adam.py
│   └── test_stable_sgd.py
├── test_integration/        # Integration tests
├── test_stability/          # Temporal stability tests  
└── test_performance/        # Performance benchmarks
```

### Writing Tests

1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test optimizer with real training loops
3. **Stability Tests**: Verify temporal stability properties
4. **Performance Tests**: Ensure no significant regressions

```python
import pytest
import torch
import torch.nn as nn
from temporal_optimizer import StableAdam


class TestNewFeature:
    """Test suite for new feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality works correctly."""
        model = nn.Linear(10, 1)
        optimizer = StableAdam(model.parameters())
        
        # Test optimization step
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        
        # Verify parameters changed
        assert any(p.grad is not None for p in model.parameters())
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        model = nn.Linear(10, 1)
        
        with pytest.raises(ValueError, match="Invalid learning rate"):
            StableAdam(model.parameters(), lr=-0.1)
    
    @pytest.mark.slow
    def test_convergence(self):
        """Test convergence on synthetic problem."""
        # Test that optimizer actually converges
        pass
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_optimizers/test_stable_adam.py -v

# Run tests with coverage
pytest tests/ --cov=temporal_optimizer --cov-report=html

# Run only fast tests (skip slow benchmarks)
pytest tests/ -m "not slow"

# Run tests in parallel
pytest tests/ -n auto
```

## Submitting Changes

### Branch Workflow

1. **Create feature branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** with clear, focused commits:
   ```bash
   git add .
   git commit -m "Add temporal stability metrics computation"
   ```

3. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   # Create PR via GitHub interface
   ```

### Commit Messages

Use clear, descriptive commit messages:

```
✅ Good:
- "Add StableLAMB optimizer with adaptive regularization"
- "Fix memory leak in parameter tracking"
- "Update benchmarks for credit scoring example"

❌ Bad:
- "Fix bug"
- "Update code"
- "Changes"
```

### Pull Request Guidelines

1. **Clear description**: Explain what the PR does and why
2. **Link issues**: Reference relevant issue numbers
3. **Add tests**: Include tests for new functionality
4. **Update docs**: Update documentation if needed
5. **Check CI**: Ensure all CI checks pass

#### PR Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (describe)

## Testing
- [ ] Added/updated unit tests
- [ ] Added/updated integration tests
- [ ] Verified benchmarks still pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Issue Reporting

### Bug Reports

Include:
- **Environment**: OS, Python version, PyTorch version
- **Reproduction steps**: Minimal code example
- **Expected vs actual behavior**
- **Error messages**: Full stack traces
- **Additional context**: Performance impact, workarounds

### Feature Requests

Include:
- **Use case**: When/why this feature is needed
- **Proposed solution**: How it should work
- **Alternatives considered**: Other approaches
- **Examples**: Code examples of desired usage

## Performance Considerations

### Optimization Guidelines

1. **Memory efficiency**: Minimize tensor allocations
2. **Vectorization**: Use PyTorch ops over Python loops
3. **In-place operations**: Use `tensor.add_()` vs `tensor + other`
4. **Device awareness**: Support CPU/GPU efficiently

### Benchmarking

Always benchmark performance changes:

```python
import time
import torch
from temporal_optimizer import StableAdam

# Benchmark optimization step
model = torch.nn.Linear(1000, 1000)
optimizer = StableAdam(model.parameters())

# Warmup
for _ in range(10):
    optimizer.zero_grad()
    loss = model(torch.randn(100, 1000)).sum()
    loss.backward()
    optimizer.step()

# Benchmark
start_time = time.time()
for _ in range(100):
    optimizer.zero_grad()
    loss = model(torch.randn(100, 1000)).sum() 
    loss.backward()
    optimizer.step()

duration = time.time() - start_time
print(f"Time per step: {duration/100*1000:.2f}ms")
```

## Documentation

### Docstring Requirements

All public functions need comprehensive docstrings:

```python
def temporal_stability_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    model: Optional[nn.Module] = None,
    stability_weight: float = 0.01
) -> torch.Tensor:
    """
    Compute loss with temporal stability regularization.
    
    This function combines a base loss with a temporal stability penalty
    that discourages rapid parameter changes, leading to more stable
    training dynamics.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Model predictions with shape (batch_size, num_classes)
    targets : torch.Tensor  
        Ground truth targets with shape (batch_size,)
    model : nn.Module, optional
        Model instance for parameter regularization. If None, 
        no stability penalty is applied.
    stability_weight : float, optional
        Weight for temporal stability penalty. Higher values
        increase stability but may slow adaptation. Default: 0.01
    
    Returns
    -------
    torch.Tensor
        Combined loss value as scalar tensor
    
    Examples
    --------
    >>> outputs = torch.randn(32, 10)  # Batch of predictions
    >>> targets = torch.randint(0, 10, (32,))  # True labels
    >>> model = torch.nn.Linear(784, 10)
    >>> 
    >>> loss = temporal_stability_loss(outputs, targets, model)
    >>> print(f"Loss: {loss.item():.4f}")
    
    See Also
    --------
    TemporalStabilityLoss : Class-based version with automatic tracking
    """
```

### README Updates

Update README.md for:
- New optimizers or major features
- Changed installation instructions
- New usage examples
- Performance improvements

## Questions?

- **GitHub Issues**: For bugs, features, and general questions
- **Discussions**: For design discussions and ideas
- **Email**: For security issues or private questions

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. Please be respectful and inclusive in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.