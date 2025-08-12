# API Reference

Complete reference for all classes and functions in temporal-optimizer.

## Optimizers

### StableAdam

Drop-in replacement for `torch.optim.Adam` with enhanced temporal stability.

```python
from temporal_optimizer import StableAdam

StableAdam(
    params,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    temporal_stability=0.01,
    momentum_decay=0.9,
    energy_conservation=True
)
```

**Parameters:**

- **params** (iterable): Iterable of parameters to optimize or dicts defining parameter groups
- **lr** (float, optional): Learning rate (default: 1e-3)
- **betas** (Tuple[float, float], optional): Coefficients for computing running averages of gradient and squared gradient (default: (0.9, 0.999))
- **eps** (float, optional): Term added to denominator for numerical stability (default: 1e-8)
- **weight_decay** (float, optional): L2 penalty coefficient (default: 0)
- **temporal_stability** (float, optional): Weight for temporal stability regularization. Higher values increase stability but may slow adaptation (default: 0.01)
- **momentum_decay** (float, optional): Momentum decay factor for Hamiltonian mechanics. Should be between 0 and 1 (default: 0.9)
- **energy_conservation** (bool, optional): Enable adaptive learning rate based on system energy. May provide better convergence but adds computational overhead (default: True)

**Methods:**

#### step(closure=None)
Perform a single optimization step.

**Parameters:**
- **closure** (callable, optional): A closure that re-evaluates the model and returns the loss

**Returns:**
- Loss value if closure is provided, None otherwise

#### zero_grad()
Zero the gradients of all optimized parameters.

#### get_temporal_stability_penalty()
Get the last computed temporal stability penalty.

**Returns:**
- torch.Tensor or None: The temporal stability penalty from the last optimization step

**Example:**

```python
import torch
import torch.nn as nn
from temporal_optimizer import StableAdam

model = nn.Linear(10, 1)
optimizer = StableAdam(model.parameters(), lr=0.001, temporal_stability=0.02)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()
    
    # Optional: monitor stability penalty
    penalty = optimizer.get_temporal_stability_penalty()
    if penalty is not None:
        print(f"Temporal stability penalty: {penalty.item()}")
```

---

### StableSGD

Drop-in replacement for `torch.optim.SGD` with enhanced temporal stability.

```python
from temporal_optimizer import StableSGD

StableSGD(
    params,
    lr=0.01,
    momentum=0,
    weight_decay=0,
    dampening=0,
    nesterov=False,
    temporal_stability=0.01,
    momentum_decay=0.9,
    energy_conservation=True
)
```

**Parameters:**

- **params** (iterable): Iterable of parameters to optimize or dicts defining parameter groups
- **lr** (float, optional): Learning rate (default: 1e-2)
- **momentum** (float, optional): Momentum factor (default: 0)
- **weight_decay** (float, optional): L2 penalty coefficient (default: 0)
- **dampening** (float, optional): Dampening for momentum (default: 0)
- **nesterov** (bool, optional): Enable Nesterov momentum (default: False)
- **temporal_stability** (float, optional): Weight for temporal stability regularization (default: 0.01)
- **momentum_decay** (float, optional): Momentum decay factor for Hamiltonian mechanics (default: 0.9)
- **energy_conservation** (bool, optional): Enable adaptive learning rate based on system energy (default: True)

**Methods:**

Same as StableAdam: `step()`, `zero_grad()`, `get_temporal_stability_penalty()`

**Example:**

```python
import torch
import torch.nn as nn
from temporal_optimizer import StableSGD

model = nn.Linear(10, 1)
optimizer = StableSGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9, 
    temporal_stability=0.015
)

# Training loop (same as standard SGD)
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()
```

---

## Loss Functions

### temporal_stability_loss

Function that combines base loss with temporal stability regularization.

```python
from temporal_optimizer.losses import temporal_stability_loss

temporal_stability_loss(
    outputs,
    targets,
    model=None,
    base_loss="cross_entropy",
    stability_weight=0.01,
    previous_params=None
)
```

**Parameters:**

- **outputs** (torch.Tensor): Model predictions
- **targets** (torch.Tensor): Ground truth targets
- **model** (nn.Module, optional): Model instance for parameter regularization
- **base_loss** (nn.Module or str, optional): Base loss function or string identifier. Supported strings: "cross_entropy", "mse", "bce" (default: "cross_entropy")
- **stability_weight** (float, optional): Weight for temporal stability penalty (default: 0.01)
- **previous_params** (dict, optional): Previous model parameters for stability computation

**Returns:**
- torch.Tensor: Combined loss value

**Example:**

```python
from temporal_optimizer.losses import temporal_stability_loss

# Basic usage
loss = temporal_stability_loss(outputs, targets, model)

# With custom base loss
import torch.nn as nn
custom_loss = nn.MSELoss()
loss = temporal_stability_loss(
    outputs, targets, model,
    base_loss=custom_loss,
    stability_weight=0.02
)
```

### TemporalStabilityLoss

Class-based temporal stability loss with automatic parameter tracking.

```python
from temporal_optimizer.losses import TemporalStabilityLoss

TemporalStabilityLoss(
    base_loss="cross_entropy",
    stability_weight=0.01,
    momentum=0.9
)
```

**Parameters:**

- **base_loss** (nn.Module or str, optional): Base loss function or string identifier (default: "cross_entropy")
- **stability_weight** (float, optional): Weight for temporal stability penalty (default: 0.01)
- **momentum** (float, optional): Exponential moving average factor for parameter tracking (default: 0.9)

**Methods:**

#### forward(outputs, targets, model)
Compute temporal stability loss with automatic parameter tracking.

#### reset_state()
Reset internal parameter tracking state.

**Example:**

```python
from temporal_optimizer.losses import TemporalStabilityLoss

criterion = TemporalStabilityLoss("cross_entropy", stability_weight=0.02)

# Training loop with automatic parameter tracking
for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(batch.x)
    loss = criterion(outputs, batch.y, model)  # Automatically tracks parameters
    loss.backward()
    optimizer.step()
```

---

## Utility Functions

### compare_optimizers

Benchmark multiple optimizers on the same task.

```python
from temporal_optimizer.utils import compare_optimizers

compare_optimizers(
    model_factory,
    optimizers,
    train_loader,
    test_loader,
    loss_fn,
    epochs=50,
    device="auto"
)
```

**Parameters:**

- **model_factory** (callable): Function that creates fresh model instances
- **optimizers** (dict): Dictionary mapping optimizer names to factory functions
- **train_loader** (DataLoader): Training data loader
- **test_loader** (DataLoader): Test data loader
- **loss_fn** (nn.Module): Loss function
- **epochs** (int, optional): Number of training epochs (default: 50)
- **device** (str, optional): Device for training. "auto" selects automatically (default: "auto")

**Returns:**
- OptimizationBenchmark: Benchmark object with results and plotting methods

**Example:**

```python
from temporal_optimizer import StableAdam
from temporal_optimizer.utils import compare_optimizers
import torch

def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(10, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2)
    )

optimizers = {
    'Adam': lambda params: torch.optim.Adam(params, lr=0.001),
    'StableAdam': lambda params: StableAdam(params, lr=0.001)
}

benchmark = compare_optimizers(
    create_model,
    optimizers,
    train_loader,
    test_loader,
    torch.nn.CrossEntropyLoss()
)

print(benchmark.generate_report())
benchmark.plot_comparison()
```

### temporal_stability_metrics

Compute stability metrics from training history.

```python
from temporal_optimizer.utils import temporal_stability_metrics

temporal_stability_metrics(
    parameter_history,
    loss_history,
    window_size=10
)
```

**Parameters:**

- **parameter_history** (dict): Dictionary mapping parameter names to lists of parameter tensors over time
- **loss_history** (list): List of loss values over training steps
- **window_size** (int, optional): Window size for stability computations (default: 10)

**Returns:**
- dict: Dictionary containing stability metrics

### convergence_metrics

Compute convergence-related metrics from training history.

```python
from temporal_optimizer.utils import convergence_metrics

convergence_metrics(
    loss_history,
    accuracy_history=None,
    target_loss=None,
    target_accuracy=None
)
```

**Parameters:**

- **loss_history** (list): List of loss values over training
- **accuracy_history** (list, optional): List of accuracy values over training
- **target_loss** (float, optional): Target loss for convergence detection
- **target_accuracy** (float, optional): Target accuracy for convergence detection

**Returns:**
- dict: Dictionary containing convergence metrics

---

## Parameter Groups

Both StableAdam and StableSGD support parameter groups for per-layer learning rate control:

```python
# Different learning rates and stability for different layers
optimizer = StableAdam([
    {'params': model.backbone.parameters(), 'lr': 1e-4, 'temporal_stability': 0.02},
    {'params': model.classifier.parameters(), 'lr': 1e-3, 'temporal_stability': 0.01}
])
```

## Error Handling

The optimizers include comprehensive error checking:

```python
# These will raise ValueError with descriptive messages
StableAdam(model.parameters(), lr=-0.1)          # Negative learning rate
StableAdam(model.parameters(), temporal_stability=-0.1)  # Negative stability
StableSGD(model.parameters(), momentum=1.5)       # Invalid momentum range
```

## Device and Precision Support

All optimizers work with:
- CPU and CUDA devices
- Mixed precision training (float16/bfloat16)
- Multi-GPU training
- Model parallelism

```python
# CUDA example
model = model.cuda()
optimizer = StableAdam(model.parameters(), lr=0.001)

# Mixed precision example
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = StableAdam(model.parameters(), lr=0.001)

for batch in dataloader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(batch.x)
        loss = criterion(outputs, batch.y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```