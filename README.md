# Hamiltonian AI

Hamiltonian AI is a Python library that implements Hamiltonian-inspired approaches for optimizing AI models. By leveraging principles from Hamiltonian mechanics, this library provides novel optimization techniques that enhance model performance and stability in both multi-hop question answering and credit scoring tasks.

# Theoretical Background

Note: For a complete visual presentation of the theoretical foundations, see our detailed presentation.

## The Optimization Problem
In machine learning, optimization seeks to find parameters $\theta^*$ that minimize an objective function:

$$\theta^* = \text{argmin } f(x) \text{ for } \theta \in \Omega$$

Traditional optimization faces several challenges:

- Non-convexity of the objective function
- High dimensionality
- Ill-conditioning

For optimization functions that are $\alpha$-strongly convex and $\beta$-strongly smooth, we have:

$$\frac{\alpha}{2}|x - y|^2 \leq f(y) - f(x) - \langle\nabla f(x), y - x\rangle \leq \frac{\beta}{2}|x - y|^2$$

## Hamiltonian Mechanics in Optimization
### Basic Principles
Hamiltonians in physics describe systems that conserve total energy, potentially leading to more
stable optimization trajectories. In quantum mechanics, Hamiltonians govern the evolution of
wavefunctions, exploring all possible states. Similarly, in optimization, this could lead to better
exploration of the parameter space. 

![Alt Text](https://raw.githubusercontent.com/Javihaus/hamiltonian_ai/main/docs/images/Basic.png)

For example, in a frictionless pendulum the total energy, it is the sum of potential and kinetic energy is conserved trough its movement.

We adapt this principle for optimization:

**State Space Variables**

Position ($q$): Represents model parameters
Momentum ($p$): Represents parameter update velocities
Time ($t$): Represents optimization steps


**Hamiltonian Function**
$H(q, p) = T(p) + V(q)$, where:

$T(p)$: Kinetic energy (parameter update costs)
$V(q)$: Potential energy (loss function)


**Hamilton's Equations**

$$\begin{align*}
\dot{q} &= \frac{\partial H}{\partial p} \
\dot{p} &= -\frac{\partial H}{\partial q}
\end{align*}$$

For systems with $n$ degrees of freedom:

$$\begin{align*}
\dot{q}_i &= \frac{\partial H}{\partial p_i}(t, q, p) \
\dot{p}_i &= -\frac{\partial H}{\partial q_i}(t, p, q)
\end{align*}$$

### Symplectic Geometry
Symplectic geometry provides the mathematical framework for understanding Hamiltonian dynamics. In local coordinates $(q_i, p_i)$, a standard symplectic form is:

$$\omega = \sum_i dq_i \wedge dp_i$$

![Alt Text](https://raw.githubusercontent.com/Javihaus/hamiltonian_ai/main/docs/images/Symplectic.png)

A frictionless pendulum is one of the most basic forms of a symplectic space. Velocity and angle are the two components that describe the movements of a pendulum. We can map this in a 2D space as a trajectory.

The Symplectic Euler method, which we implement, takes the form:

$$\begin{align*}
p_{n+1} &= p_n - \Delta t \times \frac{\partial H}{\partial q}(q_n, p_{n+1}) \
q_{n+1} &= q_n + \Delta t \times \frac{\partial H}{\partial p}(q_n, p_{n+1})
\end{align*}$$

where $\Delta t$ is analogous to the learning rate in optimization.

## Hamiltonian Optimization Algorithm
Our implementation translates these physical principles into optimization:

![Alt Text](https://raw.githubusercontent.com/Javihaus/hamiltonian_ai/main/docs/images/HamiltonianSystems.png)

**Momentum Update:**

$$m_t = \beta \times m_{t-1} + (1-\beta) \times g_t$$

where:

$\beta$: momentum decay factor
$g_t$: current gradient


**Hamiltonian Energy:**

$$\begin{align*}
K &= \frac{1}{2}v^2 \text{ (Kinetic)} \
V &= \frac{1}{2}g^2 \text{ (Potential)} \
H &= K + V \text{ (Total)}
\end{align*}$$


**Parameter Update:**

$$\theta_t = \theta_{t-1} - \frac{\eta \times m_t}{\sqrt{H_t + \epsilon}}$$

where:

$\eta$: learning rate

$\epsilon$: small constant for numerical stability

![Alt Text](https://raw.githubusercontent.com/Javihaus/hamiltonian_ai/main/docs/images/HamiltoniainOptimization.png)

## Hamiltonian Loss Function
Our custom loss function combines traditional loss with a Hamiltonian-inspired regularization term:

$$H_{loss}(\theta) = L_{base}(\theta) + \lambda \times R(\theta)$$

where:

$L_{base}(\theta)$: Base loss function (e.g., cross-entropy)

$R(\theta)$: Regularization term analogous to potential energy

$\lambda$: Regularization coefficient

The regularization term takes the form:

$$R(\theta) = \frac{1}{2}|\theta|^2$$

This is analogous to potential energy in Hamiltonian systems:

$$T(p) = \frac{1}{2}|p|^2$$

### Benefits of Hamiltonian Approach
- Energy Conservation: The conservation of the Hamiltonian leads to more stable optimization trajectories.
- Momentum-Based Exploration: The system can "roll past" local minima while maintaining exploratory behavior.
- Geometric Structure Preservation: Symplectic integration preserves the geometric properties of the optimization space.

## Features

### HamiltonianNN
A neural network architecture incorporating Hamiltonian principles:

```python
model = HamiltonianNN(
    input_dim=10,           # Input dimension
    hidden_dims=[64, 32],   # Hidden layer dimensions
    activation='leaky_relu', # Activation function ('leaky_relu' or 'relu')
    dropout_rate=0.2        # Dropout rate for regularization
)
```

### AdvancedSymplecticOptimizer
A custom optimizer based on symplectic integration:

```python
optimizer = AdvancedSymplecticOptimizer(
    params,                 # Model parameters
    lr=1e-2,               # Learning rate
    beta=0.9,              # Momentum coefficient
    epsilon=1e-8           # Small constant for numerical stability
)
```

### Hamiltonian Loss Function
Custom loss function incorporating Hamiltonian principles:

```python
loss = hamiltonian_loss(
    outputs,               # Model outputs
    labels,               # True labels
    model,                # Model instance
    reg_coeff=0.01       # Regularization coefficient
)
```

The loss function combines:
- Base loss (cross-entropy)
- Regularization term based on parameter norms

## Parameter Configuration Guide

### Model Parameters

#### HamiltonianNN Configuration
```python
model = HamiltonianNN(
    input_dim,          # Required: Integer, dimension of input features
    hidden_dims,        # Required: List[int], dimensions of hidden layers
    activation='leaky_relu',  # Optional: string, either 'leaky_relu' or 'relu'
    dropout_rate=0.2    # Optional: float between 0 and 1
)
```

| Parameter | Type | Configurable | Default | Description |
|-----------|------|--------------|---------|-------------|
| input_dim | int | ✅ Required | None | Input feature dimension |
| hidden_dims | List[int] | ✅ Required | None | List of hidden layer dimensions |
| activation | str | ✅ Optional | 'leaky_relu' | Activation function ('leaky_relu' or 'relu' only) |
| dropout_rate | float | ✅ Optional | 0.2 | Dropout probability |

### Optimizer Parameters

#### AdvancedSymplecticOptimizer Configuration
```python
optimizer = AdvancedSymplecticOptimizer(
    params,            # Required: model parameters
    lr=1e-2,          # Optional: learning rate
    beta=0.9,         # Optional: momentum coefficient
    epsilon=1e-8      # Optional: numerical stability constant
)
```

| Parameter | Type | Configurable | Default | Description |
|-----------|------|--------------|---------|-------------|
| params | Iterable | ✅ Required | None | Model parameters to optimize |
| lr | float | ✅ Optional | 1e-2 | Learning rate |
| beta | float | ✅ Optional | 0.9 | Momentum decay factor |
| epsilon | float | ✅ Optional | 1e-8 | Small constant for numerical stability |

### Loss Function Parameters

#### hamiltonian_loss Configuration
```python
loss = hamiltonian_loss(
    outputs,           # Required: model outputs
    labels,           # Required: true labels
    model,            # Required: model instance
    reg_coeff=0.01    # Optional: regularization coefficient
)
```

| Parameter | Type | Configurable | Default | Notes |
|-----------|------|--------------|---------|-------|
| outputs | Tensor | ✅ Required | None | Model predictions |
| labels | Tensor | ✅ Required | None | True labels |
| model | HamiltonianNN | ✅ Required | None | Model instance |
| reg_coeff | float | ✅ Optional | 0.01 | Regularization strength |
| base_loss | Function | ❌ Fixed | CrossEntropy | Base loss function (not configurable) |

### Data Processing Parameters

#### prepare_data Configuration
```python
train_dataset, test_dataset, scaler = prepare_data(
    X,                # Required: input features
    y,                # Required: target labels
    test_size=0.2,    # Optional: test set proportion
    apply_smote=True  # Optional: whether to apply SMOTE
)
```

| Parameter | Type | Configurable | Default | Description |
|-----------|------|--------------|---------|-------------|
| X | np.ndarray | ✅ Required | None | Input features |
| y | np.ndarray | ✅ Required | None | Target labels |
| test_size | float | ✅ Optional | 0.2 | Proportion of test set |
| apply_smote | bool | ✅ Optional | True | Whether to apply SMOTE |

### Fixed (Non-Configurable) Components

The following components are fixed and cannot be modified:

1. **Base Loss Function**:
   - Fixed to CrossEntropy
   - No option to change to other loss functions

2. **Optimizer Type**:
   - Uses Symplectic integration
   - Core optimization algorithm cannot be modified

3. **Model Architecture**:
   - Basic structure is fixed (fully connected layers)
   - Only layer dimensions and activation functions can be modified

4. **Regularization Type**:
   - Uses L2 regularization
   - Regularization method cannot be changed

### Example of Complete Configuration

```python
# Model initialization with all configurable parameters
model = HamiltonianNN(
    input_dim=10,
    hidden_dims=[64, 32],
    activation='leaky_relu',
    dropout_rate=0.3
)

# Optimizer with all configurable parameters
optimizer = AdvancedSymplecticOptimizer(
    model.parameters(),
    lr=0.01,
    beta=0.95,
    epsilon=1e-8
)

# Data preparation with all configurable parameters
train_dataset, test_dataset, scaler = prepare_data(
    X,
    y,
    test_size=0.2,
    apply_smote=True
)

# Training loop with loss configuration
outputs = model(inputs)
loss = hamiltonian_loss(
    outputs,
    labels,
    model,
    reg_coeff=0.01
)
```

## Installation

```bash
pip install hamiltonian_ai
```

## Detailed Usage Example

```python
import torch
from hamiltonian_ai.models import HamiltonianNN
from hamiltonian_ai.optimizers import AdvancedSymplecticOptimizer
from hamiltonian_ai.loss_functions import hamiltonian_loss
from hamiltonian_ai.data_processing import prepare_data
from hamiltonian_ai.utils import evaluate_model

# 1. Data Preparation
X, y = load_your_data()  # Your data loading function
train_dataset, test_dataset, scaler = prepare_data(
    X, 
    y,
    test_size=0.2,
    apply_smote=True  # Handle imbalanced datasets
)

# 2. Create Data Loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32
)

# 3. Initialize Model
model = HamiltonianNN(
    input_dim=X.shape[1],
    hidden_dims=[64, 32],
    activation='leaky_relu',
    dropout_rate=0.2
)

# 4. Setup Optimizer
optimizer = AdvancedSymplecticOptimizer(
    model.parameters(),
    lr=0.01,
    beta=0.9,
    epsilon=1e-8
)

# 5. Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = hamiltonian_loss(outputs, batch_y, model, reg_coeff=0.01)
        loss.backward()
        optimizer.step()
        
    # Evaluation
    if epoch % 2 == 0:
        metrics = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch}: Accuracy={metrics[0]:.4f}, F1={metrics[3]:.4f}")
```

## Advanced Features

### Data Processing
The `prepare_data` function includes:
- Standardization of features
- Train-test splitting
- SMOTE for handling imbalanced datasets
- Dataset creation for PyTorch

### Evaluation Metrics
The `evaluate_model` function returns:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC Score

## Performance Benefits

1. **Stability**: The Hamiltonian approach provides more stable optimization trajectories
2. **Generalization**: Better performance on out-of-time validation in credit scoring
3. **Efficiency**: Improved convergence through symplectic integration

## Research Papers

For theoretical foundations and empirical results:
- [Hamiltonian Neural Networks for Robust Out-of-Time Credit Scoring](https://arxiv.org/abs/2410.10182)
- [Optimizing AI Reasoning: A Hamiltonian Dynamics Approach to Multi-Hop Question Answering](https://arxiv.org/abs/2410.04415)

[Rest of the original README content including Examples, Contributing, etc.]

## Examples
For more detailed examples, please check the examples/ directory in our repository:

- Credit Scoring (inlcuding notebook)</br>
   Data for this example is available here: https://zenodo.org/records/8401978  (DOI 10.5281/zenodo.8401977)
- Question Answering (inlcuding notebook and data)

## Documentation
For full documentation, including API reference and tutorials, visit our documentation page.


## Contributing
We welcome contributions to Hamiltonian AI! Here are some ways you can contribute:

Report bugs and request features by opening issues.
Submit pull requests with bug fixes or new features.
Improve documentation or add examples.
Share your experience using Hamiltonian AI.

Please read our Contribution Guidelines for more details.

## Development Setup
To set up the development environment:

```bash
git clone https://github.com/yourusername/hamiltonian_ai.git
cd hamiltonian_ai
pip install -e .[dev]
```

## Run tests using

```bash
pytest
```

## Citation

If you use Hamiltonian AI in your research, please cite our paper:

```BibTex
@article{marin2024hamiltonian,
  title={Hamiltonian Neural Networks for Robust Out-of-Time Credit Scoring},
  author={Mar{\'\i}n, Javier},
  journal={arXiv preprint arXiv:2410.10182},
  year={2024}
}
```

```BibTex
@article{marin2024optimizing,
  title={Optimizing AI Reasoning: A Hamiltonian Dynamics Approach to Multi-Hop Question Answering},
  author={Marin, Javier},
  journal={arXiv preprint arXiv:2410.04415},
  year={2024}
}
```

## Contact
For any questions or feedback, please open an issue on our GitHub repository or contact us at javier@jmarin.info

