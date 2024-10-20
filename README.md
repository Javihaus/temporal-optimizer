# Hamiltonian AI

Hamiltonian AI is a Python library that implements Hamiltonian-inspired approaches for optimizing AI models. It provides tools for both multi-hop question answering and credit scoring tasks, leveraging principles from Hamiltonian mechanics to enhance model performance and stability.

The following papers provide further information regarding the theoretical development:
- https://arxiv.org/abs/2410.10182
- https://arxiv.org/abs/2410.04415

## Features

- Hamiltonian-inspired neural network architecture
- Advanced Symplectic Optimizer for efficient training
- Hamiltonian loss function for improved model convergence
- Data processing utilities for handling imbalanced datasets
- Evaluation metrics tailored for AI optimization tasks

## Installation

You can install Hamiltonian AI using pip:

```bash
pip install hamiltonian-ai
```

Or clone the repository and install it locally:

```bash
Copygit clone https://github.com/yourusername/hamiltonian-ai.git
cd hamiltonian-ai
pip install -e
```

## Quick Start Guide
Here's a simple example to get you started with Hamiltonian AI:
pythonCopyimport torch
from hamiltonian_ai import HamiltonianNN, AdvancedSymplecticOptimizer, hamiltonian_loss, prepare_data, evaluate_model
```python
# Prepare your data
X, y = load_your_data()  # Load your dataset
train_dataset, test_dataset, scaler = prepare_data(X, y)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Initialize the model
model = HamiltonianNN(input_dim=X.shape[1], hidden_dims=[64, 32])

# Set up the optimizer
optimizer = AdvancedSymplecticOptimizer(model.parameters())

# Training loop
for epoch in range(10):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = hamiltonian_loss(outputs, batch_y, model)
        loss.backward()
        optimizer.step()

# Evaluate the model
accuracy, precision, recall, f1, auc = evaluate_model(model, test_loader, device='cpu')
print(f"Accuracy: {accuracy}, F1 Score: {f1}, AUC: {auc}")
```

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
git clone https://github.com/yourusername/hamiltonian-ai.git
cd hamiltonian-ai
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

