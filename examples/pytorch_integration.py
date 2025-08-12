"""
Basic PyTorch integration examples with Temporal Optimizer.

This script demonstrates how to integrate StableAdam and StableSGD
into standard PyTorch training workflows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from temporal_optimizer import StableAdam, StableSGD


def create_synthetic_data(n_samples=1000, n_features=20, n_classes=3, noise=0.1):
    """Create synthetic classification dataset."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate structured data
    X = torch.randn(n_samples, n_features)
    
    # Create class-dependent patterns
    true_weights = torch.randn(n_features, n_classes)
    logits = X @ true_weights + torch.randn(n_samples, n_classes) * noise
    y = torch.argmax(logits, dim=1)
    
    return X, y


class SimpleClassifier(nn.Module):
    """Simple neural network classifier."""
    
    def __init__(self, input_dim=20, hidden_dims=None, num_classes=3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_model(model, optimizer, train_loader, test_loader, epochs=50):
    """
    Train model with given optimizer and return training history.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of epochs to train
    
    Returns:
        Dictionary with training history
    """
    criterion = nn.CrossEntropyLoss()
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Test phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        # Record metrics
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(100 * train_correct / train_total)
        history['test_loss'].append(test_loss / len(test_loader))
        history['test_acc'].append(100 * test_correct / test_total)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d}: Train Acc: {history["train_acc"][-1]:.2f}%, '
                  f'Test Acc: {history["test_acc"][-1]:.2f}%')
    
    return history


def example_1_basic_usage():
    """Example 1: Basic usage - drop-in replacement for Adam."""
    print("Example 1: Basic Usage")
    print("=" * 40)
    
    # Create data
    X, y = create_synthetic_data()
    dataset = TensorDataset(X, y)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    model = SimpleClassifier()
    
    # BEFORE: Standard Adam
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # AFTER: StableAdam (drop-in replacement)
    optimizer = StableAdam(model.parameters(), lr=0.001)
    
    print("Training with StableAdam...")
    history = train_model(model, optimizer, train_loader, test_loader, epochs=30)
    
    print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")
    print("✅ Basic integration successful!\n")
    
    return history


def example_2_advanced_configuration():
    """Example 2: Advanced configuration with temporal stability tuning."""
    print("Example 2: Advanced Configuration")
    print("=" * 40)
    
    # Create data
    X, y = create_synthetic_data()
    dataset = TensorDataset(X, y)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    model = SimpleClassifier()
    
    # Advanced StableAdam configuration
    optimizer = StableAdam(
        model.parameters(),
        lr=0.001,                    # Standard Adam parameter
        betas=(0.9, 0.999),         # Standard Adam parameter
        eps=1e-8,                   # Standard Adam parameter
        weight_decay=1e-4,          # L2 regularization
        
        # Temporal stability parameters
        temporal_stability=0.02,     # Higher for more stability
        momentum_decay=0.95,        # Smoother momentum updates
        energy_conservation=True    # Adaptive learning rates
    )
    
    print("Training with advanced StableAdam configuration...")
    history = train_model(model, optimizer, train_loader, test_loader, epochs=30)
    
    # Monitor temporal stability penalty
    penalty = optimizer.get_temporal_stability_penalty()
    if penalty is not None:
        print(f"Temporal stability penalty: {penalty.item():.6f}")
    
    print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")
    print("✅ Advanced configuration successful!\n")
    
    return history


def example_3_parameter_groups():
    """Example 3: Different learning rates for different layers."""
    print("Example 3: Parameter Groups")
    print("=" * 40)
    
    # Create data
    X, y = create_synthetic_data()
    dataset = TensorDataset(X, y)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    model = SimpleClassifier(hidden_dims=[128, 64, 32])
    
    # Different learning rates and stability for different layers
    optimizer = StableAdam([
        {
            'params': model.network[:3].parameters(),  # First hidden layer
            'lr': 1e-4, 
            'temporal_stability': 0.03  # Higher stability for early layers
        },
        {
            'params': model.network[3:6].parameters(),  # Second hidden layer
            'lr': 5e-4,
            'temporal_stability': 0.02
        },
        {
            'params': model.network[6:].parameters(),   # Output layers
            'lr': 1e-3,
            'temporal_stability': 0.01  # Lower stability for output
        }
    ])
    
    print("Training with parameter groups...")
    history = train_model(model, optimizer, train_loader, test_loader, epochs=30)
    
    print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")
    print("✅ Parameter groups successful!\n")
    
    return history


def example_4_stablesgd_comparison():
    """Example 4: Comparing StableAdam vs StableSGD."""
    print("Example 4: StableAdam vs StableSGD Comparison")
    print("=" * 50)
    
    # Create data
    X, y = create_synthetic_data()
    dataset = TensorDataset(X, y)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    results = {}
    
    # Test StableAdam
    print("Training with StableAdam...")
    model_adam = SimpleClassifier()
    optimizer_adam = StableAdam(model_adam.parameters(), lr=0.001, temporal_stability=0.01)
    results['StableAdam'] = train_model(model_adam, optimizer_adam, train_loader, test_loader, epochs=30)
    
    print()
    
    # Test StableSGD
    print("Training with StableSGD...")
    model_sgd = SimpleClassifier()
    optimizer_sgd = StableSGD(
        model_sgd.parameters(), 
        lr=0.01, 
        momentum=0.9, 
        temporal_stability=0.01
    )
    results['StableSGD'] = train_model(model_sgd, optimizer_sgd, train_loader, test_loader, epochs=30)
    
    # Compare results
    print("\nComparison:")
    for opt_name, history in results.items():
        print(f"{opt_name:12}: Final accuracy = {history['test_acc'][-1]:.2f}%")
    
    print("✅ Comparison successful!\n")
    
    return results


def example_5_pytorch_lightning_integration():
    """Example 5: Integration with PyTorch Lightning (conceptual)."""
    print("Example 5: PyTorch Lightning Integration (Conceptual)")
    print("=" * 60)
    
    print("Here's how you would integrate with PyTorch Lightning:")
    print()
    
    code_example = '''
import pytorch_lightning as pl
from temporal_optimizer import StableAdam

class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleClassifier()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def configure_optimizers(self):
        # Use StableAdam instead of standard Adam
        return StableAdam(
            self.parameters(),
            lr=0.001,
            temporal_stability=0.02
        )

# Usage
model = LightningModel()
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
'''
    
    print(code_example)
    print("✅ PyTorch Lightning integration example shown!\n")


def example_6_huggingface_transformers():
    """Example 6: Integration with Hugging Face Transformers (conceptual)."""
    print("Example 6: Hugging Face Transformers Integration (Conceptual)")
    print("=" * 65)
    
    print("Here's how you would integrate with Hugging Face Transformers:")
    print()
    
    code_example = '''
from transformers import Trainer, TrainingArguments
from temporal_optimizer import StableAdam

class StableAdamTrainer(Trainer):
    def create_optimizer(self):
        """Override optimizer creation to use StableAdam."""
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() 
                    if n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
                "temporal_stability": 0.02,  # Higher stability for pretrained models
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() 
                    if n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "temporal_stability": 0.01,
            },
        ]
        
        self.optimizer = StableAdam(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )

# Usage
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

trainer = StableAdamTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
'''
    
    print(code_example)
    print("✅ Hugging Face Transformers integration example shown!\n")


def plot_comparison(histories: Dict[str, Dict]):
    """Plot comparison of different optimizers."""
    plt.figure(figsize=(12, 5))
    
    # Plot test accuracy
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history['test_acc'], label=name, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot test loss
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history['test_loss'], label=name, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/pytorch_integration_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run all integration examples."""
    print("Temporal Optimizer - PyTorch Integration Examples")
    print("=" * 60)
    print()
    
    # Run examples
    history1 = example_1_basic_usage()
    history2 = example_2_advanced_configuration()
    history3 = example_3_parameter_groups()
    comparison_results = example_4_stablesgd_comparison()
    example_5_pytorch_lightning_integration()
    example_6_huggingface_transformers()
    
    # Plot comparison from example 4
    print("Generating comparison plots...")
    plot_comparison(comparison_results)
    
    print("🎉 All integration examples completed successfully!")
    print("Check 'examples/pytorch_integration_comparison.png' for visualizations.")


if __name__ == "__main__":
    main()