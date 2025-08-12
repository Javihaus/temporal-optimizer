"""
Comprehensive optimization comparison script.

This script benchmarks StableAdam and StableSGD against standard PyTorch optimizers
on various tasks to demonstrate performance improvements.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

# Add the parent directory to path to import temporal_optimizer
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from temporal_optimizer import StableAdam, StableSGD
from temporal_optimizer.utils import compare_optimizers


def create_synthetic_classification_data(n_samples=1000, n_features=20, n_classes=3):
    """Create synthetic classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features)
    
    # Create some structure in the data
    true_weights = torch.randn(n_features, n_classes)
    logits = X @ true_weights + torch.randn(n_samples, n_classes) * 0.1
    y = torch.argmax(logits, dim=1)
    
    return X, y


def create_synthetic_regression_data(n_samples=1000, n_features=10, noise=0.1):
    """Create synthetic regression dataset."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features)
    true_weights = torch.randn(n_features, 1)
    y = X @ true_weights + torch.randn(n_samples, 1) * noise
    
    return X, y.squeeze()


class ClassificationModel(nn.Module):
    """Simple neural network for classification."""
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


class RegressionModel(nn.Module):
    """Simple neural network for regression."""
    def __init__(self, input_dim=10, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


def run_classification_benchmark():
    """Run classification benchmark comparing optimizers."""
    print("Running Classification Benchmark...")
    print("=" * 50)
    
    # Create data
    X, y = create_synthetic_classification_data()
    dataset = TensorDataset(X, y)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Define optimizers
    optimizers = {
        'Adam': lambda params: torch.optim.Adam(params, lr=0.001),
        'StableAdam': lambda params: StableAdam(params, lr=0.001, temporal_stability=0.01),
        'StableAdam_NoEnergy': lambda params: StableAdam(params, lr=0.001, energy_conservation=False),
        'SGD': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
        'StableSGD': lambda params: StableSGD(params, lr=0.01, momentum=0.9, temporal_stability=0.01),
    }
    
    # Model factory
    def model_factory():
        return ClassificationModel()
    
    # Run comparison
    benchmark = compare_optimizers(
        model_factory=model_factory,
        optimizers=optimizers,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=nn.CrossEntropyLoss(),
        epochs=50
    )
    
    print("\nClassification Results:")
    print(benchmark.generate_report())
    
    # Save plots
    benchmark.plot_comparison(save_path="benchmarks/classification_comparison.png")
    
    return benchmark


def run_regression_benchmark():
    """Run regression benchmark comparing optimizers."""
    print("\nRunning Regression Benchmark...")
    print("=" * 50)
    
    # Create data
    X, y = create_synthetic_regression_data()
    dataset = TensorDataset(X, y)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Define optimizers
    optimizers = {
        'Adam': lambda params: torch.optim.Adam(params, lr=0.001),
        'StableAdam': lambda params: StableAdam(params, lr=0.001, temporal_stability=0.01),
        'SGD': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
        'StableSGD': lambda params: StableSGD(params, lr=0.01, momentum=0.9, temporal_stability=0.01),
    }
    
    # Model factory
    def model_factory():
        return RegressionModel()
    
    # Run comparison
    benchmark = compare_optimizers(
        model_factory=model_factory,
        optimizers=optimizers,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=nn.MSELoss(),
        epochs=100
    )
    
    print("\nRegression Results:")
    print(benchmark.generate_report())
    
    # Save plots
    benchmark.plot_comparison(save_path="benchmarks/regression_comparison.png")
    
    return benchmark


def run_temporal_stability_demo():
    """Demonstrate temporal stability benefits."""
    print("\nRunning Temporal Stability Demonstration...")
    print("=" * 50)
    
    # Create a challenging optimization landscape with noise
    def noisy_quadratic(x, noise_level=0.1):
        """Quadratic function with added noise to test stability."""
        base_loss = torch.sum((x - torch.tensor([1.0, -0.5, 0.8])) ** 2)
        noise = torch.randn_like(base_loss) * noise_level
        return base_loss + noise
    
    # Initialize parameters
    x1 = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    x2 = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    
    # Optimizers
    adam = torch.optim.Adam([x1], lr=0.1)
    stable_adam = StableAdam([x2], lr=0.1, temporal_stability=0.05)
    
    # Track optimization paths
    adam_path = []
    stable_adam_path = []
    adam_losses = []
    stable_adam_losses = []
    
    for i in range(200):
        # Standard Adam
        adam.zero_grad()
        loss1 = noisy_quadratic(x1)
        loss1.backward()
        adam.step()
        adam_path.append(x1.clone().detach().numpy())
        adam_losses.append(loss1.item())
        
        # Stable Adam
        stable_adam.zero_grad()
        loss2 = noisy_quadratic(x2)
        loss2.backward()
        stable_adam.step()
        stable_adam_path.append(x2.clone().detach().numpy())
        stable_adam_losses.append(loss2.item())
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(adam_losses, label='Adam', alpha=0.7)
    plt.plot(stable_adam_losses, label='StableAdam', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Evolution (Noisy Objective)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    adam_path = np.array(adam_path)
    stable_adam_path = np.array(stable_adam_path)
    
    plt.plot(adam_path[:, 0], adam_path[:, 1], 'b-', label='Adam', alpha=0.7)
    plt.plot(stable_adam_path[:, 0], stable_adam_path[:, 1], 'r-', label='StableAdam', alpha=0.7)
    plt.scatter([1.0], [-0.5], color='green', s=100, marker='*', label='True Optimum')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Optimization Path (First 2 Dimensions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("benchmarks/temporal_stability_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compute stability metrics
    adam_var = np.var(adam_losses[-50:])  # Variance in final 50 iterations
    stable_adam_var = np.var(stable_adam_losses[-50:])
    
    print(f"Final loss variance (last 50 iterations):")
    print(f"  Adam: {adam_var:.6f}")
    print(f"  StableAdam: {stable_adam_var:.6f}")
    print(f"  Stability improvement: {adam_var / stable_adam_var:.2f}x")
    
    return adam_losses, stable_adam_losses


def run_memory_and_speed_benchmark():
    """Benchmark memory usage and training speed."""
    print("\nRunning Memory and Speed Benchmark...")
    print("=" * 50)
    
    # Create larger model and dataset for meaningful benchmarks
    class LargerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Create synthetic data
    X = torch.randn(5000, 100)
    y = torch.randint(0, 10, (5000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    optimizers_to_test = {
        'Adam': torch.optim.Adam,
        'StableAdam': StableAdam,
        'SGD': torch.optim.SGD,
        'StableSGD': StableSGD
    }
    
    results = {}
    
    for opt_name, opt_class in optimizers_to_test.items():
        print(f"Testing {opt_name}...")
        
        # Create fresh model
        model = LargerModel()
        
        # Create optimizer
        if opt_name in ['Adam', 'StableAdam']:
            optimizer = opt_class(model.parameters(), lr=0.001)
        else:  # SGD variants
            optimizer = opt_class(model.parameters(), lr=0.01, momentum=0.9)
        
        criterion = nn.CrossEntropyLoss()
        
        # Measure training time
        start_time = time.time()
        
        # Initial memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # Training loop
        model.train()
        for epoch in range(3):  # Just a few epochs for speed testing
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Final memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage = (peak_memory - initial_memory) / 1024**2  # MB
        else:
            memory_usage = 0
        
        results[opt_name] = {
            'training_time': training_time,
            'memory_usage': memory_usage
        }
        
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Memory usage: {memory_usage:.2f}MB")
    
    # Compare results
    print(f"\nSpeed Comparison (relative to Adam):")
    adam_time = results['Adam']['training_time']
    for opt_name, result in results.items():
        if opt_name != 'Adam':
            relative_speed = result['training_time'] / adam_time
            print(f"  {opt_name}: {relative_speed:.2f}x")
    
    print(f"\nMemory Usage Comparison:")
    for opt_name, result in results.items():
        print(f"  {opt_name}: {result['memory_usage']:.2f}MB")
    
    return results


def main():
    """Run all benchmarks."""
    print("Temporal Optimizer Benchmark Suite")
    print("=" * 50)
    
    # Create benchmarks directory if it doesn't exist
    os.makedirs("benchmarks", exist_ok=True)
    
    # Run benchmarks
    classification_results = run_classification_benchmark()
    regression_results = run_regression_benchmark()
    stability_demo = run_temporal_stability_demo()
    speed_results = run_memory_and_speed_benchmark()
    
    print("\n" + "=" * 50)
    print("All benchmarks completed!")
    print("Check the 'benchmarks/' directory for generated plots and reports.")


if __name__ == "__main__":
    main()