"""
Advanced usage examples for Temporal Optimizer.

This script demonstrates advanced features including custom loss functions,
temporal stability analysis, and integration with modern deep learning workflows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from temporal_optimizer import StableAdam, StableSGD
from temporal_optimizer.losses import TemporalStabilityLoss, temporal_stability_loss
from temporal_optimizer.utils import temporal_stability_metrics, convergence_metrics


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(out_features)
        self.batch_norm2 = nn.BatchNorm1d(out_features)
        
        # Skip connection
        if in_features != out_features:
            self.skip_connection = nn.Linear(in_features, out_features)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x):
        identity = self.skip_connection(x)
        
        out = self.linear1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.batch_norm2(out)
        
        out += identity  # Residual connection
        out = F.relu(out)
        
        return out


class DeepResidualNet(nn.Module):
    """Deep residual network for complex tasks."""
    
    def __init__(self, input_dim=50, num_classes=5, hidden_dim=128, num_blocks=4):
        super().__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.BatchNorm1d(hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.dropout(x)
        x = self.output_layer(x)
        
        return x


def create_complex_dataset(n_samples=2000, n_features=50, n_classes=5, noise_level=0.2):
    """Create a more complex synthetic dataset."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate features with complex interactions
    X = torch.randn(n_samples, n_features)
    
    # Create non-linear relationships
    feature_interactions = torch.randn(n_features, n_features) * 0.1
    X_interaction = X @ feature_interactions
    
    # Complex target generation
    weights = torch.randn(n_features, n_classes)
    interaction_weights = torch.randn(n_features, n_classes) * 0.3
    
    logits = X @ weights + X_interaction @ interaction_weights
    
    # Add noise
    logits += torch.randn_like(logits) * noise_level
    
    y = torch.argmax(logits, dim=1)
    
    return X, y


def example_1_temporal_stability_loss():
    """Example 1: Using TemporalStabilityLoss for automatic parameter tracking."""
    print("Example 1: Temporal Stability Loss")
    print("=" * 40)
    
    # Create complex dataset
    X, y = create_complex_dataset()
    dataset = TensorDataset(X, y)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create deep model
    model = DeepResidualNet(input_dim=50, num_classes=5)
    
    # Use StableAdam optimizer
    optimizer = StableAdam(model.parameters(), lr=0.001, temporal_stability=0.01)
    
    # Use TemporalStabilityLoss instead of regular loss
    criterion = TemporalStabilityLoss(
        base_loss="cross_entropy",
        stability_weight=0.02,  # Higher weight for more stability
        momentum=0.9  # Exponential moving average for parameter tracking
    )
    
    print("Training with TemporalStabilityLoss...")
    
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(40):
        # Training
        model.train()
        epoch_train_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # TemporalStabilityLoss automatically tracks parameters
            loss = criterion(outputs, batch_y, model)
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Evaluation
        model.eval()
        epoch_test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y, model)
                
                epoch_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # Record history
        history['train_loss'].append(epoch_train_loss / len(train_loader))
        history['test_loss'].append(epoch_test_loss / len(test_loader))
        history['test_acc'].append(100 * correct / total)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d}: Test Acc: {history["test_acc"][-1]:.2f}%, '
                  f'Test Loss: {history["test_loss"][-1]:.4f}')
    
    print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")
    print("✅ Temporal stability loss example successful!\n")
    
    return history


def example_2_parameter_evolution_analysis():
    """Example 2: Analyzing parameter evolution during training."""
    print("Example 2: Parameter Evolution Analysis")
    print("=" * 40)
    
    # Create dataset
    X, y = create_complex_dataset(n_samples=1000)
    dataset = TensorDataset(X, y)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    model = DeepResidualNet(input_dim=50, num_classes=5, num_blocks=2)
    optimizer = StableAdam(model.parameters(), lr=0.001, temporal_stability=0.02)
    criterion = nn.CrossEntropyLoss()
    
    print("Training and tracking parameter evolution...")
    
    # Track parameters and metrics
    parameter_history = {name: [] for name, _ in model.named_parameters()}
    loss_history = []
    
    for epoch in range(30):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Store parameters and loss
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        for name, param in model.named_parameters():
            parameter_history[name].append(param.clone().detach())
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d}: Loss: {avg_loss:.4f}')
    
    # Analyze parameter evolution
    print("\nAnalyzing parameter evolution...")
    
    stability_metrics = temporal_stability_metrics(
        parameter_history, loss_history, window_size=5
    )
    
    convergence_info = convergence_metrics(loss_history)
    
    print(f"Parameter drift: {stability_metrics.get('parameter_drift', 0):.6f}")
    print(f"Loss plateau score: {stability_metrics.get('loss_plateau_score', 0):.4f}")
    print(f"Convergence epoch: {convergence_info.get('convergence_epoch', -1)}")
    print(f"Convergence speed: {convergence_info.get('convergence_speed', 0):.6f}")
    
    print("✅ Parameter evolution analysis successful!\n")
    
    return parameter_history, loss_history, stability_metrics


def example_3_mixed_precision_training():
    """Example 3: Mixed precision training with temporal optimizers."""
    print("Example 3: Mixed Precision Training")
    print("=" * 40)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, running on CPU (mixed precision disabled)")
    
    # Create dataset
    X, y = create_complex_dataset(n_samples=1500)
    dataset = TensorDataset(X, y)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Create model and move to device
    model = DeepResidualNet(input_dim=50, num_classes=5, num_blocks=3).to(device)
    optimizer = StableAdam(model.parameters(), lr=0.001, temporal_stability=0.015)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training (if CUDA available)
    if torch.cuda.is_available():
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        use_amp = True
        print("Using automatic mixed precision training...")
    else:
        use_amp = False
        print("Using standard precision training...")
    
    history = {'train_loss': [], 'test_acc': []}
    
    for epoch in range(25):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                # Mixed precision forward pass
                with autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                if use_amp:
                    with autocast():
                        outputs = model(batch_x)
                else:
                    outputs = model(batch_x)
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # Record metrics
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['test_acc'].append(100 * correct / total)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch:2d}: Test Acc: {history["test_acc"][-1]:.2f}%')
    
    print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")
    print("✅ Mixed precision training successful!\n")
    
    return history


def example_4_learning_rate_scheduling():
    """Example 4: Learning rate scheduling with temporal optimizers."""
    print("Example 4: Learning Rate Scheduling")
    print("=" * 40)
    
    # Create dataset
    X, y = create_complex_dataset()
    dataset = TensorDataset(X, y)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    model = DeepResidualNet(input_dim=50, num_classes=5)
    optimizer = StableAdam(model.parameters(), lr=0.01, temporal_stability=0.02)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    criterion = nn.CrossEntropyLoss()
    
    print("Training with cosine annealing learning rate schedule...")
    
    history = {'train_loss': [], 'test_acc': [], 'lr': []}
    
    for epoch in range(50):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Step scheduler
        scheduler.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # Record metrics
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['test_acc'].append(100 * correct / total)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d}: Test Acc: {history["test_acc"][-1]:.2f}%, '
                  f'LR: {history["lr"][-1]:.6f}')
    
    print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")
    print("✅ Learning rate scheduling successful!\n")
    
    return history


def example_5_multi_task_learning():
    """Example 5: Multi-task learning with different stability requirements."""
    print("Example 5: Multi-Task Learning")
    print("=" * 40)
    
    class MultiTaskModel(nn.Module):
        """Multi-task model with shared backbone and task-specific heads."""
        
        def __init__(self, input_dim=50, shared_dim=128):
            super().__init__()
            
            # Shared backbone
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                ResidualBlock(shared_dim, shared_dim),
                ResidualBlock(shared_dim, shared_dim),
            )
            
            # Task-specific heads
            self.classifier_head = nn.Linear(shared_dim, 5)  # Classification
            self.regression_head = nn.Linear(shared_dim, 1)  # Regression
        
        def forward(self, x):
            shared_features = self.backbone(x)
            
            classification_output = self.classifier_head(shared_features)
            regression_output = self.regression_head(shared_features)
            
            return classification_output, regression_output
    
    # Create multi-task dataset
    X, y_class = create_complex_dataset(n_samples=1500)
    
    # Create regression targets (correlated with classification)
    y_reg = torch.randn(len(X), 1) + y_class.float().unsqueeze(1) * 0.5
    
    dataset = TensorDataset(X, y_class, y_reg)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    model = MultiTaskModel()
    
    # Different optimization strategies for different parts
    optimizer = StableAdam([
        {
            'params': model.backbone.parameters(),
            'lr': 0.001,
            'temporal_stability': 0.03,  # High stability for shared features
        },
        {
            'params': model.classifier_head.parameters(),
            'lr': 0.002,
            'temporal_stability': 0.01,  # Lower stability for task-specific
        },
        {
            'params': model.regression_head.parameters(),
            'lr': 0.003,
            'temporal_stability': 0.005,  # Even lower for regression
        }
    ])
    
    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    
    print("Training multi-task model...")
    
    history = {'train_loss': [], 'class_acc': [], 'reg_loss': []}
    
    for epoch in range(35):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y_class, batch_y_reg in train_loader:
            optimizer.zero_grad()
            
            class_outputs, reg_outputs = model(batch_x)
            
            # Multi-task loss
            class_loss = classification_criterion(class_outputs, batch_y_class)
            reg_loss = regression_criterion(reg_outputs, batch_y_reg)
            
            # Weighted combination
            total_loss = class_loss + 0.5 * reg_loss
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        
        # Evaluation
        model.eval()
        class_correct = 0
        total = 0
        reg_test_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y_class, batch_y_reg in test_loader:
                class_outputs, reg_outputs = model(batch_x)
                
                # Classification accuracy
                _, predicted = torch.max(class_outputs.data, 1)
                total += batch_y_class.size(0)
                class_correct += (predicted == batch_y_class).sum().item()
                
                # Regression loss
                reg_test_loss += regression_criterion(reg_outputs, batch_y_reg).item()
        
        # Record metrics
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['class_acc'].append(100 * class_correct / total)
        history['reg_loss'].append(reg_test_loss / len(test_loader))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d}: Class Acc: {history["class_acc"][-1]:.2f}%, '
                  f'Reg Loss: {history["reg_loss"][-1]:.4f}')
    
    print(f"Final classification accuracy: {history['class_acc'][-1]:.2f}%")
    print(f"Final regression loss: {history['reg_loss'][-1]:.4f}")
    print("✅ Multi-task learning successful!\n")
    
    return history


def plot_advanced_results(histories: Dict[str, Dict]):
    """Plot results from advanced examples."""
    plt.figure(figsize=(15, 10))
    
    # Example 1: Temporal stability loss
    if 'temporal_loss' in histories:
        plt.subplot(2, 3, 1)
        history = histories['temporal_loss']
        plt.plot(history['test_acc'], 'b-', label='Test Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Temporal Stability Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Example 3: Mixed precision
    if 'mixed_precision' in histories:
        plt.subplot(2, 3, 2)
        history = histories['mixed_precision']
        plt.plot(history['test_acc'], 'g-', label='Test Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Mixed Precision Training')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Example 4: Learning rate scheduling
    if 'lr_schedule' in histories:
        plt.subplot(2, 3, 3)
        history = histories['lr_schedule']
        plt.plot(history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('LR Scheduling')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Learning rate subplot
        plt.subplot(2, 3, 6)
        plt.plot(history['lr'], 'r--', label='Learning Rate', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Example 5: Multi-task learning
    if 'multi_task' in histories:
        plt.subplot(2, 3, 4)
        history = histories['multi_task']
        plt.plot(history['class_acc'], 'purple', label='Classification Acc', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Multi-Task: Classification')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 3, 5)
        plt.plot(history['reg_loss'], 'orange', label='Regression Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Multi-Task: Regression')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('examples/advanced_usage_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run all advanced usage examples."""
    print("Temporal Optimizer - Advanced Usage Examples")
    print("=" * 60)
    print()
    
    histories = {}
    
    # Run examples
    histories['temporal_loss'] = example_1_temporal_stability_loss()
    param_history, loss_history, stability_metrics = example_2_parameter_evolution_analysis()
    histories['mixed_precision'] = example_3_mixed_precision_training()
    histories['lr_schedule'] = example_4_learning_rate_scheduling()
    histories['multi_task'] = example_5_multi_task_learning()
    
    # Plot results
    print("Generating advanced results plots...")
    plot_advanced_results(histories)
    
    print("🎉 All advanced usage examples completed successfully!")
    print("Check 'examples/advanced_usage_results.png' for visualizations.")


if __name__ == "__main__":
    main()