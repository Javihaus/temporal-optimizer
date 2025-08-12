"""
Performance comparison tools for temporal optimizers.

This module provides utilities to benchmark temporal optimizers against
standard PyTorch optimizers on various tasks.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class OptimizationBenchmark:
    """
    Comprehensive benchmarking suite for comparing optimizers.
    
    This class provides tools to compare temporal optimizers against
    standard PyTorch optimizers across multiple metrics including
    convergence speed, final performance, and temporal stability.
    """
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def run_comparison(
        self,
        model_factory: Callable[[], nn.Module],
        optimizers: Dict[str, Callable[[Any], torch.optim.Optimizer]],
        train_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: nn.Module,
        epochs: int = 50,
        early_stopping: bool = True,
        patience: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run comprehensive comparison between optimizers.
        
        Args:
            model_factory: Function that creates a fresh model instance
            optimizers: Dict mapping optimizer names to factory functions
            train_loader: Training data loader
            test_loader: Test data loader
            loss_fn: Loss function
            epochs: Maximum number of epochs
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
        
        Returns:
            Dictionary containing comparison results
        """
        results = {}
        
        for opt_name, opt_factory in optimizers.items():
            print(f"Benchmarking {opt_name}...")
            
            # Create fresh model and optimizer
            model = model_factory().to(self.device)
            optimizer = opt_factory(model.parameters())
            
            # Run training
            result = self._train_and_evaluate(
                model, optimizer, train_loader, test_loader, loss_fn,
                epochs, early_stopping, patience, opt_name
            )
            
            results[opt_name] = result
        
        self.results = results
        return results
    
    def _train_and_evaluate(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: nn.Module,
        epochs: int,
        early_stopping: bool,
        patience: int,
        opt_name: str
    ) -> Dict[str, Any]:
        """Train model and collect performance metrics."""
        
        # Metrics storage
        train_losses = []
        test_losses = []
        test_accuracies = []
        training_times = []
        memory_usage = []
        parameter_norms = []
        
        best_test_loss = float('inf')
        patience_counter = 0
        
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Evaluation phase
            model.eval()
            epoch_test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                    epoch_test_loss += loss.item()
                    
                    # Accuracy calculation
                    if hasattr(outputs, 'argmax'):  # Classification
                        predicted = outputs.argmax(dim=1)
                        correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            train_losses.append(epoch_train_loss / len(train_loader))
            test_losses.append(epoch_test_loss / len(test_loader))
            test_accuracies.append(correct / total if total > 0 else 0.0)
            training_times.append(epoch_time)
            
            # Memory usage (approximate)
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
            else:
                memory_usage.append(0)
            
            # Parameter norm for stability analysis
            param_norm = sum(p.norm().item() for p in model.parameters())
            parameter_norms.append(param_norm)
            
            # Early stopping
            if early_stopping:
                if epoch_test_loss < best_test_loss:
                    best_test_loss = epoch_test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch + 1}")
                        break
        
        total_training_time = time.time() - total_start_time
        
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'training_times': training_times,
            'memory_usage': memory_usage,
            'parameter_norms': parameter_norms,
            'total_training_time': total_training_time,
            'final_test_loss': test_losses[-1],
            'final_test_accuracy': test_accuracies[-1],
            'best_test_loss': min(test_losses),
            'best_test_accuracy': max(test_accuracies),
            'epochs_trained': len(train_losses),
            'convergence_epoch': self._find_convergence_epoch(test_losses),
            'stability_score': self._compute_stability_score(parameter_norms)
        }
    
    def _find_convergence_epoch(self, losses: List[float], window: int = 5) -> int:
        """Find the epoch where the model converged (loss plateau)."""
        if len(losses) < window:
            return len(losses)
        
        for i in range(window, len(losses)):
            recent_losses = losses[i-window:i]
            if max(recent_losses) - min(recent_losses) < 0.01:
                return i - window + 1
        
        return len(losses)
    
    def _compute_stability_score(self, parameter_norms: List[float]) -> float:
        """Compute stability score based on parameter norm variance."""
        if len(parameter_norms) < 2:
            return 0.0
        
        norm_array = np.array(parameter_norms)
        # Lower coefficient of variation indicates higher stability
        return 1.0 / (1.0 + np.std(norm_array) / np.mean(norm_array))
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """Generate comparison plots."""
        if not self.results:
            raise ValueError("No results to plot. Run comparison first.")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Optimizer Comparison', fontsize=16)
        
        # Training loss
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Test loss
        axes[0, 1].set_title('Test Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        
        # Test accuracy
        axes[0, 2].set_title('Test Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        
        # Training time per epoch
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        
        # Memory usage
        axes[1, 1].set_title('Memory Usage')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Memory (MB)')
        
        # Parameter stability
        axes[1, 2].set_title('Parameter Norm Evolution')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Parameter Norm')
        
        for opt_name, result in self.results.items():
            epochs = range(1, len(result['train_losses']) + 1)
            
            axes[0, 0].plot(epochs, result['train_losses'], label=opt_name, marker='o', markersize=3)
            axes[0, 1].plot(epochs, result['test_losses'], label=opt_name, marker='o', markersize=3)
            axes[0, 2].plot(epochs, result['test_accuracies'], label=opt_name, marker='o', markersize=3)
            axes[1, 0].plot(epochs, result['training_times'], label=opt_name, marker='o', markersize=3)
            axes[1, 1].plot(epochs, result['memory_usage'], label=opt_name, marker='o', markersize=3)
            axes[1, 2].plot(epochs, result['parameter_norms'], label=opt_name, marker='o', markersize=3)
        
        for ax in axes.flat:
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a text report summarizing the comparison."""
        if not self.results:
            return "No results available. Run comparison first."
        
        report = ["Optimizer Comparison Report", "=" * 30, ""]
        
        # Summary table
        headers = ["Optimizer", "Final Test Loss", "Best Test Acc", "Training Time (s)", "Stability Score", "Convergence Epoch"]
        report.append(f"{'Optimizer':<15} {'Final Loss':<12} {'Best Acc':<10} {'Train Time':<12} {'Stability':<10} {'Convergence':<12}")
        report.append("-" * 85)
        
        for opt_name, result in self.results.items():
            report.append(
                f"{opt_name:<15} {result['final_test_loss']:<12.4f} {result['best_test_accuracy']:<10.4f} "
                f"{result['total_training_time']:<12.2f} {result['stability_score']:<10.4f} {result['convergence_epoch']:<12}"
            )
        
        report.append("")
        
        # Recommendations
        best_loss = min(result['final_test_loss'] for result in self.results.values())
        best_accuracy = max(result['best_test_accuracy'] for result in self.results.values())
        best_stability = max(result['stability_score'] for result in self.results.values())
        fastest_time = min(result['total_training_time'] for result in self.results.values())
        
        report.append("Recommendations:")
        for opt_name, result in self.results.items():
            recommendations = []
            if result['final_test_loss'] == best_loss:
                recommendations.append("lowest final loss")
            if result['best_test_accuracy'] == best_accuracy:
                recommendations.append("highest accuracy")
            if result['stability_score'] == best_stability:
                recommendations.append("most stable")
            if result['total_training_time'] == fastest_time:
                recommendations.append("fastest training")
            
            if recommendations:
                report.append(f"- {opt_name}: {', '.join(recommendations)}")
        
        return "\n".join(report)


def compare_optimizers(
    model_factory: Callable[[], nn.Module],
    optimizers: Dict[str, Callable[[Any], torch.optim.Optimizer]],
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int = 50,
    device: str = "auto"
) -> OptimizationBenchmark:
    """
    Convenience function for quick optimizer comparison.
    
    Args:
        model_factory: Function that creates a fresh model instance
        optimizers: Dict mapping optimizer names to factory functions
        train_loader: Training data loader
        test_loader: Test data loader
        loss_fn: Loss function
        epochs: Number of epochs to train
        device: Device to use for training
    
    Returns:
        OptimizationBenchmark instance with results
    
    Example:
        >>> def create_model():
        ...     return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        >>> 
        >>> optimizers = {
        ...     'Adam': lambda params: torch.optim.Adam(params, lr=0.001),
        ...     'StableAdam': lambda params: StableAdam(params, lr=0.001)
        ... }
        >>> 
        >>> benchmark = compare_optimizers(
        ...     create_model, optimizers, train_loader, test_loader, nn.CrossEntropyLoss()
        ... )
        >>> print(benchmark.generate_report())
    """
    benchmark = OptimizationBenchmark(device)
    benchmark.run_comparison(
        model_factory, optimizers, train_loader, test_loader, loss_fn, epochs
    )
    return benchmark