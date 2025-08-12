"""
Credit scoring benchmark reproducing the original paper results.

This script demonstrates the temporal stability benefits of StableAdam/StableSGD
on a credit scoring task with temporal drift, similar to the original research.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to path to import temporal_optimizer
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from temporal_optimizer import StableAdam, StableSGD


def create_synthetic_credit_data(n_samples=5000, n_features=20, temporal_drift=True):
    """
    Create synthetic credit scoring dataset with optional temporal drift.
    
    This simulates the challenges in credit scoring where customer behavior
    and market conditions change over time, causing model performance degradation.
    """
    np.random.seed(42)
    
    # Generate base features
    X = np.random.randn(n_samples, n_features)
    
    # Create meaningful feature relationships
    # Income-related features
    X[:, 0] = np.abs(X[:, 0]) * 50000 + 30000  # Annual income
    X[:, 1] = X[:, 0] * 0.3 + np.random.randn(n_samples) * 5000  # Credit limit
    
    # Risk factors
    X[:, 2] = np.random.poisson(2, n_samples)  # Number of late payments
    X[:, 3] = np.random.uniform(0, 1, n_samples)  # Debt-to-income ratio
    X[:, 4] = np.random.uniform(18, 80, n_samples)  # Age
    
    # Create time component for temporal drift
    time_component = np.linspace(0, 1, n_samples)
    
    # Define true relationship (credit score depends on income, debt ratio, age, etc.)
    credit_score = (
        0.3 * (X[:, 0] / 50000) +  # Income factor
        -0.4 * X[:, 3] +  # Debt-to-income (negative)
        -0.2 * (X[:, 2] / 5) +  # Late payments (negative)
        0.1 * (X[:, 4] / 80) +  # Age (positive)
        0.1 * np.random.randn(n_samples)  # Noise
    )
    
    if temporal_drift:
        # Add temporal drift: relationships change over time
        drift_factor = 0.5 * time_component
        credit_score += drift_factor * (
            -0.2 * (X[:, 0] / 50000) +  # Income becomes less predictive
            0.1 * X[:, 3]  # Debt ratio impact changes
        )
    
    # Convert to binary classification (good/bad credit)
    y = (credit_score > np.median(credit_score)).astype(int)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return torch.FloatTensor(X), torch.LongTensor(y)


class CreditScoringModel(nn.Module):
    """Neural network model for credit scoring."""
    
    def __init__(self, input_dim=20, hidden_dims=None, dropout_rate=0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def evaluate_model(model, data_loader, criterion, device='cpu'):
    """Evaluate model performance."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    # ROC AUC (handle potential issues)
    try:
        auc = roc_auc_score(all_targets, all_predictions)
    except ValueError:
        auc = 0.5
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def train_model(model, optimizer, train_loader, test_loader, criterion, epochs=100, device='cpu'):
    """Train model and track performance over time."""
    model.to(device)
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_f1': [],
        'test_auc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Evaluation phase
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(epoch_train_loss / len(train_loader))
        history['test_loss'].append(test_metrics['loss'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics['f1'])
        history['test_auc'].append(test_metrics['auc'])
        
        # Print progress
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Test Acc: {history['test_accuracy'][-1]:.4f}, "
                  f"Test F1: {history['test_f1'][-1]:.4f}")
    
    return history


def run_temporal_stability_experiment():
    """
    Run the main temporal stability experiment.
    
    This experiment compares standard optimizers vs temporal optimizers
    on credit scoring data with temporal drift.
    """
    print("Credit Scoring Temporal Stability Experiment")
    print("=" * 60)
    
    # Create datasets with and without temporal drift
    print("Creating synthetic credit scoring datasets...")
    
    # Dataset with temporal drift (challenging)
    X_drift, y_drift = create_synthetic_credit_data(n_samples=4000, temporal_drift=True)
    
    # Dataset without temporal drift (baseline)
    X_stable, y_stable = create_synthetic_credit_data(n_samples=4000, temporal_drift=False)
    
    # Split data (maintaining temporal order for drift dataset)
    datasets = {}
    
    for name, (X, y) in [('with_drift', (X_drift, y_drift)), ('without_drift', (X_stable, y_stable))]:
        # For temporal drift, split sequentially (earlier data for training)
        if name == 'with_drift':
            split_idx = int(0.7 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            # For stable data, random split is fine
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        datasets[name] = (train_loader, test_loader)
    
    # Define optimizers to compare
    optimizers = {
        'Adam': lambda params: torch.optim.Adam(params, lr=0.001),
        'StableAdam': lambda params: StableAdam(params, lr=0.001, temporal_stability=0.02),
        'SGD': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
        'StableSGD': lambda params: StableSGD(params, lr=0.01, momentum=0.9, temporal_stability=0.02)
    }
    
    results = {}
    
    # Run experiments
    for dataset_name, (train_loader, test_loader) in datasets.items():
        print(f"\nTesting on dataset: {dataset_name}")
        print("-" * 40)
        
        results[dataset_name] = {}
        
        for opt_name, opt_factory in optimizers.items():
            print(f"\nOptimizer: {opt_name}")
            
            # Create fresh model
            model = CreditScoringModel(input_dim=20)
            optimizer = opt_factory(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            history = train_model(model, optimizer, train_loader, test_loader, criterion, epochs=80)
            
            # Store results
            results[dataset_name][opt_name] = history
    
    return results


def plot_results(results):
    """Generate comprehensive plots of the results."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Credit Scoring: Temporal Stability Comparison', fontsize=16)
    
    colors = {'Adam': 'blue', 'StableAdam': 'red', 'SGD': 'green', 'StableSGD': 'orange'}
    
    for col, dataset_name in enumerate(['without_drift', 'with_drift']):
        dataset_results = results[dataset_name]
        
        # Test accuracy over time
        axes[0, col].set_title(f'Test Accuracy - {dataset_name.replace("_", " ").title()}')
        axes[0, col].set_xlabel('Epoch')
        axes[0, col].set_ylabel('Accuracy')
        
        # Test F1 score over time  
        axes[1, col].set_title(f'Test F1 Score - {dataset_name.replace("_", " ").title()}')
        axes[1, col].set_xlabel('Epoch')
        axes[1, col].set_ylabel('F1 Score')
        
        # Test loss over time
        axes[2, col].set_title(f'Test Loss - {dataset_name.replace("_", " ").title()}')
        axes[2, col].set_xlabel('Epoch')
        axes[2, col].set_ylabel('Loss')
        
        for opt_name, history in dataset_results.items():
            epochs = range(len(history['test_accuracy']))
            
            axes[0, col].plot(epochs, history['test_accuracy'], 
                            label=opt_name, color=colors[opt_name], linewidth=2)
            axes[1, col].plot(epochs, history['test_f1'], 
                            label=opt_name, color=colors[opt_name], linewidth=2)
            axes[2, col].plot(epochs, history['test_loss'], 
                            label=opt_name, color=colors[opt_name], linewidth=2)
    
    # Add legends and grid
    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmarks/credit_scoring_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_temporal_stability(results):
    """Analyze temporal stability metrics."""
    print("\nTemporal Stability Analysis")
    print("=" * 60)
    
    for dataset_name, dataset_results in results.items():
        print(f"\nDataset: {dataset_name.replace('_', ' ').title()}")
        print("-" * 30)
        
        stability_metrics = {}
        
        for opt_name, history in dataset_results.items():
            # Calculate stability metrics for the last 20 epochs
            final_accuracies = history['test_accuracy'][-20:]
            final_f1_scores = history['test_f1'][-20:]
            
            acc_stability = 1.0 / (1.0 + np.std(final_accuracies))
            f1_stability = 1.0 / (1.0 + np.std(final_f1_scores))
            
            final_performance = {
                'accuracy': np.mean(final_accuracies),
                'f1': np.mean(final_f1_scores),
                'accuracy_stability': acc_stability,
                'f1_stability': f1_stability,
                'best_accuracy': max(history['test_accuracy']),
                'best_f1': max(history['test_f1'])
            }
            
            stability_metrics[opt_name] = final_performance
            
            print(f"{opt_name:12} | Acc: {final_performance['accuracy']:.4f} "
                  f"| F1: {final_performance['f1']:.4f} "
                  f"| Acc Stability: {final_performance['accuracy_stability']:.4f} "
                  f"| F1 Stability: {final_performance['f1_stability']:.4f}")
        
        # Identify best performers
        best_accuracy = max(stability_metrics.values(), key=lambda x: x['accuracy'])
        best_stability = max(stability_metrics.values(), key=lambda x: x['accuracy_stability'])
        
        best_acc_optimizer = [opt for opt, metrics in stability_metrics.items() 
                             if metrics['accuracy'] == best_accuracy['accuracy']][0]
        best_stability_optimizer = [opt for opt, metrics in stability_metrics.items() 
                                   if metrics['accuracy_stability'] == best_stability['accuracy_stability']][0]
        
        print(f"\nBest accuracy: {best_acc_optimizer} ({best_accuracy['accuracy']:.4f})")
        print(f"Most stable: {best_stability_optimizer} (stability: {best_stability['accuracy_stability']:.4f})")


def generate_summary_report(results):
    """Generate a comprehensive summary report."""
    print("\n" + "=" * 60)
    print("CREDIT SCORING EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print("\nKey Findings:")
    print("-" * 15)
    
    # Compare performance on drift vs non-drift data
    drift_results = results['with_drift']
    stable_results = results['without_drift']
    
    # Calculate performance degradation due to temporal drift
    for opt_name in drift_results.keys():
        stable_acc = np.mean(stable_results[opt_name]['test_accuracy'][-10:])
        drift_acc = np.mean(drift_results[opt_name]['test_accuracy'][-10:])
        degradation = stable_acc - drift_acc
        
        print(f"{opt_name:12}: {degradation:.4f} accuracy loss due to temporal drift")
    
    print("\nTemporal Stability Benefits:")
    print("-" * 30)
    
    # Compare stability of temporal vs standard optimizers
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name.replace('_', ' ').title()}:")
        
        # Compare Adam vs StableAdam
        adam_stability = 1.0 / (1.0 + np.std(dataset_results['Adam']['test_accuracy'][-20:]))
        stable_adam_stability = 1.0 / (1.0 + np.std(dataset_results['StableAdam']['test_accuracy'][-20:]))
        
        improvement = stable_adam_stability / adam_stability
        print(f"  StableAdam vs Adam stability: {improvement:.2f}x improvement")
        
        # Compare SGD vs StableSGD
        sgd_stability = 1.0 / (1.0 + np.std(dataset_results['SGD']['test_accuracy'][-20:]))
        stable_sgd_stability = 1.0 / (1.0 + np.std(dataset_results['StableSGD']['test_accuracy'][-20:]))
        
        improvement = stable_sgd_stability / sgd_stability
        print(f"  StableSGD vs SGD stability: {improvement:.2f}x improvement")


def main():
    """Run the complete credit scoring reproduction experiment."""
    # Create benchmarks directory
    os.makedirs("benchmarks", exist_ok=True)
    
    # Run the main experiment
    results = run_temporal_stability_experiment()
    
    # Analyze and visualize results
    plot_results(results)
    analyze_temporal_stability(results)
    generate_summary_report(results)
    
    print(f"\nExperiment completed! Check 'benchmarks/credit_scoring_comparison.png' for visualizations.")
    
    return results


if __name__ == "__main__":
    main()