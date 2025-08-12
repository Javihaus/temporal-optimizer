"""
Temporal stability metrics for analyzing optimization behavior.

This module provides metrics to quantify temporal stability properties
of neural network training, including parameter drift, convergence
stability, and long-term performance consistency.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from torch.nn import Module


def temporal_stability_metrics(
    parameter_history: Dict[str, List[torch.Tensor]],
    loss_history: List[float],
    window_size: int = 10
) -> Dict[str, float]:
    """
    Compute temporal stability metrics from training history.
    
    Args:
        parameter_history: Dict mapping parameter names to lists of parameter tensors over time
        loss_history: List of loss values over training steps
        window_size: Window size for computing stability metrics
    
    Returns:
        Dictionary containing various stability metrics
    
    Example:
        >>> # During training, collect parameter history
        >>> param_history = {'layer.weight': [param.clone() for param in weight_history]}
        >>> metrics = temporal_stability_metrics(param_history, loss_values)
        >>> print(f"Parameter drift: {metrics['parameter_drift']}")
    """
    metrics = {}
    
    # Parameter drift metrics
    if parameter_history:
        total_drift = 0.0
        max_drift = 0.0
        drift_variance = 0.0
        
        for param_name, param_list in parameter_history.items():
            if len(param_list) < 2:
                continue
            
            # Compute parameter changes over time
            param_diffs = []
            for i in range(1, len(param_list)):
                diff = torch.norm(param_list[i] - param_list[i-1]).item()
                param_diffs.append(diff)
                total_drift += diff
                max_drift = max(max_drift, diff)
            
            # Variance in parameter changes (stability indicator)
            if param_diffs:
                drift_variance += np.var(param_diffs)
        
        metrics['parameter_drift'] = total_drift / len(parameter_history) if parameter_history else 0.0
        metrics['max_parameter_drift'] = max_drift
        metrics['parameter_drift_variance'] = drift_variance / len(parameter_history) if parameter_history else 0.0
    
    # Loss stability metrics
    if len(loss_history) >= window_size:
        # Recent stability (coefficient of variation in recent window)
        recent_losses = loss_history[-window_size:]
        recent_mean = np.mean(recent_losses)
        recent_std = np.std(recent_losses)
        metrics['recent_loss_stability'] = 1.0 / (1.0 + recent_std / recent_mean) if recent_mean > 0 else 0.0
        
        # Overall convergence stability
        if len(loss_history) >= 2 * window_size:
            early_losses = loss_history[window_size:2*window_size]
            late_losses = loss_history[-window_size:]
            
            early_trend = np.polyfit(range(len(early_losses)), early_losses, 1)[0]
            late_trend = np.polyfit(range(len(late_losses)), late_losses, 1)[0]
            
            metrics['convergence_consistency'] = 1.0 / (1.0 + abs(early_trend - late_trend))
    
    # Loss plateau detection
    if len(loss_history) >= window_size:
        plateau_score = _detect_loss_plateau(loss_history, window_size)
        metrics['loss_plateau_score'] = plateau_score
    
    # Oscillation detection
    if len(loss_history) >= 3:
        oscillation_score = _detect_oscillations(loss_history)
        metrics['oscillation_score'] = oscillation_score
    
    return metrics


def convergence_metrics(
    loss_history: List[float],
    accuracy_history: Optional[List[float]] = None,
    target_loss: Optional[float] = None,
    target_accuracy: Optional[float] = None
) -> Dict[str, Union[float, int, bool]]:
    """
    Compute convergence-related metrics.
    
    Args:
        loss_history: List of loss values over training
        accuracy_history: Optional list of accuracy values
        target_loss: Target loss value for convergence detection
        target_accuracy: Target accuracy value for convergence detection
    
    Returns:
        Dictionary containing convergence metrics
    
    Example:
        >>> metrics = convergence_metrics(
        ...     loss_history, accuracy_history,
        ...     target_loss=0.1, target_accuracy=0.95
        ... )
        >>> print(f"Converged: {metrics['converged']}")
        >>> print(f"Convergence epoch: {metrics['convergence_epoch']}")
    """
    metrics = {}
    
    if not loss_history:
        return metrics
    
    # Basic convergence detection based on loss plateau
    convergence_epoch = _find_convergence_point(loss_history)
    metrics['convergence_epoch'] = convergence_epoch
    metrics['converged'] = convergence_epoch < len(loss_history)
    
    # Target-based convergence
    if target_loss is not None:
        target_reached_epoch = None
        for i, loss in enumerate(loss_history):
            if loss <= target_loss:
                target_reached_epoch = i
                break
        
        metrics['target_loss_reached'] = target_reached_epoch is not None
        metrics['target_loss_epoch'] = target_reached_epoch
    
    if accuracy_history and target_accuracy is not None:
        target_reached_epoch = None
        for i, acc in enumerate(accuracy_history):
            if acc >= target_accuracy:
                target_reached_epoch = i
                break
        
        metrics['target_accuracy_reached'] = target_reached_epoch is not None
        metrics['target_accuracy_epoch'] = target_reached_epoch
    
    # Convergence speed (rate of improvement)
    if len(loss_history) >= 2:
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        improvement = initial_loss - final_loss
        convergence_speed = improvement / len(loss_history)
        metrics['convergence_speed'] = convergence_speed
        metrics['total_improvement'] = improvement
    
    # Stability after convergence
    if convergence_epoch < len(loss_history) - 10:
        post_convergence_losses = loss_history[convergence_epoch:]
        post_convergence_stability = 1.0 / (1.0 + np.std(post_convergence_losses) / np.mean(post_convergence_losses))
        metrics['post_convergence_stability'] = post_convergence_stability
    
    return metrics


def parameter_evolution_analysis(
    model: Module,
    parameter_snapshots: List[Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze parameter evolution throughout training.
    
    Args:
        model: The neural network model
        parameter_snapshots: List of parameter state dictionaries over time
    
    Returns:
        Dictionary mapping parameter names to evolution metrics
    
    Example:
        >>> snapshots = []
        >>> for epoch in training_loop:
        ...     snapshots.append({name: param.clone() for name, param in model.named_parameters()})
        >>> analysis = parameter_evolution_analysis(model, snapshots)
    """
    if len(parameter_snapshots) < 2:
        return {}
    
    analysis = {}
    
    for param_name in parameter_snapshots[0].keys():
        param_evolution = [snapshot[param_name] for snapshot in parameter_snapshots]
        
        # Parameter trajectory metrics
        total_change = torch.norm(param_evolution[-1] - param_evolution[0]).item()
        max_single_change = 0.0
        changes = []
        
        for i in range(1, len(param_evolution)):
            change = torch.norm(param_evolution[i] - param_evolution[i-1]).item()
            changes.append(change)
            max_single_change = max(max_single_change, change)
        
        # Parameter stability metrics
        change_variance = np.var(changes) if changes else 0.0
        change_trend = np.polyfit(range(len(changes)), changes, 1)[0] if len(changes) > 1 else 0.0
        
        analysis[param_name] = {
            'total_change': total_change,
            'max_single_change': max_single_change,
            'average_change': np.mean(changes) if changes else 0.0,
            'change_variance': change_variance,
            'change_trend': change_trend,  # Negative means decreasing change over time (good)
            'stability_score': 1.0 / (1.0 + change_variance) if change_variance > 0 else 1.0
        }
    
    return analysis


def _detect_loss_plateau(loss_history: List[float], window_size: int = 10, threshold: float = 0.001) -> float:
    """Detect loss plateau (indicator of convergence)."""
    if len(loss_history) < window_size:
        return 0.0
    
    recent_losses = loss_history[-window_size:]
    loss_range = max(recent_losses) - min(recent_losses)
    
    # Score inversely related to loss range (lower range = higher plateau score)
    plateau_score = 1.0 / (1.0 + loss_range / threshold)
    return plateau_score


def _detect_oscillations(loss_history: List[float], window_size: int = 20) -> float:
    """Detect oscillatory behavior in loss history."""
    if len(loss_history) < window_size:
        return 0.0
    
    recent_losses = loss_history[-window_size:]
    
    # Count direction changes (sign changes in consecutive differences)
    direction_changes = 0
    for i in range(2, len(recent_losses)):
        diff1 = recent_losses[i-1] - recent_losses[i-2]
        diff2 = recent_losses[i] - recent_losses[i-1]
        
        if diff1 * diff2 < 0:  # Sign change
            direction_changes += 1
    
    # Normalize by window size
    oscillation_score = direction_changes / (window_size - 2) if window_size > 2 else 0.0
    return oscillation_score


def _find_convergence_point(
    loss_history: List[float],
    window_size: int = 10,
    threshold: float = 0.001
) -> int:
    """Find the point where loss converged (stopped improving significantly)."""
    if len(loss_history) < window_size * 2:
        return len(loss_history)
    
    for i in range(window_size, len(loss_history) - window_size):
        current_window = loss_history[i:i + window_size]
        previous_window = loss_history[i - window_size:i]
        
        current_mean = np.mean(current_window)
        previous_mean = np.mean(previous_window)
        
        # If improvement is below threshold, consider it converged
        improvement = previous_mean - current_mean
        if improvement < threshold:
            return i
    
    return len(loss_history)