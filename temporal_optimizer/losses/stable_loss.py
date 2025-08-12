"""
Temporal stability loss functions for enhanced training stability.

These loss functions can be combined with standard losses to improve
temporal stability and prevent parameter drift over time.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union


def temporal_stability_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    model: Optional[nn.Module] = None,
    base_loss: Union[nn.Module, str] = "cross_entropy",
    stability_weight: float = 0.01,
    previous_params: Optional[Dict[str, torch.Tensor]] = None
) -> torch.Tensor:
    """
    Compute loss with temporal stability regularization.
    
    This function combines a base loss with a temporal stability penalty
    that discourages rapid parameter changes, leading to more stable
    training dynamics.
    
    Args:
        outputs: Model predictions
        targets: Ground truth targets
        model: Model instance (optional, required for stability regularization)
        base_loss: Base loss function or string name (default: "cross_entropy")
        stability_weight: Weight for temporal stability penalty (default: 0.01)
        previous_params: Previous model parameters for stability computation
    
    Returns:
        Combined loss tensor
    
    Example:
        >>> loss = temporal_stability_loss(
        ...     outputs, targets, model,
        ...     base_loss="cross_entropy",
        ...     stability_weight=0.01
        ... )
    """
    # Compute base loss
    if isinstance(base_loss, str):
        if base_loss == "cross_entropy":
            base_loss_fn = nn.CrossEntropyLoss()
        elif base_loss == "mse":
            base_loss_fn = nn.MSELoss()
        elif base_loss == "bce":
            base_loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
    else:
        base_loss_fn = base_loss
    
    total_loss = base_loss_fn(outputs, targets)
    
    # Add temporal stability penalty if model and previous parameters are provided
    if model is not None and previous_params is not None and stability_weight > 0:
        stability_penalty = torch.tensor(0.0, device=outputs.device)
        
        current_params = dict(model.named_parameters())
        for name, current_param in current_params.items():
            if name in previous_params:
                param_diff = current_param - previous_params[name]
                stability_penalty += torch.sum(param_diff ** 2)
        
        total_loss += stability_weight * stability_penalty
    
    return total_loss


class TemporalStabilityLoss(nn.Module):
    """
    A loss module that combines base loss with temporal stability regularization.
    
    This class provides a stateful interface for temporal stability loss,
    automatically tracking previous parameters between calls.
    
    Args:
        base_loss: Base loss function or string name
        stability_weight: Weight for temporal stability penalty
        momentum: Exponential moving average factor for parameter tracking
    
    Example:
        >>> loss_fn = TemporalStabilityLoss("cross_entropy", stability_weight=0.01)
        >>> loss = loss_fn(outputs, targets, model)
    """
    
    def __init__(
        self,
        base_loss: Union[nn.Module, str] = "cross_entropy",
        stability_weight: float = 0.01,
        momentum: float = 0.9
    ):
        super(TemporalStabilityLoss, self).__init__()
        
        self.stability_weight = stability_weight
        self.momentum = momentum
        self._previous_params: Optional[Dict[str, torch.Tensor]] = None
        
        # Initialize base loss function
        if isinstance(base_loss, str):
            if base_loss == "cross_entropy":
                self.base_loss = nn.CrossEntropyLoss()
            elif base_loss == "mse":
                self.base_loss = nn.MSELoss()
            elif base_loss == "bce":
                self.base_loss = nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Unknown base loss: {base_loss}")
        else:
            self.base_loss = base_loss
    
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """Compute temporal stability loss."""
        # Base loss
        total_loss = self.base_loss(outputs, targets)
        
        # Temporal stability penalty
        if self.stability_weight > 0 and model is not None:
            current_params = {name: param.clone().detach() 
                             for name, param in model.named_parameters() 
                             if param.requires_grad}
            
            if self._previous_params is not None:
                stability_penalty = torch.tensor(0.0, device=outputs.device)
                
                for name, current_param in current_params.items():
                    if name in self._previous_params:
                        param_diff = current_param - self._previous_params[name]
                        stability_penalty += torch.sum(param_diff ** 2)
                
                total_loss += self.stability_weight * stability_penalty
            
            # Update previous parameters with momentum
            if self._previous_params is None:
                self._previous_params = current_params
            else:
                for name, current_param in current_params.items():
                    if name in self._previous_params:
                        self._previous_params[name] = (
                            self.momentum * self._previous_params[name] +
                            (1 - self.momentum) * current_param
                        )
                    else:
                        self._previous_params[name] = current_param
        
        return total_loss
    
    def reset_state(self):
        """Reset the internal parameter state."""
        self._previous_params = None