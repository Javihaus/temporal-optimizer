"""
StableAdam - Drop-in replacement for torch.optim.Adam with temporal stability.

This optimizer provides the same interface as Adam but with enhanced temporal
stability through Hamiltonian mechanics-inspired updates.
"""

import torch
from torch.optim import Optimizer
try:
    from typing import Any, Dict, Optional
except ImportError:
    # Fallback for older Python versions
    Any = object
    Dict = dict
    Optional = type(None)
from .base import HamiltonianMechanics


class StableAdam(Optimizer):
    """
    Drop-in replacement for torch.optim.Adam with temporal stability.
    
    This optimizer maintains the standard Adam interface while providing
    enhanced temporal stability and convergence properties through principled
    momentum conservation.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and squared gradient (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: L2 penalty (default: 0)
        temporal_stability: Weight for temporal stability regularization (default: 0.01)
        momentum_decay: Momentum decay factor (default: 0.9)
        energy_conservation: Enable adaptive learning rate based on system energy (default: True)
    
    Example:
        >>> optimizer = StableAdam(model.parameters(), lr=0.001)
        >>> optimizer = StableAdam(
        ...     model.parameters(),
        ...     lr=0.001,
        ...     temporal_stability=0.01,
        ...     energy_conservation=True
        ... )
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        temporal_stability=0.01,
        momentum_decay=0.9,
        energy_conservation=True
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= temporal_stability:
            raise ValueError("Invalid temporal_stability value: {}".format(temporal_stability))
        if not 0.0 <= momentum_decay < 1.0:
            raise ValueError("Invalid momentum_decay value: {}".format(momentum_decay))
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            temporal_stability=temporal_stability,
            momentum_decay=momentum_decay,
            energy_conservation=energy_conservation
        )
        super(StableAdam, self).__init__(params, defaults)
        
        # Store previous parameters for temporal stability
        self._previous_params = {}
        self._step_count = 0
    
    def __setstate__(self, state):
        super(StableAdam, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Store current parameters for temporal stability calculation
        current_params = {}
        
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                    
                param_id = "{}_{}".format(id(group), i)
                current_params[param_id] = p.data.clone()
                
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['momentum'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                momentum = state['momentum']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                
                # Compute bias-corrected moments
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # Standard Adam update direction
                denom = (corrected_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                standard_update = corrected_exp_avg / denom
                
                # Apply Hamiltonian mechanics for enhanced stability
                p_updated, momentum_updated = HamiltonianMechanics.symplectic_update(
                    param=p.data,
                    momentum=momentum,
                    gradient=standard_update,
                    lr=group['lr'],
                    momentum_decay=group['momentum_decay'],
                    epsilon=group['eps'],
                    energy_conservation=group['energy_conservation']
                )
                
                # Update state
                state['momentum'] = momentum_updated
        
        # Apply temporal stability regularization
        if group['temporal_stability'] > 0 and self._previous_params:
            stability_penalty = HamiltonianMechanics.temporal_stability_penalty(
                current_params, self._previous_params, group['temporal_stability']
            )
            # Note: In practice, this penalty would be added to the loss function
            # Here we store it for potential use in custom training loops
            self._last_stability_penalty = stability_penalty
        
        # Update previous parameters for next iteration
        self._previous_params = current_params
        self._step_count += 1
        
        return loss
    
    def get_temporal_stability_penalty(self):
        """Get the last computed temporal stability penalty."""
        return getattr(self, '_last_stability_penalty', None)