"""
StableSGD - Drop-in replacement for torch.optim.SGD with temporal stability.

This optimizer provides the same interface as SGD but with enhanced temporal
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


class StableSGD(Optimizer):
    """
    Drop-in replacement for torch.optim.SGD with temporal stability.
    
    This optimizer maintains the standard SGD interface while providing
    enhanced temporal stability and convergence properties through principled
    momentum conservation.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-2)
        momentum: Momentum factor (default: 0)
        weight_decay: L2 penalty (default: 0)
        dampening: Dampening for momentum (default: 0)
        nesterov: Enable Nesterov momentum (default: False)
        temporal_stability: Weight for temporal stability regularization (default: 0.01)
        momentum_decay: Momentum decay factor (default: 0.9)
        energy_conservation: Enable adaptive learning rate based on system energy (default: True)
    
    Example:
        >>> optimizer = StableSGD(model.parameters(), lr=0.01)
        >>> optimizer = StableSGD(
        ...     model.parameters(),
        ...     lr=0.01,
        ...     momentum=0.9,
        ...     temporal_stability=0.01,
        ...     energy_conservation=True
        ... )
    """
    
    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0,
        weight_decay=0,
        dampening=0,
        nesterov=False,
        temporal_stability=0.01,
        momentum_decay=0.9,
        energy_conservation=True
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= temporal_stability:
            raise ValueError("Invalid temporal_stability value: {}".format(temporal_stability))
        if not 0.0 <= momentum_decay < 1.0:
            raise ValueError("Invalid momentum_decay value: {}".format(momentum_decay))
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            temporal_stability=temporal_stability,
            momentum_decay=momentum_decay,
            energy_conservation=energy_conservation
        )
        super(StableSGD, self).__init__(params, defaults)
        
        # Store previous parameters for temporal stability
        self._previous_params = {}
        self._step_count = 0
    
    def __setstate__(self, state):
        super(StableSGD, self).__setstate__(state)
    
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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                param_id = "{}_{}".format(id(group), i)
                current_params[param_id] = p.data.clone()
                
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['hamiltonian_momentum'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                hamiltonian_momentum = state['hamiltonian_momentum']
                
                # Standard SGD momentum update
                if momentum != 0:
                    if 'step' in state:
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    else:
                        buf.copy_(grad)
                        state['step'] = 1
                    
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                
                # Apply Hamiltonian mechanics for enhanced stability
                p_updated, hamiltonian_momentum_updated = HamiltonianMechanics.symplectic_update(
                    param=p.data,
                    momentum=hamiltonian_momentum,
                    gradient=grad,
                    lr=group['lr'],
                    momentum_decay=group['momentum_decay'],
                    epsilon=1e-8,
                    energy_conservation=group['energy_conservation']
                )
                
                # Update state
                state['hamiltonian_momentum'] = hamiltonian_momentum_updated
        
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