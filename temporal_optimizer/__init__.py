"""
Temporal Optimizer - Drop-in replacements for PyTorch optimizers with temporal stability.

This package provides optimizers that maintain temporal stability in neural network
training through principled momentum conservation and adaptive learning rates.
"""

from .optimizers.stable_adam import StableAdam
from .optimizers.stable_sgd import StableSGD
from .losses.stable_loss import temporal_stability_loss

__version__ = "0.1.0"
__all__ = ["StableAdam", "StableSGD", "temporal_stability_loss"]