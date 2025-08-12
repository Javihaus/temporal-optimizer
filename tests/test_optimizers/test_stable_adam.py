"""Tests for StableAdam optimizer."""

import pytest
import torch
import torch.nn as nn
from temporal_optimizer import StableAdam


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestStableAdam:
    """Test suite for StableAdam optimizer."""
    
    def test_initialization_default_params(self):
        """Test optimizer initialization with default parameters."""
        model = SimpleModel()
        optimizer = StableAdam(model.parameters())
        
        assert len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]
        assert group['lr'] == 1e-3
        assert group['betas'] == (0.9, 0.999)
        assert group['eps'] == 1e-8
        assert group['weight_decay'] == 0
        assert group['temporal_stability'] == 0.01
        assert group['momentum_decay'] == 0.9
        assert group['energy_conservation'] is True
    
    def test_initialization_custom_params(self):
        """Test optimizer initialization with custom parameters."""
        model = SimpleModel()
        optimizer = StableAdam(
            model.parameters(),
            lr=0.001,
            betas=(0.8, 0.99),
            eps=1e-6,
            weight_decay=1e-4,
            temporal_stability=0.05,
            momentum_decay=0.95,
            energy_conservation=False
        )
        
        group = optimizer.param_groups[0]
        assert group['lr'] == 0.001
        assert group['betas'] == (0.8, 0.99)
        assert group['eps'] == 1e-6
        assert group['weight_decay'] == 1e-4
        assert group['temporal_stability'] == 0.05
        assert group['momentum_decay'] == 0.95
        assert group['energy_conservation'] is False
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        model = SimpleModel()
        
        with pytest.raises(ValueError, match="Invalid learning rate"):
            StableAdam(model.parameters(), lr=-0.1)
        
        with pytest.raises(ValueError, match="Invalid epsilon value"):
            StableAdam(model.parameters(), eps=-1e-8)
        
        with pytest.raises(ValueError, match="Invalid beta parameter"):
            StableAdam(model.parameters(), betas=(1.1, 0.999))
        
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            StableAdam(model.parameters(), weight_decay=-0.1)
        
        with pytest.raises(ValueError, match="Invalid temporal_stability"):
            StableAdam(model.parameters(), temporal_stability=-0.1)
        
        with pytest.raises(ValueError, match="Invalid momentum_decay"):
            StableAdam(model.parameters(), momentum_decay=1.1)
    
    def test_optimization_step(self):
        """Test basic optimization step."""
        model = SimpleModel()
        optimizer = StableAdam(model.parameters(), lr=0.01)
        
        # Create some dummy data
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        criterion = nn.CrossEntropyLoss()
        
        # Initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward pass and backward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Optimization step
        optimizer.step()
        
        # Check that parameters have been updated
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.equal(initial, current)
    
    def test_zero_grad(self):
        """Test gradient zeroing."""
        model = SimpleModel()
        optimizer = StableAdam(model.parameters())
        
        # Create gradients
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        criterion = nn.CrossEntropyLoss()
        
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Check gradients are zeroed
        for param in model.parameters():
            if param.grad is not None:
                assert torch.all(param.grad == 0)
    
    def test_state_initialization(self):
        """Test optimizer state initialization."""
        model = SimpleModel()
        optimizer = StableAdam(model.parameters())
        
        # Before any steps, state should be empty
        assert len(optimizer.state) == 0
        
        # Take one step
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        criterion = nn.CrossEntropyLoss()
        
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # Check state is initialized
        for param in model.parameters():
            if param.grad is not None:
                state = optimizer.state[param]
                assert 'step' in state
                assert 'exp_avg' in state
                assert 'exp_avg_sq' in state
                assert 'momentum' in state
                assert state['step'] == 1
    
    def test_convergence_quadratic_function(self):
        """Test convergence on a simple quadratic function."""
        # Define a simple quadratic function: f(x) = (x - 2)^2
        x = torch.tensor([0.0], requires_grad=True)
        optimizer = StableAdam([x], lr=0.1)
        
        losses = []
        for _ in range(100):
            optimizer.zero_grad()
            loss = (x - 2) ** 2
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Check convergence
        assert losses[-1] < 1e-3  # Should converge close to 0
        assert abs(x.item() - 2.0) < 1e-2  # x should converge close to 2
    
    def test_temporal_stability_penalty(self):
        """Test temporal stability penalty computation."""
        model = SimpleModel()
        optimizer = StableAdam(model.parameters(), temporal_stability=0.1)
        
        # Take a few optimization steps
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Check that temporal stability penalty is computed
        penalty = optimizer.get_temporal_stability_penalty()
        if penalty is not None:
            assert isinstance(penalty, torch.Tensor)
            assert penalty.item() >= 0
    
    def test_energy_conservation_toggle(self):
        """Test that energy conservation can be toggled."""
        model = SimpleModel()
        
        # Test with energy conservation enabled
        optimizer_with_energy = StableAdam(model.parameters(), energy_conservation=True)
        assert optimizer_with_energy.param_groups[0]['energy_conservation'] is True
        
        # Test with energy conservation disabled
        optimizer_without_energy = StableAdam(model.parameters(), energy_conservation=False)
        assert optimizer_without_energy.param_groups[0]['energy_conservation'] is False
    
    def test_mixed_precision_compatibility(self):
        """Test compatibility with different tensor dtypes."""
        model = SimpleModel()
        optimizer = StableAdam(model.parameters())
        
        # Test with float16 gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param, dtype=torch.float16)
        
        # Should not raise an error
        optimizer.step()
    
    def test_closure_function(self):
        """Test optimization with closure function."""
        model = SimpleModel()
        optimizer = StableAdam(model.parameters())
        
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            return loss
        
        # Should not raise an error
        returned_loss = optimizer.step(closure)
        assert isinstance(returned_loss, torch.Tensor)
    
    def test_parameter_groups(self):
        """Test optimizer with multiple parameter groups."""
        model = SimpleModel()
        
        # Create separate parameter groups
        optimizer = StableAdam([
            {'params': model.linear1.parameters(), 'lr': 0.01},
            {'params': model.linear2.parameters(), 'lr': 0.001}
        ])
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[1]['lr'] == 0.001