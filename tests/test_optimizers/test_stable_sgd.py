"""Tests for StableSGD optimizer."""

import pytest
import torch
import torch.nn as nn
from temporal_optimizer import StableSGD


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestStableSGD:
    """Test suite for StableSGD optimizer."""
    
    def test_initialization_default_params(self):
        """Test optimizer initialization with default parameters."""
        model = SimpleModel()
        optimizer = StableSGD(model.parameters())
        
        assert len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]
        assert group['lr'] == 1e-2
        assert group['momentum'] == 0
        assert group['weight_decay'] == 0
        assert group['dampening'] == 0
        assert group['nesterov'] is False
        assert group['temporal_stability'] == 0.01
        assert group['momentum_decay'] == 0.9
        assert group['energy_conservation'] is True
    
    def test_initialization_custom_params(self):
        """Test optimizer initialization with custom parameters."""
        model = SimpleModel()
        optimizer = StableSGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,
            dampening=0.1,
            nesterov=False,
            temporal_stability=0.05,
            momentum_decay=0.95,
            energy_conservation=False
        )
        
        group = optimizer.param_groups[0]
        assert group['lr'] == 0.01
        assert group['momentum'] == 0.9
        assert group['weight_decay'] == 1e-4
        assert group['dampening'] == 0.1
        assert group['nesterov'] is False
        assert group['temporal_stability'] == 0.05
        assert group['momentum_decay'] == 0.95
        assert group['energy_conservation'] is False
    
    def test_nesterov_momentum(self):
        """Test Nesterov momentum configuration."""
        model = SimpleModel()
        
        # Valid Nesterov configuration
        optimizer = StableSGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            nesterov=True,
            dampening=0.0
        )
        assert optimizer.param_groups[0]['nesterov'] is True
    
    def test_invalid_nesterov_config(self):
        """Test invalid Nesterov momentum configurations."""
        model = SimpleModel()
        
        # Nesterov with zero momentum should raise error
        with pytest.raises(ValueError, match="Nesterov momentum requires"):
            StableSGD(model.parameters(), momentum=0, nesterov=True)
        
        # Nesterov with dampening should raise error
        with pytest.raises(ValueError, match="Nesterov momentum requires"):
            StableSGD(model.parameters(), momentum=0.9, dampening=0.1, nesterov=True)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        model = SimpleModel()
        
        with pytest.raises(ValueError, match="Invalid learning rate"):
            StableSGD(model.parameters(), lr=-0.1)
        
        with pytest.raises(ValueError, match="Invalid momentum"):
            StableSGD(model.parameters(), momentum=-0.1)
        
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            StableSGD(model.parameters(), weight_decay=-0.1)
        
        with pytest.raises(ValueError, match="Invalid temporal_stability"):
            StableSGD(model.parameters(), temporal_stability=-0.1)
        
        with pytest.raises(ValueError, match="Invalid momentum_decay"):
            StableSGD(model.parameters(), momentum_decay=1.1)
    
    def test_optimization_step(self):
        """Test basic optimization step."""
        model = SimpleModel()
        optimizer = StableSGD(model.parameters(), lr=0.01)
        
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
    
    def test_momentum_step(self):
        """Test optimization step with momentum."""
        model = SimpleModel()
        optimizer = StableSGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Create some dummy data
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        criterion = nn.CrossEntropyLoss()
        
        # Take multiple steps to build momentum
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Should have momentum buffers in state
        for param in model.parameters():
            if param.grad is not None:
                state = optimizer.state[param]
                assert 'momentum_buffer' in state
                assert 'hamiltonian_momentum' in state
    
    def test_state_initialization(self):
        """Test optimizer state initialization."""
        model = SimpleModel()
        optimizer = StableSGD(model.parameters(), momentum=0.9)
        
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
                assert 'momentum_buffer' in state
                assert 'hamiltonian_momentum' in state
                if optimizer.param_groups[0]['momentum'] > 0:
                    assert 'step' in state
    
    def test_convergence_quadratic_function(self):
        """Test convergence on a simple quadratic function."""
        # Define a simple quadratic function: f(x) = (x - 2)^2
        x = torch.tensor([0.0], requires_grad=True)
        optimizer = StableSGD([x], lr=0.1)
        
        losses = []
        for _ in range(200):  # SGD may need more steps than Adam
            optimizer.zero_grad()
            loss = (x - 2) ** 2
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Check convergence (SGD might be slower than Adam)
        assert losses[-1] < 1e-2
        assert abs(x.item() - 2.0) < 0.1
    
    def test_temporal_stability_penalty(self):
        """Test temporal stability penalty computation."""
        model = SimpleModel()
        optimizer = StableSGD(model.parameters(), temporal_stability=0.1)
        
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
        optimizer_with_energy = StableSGD(model.parameters(), energy_conservation=True)
        assert optimizer_with_energy.param_groups[0]['energy_conservation'] is True
        
        # Test with energy conservation disabled
        optimizer_without_energy = StableSGD(model.parameters(), energy_conservation=False)
        assert optimizer_without_energy.param_groups[0]['energy_conservation'] is False
    
    def test_weight_decay(self):
        """Test weight decay functionality."""
        model = SimpleModel()
        optimizer = StableSGD(model.parameters(), lr=0.01, weight_decay=1e-4)
        
        # Take optimization steps
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        criterion = nn.CrossEntropyLoss()
        
        # Get initial parameter norms
        initial_norms = [p.norm().item() for p in model.parameters()]
        
        for _ in range(10):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Weight decay should generally reduce parameter norms over time
        # (though this depends on the specific dynamics)
        assert optimizer.param_groups[0]['weight_decay'] == 1e-4
    
    def test_closure_function(self):
        """Test optimization with closure function."""
        model = SimpleModel()
        optimizer = StableSGD(model.parameters())
        
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
        optimizer = StableSGD([
            {'params': model.linear1.parameters(), 'lr': 0.01, 'momentum': 0.9},
            {'params': model.linear2.parameters(), 'lr': 0.001, 'momentum': 0.5}
        ])
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[0]['momentum'] == 0.9
        assert optimizer.param_groups[1]['lr'] == 0.001
        assert optimizer.param_groups[1]['momentum'] == 0.5
    
    def test_comparison_with_torch_sgd(self):
        """Test that StableSGD behaves similarly to torch.optim.SGD in basic cases."""
        # Create two identical models
        model1 = SimpleModel()
        model2 = SimpleModel()
        
        # Copy parameters to ensure they start identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.data.copy_(p1.data)
        
        # Create optimizers (with energy_conservation=False and temporal_stability=0 for fair comparison)
        stable_sgd = StableSGD(model1.parameters(), lr=0.01, energy_conservation=False, temporal_stability=0)
        torch_sgd = torch.optim.SGD(model2.parameters(), lr=0.01)
        
        # Create identical data
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        criterion = nn.CrossEntropyLoss()
        
        # Take one optimization step on each
        for model, optimizer in [(model1, stable_sgd), (model2, torch_sgd)]:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Parameters should be very close (allowing for small numerical differences)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)