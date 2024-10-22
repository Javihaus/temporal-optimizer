import pytest
import torch
from hamiltonian_ai.optimizers import AdvancedSymplecticOptimizer


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def model_and_optimizer():
    model = SimpleModel()
    optimizer = AdvancedSymplecticOptimizer(model.parameters())
    return model, optimizer


def test_optimizer_step(model_and_optimizer):
    model, optimizer = model_and_optimizer

    # Generate dummy data
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)

    # Perform one optimization step
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        return loss

    initial_params = [p.clone() for p in model.parameters()]
    loss = optimizer.step(closure)

    # Check that parameters have been updated
    for p, initial_p in zip(model.parameters(), initial_params):
        assert not torch.allclose(p, initial_p)


def test_optimizer_state(model_and_optimizer):
    model, optimizer = model_and_optimizer

    # Check initial state
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert "momentum" not in optimizer.state[p]

    # Perform one step to initialize state
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Check that state has been initialized
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert "momentum" in optimizer.state[p]
            assert "step" in optimizer.state[p]
