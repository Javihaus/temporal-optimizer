import pytest
import torch
from hamiltonian_ai.loss_functions import hamiltonian_loss
from hamiltonian_ai.models import HamiltonianNN


@pytest.fixture
def model_and_data():
    model = HamiltonianNN(input_dim=10, hidden_dims=[64, 32])
    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    return model, x, y


def test_hamiltonian_loss(model_and_data):
    model, x, y = model_and_data
    outputs = model(x)
    loss = hamiltonian_loss(outputs, y, model)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss.requires_grad


def test_hamiltonian_loss_regularization(model_and_data):
    model, x, y = model_and_data
    outputs = model(x)

    loss_with_reg = hamiltonian_loss(outputs, y, model, reg_coeff=0.1)
    loss_without_reg = hamiltonian_loss(outputs, y, model, reg_coeff=0.0)

    assert loss_with_reg > loss_without_reg


def test_hamiltonian_loss_backprop(model_and_data):
    model, x, y = model_and_data
    outputs = model(x)
    loss = hamiltonian_loss(outputs, y, model)

    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
