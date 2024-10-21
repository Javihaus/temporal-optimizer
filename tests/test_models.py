import pytest
import torch
from hamiltonian_ai.models import HamiltonianNN


@pytest.fixture
def model():
    return HamiltonianNN(input_dim=10, hidden_dims=[64, 32])


def test_model_initialization(model):
    assert isinstance(model, HamiltonianNN)
    assert len(model.layers) == 4  # input, 2 hidden, output
    assert model.layers[0].in_features == 10
    assert model.layers[0].out_features == 64
    assert model.layers[1].in_features == 64
    assert model.layers[1].out_features == 32
    assert model.layers[2].in_features == 32
    assert model.layers[2].out_features == 2


def test_model_forward(model):
    x = torch.randn(5, 10)
    output = model(x)

    assert output.shape == (5, 2)
    assert output.requires_grad


def test_model_backward(model):
    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, y)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None


def test_model_with_different_activation(model):
    model_relu = HamiltonianNN(input_dim=10, hidden_dims=[64, 32], activation='relu')
    x = torch.randn(5, 10)

    output_leaky = model(x)
    output_relu = model_relu(x)

    assert not torch.allclose(output_leaky, output_relu)


def test_model_dropout():
    model_with_dropout = HamiltonianNN(input_dim=10, hidden_dims=[64, 32], dropout_rate=0.5)
    x = torch.randn(5, 10)

    # Set model to eval mode
    model_with_dropout.eval()
    with torch.no_grad():
        output1 = model_with_dropout(x)
        output2 = model_with_dropout(x)

    assert torch.allclose(output1, output2)

    # Set model to train mode
    model_with_dropout.train()
    output1 = model_with_dropout(x)
    output2 = model_with_dropout(x)

    assert not torch.allclose(output1, output2)
