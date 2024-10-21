import pytest
import torch
import numpy as np
from hamiltonian_ai.utils import evaluate_model
from hamiltonian_ai.models import HamiltonianNN
from torch.utils.data import TensorDataset, DataLoader

@pytest.fixture
def model_and_data():
    torch.manual_seed(42)  # Set a fixed seed for reproducibility
    model = HamiltonianNN(input_dim=10, hidden_dims=[64, 32])
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    print("Label distribution:", torch.bincount(y))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # Set shuffle to False
    return model, dataloader

def test_evaluate_model_perfect_prediction(model_and_data):
    model, dataloader = model_and_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a dictionary to store the correct labels for each input
    correct_labels = {}
    for x, y in dataloader:
        for i in range(x.shape[0]):
            correct_labels[tuple(x[i].tolist())] = y[i].item()

    # Override model's forward method to always predict correctly
    def perfect_forward(self, x):
        batch_size = x.shape[0]
        predictions = torch.zeros((batch_size, 2)).float().to(device)
        for i in range(batch_size):
            label = correct_labels[tuple(x[i].tolist())]
            predictions[i, label] = 1
        return predictions

    # Replace the model's forward method
    model.forward = perfect_forward

    accuracy, precision, recall, f1, auc = evaluate_model(model, dataloader, device)

    assert accuracy == 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0
    assert auc == 1.0


def test_evaluate_model_random_prediction(model_and_data):
    model, dataloader = model_and_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Override model's forward method to predict randomly
    def random_forward(self, x):
        batch_size = x.shape[0]
        return torch.rand((batch_size, 2)).to(device)
    
    model.forward = lambda x: random_forward(model, x)
    
    accuracy, precision, recall, f1, auc = evaluate_model(model, dataloader, device)
    
    assert 0.4 <= accuracy <= 0.6
    assert 0.4 <= precision <= 0.6
    assert 0.4 <= recall <= 0.6
    assert 0.4 <= f1 <= 0.6
    assert 0.4 <= auc <= 0.6
