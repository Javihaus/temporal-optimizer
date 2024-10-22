import pytest
import torch
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
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False
    )  # Set shuffle to False
    return model, dataloader


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
