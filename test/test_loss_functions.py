import pytest
import torch
import numpy as np
from hamiltonian_ai.data_processing import HamiltonianDataset, prepare_data

@pytest.fixture
def sample_data():
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_hamiltonian_dataset(sample_data):
    X, y = sample_data
    dataset = HamiltonianDataset(X, y)
    
    assert len(dataset) == len(X)
    
    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert features.shape == (10,)
    assert label.shape == ()

def test_prepare_data(sample_data):
    X, y = sample_data
    train_dataset, test_dataset, scaler = prepare_data(X, y, test_size=0.2, apply_smote=True)
    
    assert isinstance(train_dataset, HamiltonianDataset)
    assert isinstance(test_dataset, HamiltonianDataset)
    
    # Check if SMOTE was applied (train set should be balanced)
    train_labels = [label.item() for _, label in train_dataset]
    unique, counts = np.unique(train_labels, return_counts=True)
    assert len(unique) == 2
    assert counts[0] == counts[1]
    
    # Check if scaling was applied
    assert scaler.mean_ is not None
    assert scaler.scale_ is not None

def test_prepare_data_without_smote(sample_data):
    X, y = sample_data
    train_dataset, test_dataset, scaler = prepare_data(X, y, test_size=0.2, apply_smote=False)

    # Check that class distribution is preserved in train set
    train_labels = [label.item() for _, label in train_dataset]
    unique, counts = np.unique(train_labels, return_counts=True)
    assert len(unique) == 2
    assert np.abs(counts[0] - counts[1]) <= 1  # Allow for a difference of at most 1 due to odd number of samples
