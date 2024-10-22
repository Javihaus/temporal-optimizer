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


def test_prepare_data_without_smote(sample_data):
    X, y = sample_data
    train_dataset, test_dataset, scaler = prepare_data(
        X, y, test_size=0.2, apply_smote=False
    )

    # Check that class distribution is preserved in train set
    train_labels = [label.item() for _, label in train_dataset]
    unique, counts = np.unique(train_labels, return_counts=True)
    assert len(unique) == 2

    # Calculate the imbalance ratio
    imbalance_ratio = max(counts) / min(counts)

    # Allow for some imbalance, but not too much
    assert imbalance_ratio < 1.75, f"Class imbalance too high: {imbalance_ratio}"

    # Check that the total number of samples is correct
    assert len(train_dataset) == int(0.8 * len(sample_data[0]))
    assert len(test_dataset) == int(0.2 * len(sample_data[0]))

    # Check that the scaler is fitted
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "scale_")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Class distribution in train set: {dict(zip(unique, counts))}")
    print(f"Imbalance ratio: {imbalance_ratio}")
