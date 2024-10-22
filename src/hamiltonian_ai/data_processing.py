import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class HamiltonianDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def prepare_data(X, y, test_size=0.2, apply_smote=True):
    print(f"Input shapes: X: {X.shape}, y: {y.shape}")
    print(f"apply_smote: {apply_smote}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"Train shapes after split: X: {X_train.shape}, y: {y_train.shape}")
    print(f"Test shapes after split: X: {X_test.shape}, y: {y_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE if needed
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_scaled, y_train
        )
        print(
            f"Train shapes after SMOTE: X: {X_train_resampled.shape}, y: {y_train_resampled.shape}"
        )
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train

    # Create datasets
    train_dataset = HamiltonianDataset(X_train_resampled, y_train_resampled)
    test_dataset = HamiltonianDataset(X_test_scaled, y_test)

    print(
        f"Output shapes: train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}"
    )
    return train_dataset, test_dataset, scaler
