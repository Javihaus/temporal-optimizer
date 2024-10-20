import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
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
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE if needed
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train
    
    # Create datasets
    train_dataset = HamiltonianDataset(X_train_resampled, y_train_resampled)
    test_dataset = HamiltonianDataset(X_test_scaled, y_test)
    
    return train_dataset, test_dataset, scaler