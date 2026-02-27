import os
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    """Custom PyTorch Dataset for loading data."""
    def __init__(self, X, Y, Z=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        if Z is not None:
            self.Z = torch.tensor(Z, dtype=torch.float32)
        else:
            self.Z = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Z is None:
            return self.X[idx], self.Y[idx]
        else:
            return self.X[idx], self.Y[idx], self.Z[idx]
    
def get_next_run_path(base_dir, experiment):
    exp_dir = os.path.join(base_dir, "training-results/", experiment)
    os.makedirs(exp_dir, exist_ok=True)
    runs = [d for d in os.listdir(exp_dir) if d.startswith("run_")]
    run_ids = [int(r.split("_")[1]) for r in runs if r.split("_")[1].isdigit()]
    next_id = max(run_ids, default=0) + 1
    run_name = f"run_{next_id:03d}"
    run_dir = os.path.join(exp_dir, run_name)
    os.makedirs(run_dir)
    return run_name, run_dir


def load_data(base_path):
    warnings.warn("Only works with interpolated data.")
    gamma_data = np.load(base_path + 'activations_gamma.npz', allow_pickle=True)
    z_data = np.load(base_path + 'coordinate_z.npz', allow_pickle=True)
    r_data = np.load(base_path + 'centerline_r.npz', allow_pickle=True)

    gamma = gamma_data["gamma"]
    r_raw = r_data["r"]
    z_raw = z_data["z"]
    
    gamma = torch.tensor(gamma, dtype=torch.float32)
    r = torch.tensor(r_raw, dtype=torch.float32)
    z = torch.tensor(z_raw, dtype=torch.float32)

    return gamma, r, z


def preprocess_data(X, Y, z, test_size=0.2, val_ratio=0.2, random_state=42):
    sparsifier = 1 # in case we want to reduce the number of samples for faster training
    Z_sparse = z[::sparsifier, :]
    X_sparse = X[::sparsifier, :]
    Y_sparse = Y[::sparsifier, :, :]
    # Fixed test set
    X_remain, X_test, Y_remain, Y_test, Z_remain, Z_test = train_test_split(X_sparse, Y_sparse, Z_sparse, test_size=test_size, random_state=random_state)
    X_train, X_valid, Y_train, Y_valid, Z_train, Z_valid = train_test_split(X_remain, Y_remain, Z_remain, test_size=val_ratio, random_state=random_state)
    
    X_train = np.repeat(X_train, repeats=z.shape[-1], axis=0)
    X_valid = np.repeat(X_valid, repeats=z.shape[-1], axis=0)
    X_test = np.repeat(X_test, repeats=z.shape[-1], axis=0)
    Y_train = Y_train.reshape(-1, Y_train.shape[-1])
    Y_valid = Y_valid.reshape(-1, Y_valid.shape[-1])
    Y_test = Y_test.reshape(-1, Y_test.shape[-1])
    Z_train = Z_train.reshape(-1, 1)
    Z_valid = Z_valid.reshape(-1, 1)
    Z_test = Z_test.reshape(-1, 1)
    print(f"Training data: Dimensions for DON are...")
    print("gamma shape:", X_train.shape)
    print("r shape:", Y_train.shape)
    print("z shape:", Z_train.shape)
    print("-------------------")
    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)
    scaler_Z = StandardScaler().fit(Z_train)
    X_train = scaler_X.transform(X_train)
    X_valid = scaler_X.transform(X_valid)
    X_test = scaler_X.transform(X_test)
    Y_train = scaler_Y.transform(Y_train)
    Y_valid = scaler_Y.transform(Y_valid)
    Y_test = scaler_Y.transform(Y_test)
    Z_train = scaler_Z.transform(Z_train)
    Z_valid = scaler_Z.transform(Z_valid)
    Z_test = scaler_Z.transform(Z_test)
    scalers = {
        "X": scaler_X,
        "Y": scaler_Y,
        "Z": scaler_Z
    }
    return (X_train, Y_train, Z_train), (X_valid, Y_valid, Z_valid), (X_test, Y_test, Z_test), scalers



def create_dataloader(X, Y, shuffle, Z=None, batch_size=32):
    """Create a DataLoader for the dataset."""
    dataset = CustomDataset(X, Y, Z)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader