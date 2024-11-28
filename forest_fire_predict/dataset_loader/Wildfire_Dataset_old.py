import torch
from torch.utils.data import Dataset
import numpy as np

class WildfireDataset(Dataset):
    def __init__(self, data_2d_path, data_1d_path, Y_path):
        # Load the data from npy files
        self.data_2d = np.load(data_2d_path)  # Shape: (N, C, H, W)
        self.data_1d = np.load(data_1d_path)  # Shape: (N, L)
        self.Y = np.load(Y_path)    # Shape: (N, H, W)

    def __len__(self):
        return len(self.data_2d)

    def __getitem__(self, idx):
        # Retrieve the sample at the given index
        sample_2d = torch.from_numpy(self.data_2d[idx]).float()
        sample_1d = torch.from_numpy(self.data_1d[idx]).float()
        Y = torch.from_numpy(self.Y[idx]).float()

        return [sample_2d, sample_1d, Y]
