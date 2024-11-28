import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WildfireDataset(Dataset):
    def __init__(self, data_2d_path, data_1d_path, Y_path, stats_csv_path, standardize_features_2d=None,
                 standardize_features_1d=None ,std_flag=True):
        # Load the data from npy files
        self.data_2d = np.load(data_2d_path)  # Shape: (N, C_2D, 64, 64)
        self.data_1d = np.load(data_1d_path)  # Shape: (N, C_1D, [length of 1D])
        self.Y = np.load(Y_path)  # Shape: (N, H, W)

        # Load the mean and standard deviation values from the CSV file
        stats_df = pd.read_csv(stats_csv_path)
        self.means = dict(zip(stats_df['Feature'], stats_df['Mean']))
        self.stds = dict(zip(stats_df['Feature'], stats_df['Standard_Deviation']))

        self.std_flag = std_flag

        # Define which features to standardize (default to None for no standardization)
        self.standardize_features_2d = standardize_features_2d
        self.standardize_features_1d = standardize_features_1d
        if standardize_features_2d is None:
            self.standardize_features_2d = [2,4]
        if standardize_features_1d is None:
            self.standardize_features_1d = [50]

    def __len__(self):
        return len(self.data_2d)

    def __getitem__(self, idx):
        # Retrieve the sample at the given index
        sample_2d = torch.from_numpy(self.data_2d[idx]).float().clone()
        sample_1d = torch.from_numpy(self.data_1d[idx]).float().clone()
        Y = torch.from_numpy(self.Y[idx]).float()


        if self.std_flag:
            # Apply standardization to selected 2D features
            for feature_idx in self.standardize_features_2d:
                feature_name_2d = f"feature_2D_{feature_idx}"
                if feature_name_2d in self.means and feature_name_2d in self.stds:
                    mean_2d = self.means[feature_name_2d]
                    std_dev_2d = self.stds[feature_name_2d]
                    sample_2d[feature_idx, :, :] = (sample_2d[feature_idx, :, :] - mean_2d) / std_dev_2d

            # Apply standardization to selected 1D features
            for feature_idx in self.standardize_features_1d:
                feature_name_1d = f"feature_1D_{feature_idx}"
                if feature_name_1d in self.means and feature_name_1d in self.stds:
                    mean_1d = self.means[feature_name_1d]
                    std_dev_1d = self.stds[feature_name_1d]
                    sample_1d[feature_idx] = (sample_1d[feature_idx] - mean_1d) / std_dev_1d

        return [sample_2d, sample_1d, Y]