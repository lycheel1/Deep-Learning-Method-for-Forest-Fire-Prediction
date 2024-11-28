import numpy as np
import pandas as pd


wf_path= '../../data/dataset_wf03'

X2D = np.load(f"{wf_path}/X_2D_train.npy")  # (N, C_2D, 64, 64)
X1D = np.load(f"{wf_path}/X_1D_train.npy")  # (N, C_1D, [length of 1D])


# Get the number of channels for each input
_, C_2D, _, _ = X2D.shape
_, C_1D = X1D.shape

# Calculate means and standard deviations
# For 2D data
mean_2D = np.mean(X2D, axis=(0, 2, 3))
std_2D = np.std(X2D, axis=(0, 2, 3))

# For 1D data
mean_1D = np.mean(X1D, axis=(0,))
std_1D = np.std(X1D, axis=(0,))

# in case std is 0
std_2D = np.where(std_2D == 0, 1e-6, std_2D)
std_1D = np.where(std_1D == 0, 1e-6, std_1D)

# Combine means and standard deviations into one array
means = np.concatenate([mean_2D, mean_1D])
stds = np.concatenate([std_2D, std_1D])

# Feature names
feature_names = [f"feature_2D_{i}" for i in range(C_2D)] + [f"feature_1D_{i}" for i in range(C_1D)]

# Save to CSV
stats_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean': means,
    'Standard_Deviation': stds
})
stats_df.to_csv(f"{wf_path}/trainSet_std.csv", index=False)
