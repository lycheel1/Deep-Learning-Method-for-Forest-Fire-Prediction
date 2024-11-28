import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from NN_models.models_wf02.ASPCUnet11 import Custom_ASPCUNet11


def standardization(array_input, mean, std):
    return (array_input - mean) / std

def plot_single_sample(model_make, X_2d, X_1d, y, csv_path, model_save_path, model_name, output_dir, sample_name):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file containing the best epochs for each fold
    df = pd.read_csv(csv_path)

    # Convert to torch tensors and move to GPU if available
    X_2d = torch.tensor(X_2d, dtype=torch.float32).cuda()
    X_1d = torch.tensor(X_1d, dtype=torch.float32).unsqueeze(0).cuda()
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(0).cuda()

    # Iterate through each best fold
    for index, row in df.iterrows():
        fold = row['fold']
        epoch = row['epoch']
        model_weights_path = f'{model_save_path}/{model_name}_fold{fold}_epoch{epoch}.pth'

        if not os.path.exists(model_weights_path):
            print(f"Model weights not found for fold {fold}, epoch {epoch}, {model_weights_path}")
            continue

        # Load model weights
        model = model_make().cuda()
        model.load_state_dict(torch.load(model_weights_path))
        model.eval()

        with torch.no_grad():
            print(f"Processing sample '{sample_name}'")

            outputs = model(X_2d, X_1d)

            # Convert to numpy arrays
            labels_np = y.squeeze().cpu().numpy()
            outputs_np = (outputs.squeeze().cpu().numpy() > 0.5).astype(int)  # Apply threshold to get binary output

            # Create the plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Create a composite image for the ground truth and predictions with black background
            ground_truth_image = np.zeros((labels_np.shape[0], labels_np.shape[1], 3))  # Default to black background
            prediction_image = np.zeros((labels_np.shape[0], labels_np.shape[1], 3))  # Default to black background

            # Assign colors for ground truth
            ground_truth_image[labels_np == 1] = [0, 1, 0]  # Green for ground truth positive

            # Assign colors for predictions
            correct_pred = (outputs_np == 1) & (labels_np == 1)
            wrong_positive = (outputs_np == 1) & (labels_np == 0)
            wrong_negative = (outputs_np == 0) & (labels_np == 1)

            prediction_image[correct_pred] = [0, 0, 1]  # Blue for correct positive
            prediction_image[wrong_positive] = [1, 1, 0]  # Yellow for wrong positive
            prediction_image[wrong_negative] = [1, 0, 0]  # Red for wrong negative

            axes[0].imshow(ground_truth_image)
            axes[0].set_title('Ground Truth')
            axes[1].imshow(prediction_image)
            axes[1].set_title('Prediction')

            plt.suptitle(f'Sample {sample_name}, Fold {fold}, Epoch {epoch}')
            plt.savefig(os.path.join(output_dir, f'{sample_name}_fold{fold}_epoch{epoch}.png'))
            plt.close()

    print(f"Plots saved in {output_dir}")

def plot_samples_by_fire_id(model_make, fire_id, X_2D_path, X_1D_path, Y_path, meta_csv_path, csv_path, model_save_path, model_name, output_dir, mean_2D_path, std_2D_path):
    # Ensure output directory exists
    fire_id_dir = os.path.join(output_dir, str(fire_id))
    os.makedirs(fire_id_dir, exist_ok=True)

    # Load the datasets
    X_2D = np.load(X_2D_path)
    X_1D = np.load(X_1D_path)
    Y = np.load(Y_path)

    # Load the metadata CSV
    meta_df = pd.read_csv(meta_csv_path)

    # Load mean and std for standardization
    mean_2D = np.load(mean_2D_path)
    std_2D = np.load(std_2D_path)

    # Find indices of samples with the given fire_id
    fire_id_indices = meta_df.index[meta_df['fireID'] == fire_id].tolist()

    # Plot each sample
    for idx in fire_id_indices:
        X_2d_sample = X_2D[idx]
        X_1d_sample = X_1D[idx]
        y_sample = Y[idx]

        # Standardize the sample
        X_2d_sample = standardization(X_2d_sample, mean_2D, std_2D)

        sample_name = f'{fire_id}_{idx}'

        plot_single_sample(
            model_make,
            X_2d_sample,
            X_1d_sample,
            y_sample,
            csv_path,
            model_save_path,
            model_name,
            fire_id_dir,
            sample_name
        )

    print(f"Plots saved in {fire_id_dir}")

def model_make():
    return Custom_ASPCUNet11()

# Usage example
if __name__ == '__main__':
    fire_id = '2021_400'  # Replace with the actual fire_id you want to plot
    ds_name = 'wf02'
    aug = '' #_augXX

    ### CHANGE NETWORK NAME ###
    network_name = 'ASPCUNet11'
    net_suffix = aug
    ds_path = f'../../data/dataset_{ds_name}'
    save_path = f'F:/Code/model_training/model_performance_{ds_name}/{network_name}/'
    output_dir = f'{ds_path}/{network_name}/'

    plot_samples_by_fire_id(
        model_make,
        fire_id,
        f'{ds_path}/{ds_name}_2D.npy',
        f'{ds_path}/{ds_name}_1D.npy',
        f'{ds_path}/{ds_name}_Y.npy',
        f'{ds_path}/{ds_name}_samples_meta.csv',
        f'{save_path}/best_{network_name}{net_suffix}_kfold.csv',  # Replace with the actual CSV path
        f'{save_path}/nn_save/kfold/',    # Replace with the actual model weights path
        network_name,
        output_dir,
        f'{ds_path}/mean_2D.npy',      # Path to the mean 2D file
        f'{ds_path}/std_2D.npy'        # Path to the std 2D file
    )
