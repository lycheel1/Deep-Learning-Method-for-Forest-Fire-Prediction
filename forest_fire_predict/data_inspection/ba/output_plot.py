import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import os
from NN_models.models_wf02.ASPCUnet11 import Custom_ASPCUNet11
from dataset_loader.Wildfire_Dataset_old import WildfireDataset

def plot_single_sample(model_make, X_2d, X_1d, y, csv_path, model_save_path, model_name, output_dir, sample_name):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file containing the best epochs for each fold
    df = pd.read_csv(csv_path)

    # Convert to torch tensors and move to GPU if available
    X_2d = torch.tensor(X_2d).unsqueeze(0).cuda()
    X_1d = torch.tensor(X_1d).unsqueeze(0).cuda()
    y = torch.tensor(y).unsqueeze(0).cuda()

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



def model_make():
    return Custom_ASPCUNet11()

# Usage example
if __name__ == '__main__':
    ds_name = 'wf02'
    aug = '' #_augXX

    ### CHANGE NETWORK NAME ###
    network_name = 'ASPCUNet1105'
    net_suffix = aug

    dataset_folder = f'../data/dataset_{ds_name}'
    save_path = f'F:/Code/model_training/model_performance_{ds_name}/{network_name}/'

    test_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_std_test.npy',
                                  f'{dataset_folder}/X_1D_std_test.npy',
                                  f'{dataset_folder}/Y_std_test.npy')



    plot_single_sample(model_make, test_set_ds,f'{save_path}/best_{network_name}{net_suffix}_kfold.csv',
            f'{save_path}/nn_save/kfold/',  network_name,
            f'{save_path}/output_plot/',
            [0,1,2,3])