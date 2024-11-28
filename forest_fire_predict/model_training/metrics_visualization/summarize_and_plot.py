import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util.load_config import load_config


def plot_loss_curve(metrics_path, phase_list, output_path, output_name):
    # Read the data from the CSV
    data = pd.read_csv(metrics_path)

    # Filter based on the provided phase list
    data = data[data['phase'].isin(phase_list)]

    # Set the figure size for the plot
    plt.figure(figsize=(10, 5))

    # Define colors for each phase for consistency
    colors = {
        'train': 'red',
        'val': 'blue',
        'test': 'green'
    }

    # Set up a color palette for each fold (lighter to darker shades)
    sns.set_palette(sns.color_palette("husl", 5))

    for phase in phase_list:
        phase_data = data[data['phase'] == phase]
        # Get a consistent color for the phase
        color = colors.get(phase, 'gray')
        for fold in range(5):  # Assuming fold numbers are 0 to 4
            fold_data = phase_data[phase_data['fold'] == fold]
            # Use a lighter shade for each subsequent fold
            plt.plot(fold_data['epoch'], fold_data['loss'], label=f'Fold {fold} {phase}', color=color,
                     alpha=0.5 + (fold * 0.1))

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve per Fold')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/{output_name}_kfold_loss.png')
    plt.show()

    # Find the epoch with the smallest val loss and extract corresponding test metrics
    idx_val_f1_max = data[data['phase'] == 'val'].groupby('fold')['f1_score'].idxmax()
    test_metrics_at_max_val_f1 = pd.DataFrame()

    for idx in idx_val_f1_max:
        fold = data.loc[idx, 'fold']
        epoch = data.loc[idx, 'epoch']
        # Extracting the corresponding test metrics
        test_metrics = data[(data['fold'] == fold) & (data['epoch'] == epoch) & (data['phase'] == 'test')]
        test_metrics_at_max_val_f1 = pd.concat([test_metrics_at_max_val_f1, test_metrics], ignore_index=True)

    # Save the new CSV
    new_csv_path = f'{output_path}/best_{output_name}_kfold.csv'
    test_metrics_at_max_val_f1.to_csv(new_csv_path, index=False)
    return new_csv_path


def calculate_and_save_avg_metrics(best_kfold_csv_path, save_path, output_name):
    # Read the CSV file containing the best metrics for 5 folds
    df = pd.read_csv(best_kfold_csv_path)

    # Drop 'fold', 'epoch', and 'phase' columns if they exist
    columns_to_drop = ['fold', 'epoch', 'phase']
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Calculate the average of the metrics across rows for the remaining columns
    avg_metrics = df.mean(axis=0)

    # Create a new DataFrame for the average metrics
    avg_metrics_df = pd.DataFrame([avg_metrics])

    # Save the average metrics DataFrame to a new CSV file
    avg_csv_path = f'{save_path}/best_{output_name}_avg.csv'
    avg_metrics_df.to_csv(avg_csv_path, index=False)

    print(f"Average metrics saved to {avg_csv_path}")

if __name__ == '__main__':
    config = load_config('../execution_wf03/model_configs/config_AAUnet11.yaml')

    ds_name = config['data']['ds_name']
    aug = '' #_augXX

    # network_name = 'UNet1'
    model_name = config['model']['model_name']


    # path = f'../model_performance_{ds_name}/{model_name}'
    path = f'F:/Code/model_training/model_performance_{ds_name}/{model_name}'
    metrics_path = f'{path}/{model_name}_kfold{aug}.csv'
    best_kfold_csv_path = plot_loss_curve(metrics_path, ['train', 'val', 'test'], f'{path}', output_name=f'{model_name}{aug}')
    calculate_and_save_avg_metrics(best_kfold_csv_path, path, f'{model_name}{aug}')


