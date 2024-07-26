import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    idx_val_loss_min = data[data['phase'] == 'val'].groupby('fold')['loss'].idxmin()
    test_metrics_at_min_val_loss = pd.DataFrame()

    for idx in idx_val_loss_min:
        fold = data.loc[idx, 'fold']
        epoch = data.loc[idx, 'epoch']
        # Extracting the corresponding test metrics
        test_metrics = data[(data['fold'] == fold) & (data['epoch'] == epoch) & (data['phase'] == 'test')]
        test_metrics_at_min_val_loss = pd.concat([test_metrics_at_min_val_loss, test_metrics], ignore_index=True)

    # Save the new CSV
    test_metrics_at_min_val_loss.to_csv(f'{output_path}/best_{output_name}_kfold.csv', index=False)


def plot_avg_loss_curve(metrics_path, phase_list, output_path, output_name):
    # Read the data from the CSV
    data = pd.read_csv(metrics_path)

    # Filter based on the provided phase list
    data = data[data['phase'].isin(phase_list)]

    # Calculate the average metrics for each epoch across all folds
    avg_metrics = data.groupby(['epoch', 'phase']).mean().reset_index()

    # Set the figure size for the plot
    plt.figure(figsize=(10, 5))

    # Define colors for each phase for consistency
    colors = {
        'train': 'red',
        'val': 'blue',
        'test': 'green'
    }

    # Plot average loss for each phase
    for phase in phase_list:
        phase_data = avg_metrics[avg_metrics['phase'] == phase]
        plt.plot(phase_data['epoch'], phase_data['loss'], label=f'Average {phase}', color=colors[phase])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Loss Curve per Phase')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/{output_name}_avg_loss.png')
    plt.show()

    # Determine the best epoch based on the lowest average validation loss
    idx_val_loss_min = avg_metrics[avg_metrics['phase'] == 'val']['loss'].idxmin()
    best_epoch = avg_metrics.loc[idx_val_loss_min, 'epoch']

    # Prepare the metrics for the best epoch
    best_epoch_metrics = avg_metrics[avg_metrics['epoch'] == best_epoch].copy()
    best_epoch_metrics['fold'] = -1  # Set fold to -1

    # Save the metrics at the best epoch
    best_epoch_metrics.to_csv(f'{output_path}/best_{output_name}_avg.csv', index=False)

if __name__ == '__main__':
    ds_name = 'wf02'
    model_name = 'SwinUnet3'
    aug_list = [''] #____

    for aug_name in aug_list:
        # path = f'../model_performance_{ds_name}/{model_name}'
        path = f'F:/Code/model_training/model_performance_{ds_name}/{model_name}'
        metrics_path = f'{path}/kfold_metrics{aug_name}.csv'
        plot_loss_curve(metrics_path, ['train', 'val', 'test'], f'{path}', output_name=f'{model_name}{aug_name}')
        # plot_avg_loss_curve(metrics_path, ['train', 'val', 'test'], f'{path}', output_name=f'{model_name}{aug_name}')


