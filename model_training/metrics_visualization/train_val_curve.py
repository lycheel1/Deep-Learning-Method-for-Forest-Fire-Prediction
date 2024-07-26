import pandas as pd
import matplotlib.pyplot as plt


class ModelMetricsPlotter:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def plot_k_folds(self, metric_name, save_path=None):
        plt.figure(figsize=(14, 7))
        for fold in self.df['fold'].unique():
            for phase in ['train', 'val']:
                subset = self.df[(self.df['fold'] == fold) & (self.df['phase'] == phase)]
                plt.plot(subset['epoch'], subset[metric_name], label=f'Fold {fold} {phase}')
        plt.title(f'K-Fold {metric_name} per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        # plt.savefig(save_path)
        plt.show()

    def plot_average_curve(self, metric_name, save_path=None):
        plt.figure(figsize=(14, 7))
        for phase in ['train', 'val']:
            mean_metrics = self.df[self.df['phase'] == phase].groupby('epoch')[metric_name].mean()
            plt.plot(mean_metrics.index, mean_metrics, label=f'Average {phase}')
            print(f'{phase} {metric_name} {max(mean_metrics)}')
        plt.title(f'Average {metric_name} per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        # plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':

    suffix = ''
    suffix2 = '_augA'

    # Model Path List
    nn_path_list = {'resnet18_fc': '../../NN_Models/resnet18_fc',
                    'resnet18_trans': '../../NN_Models/resnet18_trans',
                    'unet': '../../NN_Models/UNet',
                    'unet_dilated': '../../NN_Models/UNet_dilated',
                    'unet_dilated 2': '../../NN_Models/UNet_dilated firstL'}

    plotter = ModelMetricsPlotter(f'{nn_path_list["unet"]}{suffix}/kfold_metrics{suffix2}.csv')
    plotter.plot_k_folds('f1_score')
    plotter.plot_average_curve('f1_score')
