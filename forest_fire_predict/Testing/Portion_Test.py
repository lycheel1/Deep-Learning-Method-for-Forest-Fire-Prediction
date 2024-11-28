import torch
import pandas as pd
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import os

from dataset_loader.Wildfire_Dataset_old import WildfireDataset


class MetricsCalculator():
    def __init__(self, model_name, model_make, test_set, root_save, batch_size=32):
        self.root_save = root_save
        self.model_name = model_name
        self.model = None
        self.model_make = model_make
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    def model_paras_path(self, model_name, fold, net_suffix='', phase='test'):
        save_path = os.path.join(self.root_save, model_name)
        paras_path = os.path.join(save_path, 'nn_save', 'kfold')
        df = pd.read_csv(f'{save_path}/best_{model_name}{net_suffix}_kfold_AUC.csv')

        filtered_df = df[(df['fold'] == fold) & (df['phase'] == phase)]
        epoch = filtered_df['epoch'].values[0]

        return f'{paras_path}/{model_name}_fold{fold}_epoch{epoch}.pth'

    def load_model(self, fold):
        model_path = self.model_paras_path(self.model_name, fold)
        self.model = self.model_make().cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def calculate_metrics(self, fold, device='cuda', threshold=0.5, ratio=0.1):
        self.load_model(fold)

        overall_f1_scores = []
        large_area_f1_scores = []
        small_area_f1_scores = []

        large_area_count = 0
        small_area_count = 0

        with torch.no_grad():
            for X_2D, X_1D, targets in self.test_loader:
                X_2D, X_1D, targets = X_2D.to(device), X_1D.to(device), targets.to(device)
                outputs = self.model(X_2D, X_1D)
                preds = (outputs > threshold).float()

                for pred, target in zip(preds, targets):
                    # Flatten the maps to calculate metrics
                    pred_flat = pred.view(-1).cpu().numpy()
                    target_flat = target.view(-1).cpu().numpy()

                    overall_f1 = f1_score(target_flat, pred_flat, average='binary')
                    overall_f1_scores.append(overall_f1)

                    # Calculate the number of positive pixels in the ground truth
                    positive_pixels = target_flat.sum()

                    if positive_pixels >= (64 * 64 * ratio):
                        large_area_f1 = f1_score(target_flat, pred_flat, average='binary')
                        large_area_f1_scores.append(large_area_f1)
                        large_area_count += 1
                    else:
                        small_area_f1 = f1_score(target_flat, pred_flat, average='binary')
                        small_area_f1_scores.append(small_area_f1)
                        small_area_count += 1

        # Calculate percentages
        total_count = large_area_count + small_area_count
        large_area_percentage = large_area_count / total_count * 100
        small_area_percentage = small_area_count / total_count * 100

        metrics = {
            'fold': fold,
            'overall_f1_score': sum(overall_f1_scores) / len(overall_f1_scores),
            'large_area_f1_score': sum(large_area_f1_scores) / len(large_area_f1_scores) if large_area_f1_scores else 0,
            'small_area_f1_score': sum(small_area_f1_scores) / len(small_area_f1_scores) if small_area_f1_scores else 0,
            'large_area_percentage': large_area_percentage,
            'small_area_percentage': small_area_percentage
        }

        return metrics

    def print_fold_metrics(self, fold_metrics):
        print(f"Fold {fold_metrics['fold']} Metrics:")
        print(f"  Overall F1 Score: {fold_metrics['overall_f1_score']:.4f}")
        print(f"  Large Area F1 Score: {fold_metrics['large_area_f1_score']:.4f}")
        print(f"  Small Area F1 Score: {fold_metrics['small_area_f1_score']:.4f}")
        print(f"  Large Area Percentage: {fold_metrics['large_area_percentage']:.2f}%")
        print(f"  Small Area Percentage: {fold_metrics['small_area_percentage']:.2f}%")
        print("")

    def run_evaluation(self, k_fold=5, device='cuda'):
        all_fold_metrics = []

        for fold in range(k_fold):
            metrics = self.calculate_metrics(fold, device)
            self.print_fold_metrics(metrics)
            all_fold_metrics.append(metrics)

        overall_metrics = {
            'overall_f1_score': sum([m['overall_f1_score'] for m in all_fold_metrics]) / k_fold,
            'large_area_f1_score': sum([m['large_area_f1_score'] for m in all_fold_metrics]) / k_fold,
            'small_area_f1_score': sum([m['small_area_f1_score'] for m in all_fold_metrics]) / k_fold,
            'large_area_percentage': sum([m['large_area_percentage'] for m in all_fold_metrics]) / k_fold,
            'small_area_percentage': sum([m['small_area_percentage'] for m in all_fold_metrics]) / k_fold,
        }

        print("Overall Metrics (Averaged over all folds):")
        print(f"  Overall F1 Score: {overall_metrics['overall_f1_score']:.4f}")
        print(f"  Large Area F1 Score: {overall_metrics['large_area_f1_score']:.4f}")
        print(f"  Small Area F1 Score: {overall_metrics['small_area_f1_score']:.4f}")
        print(f"  Large Area Percentage: {overall_metrics['large_area_percentage']:.2f}%")
        print(f"  Small Area Percentage: {overall_metrics['small_area_percentage']:.2f}%")


### CHANGE ###
from NN_models.models_wf02.ASPCUnet11 import Custom_ASPCUNet11


def model_make():
    return Custom_ASPCUNet11()


if __name__ == '__main__':
    ds_name = 'wf02'
    aug = ''  # _augXX

    ### CHANGE NETWORK NAME ###
    network_name = 'ASPCUNet11'
    net_suffix = aug

    dataset_folder = f'../data/dataset_{ds_name}/'
    save_root = f'F:/Code/model_training/model_performance_{ds_name}/'

    test_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_std_test.npy',
                                  f'{dataset_folder}/X_1D_std_test.npy',
                                  f'{dataset_folder}/Y_std_test.npy')

    metrics_calculator = MetricsCalculator(model_name=network_name, model_make=model_make, test_set=test_set_ds,
                                           root_save=save_root)

    metrics_calculator.run_evaluation()

