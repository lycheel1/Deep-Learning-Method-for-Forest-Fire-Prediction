import torch
import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader
import os

from dataset_loader.Wildfire_Dataset_old import WildfireDataset


class TTester():
    def __init__(self, model1_name, model2_name, model_make1, model_make2, test_set, root_save, batch_size=32):
        self.root_save = root_save
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.model = None
        self.model_make1 = model_make1
        self.model_make2 = model_make2
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    def model_paras_path(self, model_name, fold, net_suffix='', phase='test'):
        save_path = os.path.join(self.root_save, model_name)
        paras_path = os.path.join(save_path, 'nn_save', 'kfold')
        df = pd.read_csv(f'{save_path}/best_{model_name}{net_suffix}_kfold_AUC.csv')

        filtered_df = df[(df['fold'] == fold) & (df['phase'] == phase)]
        epoch = filtered_df['epoch'].values[0]

        return f'{paras_path}/{model_name}_fold{fold}_epoch{epoch}.pth'

    def load_model(self, num, fold):
        if num == 1:
            self.model = self.model_make1().cuda()
            model_load_path1 = self.model_paras_path(self.model1_name, fold)
            self.model.load_state_dict(torch.load(model_load_path1))
        elif num == 2:
            self.model = self.model_make2().cuda()
            model_load_path2 = self.model_paras_path(self.model2_name, fold)
            self.model.load_state_dict(torch.load(model_load_path2))
        # set model to correct mode
        self.model.eval()

    def calculate_f1_per_sample(self, model_num, dataloader, device, threshold=0.5, k_fold=5):
        k_fold_f1 = []
        with torch.no_grad():
            for fold in range(k_fold):
                f1_scores = []
                self.load_model(model_num, fold)
                for inputs_2D, inputs_1D, targets in dataloader:
                    inputs_2D, inputs_1D, targets = inputs_2D.to(device), inputs_1D.to(device), targets.to(device)
                    outputs = self.model(inputs_2D, inputs_1D)
                    preds = (outputs > threshold).float()
                    for pred, target in zip(preds, targets):
                        # Flatten the maps to calculate F1 score
                        pred_flat = pred.view(-1).cpu().numpy()
                        target_flat = target.view(-1).cpu().numpy()
                        f1 = f1_score(target_flat, pred_flat, average='binary')
                        f1_scores.append(f1)
                k_fold_f1.append(f1_scores)
        return k_fold_f1

    def compare_models_f1(self, device='cuda'):
        f1_scores_model1 = self.calculate_f1_per_sample(1, self.test_loader, device)
        f1_scores_model2 = self.calculate_f1_per_sample(2, self.test_loader, device)

        results = []

        # Calculate t-test for each fold
        for fold_idx in range(len(f1_scores_model1)):
            t_stat, p_value = ttest_rel(f1_scores_model1[fold_idx], f1_scores_model2[fold_idx])
            significance = 'yes' if (t_stat > 0 and p_value < 0.05) else 'no'
            results.append([self.model1_name, self.model2_name, fold_idx, t_stat, p_value, significance])
            print(f"Fold {fold_idx}: t-statistic: {t_stat}, p-value: {p_value}")

        # Calculate the overall t-test
        all_f1_scores_model1 = [score for fold_scores in f1_scores_model1 for score in fold_scores]
        all_f1_scores_model2 = [score for fold_scores in f1_scores_model2 for score in fold_scores]

        overall_t_stat, overall_p_value = ttest_rel(all_f1_scores_model1, all_f1_scores_model2)
        overall_significance = 'yes' if (overall_t_stat > 0 and overall_p_value < 0.05) else 'no'
        results.append([self.model1_name, self.model2_name, 'all', overall_t_stat, overall_p_value, overall_significance])
        print(f"Overall: t-statistic: {overall_t_stat}, p-value: {overall_p_value}")

        # Save results to CSV
        results_df = pd.DataFrame(results, columns=['model1', 'model2', 'fold', 't-statistic', 'p-value', 'significance'])
        results_df.to_csv(os.path.join(self.root_save, f'{self.model1_name}_{self.model2_name}_ttest.csv'), index=False)
        print(f'Results saved to {os.path.join(self.root_save, f"{self.model1_name}_{self.model2_name}_ttest.csv")}')



### CHANGE ###
# from NN_models.ASPCUnet11 import Custom_ASPCUNet11
from NN_models.models_wf02.ASPCUnet17 import Custom_ASPCUNet17
from NN_models.models_wf02.Unet import Custom_UNet


def model_make1():
    return Custom_ASPCUNet17()

def model_make2():
    return Custom_UNet()

if __name__ == '__main__':
    ds_name = 'wf02'
    aug = '' #_augXX

    ### CHANGE NETWORK NAME ###
    network1_name = 'ASPCUNet17'
    network2_name = 'UNet'
    net_suffix = aug

    dataset_folder = f'../data/dataset_{ds_name}/'
    save_root = f'F:/Code/model_training/model_performance_{ds_name}/'

    test_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_std_test.npy',
                                  f'{dataset_folder}/X_1D_std_test.npy',
                                  f'{dataset_folder}/Y_std_test.npy')

    TTester1 = TTester(model1_name=network1_name, model2_name=network2_name,
                       model_make1=model_make1, model_make2=model_make2, test_set=test_set_ds, root_save=save_root)

    TTester1.compare_models_f1()



