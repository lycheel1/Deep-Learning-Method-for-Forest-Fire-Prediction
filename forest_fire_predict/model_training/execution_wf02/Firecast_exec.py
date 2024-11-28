from NN_models.models_wf02.Firecast import custom_Firecast
from dataset_loader.Wildfire_Dataset_old import WildfireDataset
from model_training.Trainer import WF01_Trainer
import torch
import torch.nn as nn
from torch.optim import AdamW
from model_training.metrics_visualization.update_csv_with_AUC import update_csv_with_auc_scores


# Factory functions for instances that would be reset in new folds

### change model_make() ###
def model_make():
    return custom_Firecast()


def criterion_make():
    return nn.BCELoss()


def optimizer_make(model):
    return AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)


def scheduler_make(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)


if __name__ == '__main__':
    ds_name = 'wf02'
    aug = '' #_augXX

    ### CHANGE NETWORK NAME ###
    network_name = 'Firecast'
    net_suffix = aug

    dataset_folder = f'../../data/dataset_{ds_name}'
    # save_path = f'../../model_training/model_performance_{ds_name}/{network_name}'
    save_path = f'F:/Code/model_training/model_performance_{ds_name}/{network_name}'
    train_set_suffix = f'std_train{aug}'






    train_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_{train_set_suffix}.npy',
                                   f'{dataset_folder}/X_1D_{train_set_suffix}.npy',
                                   f'{dataset_folder}/Y_{train_set_suffix}.npy')

    test_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_std_test.npy',
                                  f'{dataset_folder}/X_1D_std_test.npy',
                                  f'{dataset_folder}/Y_std_test.npy')

    trainer1 = WF01_Trainer(model_make=model_make, train_set=train_set_ds, test_set=test_set_ds,
                            k_folds=5, epochs=100, batch_size=32,
                            criterion_make=criterion_make, optimizer_make=optimizer_make,
                            scheduler_make=scheduler_make, random_seed=20974241)


    # trainer1.K_fold_train_and_val(f'{save_path}/nn_save/kfold/',
    #                               f'{network_name}{net_suffix}')
    # trainer1.save_metrics(f'{save_path}',
    #                       f'kfold_metrics{net_suffix}.csv')


    update_csv_with_auc_scores(f'{save_path}/best_{network_name}{net_suffix}_kfold.csv',
                               f'{save_path}/best_{network_name}{net_suffix}_kfold_AUC.csv',
                               trainer1,
                               f'{network_name}{net_suffix}', f'{save_path}/nn_save/kfold/')







    # trainer1.train_on_full_dataset('../NN_models/resnet18_trans/','resnet18_trans')
    # trainer1.save_metrics('../NN_models/resnet18_trans/', f'test_metrics.csv')
