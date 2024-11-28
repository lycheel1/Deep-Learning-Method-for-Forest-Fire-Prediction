from NN_models.models_wf02.Unet_dilated import Custom_UNet_dilated
from dataset_loader.Wildfire_Dataset_old import WildfireDataset
from model_training.Trainer import WF01_Trainer
import torch
import torch.nn as nn
from torch.optim import Adam

# Factory functions for instances that would be reset in new folds
def model_make():
    return Custom_UNet_dilated()


def criterion_make():
    return nn.BCELoss()


def optimizer_make(model):
    return Adam(model.parameters(), lr=0.001)


def scheduler_make(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

if __name__ == '__main__':
    dataset_folder = '../../data/dataset_hs_hsp/'
    train_set_suffix = 'std_train_augA'

    train_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_{train_set_suffix}.npy',
                                   f'{dataset_folder}/X_1D_{train_set_suffix}.npy',
                                   f'{dataset_folder}/Y_std_{train_set_suffix}.npy')

    test_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_std_test.npy',
                                  f'{dataset_folder}/X_1D_std_test.npy',
                                  f'{dataset_folder}/Y_std_test.npy')

    trainer1 = WF01_Trainer(model_make=model_make, train_set=train_set_ds, test_set=test_set_ds,
                            k_folds=5, epochs=30, batch_size=32,
                            criterion_make=criterion_make, optimizer_make=optimizer_make,
                            scheduler_make=scheduler_make, random_seed=20974241)

    ### CHANGE NETWORK NAME ###
    network_name = 'UNet_dilated'
    net_suffix = 'augA'

    trainer1.K_fold_train_and_val(f'../../NN_models/{network_name}/nn_save/kfold/',f'{network_name}_{net_suffix}')
    trainer1.save_metrics(f'../../NN_models/{network_name}/', f'kfold_metrics_{net_suffix}.csv')

    # trainer1.train_on_full_dataset('../NN_models/resnet18_trans/','resnet18_trans')
    # trainer1.save_metrics('../NN_models/resnet18_trans/', f'test_metrics.csv')