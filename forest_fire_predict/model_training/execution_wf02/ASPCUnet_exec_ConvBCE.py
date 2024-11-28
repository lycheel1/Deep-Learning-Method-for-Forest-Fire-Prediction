from NN_models.models_wf02.ASPCUnet11 import Custom_ASPCUNet11
from dataset_loader.Wildfire_Dataset_old import WildfireDataset
from model_training.Trainer import WF01_Trainer
import torch
from torch.optim import AdamW
from model_training.metrics_visualization.update_csv_with_AUC import update_csv_with_auc_scores
from loss_functions.ConvMSELoss_noMask import ConvolutionalMSELoss_noMask


# Factory functions for instances that would be reset in new folds

### change model_make() ###
def model_make():
    return Custom_ASPCUNet11()


def criterion_make():
    return ConvolutionalMSELoss_noMask(kernel_size=5, attenuation=0.5, scaling_factor=1)


def optimizer_make(model):
    return AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)


def scheduler_make(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)


if __name__ == '__main__':
    ds_name = 'wf02'
    aug = ''  # _augXX

    ### CHANGE NETWORK NAME ###
    network_name = 'ASPCUNet001105'
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
                            k_folds=5, epochs=30, batch_size=32,
                            criterion_make=criterion_make, optimizer_make=optimizer_make,
                            scheduler_make=scheduler_make, random_seed=20974241)

    part = 2
    if part == 1:
        trainer1.K_fold_train_and_val(f'{save_path}/nn_save/kfold/',
                                      f'{network_name}{net_suffix}')
        trainer1.save_metrics(f'{save_path}',
                              f'kfold_metrics{net_suffix}.csv')
    #
    # ### only do this when you have the metrics ###
    #
    if part == 2:
        update_csv_with_auc_scores(f'{save_path}/best_{network_name}{net_suffix}_kfold.csv',
                                   f'{save_path}/best_{network_name}{net_suffix}_kfold_AUC.csv',
                                   trainer1,
                                   f'{network_name}{net_suffix}', f'{save_path}/nn_save/kfold/')
