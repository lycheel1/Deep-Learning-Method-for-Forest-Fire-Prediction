from NN_models.models_wf03.Unet import Custom_UNet
from dataset_loader.Wildfire_Dataset import WildfireDataset
from model_training.Trainer import WF01_Trainer
import torch
import torch.nn as nn
from torch.optim import AdamW
import importlib
from util.load_config import load_config




# Factory functions for instances that would be reset in new folds

def load_model_class(config):
    """
    adaptively impport NN
    """
    module_name = config['model']['module']
    class_name = config['model']['class']
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

def model_make():
    """
    Call load_model_class and return an instance of the returned class
    """
    return load_model_class(config)()


def criterion_make():
    return nn.BCELoss()


# def optimizer_make(model):
#     return AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
def optimizer_make(model):
    return AdamW(model.parameters(), lr=config['model_train']['learning_rate'], weight_decay=config['model_train']['weight_decay'])


def scheduler_make(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)


if __name__ == '__main__':
    ### you could choose to overwrite the config read from the yaml ###
    dataset_folder_overwrite = "D:/Waterloo/Masc/Wildfire/Wildfire_Project/forest_fire_predict/data/dataset_wf03"
    save_path_without_model_name_overwrite = "F:/Code/model_training/model_performance_wf03"


    config = load_config('model_configs/config_AAUnet11.yaml')

    ds_name = config['data']['ds_name']
    aug = '' #_augXX

    # network_name = 'UNet1'
    network_name = config['model']['model_name']
    net_suffix = aug

    if dataset_folder_overwrite is None:
        dataset_folder = config['data']['train_path']
    else:
        dataset_folder = dataset_folder_overwrite

    if save_path_without_model_name_overwrite is None:
        save_path = config['data']['save_path']
    else:
        save_path = f'{save_path_without_model_name_overwrite}{network_name}{net_suffix}'

    train_set_suffix = f'train{aug}'



    train_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_{train_set_suffix}.npy',
                                   f'{dataset_folder}/X_1D_{train_set_suffix}.npy',
                                   f'{dataset_folder}/Y_{train_set_suffix}.npy',
                                   f'{dataset_folder}/trainSet_std.csv')

    test_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_test.npy',
                                  f'{dataset_folder}/X_1D_test.npy',
                                  f'{dataset_folder}/Y_test.npy',
                                  f'{dataset_folder}/trainSet_std.csv',)


    # trainer1 = WF01_Trainer(model_make=model_make, train_set=train_set_ds, test_set=test_set_ds,
    #                         k_folds=5, epochs=30, batch_size=32,
    #                         criterion_make=criterion_make, optimizer_make=optimizer_make,
    #                         scheduler_make=scheduler_make, random_seed=20974241)
    trainer1 = WF01_Trainer(model_make=model_make, train_set=train_set_ds, test_set=test_set_ds,
                            k_folds=config['model_train']['k_folds'], epochs=config['model_train']['epochs'],
                            batch_size=config['model_train']['batch_size'],
                            criterion_make=criterion_make, optimizer_make=optimizer_make,
                            scheduler_make=scheduler_make, random_seed=20974241)


    trainer1.K_fold_train_and_val(f'{save_path}/nn_save/kfold/',
                                  f'{network_name}{net_suffix}')
    trainer1.save_metrics(f'{save_path}',
                          f'{network_name}{net_suffix}_kfold.csv')








