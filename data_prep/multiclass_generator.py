from NN_models.Unet import Custom_UNet
from dataset_loader.Wildfire_Dataset import WildfireDataset
from model_training.WF01_Trainer import WF01_Trainer
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from model_training.metrics_visualization.update_csv_with_AUC import update_csv_with_auc_scores


if __name__ == '__main__':
    ds_name = 'wf02'
    aug = '' #_augXX

    ### CHANGE NETWORK NAME ###
    network_name = 'UNet'
    net_suffix = aug

    dataset_folder = f'../data/dataset_{ds_name}'
    # save_path = f'../../model_training/model_performance_{ds_name}/{network_name}'
    save_path = f'../model_training/model_performance_{ds_name}/{network_name}'
    train_set_suffix = f'std_train{aug}'






    train_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_{train_set_suffix}.npy',
                                   f'{dataset_folder}/X_1D_{train_set_suffix}.npy',
                                   f'{dataset_folder}/Y_{train_set_suffix}.npy')
    d2 = train_set_ds.data_2d
    y = train_set_ds.Y
    # print(d2.dtype, d2.shape)
    # print(d2[100][0])
    # print(d2[100][1])
    # print(d2[0][4])
    def save_new_class(d2, flag):
        current_burn = d2[:, 0, :, :]
        cumulative_burn = d2[:, 1, :, :]
        new_class = np.empty((len(d2), 64, 64))
        for i in range(len(d2)):
            for x in range(64):
                for y in range(64):
                    if current_burn[i][x][y] == -0.3129374466413696 and cumulative_burn[i][x][y] == 1.967173621471376:
                        new_class[i][x][y] = 1
                    else:
                        new_class[i][x][y] = 0
        print(new_class[0])
        print(current_burn[0])
        print(cumulative_burn[0])
        new_class_exp = np.expand_dims(new_class, axis=1)
        new_class = np.concatenate((d2, new_class_exp), axis=1)
        """
        raw [
        
        chsp [0, 2, 4, 6
        yp [1, 3, 5, 7
        """
        if flag == "train" or flag == "test":
            np.save(dataset_folder + f"/multiclass/X_2D_std_multiclass_{flag}.npy", new_class)
        elif flag == "Y_test" or flag == "Y_train":
            np.save(dataset_folder + f"/multiclass/Y_std_multiclass{flag[2:]}.npy", new_class)

    def save_new_class_y(d2, y_data, flag):
        chsp = d2[:, 1, :, :]
        new_class = np.empty((len(d2), 64, 64))
        for i in range(len(d2)):
            for x in range(64):
                for y in range(64):
                    if y_data[i][x][y] == 1 and chsp[i][x][y] == 1.967173621471376:
                        new_class[i][x][y] = 1
                    else:
                        new_class[i][x][y] = 0
        new_class_exp = np.expand_dims(new_class, axis=1)
        y_data = np.expand_dims(y_data, axis=1)
        new_class = np.concatenate((new_class_exp, y_data), axis=1)
        np.save(dataset_folder + f"/multiclass/Y_std_multiclass_{flag}.npy", new_class)

    test_data = np.load(dataset_folder + "/X_2D_std_test.npy")
    train_data = np.load(dataset_folder + "/X_2D_std_train.npy")
    y_test = np.load(dataset_folder + "/Y_std_test.npy")
    y_train = np.load(dataset_folder + "/Y_std_train.npy")
    save_new_class(train_data, "train")
    save_new_class(test_data, "test")
    save_new_class_y(train_data, y_train, flag="train")
    save_new_class_y(test_data, y_test, flag="test")
    # test_set_ds = WildfireDataset(f'{dataset_folder}/X_2D_std_test.npy',
    #                               f'{dataset_folder}/X_1D_std_test.npy',
    #                               f'{dataset_folder}/Y_std_test.npy')

    # trainer1 = WF01_Trainer(model_make=model_make, train_set=train_set_ds, test_set=test_set_ds,
    #                         k_folds=5, epochs=30, batch_size=32,
    #                         criterion_make=criterion_make, optimizer_make=optimizer_make zX,
    #                         scheduler_make=scheduler_make, random_seed=20974241)


    # trainer1.K_fold_train_and_val(f'{save_path}/nn_save/kfold/',
    #                               f'{network_name}{net_suffix}')
    # trainer1.save_metrics(f'{save_path}',
    #                       f'kfold_metrics{net_suffix}.csv')


    # update_csv_with_auc_scores(f'{save_path}/best_{network_name}{net_suffix}_kfold.csv',
    #                            f'{save_path}/best_{network_name}{net_suffix}_kfold_AUC.csv',
    #                            trainer1,
    #                            f'{network_name}{net_suffix}', f'{save_path}/nn_save/kfold/')







    # trainer1.train_on_full_dataset('../NN_models/resnet18_trans/','resnet18_trans')
    # trainer1.save_metrics('../NN_models/resnet18_trans/', f'test_metrics.csv')
