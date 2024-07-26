from shutil import copyfile

import pandas as pd
import numpy as np
import os


def concat_2D_features_single_sample(sample_path, index, feature_list_2D):
    # concat 2D features
    list2D = []

    for suffix_2D in feature_list_2D:
        list2D.append(np.load(f'{sample_path}/{index} {suffix_2D}.npy'))

    array_2D = np.stack(list2D, axis=0)

    return array_2D


def concat_1D_features_single_sample(sample_path, index, feature_list_1D):
    # concat 1D features
    list1D = []

    for suffix_1D in feature_list_1D:
        temp_1D = np.load(f'{sample_path}/{index} {suffix_1D}.npy')
        if temp_1D.ndim == 0:
            temp_1D = temp_1D.reshape(1)
        list1D.append(temp_1D)

    array_1D = np.concatenate(list1D, axis=0)

    return array_1D


def concat_Y_features_single_sample(sample_path, index, Y_list):
    # return Y
    Y = np.load(f'{sample_path}/{index} {Y_list[0]}.npy')

    if not np.any(Y):
        raise Exception(f'Y only contains zeros for sample {index}')

    return Y


def number_of_samples(file_path):
    df = pd.read_csv(file_path)
    return len(df)


def concat_samples(folder, data_type, feature_list):
    N = number_of_samples(f'{folder}/samples_meta.csv')

    list_arrays = []

    for index in np.arange(0, N):
        print(f'index: {index}')

        sample_path = f'{folder}/{index}'
        match data_type:
            case '2D':
                array = concat_2D_features_single_sample(sample_path, index, feature_list)
            case '1D':
                array = concat_1D_features_single_sample(sample_path, index, feature_list)
            case 'Y':
                array = concat_Y_features_single_sample(sample_path, index, feature_list)
            case _:
                raise Exception(f'Datatype {data_type} not found in sample {index}')

        list_arrays.append(array)

    return np.stack(list_arrays, axis=0)


def save_dataset(folder, save_path, list_2d, list_1d, list_Y, ds_name):
    print('start to generate dataset for 2D feautres')
    np.save(f'{save_path}/{ds_name}_2D.npy', concat_samples(folder, data_type='2D', feature_list=list_2d))

    print('start to generate dataset for 1D feautres')
    np.save(f'{save_path}/{ds_name}_1D.npy', concat_samples(folder, data_type='1D', feature_list=list_1d))

    print('start to generate dataset for Y')
    np.save(f'{save_path}/{ds_name}_Y.npy', concat_samples(folder, data_type='Y', feature_list=list_Y))


if __name__ == '__main__':
    ### change the name of input ###
    ds_name = 'wf03'

    folder_path = '../data/ds samples 64 200'
    save_path = f'../data/dataset_{ds_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ### change the input and output ###
    list_2D = ['hsp', 'chsp', 'elev', 'fuel', 'fwi'] # hsp: burned area of the day , chsp: all burned area from day0 to dayi, 
    list_1D = ['month', 'agency', 'fc']
    list_Y = ['Yp'] # 

    save_dataset(folder_path, save_path, list_2d=list_2D, list_1d=list_1D, list_Y=list_Y, ds_name=ds_name)

    # copy meta but add augmentation_flag
    df = pd.read_csv(f'{folder_path}/samples_meta.csv')

    # augmentation flag, -1 means no augmentation
    df['aug'] = 'None'

    # Write the modified DataFrame to a new CSV file
    df.to_csv(f'{save_path}/{ds_name}_samples_meta.csv', index=False)

