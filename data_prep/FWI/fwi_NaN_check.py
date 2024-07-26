import numpy as np
import os
import pandas as pd


def contains_NaN(file_path):
    array = np.load(file_path)
    return np.isnan(array).any()

if __name__ == '__main__':
    instances_path = '../../data/fire instances 64 200/'

    years = np.arange(1994, 2022)

    year_list = []
    instance_list = []
    filename_list = []

    for year in years:
        print(f'year {year}')
        yearly_instances_path = f'{instances_path}/{str(year)}/'
        for fire_instance_folder in os.listdir(yearly_instances_path):
            fire_instance_path = f'{yearly_instances_path}/{fire_instance_folder}'
            for file in os.listdir(fire_instance_path):
                file_path = f'{fire_instance_path}/{file}'
                if file.endswith('fwi.npy'):
                    if contains_NaN(file_path):
                        year_list.append(year)
                        instance_list.append(fire_instance_folder)
                        filename_list.append(file)

    df = pd.DataFrame({'year': year_list, 'instance': instance_list, 'filename': filename_list})
    df.to_csv('fwi_nan_check.csv', index=False)