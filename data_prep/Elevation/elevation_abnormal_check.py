import os
import pandas as pd
import numpy as np

def fill_abn_with_neighbor_average(old_array, lower_th, upper_th):
    # Find the indices of NaN values
    array = old_array.copy()
    abnormal_indices = np.argwhere((array <= lower_th) | (array >= upper_th))

    for index in abnormal_indices:
        i, j = index
        neighbors = []
        # Check all eight neighbors
        for x in [i-1, i, i+1]:
            for y in [j-1, j, j+1]:
                if x >= 0 and x < array.shape[0] and y >= 0 and y < array.shape[1] and (x != i or y != j):
                    if (lower_th < array[x, y]) and (array[x, y] < upper_th):
                        neighbors.append(array[x, y])
        if neighbors:
            array[i, j] = np.mean(neighbors)
        else:
            print(f'Abnormal values at all the neighbors')

    return array

if __name__ == '__main__':
    instances_path = '../../data/fire instances 64 200/'

    years = np.arange(1994, 2022)

    year_list = []
    instance_list = []
    abn_count_list = []

    for year in years:
        print(f'year {year}')
        yearly_instances_path = f'{instances_path}/{str(year)}/'
        for fire_instance_folder in os.listdir(yearly_instances_path):

            fire_instance_path = f'{yearly_instances_path}/{fire_instance_folder}'
            file_path = f'{fire_instance_path}/{fire_instance_folder} elev.npy'
            elevation = np.load(file_path)

            # locate the files containing abnormal values
            lower_threshold = -500
            upper_threshold = 9000
            if (elevation<=lower_threshold).any() or (elevation>=upper_threshold).any():
                print(f'Abnormal value in {fire_instance_folder}')
                new_elevation = fill_abn_with_neighbor_average(elevation, lower_threshold, upper_threshold)

                np.save(f'{fire_instance_path}/{fire_instance_folder} elev_abn.npy', elevation)
                np.save(file_path, new_elevation)

            # in consideration of repetitively running
            if os.path.isfile(f'{fire_instance_path}/{fire_instance_folder} elev_abn.npy'):
                # append fire instance info
                year_list.append(year)
                instance_list.append(fire_instance_folder)

                # append abn count
                old_data = np.load(f'{fire_instance_path}/{fire_instance_folder} elev_abn.npy')
                abnormal_count = np.sum((old_data <= lower_threshold) | (old_data >= upper_threshold))
                abn_count_list.append(abnormal_count)

    df = pd.DataFrame({'year': year_list, 'instance': instance_list, 'abn_count': abn_count_list})
    df.to_csv('elev_abn_check.csv', index=False)



