import numpy as np
import os
import pandas as pd

def fill_inf_with_neighbor_average(old_array):
    # Find the indices of NaN values
    array = old_array.copy()
    inf_indices = np.argwhere(np.isinf(array))

    for index in inf_indices:
        i, j = index
        neighbors = []
        # Check all eight neighbors
        for x in [i-1, i, i+1]:
            for y in [j-1, j, j+1]:
                if x >= 0 and x < array.shape[0] and y >= 0 and y < array.shape[1] and (x != i or y != j):
                    if not np.isinf(array[x, y]):
                        neighbors.append(array[x, y])
        if neighbors:
            array[i, j] = np.mean(neighbors)
        else:
            print(f'Inf at all the neighbors')

    return array

if __name__ == '__main__':
    instances_path = '../../data/fire instances 64 200/'

    years = np.arange(1994, 2022)

    year_list = []
    instance_list = []

    for year in years:
        print(f'year {year}')
        yearly_instances_path = f'{instances_path}/{str(year)}/'
        for fire_instance_folder in os.listdir(yearly_instances_path):

            fire_instance_path = f'{yearly_instances_path}/{fire_instance_folder}'
            file_path = f'{fire_instance_path}/{fire_instance_folder} elev.npy'
            elevation = np.load(file_path)

            if np.isinf(elevation).any():
                print(fire_instance_folder)
                new_elevation = fill_inf_with_neighbor_average(elevation)

                np.save(f'{fire_instance_path}/{fire_instance_folder} elev_inf.npy', elevation)
                np.save(file_path, new_elevation)

            # in consideration of repetitively running
            if os.path.exists(f'{fire_instance_path}/{fire_instance_folder} elev_inf.npy'):
                year_list.append(year)
                instance_list.append(fire_instance_folder)

    df = pd.DataFrame({'year': year_list, 'instance': instance_list})
    df.to_csv('elev_inf_check.csv', index=False)