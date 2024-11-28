import numpy as np
import os
import geopandas as gpd
from util.one_hot_encoding import one_hot_encode


def fire_cause_to_onehot(gdf_name, folder_path):
    ba_gdf = gpd.read_file(f'{folder_path}/{gdf_name}')
    ba_row = ba_gdf.iloc[0]

    fc_value = int(ba_row['FIRECAUS'])
    fc_onehot = one_hot_encode(fc_value, 5)

    filename = gdf_name.replace('ba.shp',f'fc.npy')
    np.save(f'{folder_path}/{filename}', fc_onehot)



instances_path = '../../../data/fire instances 64 200'
years = np.arange(1994, 2022)
for year in years:
    print(f'Year {year}')

    yearly_instances_path = f'{instances_path}/{str(year)}'
    for fire_instance_folder in os.listdir(yearly_instances_path):
        print(f'start to process {fire_instance_folder}')

        fire_instance_path = f'{yearly_instances_path}/{fire_instance_folder}'

        ba_gdf = f'{fire_instance_folder} ba.shp'

        fire_cause_to_onehot(ba_gdf, fire_instance_path)


print('finished')