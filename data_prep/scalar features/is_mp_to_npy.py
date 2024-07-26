import numpy as np
import os
import geopandas as gpd


def is_multipolygon_to_npy(gdf_name, folder_path):
    ba_gdf = gpd.read_file(f'{folder_path}/{gdf_name}')
    ba_row = ba_gdf.iloc[0]

    mp_value = ba_row['is_mp']
    # grid = np.full((64, 64), mp_value)

    filename = gdf_name.replace('ba.shp',f'mp.npy')
    np.save(f'{folder_path}/{filename}', mp_value)



instances_path = '../../data/fire instances 64 200'
years = np.arange(1994, 2022)
for year in years:
    print(f'Year {year}')

    yearly_instances_path = f'{instances_path}/{str(year)}'
    for fire_instance_folder in os.listdir(yearly_instances_path):
        print(f'start to process {fire_instance_folder}')

        fire_instance_path = f'{yearly_instances_path}/{fire_instance_folder}'

        ba_gdf = f'{fire_instance_folder} ba.shp'

        is_multipolygon_to_npy(ba_gdf, fire_instance_path)


print('finished')