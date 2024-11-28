import numpy as np
import os
import geopandas as gpd
from util.one_hot_encoding import one_hot_encode


def agency_to_onehot(gdf_name, folder_path, sorted_agency_list, num_categories):
    ba_gdf = gpd.read_file(f'{folder_path}/{gdf_name}')
    ba_row = ba_gdf.iloc[0]

    fc_value = sorted_agency_list.index(ba_row['AGENCY'])
    fc_onehot = one_hot_encode(fc_value, num_categories)

    filename = gdf_name.replace('ba.shp',f'agency.npy')
    np.save(f'{folder_path}/{filename}', fc_onehot)

def get_length_of_agency_onehot(filepath='../raw_data/burned_areas/nbac_1986_to_2022_20230630.shp'):
    gdf = gpd.read_file(filepath)
    agency_list = gdf['AGENCY'].unique().tolist()
    return agency_list

agency_list_sorted = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'QC', 'SK', 'YT', 'PC-BA', 'PC-EI', 'PC-GL', 'PC-GR', 'PC-JA', 'PC-KG', 'PC-KO', 'PC-LM', 'PC-NA', 'PC-PA', 'PC-PP', 'PC-PU', 'PC-RE', 'PC-RM', 'PC-TH', 'PC-TN', 'PC-VU', 'PC-WB', 'PC-WL', 'PC-WP', 'PC-YO']
num_of_categories = len(agency_list_sorted)


instances_path = '../../../data/fire instances 64 200'
years = np.arange(1994, 2022)
for year in years:
    print(f'Year {year}')

    yearly_instances_path = f'{instances_path}/{str(year)}'
    for fire_instance_folder in os.listdir(yearly_instances_path):
        print(f'start to process {fire_instance_folder}')

        fire_instance_path = f'{yearly_instances_path}/{fire_instance_folder}'

        ba_gdf = f'{fire_instance_folder} ba.shp'

        agency_to_onehot(ba_gdf, fire_instance_path, agency_list_sorted, num_of_categories)


print('finished')
