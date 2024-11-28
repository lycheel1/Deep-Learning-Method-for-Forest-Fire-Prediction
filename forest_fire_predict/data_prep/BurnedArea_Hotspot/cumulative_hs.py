import os
import numpy as np
import pandas as pd
import geopandas as gpd

from util.date_list import generate_date_list

""" 2 cumulative/aggregated hotspots """

def cumulative_hs_of_one_fire(folder, path):
    ba_gdf = gpd.read_file(f'{path}/{folder} ba.shp')
    ba_row = ba_gdf.iloc[0]

    cumulative_hs = np.zeros((64,64), dtype=np.float32)

    start_date = pd.to_datetime(ba_row['sDate'], format='%Y-%m-%d').date()
    end_date = pd.to_datetime(ba_row['eDate'], format='%Y-%m-%d').date()
    fire_timerange = generate_date_list(start_date, end_date)

    for date_of_fire in fire_timerange:
        if os.path.isfile(f'{path}/{folder} {str(date_of_fire)} hs.npy'):
            # add daily hotspots to the cumulative burned situation
            daily_hs = np.load(f'{path}/{folder} {str(date_of_fire)} hs.npy')

            # the cumulative_hs records the day of burning but we transform it into binary data
            cumulative_hs += daily_hs
            cumulative_hs = (cumulative_hs > 0).astype(np.float32)

            np.save(f'{path}/{folder} {str(date_of_fire)} chs.npy', cumulative_hs)



if __name__ == '__main__':
    instances_path = '../../../data/fire instances 64 200/'
    years = np.arange(1994, 2022)
    for year in years:
        print(f'Year {year}')
        yearly_instance_path = instances_path + str(year) + '/'

        for fire_instance_folder in os.listdir(yearly_instance_path):
            print(f'Start to do {fire_instance_folder}')
            fire_instance_path = f'{yearly_instance_path}/{fire_instance_folder}'
            cumulative_hs_of_one_fire(fire_instance_folder, fire_instance_path)

    print('Finished')



