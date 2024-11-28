import os
import numpy as np

fires_path = '../../data/fire/'
years = np.arange(1994, 2022)

for year in years:
    year_path = fires_path + str(year)
    yearly_empty_shp_files_count = 0
    yearly_total_shp_files_count = 0

    for fire_folder in os.listdir(year_path):
        fire_path = year_path + '/' + fire_folder
        empty_shp_files_count = 0
        total_shp_files_count = -1

        for file in os.listdir(fire_path):
            if file.lower().endswith('.shp'):
                total_shp_files_count += 1
                if "EMPTY" in file:
                    empty_shp_files_count += 1

        yearly_empty_shp_files_count += empty_shp_files_count
        yearly_total_shp_files_count += total_shp_files_count

    print(str(year) + " has non-empty hs " + str(yearly_total_shp_files_count-yearly_empty_shp_files_count) + "/" + str(yearly_total_shp_files_count))
