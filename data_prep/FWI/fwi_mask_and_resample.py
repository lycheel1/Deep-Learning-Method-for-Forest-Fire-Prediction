import os.path
import pyproj
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from util.date_list import generate_date_list
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio.errors
import warnings


def fwi_joint_of_one_fire(fwi_path, save_path, left, bottom, right, top, raster_shape = (64, 64), expansion_factor=0.1):
    # Create a bounding boxs
    fwi_bbox = [left, bottom, right, top]
    view_bbox = list(fwi_bbox)

    # the flag for view bbox
    average_value = 0
    found_values = False

    fwi_gdf = gpd.read_file(fwi_path)

    (width, height) = raster_shape

    # if current view bbox contains no non-empty data, expand the length of border to 1.2L
    max_expansions = 10
    for expansion in range(max_expansions):
        # create the view
        view_transform = from_bounds(*view_bbox, width, height)
        shapes = ((geometry, value) for geometry, value in zip(fwi_gdf.geometry, fwi_gdf['FWI']))
        resampled_view = rasterize(shapes, out_shape=(width, height), transform=view_transform, fill=0)

        # if current view bbox contains any non-empty value, calculate the average then quit the loop
        if np.any(resampled_view != 0):
            average_value = np.mean(resampled_view[resampled_view != 0])
            found_values = True
            break  # Found non-zero values, no need to expand further

        # Expand the view bounding box
        view_bbox = [
            view_bbox[0] - expansion_factor * (view_bbox[2] - view_bbox[0]),
            view_bbox[1] - expansion_factor * (view_bbox[3] - view_bbox[1]),
            view_bbox[2] + expansion_factor * (view_bbox[2] - view_bbox[0]),
            view_bbox[3] + expansion_factor * (view_bbox[3] - view_bbox[1])
        ]

    final_raster = np.full((width, height), average_value if found_values else 0)

    if not found_values:
        print(f"{fwi_path} Failed to find any values in 10 iters")

    # Save the final raster data to a numpy file
    np.save(save_path, final_raster)



instances_path = '../../data/fire instances 64 200/'
fwi_path = '../../raw_data/weather/fwi_EPSG3978/'

years = np.arange(1994, 2022)

for year in years:
    print("start masking year " + str(year))
    yearly_instances_path = f'{instances_path}/{str(year)}/'
    for fire_instance_folder in os.listdir(yearly_instances_path):
        print("masking " + fire_instance_folder)

        fire_instance_path = f'{yearly_instances_path}/{fire_instance_folder}/'
        ba_gdf = gpd.read_file(f'{fire_instance_path}/{fire_instance_folder} ba.shp')
        ba_row = ba_gdf.iloc[0]

        left, bottom, right, top = ba_row['WB'], ba_row['SB'], ba_row['EB'], ba_row['NB']
        start_date = pd.to_datetime(ba_row['sDate'], format='%Y-%m-%d').date()
        end_date = pd.to_datetime(ba_row['eDate'], format='%Y-%m-%d').date()
        fire_timerange = generate_date_list(start_date, end_date)

        with ThreadPoolExecutor(max_workers=os.cpu_count() - 4) as executor:
            futures = {}

            for date_of_fire in fire_timerange:
                if os.path.isfile(f'{fire_instance_path}/{fire_instance_folder} {str(date_of_fire)} hs.npy'):
                    save_path_of_the_day = f'{fire_instance_path}/{fire_instance_folder} {str(date_of_fire)} fwi.npy'
                    fwi_of_the_day = f'{fwi_path}/{year}/{str(date_of_fire)} fwi_EPSG3978.shp'

                    # Submitting the task to the executor
                    future = executor.submit(fwi_joint_of_one_fire, fwi_of_the_day, save_path_of_the_day, left, bottom, right, top)
                    futures[future] = f'{fire_instance_folder} {str(date_of_fire)}'

            # Wait for all futures to complete
            for future in as_completed(futures):
                date_of_fire_instance = futures[future]
                try:
                    result = future.result()
                    print(f"{date_of_fire_instance} completed successfully.")
                except Exception as exc:
                    print(f"{date_of_fire_instance} generated an exception: {exc}")
