import os
import geopandas as gpd
import pyproj
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling


def mask_and_resample_fuel(input_tif_loc, output_npy_loc, left, bottom, right, top, width=64, height=64):
    try:
        # Open the raster file
        with rasterio.open(input_tif_loc) as src:
            # Create a bounding box
            bbox = [{'type': 'Polygon', 'coordinates': [[
                [left, bottom],
                [right, bottom],
                [right, top],
                [left, top],
                [left, bottom]
            ]]}]

            # Mask the raster with the bounding box
            out_image, out_transform = mask(src, bbox, crop=True)
            dst_transform, dst_width, dst_height = calculate_default_transform(src.crs, src.crs, width, height, left, bottom, right, top)

            # Create an array for the resampled data
            resampled_fuel = np.empty(shape=(height, width), dtype=src.dtypes[0])

            # Reproject (resample) the data
            reproject(
                source=out_image[0],  # Process the single band
                destination=resampled_fuel,
                src_transform=out_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest)

            # Save the resampled data to a .npy file
            if np.all(resampled_fuel == 0):
                print("the resampled fuel contains only 0")
            np.save(output_npy_loc, resampled_fuel)

    except ValueError as e:
        print(f"Skipping due to error: {e}")
        print(f"Problematic Zone: Left: {left}, Bottom: {bottom}, Right: {right}, Top: {top}")
        return  # Skip the rest of this function


years = np.arange(1994, 2022)
pyproj.datadir.set_data_dir('E:/anaconda3/envs/wildfire2/Library/share/proj')
os.environ['PROJ_LIB'] = 'E:/anaconda3/envs/wildfire2/Library/share/proj'

instances_path = '../../../data/fire instances 64 200/'
fuel_map_loc = '../raw_data/fuel/FBP_FuelLayer_EPSG3978_fillingzero.tif'

for year in years:
    print("start masking year " + str(year))
    yearly_instances_path = f'{instances_path}/{str(year)}/'
    for fire_instance_folder in os.listdir(yearly_instances_path):
        print("masking " + fire_instance_folder)

        fire_instance_path = f'{yearly_instances_path}/{fire_instance_folder}/'
        ba_gdf = gpd.read_file(f'{fire_instance_path}/{fire_instance_folder} ba.shp')
        ba_row = ba_gdf.iloc[0]

        left, bottom, right, top = ba_row['WB'], ba_row['SB'], ba_row['EB'], ba_row['NB']
        mask_and_resample_fuel(fuel_map_loc, f'{fire_instance_path}/{fire_instance_folder} fuel.npy', left, bottom,
                               right, top)
