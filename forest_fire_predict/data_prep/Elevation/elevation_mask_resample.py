import os
import numpy as np
from util.bbox_crs_transformation import transform_bbox_crs
import pyproj
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import pandas as pd
from util.date_list import generate_date_list


def get_relevant_elevation_tiles_v2(EPSG3978_bbox, tile_folder, file_suffix='_elev_EPSG3978_fz.tif'):

    WGS84 = pyproj.CRS("EPSG:4326")
    EPSG3978_Canada_Atlas_Lambert = pyproj.CRS(
        "+proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs")

    left, bottom, right, top = transform_bbox_crs(EPSG3978_bbox, EPSG3978_Canada_Atlas_Lambert, WGS84)


    relevant_tiles = []

    # Calculate the range of tiles that intersect with the bbox
    min_x_tile = int(np.floor(left))
    max_x_tile = int(np.ceil(right))
    min_y_tile = int(np.floor(bottom))
    max_y_tile = int(np.ceil(top))

    # Iterate over the range and add the file paths for the relevant tiles
    for x in range(min_x_tile, max_x_tile):
        for y in range(min_y_tile, max_y_tile):
            # Determine the Northing and Easting prefixes
            ns_prefix = 'N' if y >= 0 else 'S'
            ew_prefix = 'W' if x <= 0 else 'E'

            # Format the filename
            tile_file_name = f"{ns_prefix}{abs(y):02d}{ew_prefix}{abs(x):03d}{file_suffix}"
            if os.path.exists(f'{tile_folder}/{tile_file_name}'):
                relevant_tiles.append(tile_file_name)

    return relevant_tiles

def elevation_mask_and_resample(relevant_tiles, tile_folder_path, output_npy_loc, left, bottom, right, top, width=64, height=64):
    """
    Mask and resample the elevation data for a given bounding box with averaging for overlaps.

    Parameters:
    relevant_tiles (list): List of file paths for the relevant elevation tiles.
    bbox (tuple): A tuple of (left, bottom, right, top) coordinates.
    output_npy_loc (str): Location to save the output numpy file.
    width (int): Width of the resampled grid.
    height (int): Height of the resampled grid.
    """
    # Create a bounding box polygon
    bbox= [{
        'type': 'Polygon', 'coordinates': [[
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
            [left, bottom]
        ]]
    }]

    # Initialize arrays to accumulate the data and count the overlaps
    accumulated_data = np.zeros((height, width), dtype=np.float32)
    overlap_count = np.zeros((height, width), dtype=np.int32)

    for tile_name in relevant_tiles:
        tile_path = f'{tile_folder_path}/{tile_name}'
        with rasterio.open(tile_path) as src:
            # Mask the raster with the bounding box
            try:
                out_image, out_transform = mask(src, bbox, crop=True, filled=False)
            except ValueError as e:
                print(f"Skipping tile {tile_name} due to no overlap with the bounding box.")
                continue  # Skip to the next tile

            if out_image.size == 0:
                continue  # Skip if the masked image is empty

            # Calculate the transform and dimensions for the resampled data
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, src.crs, width, height, left, bottom, right, top)

            # Resample the data
            resampled_data = np.empty(shape=(height, width), dtype=src.dtypes[0])

            reproject(
                source=out_image[0],
                destination=resampled_data,
                src_transform=out_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest)

            # Accumulate the resampled data and update the overlap count
            accumulated_data += resampled_data
            overlap_count += (resampled_data != 0)  # Increment count where we have data

    # Average the accumulated data where there are overlaps
    with np.errstate(divide='ignore', invalid='ignore'):
        averaged_data = np.true_divide(accumulated_data, overlap_count, where=overlap_count!=0)

    # Save the averaged data to a .npy file
    np.save(output_npy_loc, averaged_data)


instances_path = '../../../data/fire instances 64 200'
elevation_path = 'F:/ElevationCanada_singles_EPSG3978_fillingzeros'

years = np.arange(1994, 2022)

for year in years:
    print("start masking year " + str(year))
    yearly_instances_path = f'{instances_path}/{str(year)}/'

    for fire_instance_folder in os.listdir(yearly_instances_path):
        print(f"start to mask and resample {fire_instance_folder}")

        fire_instance_path = f'{yearly_instances_path}/{fire_instance_folder}/'
        ba_gdf = gpd.read_file(f'{fire_instance_path}/{fire_instance_folder} ba.shp')
        ba_row = ba_gdf.iloc[0]

        left, bottom, right, top = ba_row['WB'], ba_row['SB'], ba_row['EB'], ba_row['NB']
        ba_bbox = (left, bottom, right, top)
        start_date = pd.to_datetime(ba_row['sDate'], format='%Y-%m-%d').date()
        end_date = pd.to_datetime(ba_row['eDate'], format='%Y-%m-%d').date()
        fire_timerange = generate_date_list(start_date, end_date)

        tile_name_list = get_relevant_elevation_tiles_v2(ba_bbox, elevation_path)
        elevation_mask_and_resample(tile_name_list, elevation_path, f'{yearly_instances_path}/{fire_instance_folder}/{fire_instance_folder} elev.npy', left, bottom, right, top)


