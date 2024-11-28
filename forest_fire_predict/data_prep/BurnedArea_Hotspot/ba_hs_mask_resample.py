import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import os
import numpy as np
import pandas as pd
from util.date_list import generate_date_list

""" masks """


# given the transform and target geometry, rasterize it to a 128*128 grid
def rasterize_geometry(geometry, transform, raster_shape=(64, 64)):
    # Rasterize the geometry using the provided transform
    raster = rasterize([(geometry, 1)], out_shape=raster_shape, transform=transform, fill=0, all_touched=True)
    return raster

def get_raster_of_daily_hotspots(hs_gdf, transform, raster_shape=(64, 64)):
    raster = np.zeros(raster_shape, dtype=np.uint8)
    for idx, hs_row in hs_gdf.iterrows():
        raster_single_hs = rasterize_geometry(hs_row['geometry'], transform)
        raster |= raster_single_hs
    return raster

# generate 64*64 cells, each is resolution meters 200m
def get_transform_of_burned_area(geometry, resolution=200, raster_shape=(64, 64)):
    # Calculate the centroid
    centroid = geometry.centroid

    # Calculate the bounding box based on the resolution and raster shape
    half_width = resolution * raster_shape[1] / 2
    half_height = resolution * raster_shape[0] / 2
    bbox = box(centroid.x - half_width, centroid.y - half_height, centroid.x + half_width, centroid.y + half_height)

    # Define the raster's transform and shape
    transform = rasterio.transform.from_bounds(*bbox.bounds, width=raster_shape[1], height=raster_shape[0])

    print(f"centroid: ({centroid.x},{centroid.y})")

    return *bbox.bounds, transform


fires_path = '../../../data/fire/'
instances_path = '../../../data/fire instances 64 200/'
years = np.arange(1994, 2022)

if not os.path.exists(instances_path):
    os.makedirs(instances_path)

for year in years:
    fires_yearly_path = fires_path + str(year)
    instances_yearly_path = instances_path + str(year)
    if not os.path.exists(instances_yearly_path):
        os.makedirs(instances_yearly_path)

    # rasterizing each fire in a year
    print("start rasterizing year "+str(year))
    for fire_folder in os.listdir(fires_yearly_path):
        # create instances folders
        print("fire "+fire_folder+" processing")
        fire_path = fires_yearly_path + '/' + fire_folder
        instance_path = instances_yearly_path + '/' + fire_folder
        if not os.path.exists(instance_path):
            os.makedirs(instance_path)

        # get the transform from a burned_area
        # add boundaries into burned area rows and store it in instance storage path
        ba_gdf = gpd.read_file(fire_path+'/'+fire_folder+'.shp')
        ba_row = ba_gdf.iloc[0]
        W, S, E, N, transform = get_transform_of_burned_area(ba_row['geometry'])
        row_gdf = gpd.GeoDataFrame([ba_row], geometry='geometry')
        row_gdf['WB'] = W
        row_gdf['SB'] = S
        row_gdf['EB'] = E
        row_gdf['NB'] = N
        row_gdf.to_file(f'{instance_path}/{fire_folder} ba.shp',driver='ESRI Shapefile')

        # rasterizing ba
        raster_ba = rasterize_geometry(ba_row['geometry'], transform)
        np.save(f'{instance_path}/{fire_folder} ba.npy', raster_ba)

        start_date = pd.to_datetime(ba_row['sDate'], format='%Y-%m-%d').date()
        end_date = pd.to_datetime(ba_row['eDate'], format='%Y-%m-%d').date()
        fire_timerange = generate_date_list(start_date, end_date)
        for date_of_fire in fire_timerange:
            current_hs = fire_path+'/'+str(date_of_fire) + '.shp'
            current_hs = f'{fire_path}/{str(date_of_fire)}.shp'
            # if the hotspot file is not empty
            if os.path.exists(current_hs):
                hs_gdf = gpd.read_file(current_hs)
                raster_hs = get_raster_of_daily_hotspots(hs_gdf, transform)

                if not np.all(raster_hs == 0):
                    np.save(f'{instance_path}/{fire_folder} {str(date_of_fire)} hs.npy', raster_hs)









