import pandas as pd
import pyproj
import os
import concurrent.futures
import osgeo
import fiona
import rioxarray

def fuel_transform(fuelpath, targetpath, target_crs):
    # Open the raster data with its original CRS
    fuel_rds = rioxarray.open_rasterio(fuelpath)

    # Reproject the raster data to the target CRS
    rds_reprojected = fuel_rds.rio.reproject(target_crs)

    # Write the reprojected data to a new TIF file
    rds_reprojected.rio.to_raster(targetpath)




pyproj.datadir.set_data_dir('E:/anaconda3/envs/wildfire2/Library/share/proj')
os.environ['PROJ_LIB'] = 'E:/anaconda3/envs/wildfire2/Library/share/proj'

EPSG3978_Canada_Atlas_Lambert = pyproj.CRS("+proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs")


### FUEL
print("start to transform fuel to EPSG3978")
fuel_from = '../raw_data/fuel/Canadian_Forest_FBP_Fuel_Types_v20191114/fuel_layer/FBP_FuelLayer.tif'
fuel_to = '../raw_data/fuel/FBP_FuelLayer_EPSG3978.tif'
fuel_transform(fuel_from, fuel_to, EPSG3978_Canada_Atlas_Lambert)
print("fuel transformation finished")






