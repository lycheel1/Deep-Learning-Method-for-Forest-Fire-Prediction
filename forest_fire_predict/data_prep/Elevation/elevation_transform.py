import osgeo
import fiona
import rioxarray
import pyproj
import os

#### remember non-value!!!!!!!!!!!
#### remember non-value!!!!!!!!!!!
#### remember non-value!!!!!!!!!!!
#### remember non-value!!!!!!!!!!!
#### remember non-value!!!!!!!!!!!
#### remember non-value!!!!!!!!!!!


def elevation_coor_transform(input_path, save_path, target_crs):
    print(f"start transforming {input_path}")
    rds = rioxarray.open_rasterio(input_path)

    # Perform the reprojection to the target CRS
    rds_reprojected = rds.rio.reproject(target_crs)

    rds_reprojected.rio.to_raster(save_path)

WGS84 = pyproj.CRS("EPSG:4326")
EPSG3978_Canada_Atlas_Lambert = pyproj.CRS("+proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs")

elevation_path = 'F:/ElevationCanada_singles/'
save_path = 'F:/ElevationCanada_singles_EPSG3978/'

print("start to transform elevation files")
for elevation_part_single in os.listdir(elevation_path):
    new_filename = elevation_part_single.replace('FABDEM_V1-2.tif', 'elev_EPSG3978.tif')
    elevation_coor_transform(f'{elevation_path}/{elevation_part_single}', f'{save_path}/{new_filename}', EPSG3978_Canada_Atlas_Lambert)




