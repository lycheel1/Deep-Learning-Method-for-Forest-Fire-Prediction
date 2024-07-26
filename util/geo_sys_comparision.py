import pandas as pd
import pyproj
import os
import concurrent.futures
import osgeo
import fiona
import rioxarray


# Define CRS using EPSG codes
WGS84 = pyproj.CRS("EPSG:4326")
NAD83 = pyproj.CRS("EPSG:4269")
EPSG3347_Statistics_Canada_Lambert = pyproj.CRS("+proj=lcc +lat_0=63.390675 +lon_0=-91.8666666666667 +lat_1=49 +lat_2=77 +x_0=6200000 +y_0=3000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs")
ESRI102002_NAD_1983_Lambert_Canada = pyproj.CRS("+proj=lcc +lat_0=40 +lon_0=-96 +lat_1=50 +lat_2=70 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs +type=crs")
EPSG3978_Canada_Atlas_Lambert = pyproj.CRS("+proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs")

str2009_0 = 'PROJCS["Canada_Lambert_Conformal_Conic",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",49],PARAMETER["central_meridian",-95],PARAMETER["standard_parallel_1",49],PARAMETER["standard_parallel_2",77],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
strS = 'PROJCS["NAD83 / Canada Atlas Lambert",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["standard_parallel_1",49],PARAMETER["standard_parallel_2",77],PARAMETER["latitude_of_origin",49],PARAMETER["central_meridian",-95],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3978"]]'
strA = 'PROJCS["Canada_Lambert_Conformal_Conic",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",49],PARAMETER["central_meridian",-95],PARAMETER["standard_parallel_1",49],PARAMETER["standard_parallel_2",77],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
strFUEL = 'PROJCS["Canada_Lambert_Conformal_Conic",GEOGCS["NAD83",DATUM["North American Datum 1983",SPHEROID["GRS 1980",6378137,298.257222101004]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",40],PARAMETER["central_meridian",-96],PARAMETER["standard_parallel_1",50],PARAMETER["standard_parallel_2",70],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'




# Define corners of the area (in WGS84)
left, right = -141.005549, -52.616607  # Longitudes
top, bottom = 83.116523, 41.669086  # Latitudes

# 102002
# left, bottom = -141.01, 38.21
# right, top = -40.73, 86.46

# EPSG3347
# left, bottom = -172.54, 23.81
# right, top = -47.74, 86.46

# Create a transformer object to convert from NAD83 to Lambert
transformer3978 = pyproj.Transformer.from_crs(WGS84, EPSG3978_Canada_Atlas_Lambert, always_xy=True)
transformer2 = pyproj.Transformer.from_crs(WGS84, str2009_0, always_xy=True)
transformer3 = pyproj.Transformer.from_crs(WGS84, strS, always_xy=True)
transformer4 = pyproj.Transformer.from_crs(WGS84, strA, always_xy=True)
transformerFUEL = pyproj.Transformer.from_crs(WGS84, strFUEL, always_xy=True)

def print4points(left, bottom, right, top, transformer):
    # Define points to transform
    lon_list = [left, right, left, right]
    lat_list = [top, top, bottom, bottom]

    # Transform points
    x_list, y_list = transformer.transform(lon_list, lat_list)

    # Print transformed points
    for (x, y) in zip(x_list, y_list):
        print(f"({int(x)},{int(y)})")  # Casting to int will truncate decimals
    print("\n")

print4points(left, bottom, right, top, transformer3978)
print4points(left, bottom, right, top, transformer2)
print4points(left, bottom, right, top, transformer3)
print4points(left, bottom, right, top, transformer4)
print4points(left, bottom, right, top, transformerFUEL)