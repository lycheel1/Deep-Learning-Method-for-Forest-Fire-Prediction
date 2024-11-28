import os

import rioxarray
import rasterio
import numpy as np

def elevation_non_value_transformation(elevation_from, elevation_to):
    with rasterio.open(elevation_from) as src:
        # Read the single band in fuel file
        band1 = src.read(1)

        # Get the 'no data' value from the metadata
        no_data_value = src.nodata

        # Replace 'no data' values with 0
        band1[band1 == no_data_value] = 0

        # Copy the metadata and update the 'no data' value
        out_meta = src.meta.copy()
        out_meta.update(nodata=0)

        # Write out to a new file with the updated 'no data' value and modified band
        with rasterio.open(elevation_to, 'w', **out_meta) as dst:
            dst.write(band1, 1)  # Write the modified band

print("Start to change all no_data_value in fuel (65535) to 0")
elevation_from ='F:/ElevationCanada_singles_EPSG3978'
elevation_to = 'F:/ElevationCanada_singles_EPSG3978_fillingzeros'

for elevation_file in os.listdir(elevation_from):
    new_filename = elevation_file.replace('elev_EPSG3978.tif', 'elev_EPSG3978_fz.tif')
    elevation_non_value_transformation(f'{elevation_from}/{elevation_file}', f'{elevation_to}/{new_filename}')
print("finished changing no_data_value in elevation")