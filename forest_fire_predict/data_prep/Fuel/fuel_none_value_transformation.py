import rioxarray
import rasterio
import numpy as np

def fuel_non_value_transformation(fuel_from, fuel_to):
    with rasterio.open(fuel_from) as src:
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
        with rasterio.open(fuel_to, 'w', **out_meta) as dst:
            dst.write(band1, 1)  # Write the modified band

print("Start to change all no_data_value in fuel (65535) to 0")
fuel_from = '../raw_data/fuel/FBP_FuelLayer_EPSG3978.tif'
fuel_to = '../raw_data/fuel/FBP_FuelLayer_EPSG3978_fillingzero.tif'
fuel_non_value_transformation(fuel_from, fuel_to)
print("finished changing no_data_value in fuel")
