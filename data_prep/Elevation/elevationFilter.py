import os
import pickle
import shutil

# Please change the path to the folder where elevation pieces are
# For myself, it is an external SSD
directory_path = 'H:/Elevation/ElevationSingle'
destination_directory = 'F:/ElevationCanada_singles'

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)


# Canada extent boundaries
top = 83.116523
bottom = 41.669086
left = -141.005549
right = -52.616607

# List to store the names of files within the Canada extent
filtered_files = []

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith("_FABDEM_V1-2.tif"):
        basename = filename[:-len("_FABDEM_V1-2.tif")]
        # Extract latitude and longitude from the filename
        # lat and lon has a fixed position and length in here
        lat_str = basename[1:3]
        lon_str = basename[4:7]
        # W and S are negative value in file names
        lat = float(lat_str) if 'N' in basename else -float(lat_str)
        lon = float(lon_str) if 'E' in basename else -float(lon_str)

        # Check if the file is within the Canada extent
        # the lat of lon is the west south point of a square so the boundaries of N/E side should +1
        left_bound, right_bound = lon, lon+1
        bottom_bound, top_bound = lat, lat+1
        #
        if not (right_bound < left or left_bound > right or bottom_bound > top or top_bound < bottom):
            filtered_files.append(filename)

# Print the filtered files
for file in filtered_files:
    shutil.copy(os.path.join(directory_path, file), destination_directory)
    print(file+" copied")
print("range length: " + str(len(filtered_files)))
print("total length: " + str(len(os.listdir(directory_path))))

