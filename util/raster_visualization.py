import numpy as np
import matplotlib.pyplot as plt


def raster_visualization(file_path):

    # Load the 2D array from the .npy file
    data = np.load(file_path)

    print(data)
    print('\n')
    print(np.sum(data))

    # Create a figure and an axis to plot
    fig, ax = plt.subplots()

    # Display the data as an image, i.e., on a 2D regular raster
    cax = ax.imshow(data, cmap='viridis')  # 'viridis' is a commonly used colormap; you can choose others

    # Add a colorbar to provide a scale for the image
    fig.colorbar(cax)

    # Optionally, set title and labels
    ax.set_title('2D Distribution')
    ax.set_xlabel('X Pixels')
    ax.set_ylabel('Y Pixels')

    # Show the plot
    plt.show()

fwi_npy_path = f'../data/fire instances 64 200'
#
# raster_visualization(f'{fwi_npy_path}/2016/2016_20/2016_20 elev.npy')
# raster_visualization(f'{fwi_npy_path}/2016/2016_20/2016_20 ba.npy')
#
# raster_visualization(f'{fwi_npy_path}/2016/2016_20/2016_20 2016-05-02 hs.npy')
# raster_visualization(f'{fwi_npy_path}/2016/2016_20/2016_20 2016-05-03 hs.npy')
#
# raster_visualization(f'{fwi_npy_path}/2016/2016_20/2016_20 2016-05-02 chs.npy')
# raster_visualization(f'{fwi_npy_path}/2016/2016_20/2016_20 2016-05-03 chs.npy')

# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-13 hs.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-13 hs_poly.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 ba.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-14 hs.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-14 hs_poly.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-15 hs.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-15 hs_poly.npy')

# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-13 chs.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-13 chs_poly.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-14 chs.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-14 chs_poly.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-15 chs.npy')
# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 2006-06-15 chs_poly.npy')

# raster_visualization(f'{fwi_npy_path}/2006/2006_11/2006_11 ba.npy')


# raster_visualization(f'{fwi_npy_path}/1994/1994_1/1994_1 1994-07-27 hs.npy')
# raster_visualization(f'{fwi_npy_path}/1994/1994_1/1994_1 1994-07-27 hs_poly.npy')

# raster_visualization(f'{fwi_npy_path}/1994/1994_1/1994_1 1994-07-26 fwi.npy')
# raster_visualization(f'{fwi_npy_path}/1994/1994_1/1994_1 1994-07-27 fwi.npy')

raster_visualization(f'{fwi_npy_path}/1994/1994_22/1994_22 fuel.npy')
#
# raster_visualization(f'{fwi_npy_path}/1994/1994_114/1994_114 1994-06-27 fwi.npy')
#
# raster_visualization(f'{fwi_npy_path}/1994/1994_114/1994_114 1994-06-28 fwi.npy')
#
# raster_visualization(f'{fwi_npy_path}/1994/1994_114/1994_114 1994-06-29 fwi.npy')

# raster_visualization(f'../data/ds samples 64 200/24841/24841 hsp.npy')
# raster_visualization(f'../data/ds samples 64 200/24841/24841 chsp.npy')
# raster_visualization(f'../data/ds samples 64 200/24841/24841 elev.npy')
# raster_visualization(f'../data/ds samples 64 200/24841/24841 fuel.npy')
# raster_visualization(f'../data/ds samples 64 200/24841/24841 fwi.npy')
# raster_visualization(f'../data/ds samples 64 200/24841/24841 Ycp.npy')

# raster_visualization(f'{fwi_npy_path}/2020/2020_223/2020_223 2020-06-27 hs.npy')
# raster_visualization(f'{fwi_npy_path}/2020/2020_223/2020_223 2020-06-27 hs_poly.npy')
# raster_visualization(f'{fwi_npy_path}/2020/2020_223/2020_223 ba.npy')
# raster_visualization(f'{fwi_npy_path}/2020/2020_223/2020_223 2020-06-28 hs.npy')
# raster_visualization(f'{fwi_npy_path}/2020/2020_223/2020_223 2020-06-28 hs_poly.npy')
# raster_visualization(f'{fwi_npy_path}/2020/2020_223/2020_223 2020-06-29 hs.npy')
# raster_visualization(f'{fwi_npy_path}/2020/2020_223/2020_223 2020-06-29 hs_poly.npy')
