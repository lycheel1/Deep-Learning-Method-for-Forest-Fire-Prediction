import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

""" 3 Run after cumulative_hs. Turn hs/chs into polygon"""

def hs_points_to_polygon(hs_points_path, hs_polygon_path, ba_path, np_to_cv=True):
    hs_grid = np.load(hs_points_path)
    hs_grid = np.round(hs_grid)

    actual_burned_area = np.load(ba_path)

    # Find the coordinates of points marked as '1'
    points = np.column_stack(np.where(hs_grid == 1))

    # Create an empty grid image to draw the polygon
    polygon_image = np.zeros_like(hs_grid)

    output = None

    # Check if there are enough points to form a polygon
    if len(points) > 2:
        # Calculate the convex hull of the points
        hull = cv2.convexHull(points)

        # Draw and fill the polygon (convex hull)
        cv2.drawContours(polygon_image, [hull], -1, color=1, thickness=-1)

        output = polygon_image

        # np and cv has different order of reading dimensions (y first for cv)
        if np_to_cv:
            output = np.transpose(output, (1, 0))

        output = output*actual_burned_area

    else:
        output = hs_grid

    np.save(hs_polygon_path, output)


def plot_binary_image(image):
    # Plotting the binary image (0s will be black, 1s will be white)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.axis('on')  # Turn off axis numbers and labels
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def points_to_polygon_one_instance(current_instance_path, current_folder, old_suffix, new_suffix):
    for filename in os.listdir(current_instance_path):
        if filename.endswith(old_suffix):
            print(f'{filename}')
            new_filename = filename.replace(old_suffix, new_suffix)
            hs_points_to_polygon(f'{current_instance_path}/{filename}',
                                 f'{current_instance_path}/{new_filename}',
                                 f'{current_instance_path}/{current_folder} ba.npy')


if __name__ == '__main__':
    old_suffix = ' hs.npy'
    new_suffix = ' hs_poly.npy'

    # old_suffix = ' chs.npy'
    # new_suffix = ' chs_poly.npy'

    instances_path = '../../data/fire instances 64 200/'
    years = np.arange(1994, 2022)
    for year in years:
        print(f'Year {year}')
        yearly_instance_path = instances_path + str(year) + '/'
        for fire_instance_folder in os.listdir(yearly_instance_path):
            print(f'Start to do {fire_instance_folder}')
            fire_instance_path = f'{yearly_instance_path}/{fire_instance_folder}'

            points_to_polygon_one_instance(fire_instance_path, fire_instance_folder, old_suffix, new_suffix)

    print('finished')

# zeros = np.zeros((10,10))
# test1_data = zeros.copy()
# test1_data[1, 2] = 1
# test1_data[5, 5] = 1
# test1_data[0, 7] = 1
# test1_data[7, 3] = 1
# test1_data[7, 8] = 1
#
# plot_binary_image(test1_data)
# result = hs_points_to_polygon_test(test1_data)
# plot_binary_image(result)
