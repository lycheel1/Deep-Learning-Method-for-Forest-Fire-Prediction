import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from data_prep.crop_augmentation import CropDataAugmenter
from skimage.draw import rectangle_perimeter


def test_crop_and_resize(input_grid, rate, dz, scale_size=(64, 64)):
    # Use the static method directly without creating an instance
    crop_area = CropDataAugmenter.crop_window(input_grid, rate, dz, scale_size)

    if crop_area is not None:
        min_x, min_y, max_x, max_y = crop_area
        cropped = input_grid[min_x:max_x + 1, min_y:max_y + 1]
        resized = resize(cropped, scale_size, order=0, anti_aliasing=False, mode='edge', preserve_range=True)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(input_grid, cmap='gray')
        axes[0].set_title('Original Input')
        # Draw rectangle on the original input to show crop area
        rr, cc = rectangle_perimeter(start=(min_x, min_y), end=(max_x, max_y), shape=input_grid.shape)
        input_grid[rr, cc] = input_grid.max()  # Highlight the crop area
        axes[1].imshow(resized, cmap='gray')
        axes[1].set_title('Cropped & Resized Output')
    else:
        # Plotting only the input if cropping is not feasible
        plt.imshow(input_grid, cmap='gray')
        plt.title('Original Input Only (Cropping Not Feasible)')
        print('Not feasible')

    plt.show()

if __name__ == '__main__':
    input_grid1 = np.load('../data/ds samples 64 200/112/112 Ycp.npy')
    input_grid2 = np.load('../data/ds samples 64 200/19256/19256 Ycp.npy')

    test_crop_and_resize(input_grid1, rate=1.2, dz=1, scale_size=(64, 64))
