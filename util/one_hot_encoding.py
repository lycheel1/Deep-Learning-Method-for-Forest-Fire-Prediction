import numpy as np


def one_hot_encode(value, num_categories):
    # Create an array of zeros with length equal to the number of categories
    one_hot_encoded = np.zeros(num_categories, dtype=int)

    # Set the index corresponding to the specific value to 1
    one_hot_encoded[value] = 1

    return one_hot_encoded

def one_hot_to_grid(one_hot_1d, H, W):
    # (one_hot_d, 1, 1)
    one_hot_3d = np.expand_dims(np.expand_dims(one_hot_1d, axis=-1), axis=-1)
    # (one_hot_d, H, W)
    one_hot_grid = np.tile(one_hot_3d, (1, H, W))

    return one_hot_grid


# for i in np.arange(0, 5):
#
#     print('i: '+str(i))
#     one_hot_grid = one_hot_to_grid(one_hot_encode(i, 5), 3,3)
#     print(one_hot_grid)