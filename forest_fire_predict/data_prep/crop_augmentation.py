import os
import math
import numpy as np
from skimage.transform import resize
import pandas as pd

class CropDataAugmenter:
    def __init__(self, train_2D_path, train_1D_path, train_Y_path, meta_path):
        self.train_2D = np.load(train_2D_path)
        self.train_1D = np.load(train_1D_path)
        self.train_Y = np.load(train_Y_path)
        self.train_meta = pd.read_csv(meta_path)

    @staticmethod
    def crop_window(target_grid, rate, dz, scale_size=(64,64)):
        if scale_size[0] != scale_size[1] or scale_size[0]<1:
            raise Exception('Scale area should be a square, with size (a,a), where a is non-negative int.')
        if rate < 1:
            raise Exception('Invalid rate. Rate should be not less than 1.')

        # Find hotspot coordinates
        hotspot_coords = np.argwhere(target_grid == 1)

        max_border_index = scale_size[0] - 1

        # Determine the area window
        min_coords = hotspot_coords.min(axis=0)
        max_coords = hotspot_coords.max(axis=0)
        min_x, min_y = min_coords[0], min_coords[1]
        max_x, max_y = max_coords[0], max_coords[1]

        # Calculate the cropping window if possible
        x_length = max_x - min_x + 1
        y_length = max_y - min_y + 1
        max_edge_length = max(x_length, y_length)
        if math.ceil(max_edge_length * rate) > scale_size[0] - 2*dz:
            return None

        # Northing direction (x)
        buffered_length = math.ceil(scale_size[0]/rate)

        x_buffer = (buffered_length - x_length) // 2
        crop_min_x = min_x - x_buffer
        crop_max_x = max_x + x_buffer

        # keep the buffered border in range
        if crop_min_x < 0:
            crop_max_x += -crop_min_x
            crop_min_x = 0
        elif crop_max_x > max_border_index:
            crop_min_x -= (crop_max_x - max_border_index)
            crop_max_x = max_border_index

        # Easting direction
        y_buffer = (buffered_length - y_length) // 2
        crop_min_y = min_y - y_buffer
        crop_max_y = max_y + y_buffer

        # keep the buffered border in range
        if crop_min_y < 0:
            crop_max_y += -crop_min_y
            crop_min_y = 0
        elif crop_max_y > max_border_index:
            crop_min_y -= (crop_max_y - max_border_index)
            crop_max_y = max_border_index

        return (crop_min_x, crop_min_y, crop_max_x, crop_max_y)


    def crop_and_scale(self, save_path, rate, dz, scale_size=(64, 64)):
        train_len = len(self.train_meta)

        aug_df = pd.DataFrame(columns=self.train_meta.columns)

        list_2D = []
        list_1D = []
        list_Y = []

        for sample_index in range(0, train_len):
            print(f"Augmenting {sample_index}/{train_len-1} rate={rate} dz={dz}", end=' --- ')

            sample_2D = self.train_2D[sample_index].copy()
            sample_1D = self.train_1D[sample_index].copy()
            sample_Y = self.train_Y[sample_index].copy()
            crop_area = self.crop_window(target_grid=sample_Y, rate=rate, dz=dz)

            # if the area is ovetr 3/4 original or cropping is invalid
            if crop_area is None:
                print("Not feasible, SKIP")
                continue

            # if cropping is valid
            min_x, min_y, max_x, max_y = crop_area
            cropped_2D = sample_2D[:, min_x:max_x + 1, min_y:max_y + 1]
            cropped_Y = sample_Y[min_x:max_x + 1, min_y:max_y + 1]

            scaled_2D = np.array(
                [resize(feature, scale_size, mode='edge',order=0, anti_aliasing=False, preserve_range=True) for feature in cropped_2D])
            scaled_Y = resize(cropped_Y, scale_size, mode='edge', order=0, anti_aliasing=False, preserve_range=True)

            # save the augmented sample
            list_2D.append(scaled_2D)
            list_1D.append(sample_1D)
            list_Y.append(scaled_Y)

            # copy the meta but set the augmentation flag to the size of buffer
            temp_df = self.train_meta.iloc[[sample_index]].copy()
            temp_df['aug'] = rate
            aug_df = pd.concat([aug_df, temp_df], ignore_index=True)

            print('Success')

        # stack data by creating new axis
        array_2D = np.stack(list_2D, axis=0)
        array_1D = np.stack(list_1D, axis=0)
        array_Y = np.stack(list_Y, axis=0)
        np.save(f'{save_path}/X_2D_std_aug{rate}.npy', array_2D)
        np.save(f'{save_path}/X_1D_std_aug{rate}.npy', array_1D)
        np.save(f'{save_path}/Y_std_aug{rate}.npy', array_Y)
        aug_df.to_csv(f'{save_path}/aug{rate}_std_meta.csv', index=False)

        print(f'Augmentation with rate {rate} and deadzone {dz} Done. {len(array_Y)} samples generated')


    def add_augmented_data(self, path, list_rate, name):
        np.random.seed(20974241)

        combined_2D = self.train_2D.copy()
        combined_1D = self.train_1D.copy()
        combined_Y = self.train_Y.copy()
        combined_meta = self.train_meta.copy()

        for rate in list_rate:
            print(f'adding rate={rate}')

            # load augmented data
            aug_2D = np.load(f'{path}/X_2D_std_aug{rate}.npy')
            aug_1D = np.load(f'{path}/X_1D_std_aug{rate}.npy')
            aug_Y = np.load(f'{path}/Y_std_aug{rate}.npy')
            aug_meta = pd.read_csv(f'{path}/aug{rate}_std_meta.csv')

            # Concatenate original and augmented data along existing axis 0
            combined_2D = np.concatenate([combined_2D, aug_2D], axis=0)
            combined_1D = np.concatenate([combined_1D, aug_1D], axis=0)
            combined_Y = np.concatenate([combined_Y, aug_Y], axis=0)
            combined_meta = pd.concat([combined_meta, aug_meta], ignore_index=True)

        # Generate a shared index to shuffle data and metadata in unison
        indices = np.arange(combined_2D.shape[0])
        np.random.shuffle(indices)

        # Apply the same shuffle to data and metadata
        shuffled_2D = combined_2D[indices]
        shuffled_1D = combined_1D[indices]
        shuffled_Y = combined_Y[indices]
        shuffled_meta = combined_meta.iloc[indices].reset_index(drop=True)

        np.save(f'{path}/X_2D_std_train_aug{name}.npy', shuffled_2D)
        np.save(f'{path}/X_1D_std_train_aug{name}.npy', shuffled_1D)
        np.save(f'{path}/Y_std_train_aug{name}.npy', shuffled_Y)
        shuffled_meta.to_csv(f'{path}/train_aug{name}_std_meta.csv', index=False)

        print('Added and shuffled')







if __name__ == '__main__':
    ds_name = 'wf01'
    ds_path = '../data/dataset_wf01'

    da = CropDataAugmenter(train_2D_path=f'{ds_path}/X_2D_std_train.npy',
                           train_1D_path=f'{ds_path}/X_1D_std_train.npy',
                           train_Y_path=f'{ds_path}/Y_std_train.npy',
                           meta_path=f'{ds_path}/train_std_meta.csv')

    # generating the augmented data with different rate
    # da.crop_and_scale(save_path=ds_path, rate=1.25, dz=1)

    # da.add_augmented_data(path=ds_path, list_rate=[1.5, 2, 3, 4, 5], name='A')
    # da.add_augmented_data(path=ds_path, list_rate=[1.5, 2, 3], name='B')
    da.add_augmented_data(path=ds_path, list_rate=[1.25, 1.5, 2], name='C')

























