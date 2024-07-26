import numpy as np
import pandas as pd
from skimage.transform import resize

class NoiseDataAugmenter:
    def __init__(self, train_2D_path, train_1D_path, train_Y_path, meta_path):
        self.train_2D = np.load(train_2D_path)
        self.train_1D = np.load(train_1D_path)
        self.train_Y = np.load(train_Y_path)
        self.train_meta = pd.read_csv(meta_path)

    def add_noise_to_2D_feature(self, feature, feature_type, trigger):
        if trigger is False:
            return feature.copy()
        if feature_type == 'binary':
            return self.add_noise_binary(feature)
        elif feature_type == 'categorical':
            return self.add_noise_categorical(feature)  # Define your categories
        elif feature_type == 'continuous':
            return self.add_noise_continuous(feature)
        else:
            raise ValueError("Unknown feature type")

    def generate_noise_augmented_data(self, save_path, feature_types, triggers, aug_name):
        aug_2D = []
        BASE_SEED = 20974241
        for index in range(0, self.train_2D.shape[0]):
            sample_seed = BASE_SEED + index  # Unique seed for each sample
            np.random.seed(sample_seed)

            print(f'adding noise to sample {index}/{self.train_2D.shape[0]-1}')
            sample = self.train_2D[index]
            noisy_sample = np.array([self.add_noise_to_2D_feature(feature, ft, trigger) for feature, ft, trigger in zip(sample, feature_types, triggers)])
            aug_2D.append(noisy_sample)

        # Copying 1D features and labels as they are not modified
        aug_1D = self.train_1D.copy()
        aug_Y = self.train_Y.copy()

        # Generating new metadata with an indicator for noise augmentation
        aug_meta = self.train_meta.copy()
        aug_meta['aug'] = aug_name

        # Saving augmented data and new metadata
        np.save(f'{save_path}/X_2D_std_aug{aug_name}.npy', np.array(aug_2D))
        np.save(f'{save_path}/X_1D_std_aug{aug_name}.npy', aug_1D)
        np.save(f'{save_path}/Y_std_aug{aug_name}.npy', aug_Y)
        aug_meta.to_csv(f'{save_path}/aug{aug_name}_std_meta.csv', index=False)

        print(f'Noise-augmented data generated and saved to {save_path}.')

# Helper functions for adding noise, adjust based on your dataset and requirements
    def add_noise_binary(self, feature, flip_prob=0.01):
        feature_bool = feature.astype(bool)

        # Generate a noise mask of the same shape as `feature`
        noise_mask = np.random.rand(*feature.shape) < flip_prob

        noisy_feature_bool = np.bitwise_xor(feature_bool, noise_mask)
        noisy_feature = noisy_feature_bool.astype(feature.dtype)

        return noisy_feature

    def add_noise_categorical(self, feature, subst_rate=0.01):
        # Calculate the set of unique categorical values in the feature
        unique_categories = np.unique(feature)

        # Determine the number of pixels to substitute based on the substitution rate
        total_pixels = feature.size
        n_subst = int(subst_rate * total_pixels)

        # Randomly select pixel indices to substitute
        subst_indices = np.random.choice(total_pixels, size=n_subst, replace=False)

        # Generate new random category values for each selected pixel
        new_values = np.random.choice(unique_categories, size=n_subst)

        # Create a copy of the feature to avoid modifying the original data
        feature_copy = feature.copy()

        # Apply the substitutions
        for idx, new_val in zip(subst_indices, new_values):
            # Convert the 1D index back to 2D indices
            row_idx, col_idx = np.unravel_index(idx, feature.shape)
            feature_copy[row_idx, col_idx] = new_val

        return feature_copy

    def add_noise_continuous(self, feature, mean=0, std=0.05):
        gaussian_noise = np.random.normal(mean, std, feature.shape)
        noisy_feature = feature + gaussian_noise
        return noisy_feature

    def add_augmented_data(self, path, list_type, name):
        np.random.seed(20974241)

        combined_2D = self.train_2D.copy()
        combined_1D = self.train_1D.copy()
        combined_Y = self.train_Y.copy()
        combined_meta = self.train_meta.copy()

        for aug_type in list_type:
            print(f'adding aug_type={aug_type}')

            # load augmented data
            aug_2D = np.load(f'{path}/X_2D_std_aug{aug_type}.npy')
            aug_1D = np.load(f'{path}/X_1D_std_aug{aug_type}.npy')
            aug_Y = np.load(f'{path}/Y_std_aug{aug_type}.npy')
            aug_meta = pd.read_csv(f'{path}/aug{aug_type}_std_meta.csv')

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
    ds_name = 'wf02'
    ds_path = '../data/dataset_wf02'

    da = NoiseDataAugmenter(train_2D_path=f'{ds_path}/X_2D_std_train.npy',
                           train_1D_path=f'{ds_path}/X_1D_std_train.npy',
                           train_Y_path=f'{ds_path}/Y_std_train.npy',
                           meta_path=f'{ds_path}/train_std_meta.csv')

    # hsp, chsp, ele, fuel, fwi
    # da.generate_noise_augmented_data(save_path=ds_path,
    #                                  feature_types=['binary','binary','continuous','categorical','continuous'],
    #                                  triggers=[True,False,False,False,False],
    #                                  aug_name='N1')

    da.add_augmented_data(path=ds_path, list_type=['N3'], name='N3')






