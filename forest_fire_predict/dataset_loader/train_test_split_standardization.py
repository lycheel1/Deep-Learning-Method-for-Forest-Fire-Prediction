import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def standardization(array_input, mean, std):
    return (array_input - mean) / std

def update_csv_meta(csv_path, train_indices, test_indices, train_csv_path, test_csv_path):
    # Read the original CSV
    df = pd.read_csv(csv_path)

    # Create train and test DataFrames using the indices
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()

    # Save the new CSV files
    df_train.to_csv(train_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)


def split_and_standardize(X_2D_path, X_1D_path, Y_path, save_folder):

    X_2D = np.load(X_2D_path)
    X_1D = np.load(X_1D_path)
    Y = np.load(Y_path)

    nan_in_2D = np.isnan(X_2D).any()
    nan_in_1D = np.isnan(X_1D).any()
    print("NaN in 2D data:", nan_in_2D)
    print("NaN in 1D data:", nan_in_1D)

    nan_positions_2D = np.argwhere(np.isnan(X_2D))

    # Print the positions of NaN values
    for pos in nan_positions_2D:
        raise Exception(f"NaN at sample {pos[0]}, channel {pos[1]}, position ({pos[2]}, {pos[3]})")

    inf_positions = np.argwhere(np.isinf(X_2D))

    # Print the sample names and positions of -inf values
    for pos in inf_positions:
        sample_index, channel, h, w = pos
        value = X_2D[sample_index, channel, h, w]
        if value == -np.inf:
            raise Exception(f"-INF at sample {pos[0]}, channel {pos[1]}, position ({pos[2]}, {pos[3]})")
        elif value == np.inf:
            raise Exception(f"+INF at sample {pos[0]}, channel {pos[1]}, position ({pos[2]}, {pos[3]})")


    indices = np.arange(0, len(X_2D))

    # shuffle and split indices
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=20974241)

    # split dataset by indices
    X_2D_train, X_2D_test = X_2D[train_indices], X_2D[test_indices]
    X_1D_train, X_1D_test = X_1D[train_indices], X_1D[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]

    # Compute mean and std from the training set
    mean_2D = np.mean(X_2D_train, axis=(0, 2, 3), keepdims=True)
    std_2D = np.std(X_2D_train, axis=(0, 2, 3), keepdims=True)

    print(mean_2D)
    print(std_2D)

    # Standardize the training data
    X_2D_train = standardization(X_2D_train, mean_2D, std_2D)
    # Standardize the test data using training mean and std
    X_2D_test = standardization(X_2D_test, mean_2D, std_2D)

    nan_in_2D = np.isnan(X_2D_train).any()
    print("NaN in 2D_train data:", nan_in_2D)


    nan_in_2D = np.isnan(X_2D_test).any()
    print("NaN in 2D_test data:", nan_in_2D)


    np.save(f'{save_folder}/mean_2D.npy', mean_2D)
    np.save(f'{save_folder}/std_2D.npy', std_2D)


    # Save the preprocessed datasets
    np.save(f'{save_folder}/X_2D_std_train.npy', X_2D_train)
    np.save(f'{save_folder}/X_2D_std_test.npy', X_2D_test)
    np.save(f'{save_folder}/X_1D_std_train.npy', X_1D_train)
    np.save(f'{save_folder}/X_1D_std_test.npy', X_1D_test)
    np.save(f'{save_folder}/Y_std_train.npy', Y_train)
    np.save(f'{save_folder}/Y_std_test.npy', Y_test)

    print('dataset split done')
    return (train_indices, test_indices)


if __name__ == '__main__':
    ### change the name of dataset ###
    ds_name = 'exp'

    ds_path = f'../data/dataset_{ds_name}'
    train_idx, test_idx = split_and_standardize(f'{ds_path}/{ds_name}_2D.npy', f'{ds_path}/{ds_name}_1D.npy',f'{ds_path}/{ds_name}_Y.npy', ds_path)
    update_csv_meta(f'{ds_path}/{ds_name}_samples_meta.csv', train_idx, test_idx,
                    f'{ds_path}/train_std_meta.csv', f'{ds_path}/test_std_meta.csv')
