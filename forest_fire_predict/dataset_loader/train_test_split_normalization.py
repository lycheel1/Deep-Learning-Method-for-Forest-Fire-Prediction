from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def normalization(array_input, min_val, max_val):
    return (array_input - min_val) / (max_val - min_val)

def update_csv_meta(csv_path, train_indices, test_indices, train_csv_path, test_csv_path):
    # Read the original CSV
    df = pd.read_csv(csv_path)

    # Create train and test DataFrames using the indices
    df_train = df.iloc[train_indices]
    df_test = df.iloc[test_indices]

    # Save the new CSV files
    df_train.to_csv(train_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)


def split_and_normalize(X_2D_path, X_1D_path, Y_path, save_folder):

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
        print(f"NaN at sample {pos[0]}, channel {pos[1]}, position ({pos[2]}, {pos[3]})")
    temp = set(nan_positions_2D[:,0])
    print(temp)
    print(len(temp))





    indices = np.arange(0, len(X_2D))

    # shuffle and split indices
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=20974241)

    # split dataset by indices
    X_2D_train, X_2D_test = X_2D[train_indices], X_2D[test_indices]
    X_1D_train, X_1D_test = X_1D[train_indices], X_1D[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]

    # Compute min and max
    min_2D = np.min(X_2D_train, axis=(0, 2, 3), keepdims=True)
    max_2D = np.max(X_2D_train, axis=(0, 2, 3), keepdims=True)

    # Standardize the training data
    X_2D_train = normalization(X_2D_train, min_2D, max_2D)
    # Standardize the test data using training mean and std
    X_2D_test = normalization(X_2D_test, min_2D, max_2D)

    # Save the preprocessed datasets
    np.save(f'{save_folder}/X_2D_train.npy', X_2D_train)
    np.save(f'{save_folder}/X_2D_test.npy', X_2D_test)
    np.save(f'{save_folder}/X_1D_train.npy', X_1D_train)
    np.save(f'{save_folder}/X_1D_test.npy', X_1D_test)
    np.save(f'{save_folder}/Y_train.npy', Y_train)
    np.save(f'{save_folder}/Y_test.npy', Y_test)

    print('dataset split done')
    return (train_indices, test_indices)


if __name__ == '__main__':
    ds_path = '../data/dataset_hs_hsp'
    train_idx, test_idx = split_and_normalize(f'{ds_path}/hs_hsp_2D.npy', f'{ds_path}/hs_hsp_1D.npy',f'{ds_path}/hs_hsp_Y.npy', ds_path)
    update_csv_meta(f'{ds_path}/hs_hsp_samples_meta.csv', train_idx, test_idx,
                    f'{ds_path}/train_meta.csv', f'{ds_path}/test_meta.csv')
