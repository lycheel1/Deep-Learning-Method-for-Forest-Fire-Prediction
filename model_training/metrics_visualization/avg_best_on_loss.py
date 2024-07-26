import pandas as pd


def calculate_and_save_avg_metrics(csv_path, output_csv_path):
    # Read the CSV file containing the best metrics for 5 folds
    df = pd.read_csv(csv_path)

    # Drop 'fold', 'epoch', and 'phase' columns if they exist
    columns_to_drop = ['fold', 'epoch', 'phase']
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Calculate the average of the metrics across rows for the remaining columns
    avg_metrics = df.mean(axis=0)

    # Create a new DataFrame for the average metrics
    avg_metrics_df = pd.DataFrame([avg_metrics])

    # Save the average metrics DataFrame to a new CSV file
    avg_metrics_df.to_csv(output_csv_path, index=False)

    print(f"Average metrics saved to {output_csv_path}")

if __name__ == '__main__':
    ds_name = 'wf02'
    model_name = 'MLP1'
    aug_list = [''] #____

    for aug_name in aug_list:
        # path = f'../model_performance_{ds_name}/{model_name}'
        path = f'F:/Code/model_training/model_performance_{ds_name}/{model_name}'
        metrics_path = f'{path}/kfold_metrics{aug_name}.csv'

        calculate_and_save_avg_metrics(f'{path}/best_{model_name}{aug_name}_kfold_AUC.csv', f'{path}/best_{model_name}{aug_name}_avg_AUC.csv')