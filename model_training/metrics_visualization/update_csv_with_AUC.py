import os
import pandas as pd

def update_csv_with_auc_scores(csv_path, new_csv_path, trainer, model_name, weight_save_path):
    # Load the CSV file containing the best metrics for 5 folds
    df = pd.read_csv(csv_path)

    # Check if AUC columns exist, if not, initialize them
    if 'ROC_AUC' not in df.columns:
        df['ROC_AUC'] = None
    if 'PR_AUC' not in df.columns:
        df['PR_AUC'] = None

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        fold = row['fold']
        epoch = row['epoch']
        phase = row['phase']
        print(f'Starting AUC calculation for fold {fold}, epoch {epoch}, phase {phase}')

        # Generate the path for the model weights
        model_weights_path = f'{weight_save_path}/{model_name}_fold{fold}_epoch{epoch}.pth'

        # Check if the model weights file exists
        if os.path.exists(model_weights_path):
            # Call get_AUC to get the ROC AUC and PR AUC values
            roc_auc, pr_auc = trainer.get_AUC(model_weights_path=model_weights_path)
            # Update the DataFrame with the AUC scores
            df.at[index, 'ROC_AUC'] = roc_auc
            df.at[index, 'PR_AUC'] = pr_auc
        else:
            raise Exception(f'model parmeters not found {model_weights_path}')

    # Save the updated DataFrame to the same CSV file
    df.to_csv(new_csv_path, index=False)
    print("Updated CSV saved.")
