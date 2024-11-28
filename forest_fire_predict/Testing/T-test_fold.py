import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from scipy.stats import ttest_rel
import pandas as pd


def compare_models_from_csv(file1, file2):
    """
    比较两个CSV文件中的模型F1分数，并进行配对t检验以确定第一个模型是否显著优于第二个模型。

    参数:
    - file1: 第一个模型的CSV文件路径
    - file2: 第二个模型的CSV文件路径

    返回:
    - t统计量和p值
    """
    # 读取CSV文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 确保两个文件具有相同的折数
    assert len(df1) == len(df2), "两个模型的折数应相同"

    # 提取F1分数
    f1_scores_model1 = df1['f1_score']
    f1_scores_model2 = df2['f1_score']

    # 计算t检验
    t_stat, p_value = ttest_rel(f1_scores_model1, f1_scores_model2)

    return t_stat, p_value

if __name__ == '__main__':
    ds_name = 'wf02'
    aug = '' #_augXX

    ### CHANGE NETWORK NAME ###
    network1_name = 'ASPCUNet11'
    network2_name = 'Unet'
    net_suffix = aug

    dataset_folder = f'../data/dataset_{ds_name}/'
    csv_path1 = f'F:/Code/model_training/model_performance_{ds_name}/{network1_name}/best_{network1_name}_kfold.csv'
    csv_path2 = f'F:/Code/model_training/model_performance_{ds_name}/{network2_name}/best_{network2_name}_kfold.csv'

    print(compare_models_from_csv(csv_path1, csv_path2))

