import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def calculate_metrics(outputs_list, labels_list):

    outputs = torch.cat(tuple(outputs_list), dim=0)
    labels = torch.cat(tuple(labels_list), dim=0)

    # Flatten the tensors
    outputs_flat = (outputs > 0.5).view(-1).float()
    labels_flat = labels.view(-1).float()

    # Calculate True Positives, False Positives, True Negatives, and False Negatives
    tp = torch.sum(labels_flat * outputs_flat)
    fp = torch.sum((1 - labels_flat) * outputs_flat)
    tn = torch.sum((1 - labels_flat) * (1 - outputs_flat))
    fn = torch.sum(labels_flat * (1 - outputs_flat))

    # Calculate metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    confusion_matrix = [tp.item(), fp.item(), tn.item(), fn.item()]

    return accuracy.item(), precision.item(), recall.item(), specificity.item(), f1.item(), confusion_matrix


def calculate_AUC(all_output, all_label):
    # Concatenate all the tensors while still on the GPU
    labels_tensor = torch.cat(all_label).view(-1)  # Flatten to 1D (B * H * W)
    outputs_tensor = torch.cat(all_output).view(-1)  # Flatten to 1D (B * H * W)

    # Move the concatenated tensors to CPU and convert to numpy
    labels_array = labels_tensor.detach().cpu().numpy()
    outputs_array = outputs_tensor.detach().cpu().numpy()

    # Flatten the 2D grid (H, W) into 1D for each batch
    labels_array = labels_array.reshape(-1).astype(int)  # Flatten to 1D (B * H * W)
    outputs_array = outputs_array.reshape(-1)  # Flatten to 1D (B * H * W)

    roc_auc = roc_auc_score(labels_array, outputs_array)
    precision, recall, _ = precision_recall_curve(labels_array, outputs_array)
    pr_auc = auc(recall, precision)

    return roc_auc, pr_auc