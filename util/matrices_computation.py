import torch

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