import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

def compute_class_weights(labels):
    import numpy as np
    class_counts = np.bincount([1 if label == 'PNEUMONIA' else 0 for label in labels])
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    return weights, class_counts

def evaluate_model(model, dataloader, device, class_weight):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_probs = []

    criterion = nn.BCELoss(weight=class_weight)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.round(outputs)
            running_corrects += torch.sum(preds == labels.data)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects.double() / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_probs)

    return loss, acc, auc
