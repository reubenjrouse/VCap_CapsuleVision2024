# validation.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from training import EndoscopyDataset, get_transforms, EndoscopyClassifier, create_dataloaders

def validate_model(model: torch.nn.Module, val_loader: DataLoader, criterion, device: str = 'cuda') -> dict:
    model.eval()
    binary_preds, multi_preds, binary_targets, multi_targets = [], [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            binary_output, multi_output = model(images, inference=True)
            binary_targets.extend((labels > 0).cpu().numpy())
            multi_targets.extend(labels.cpu().numpy())
            binary_preds.extend(torch.sigmoid(binary_output).cpu().numpy() > 0.5)
            multi_preds.extend(multi_output.argmax(dim=1).cpu().numpy())
            loss = criterion(binary_output, multi_output, labels)
            running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / len(val_loader.dataset)
    binary_accuracy = accuracy_score(binary_targets, binary_preds)
    multi_accuracy = accuracy_score(multi_targets, multi_preds)
    report = classification_report(multi_targets, multi_preds, output_dict=True)
    cm = confusion_matrix(multi_targets, multi_preds)

    metrics = {
        'val_loss': avg_loss,
        'binary_accuracy': binary_accuracy,
        'multi_accuracy': multi_accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }
    return metrics

def display_metrics(metrics: dict):
    print(f"Validation Loss: {metrics['val_loss']:.4f}")
    print(f"Binary Classification Accuracy: {metrics['binary_accuracy']:.4f}")
    print(f"Multi-Class Classification Accuracy: {metrics['multi_accuracy']:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(metrics['classification_report']).transpose())
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

def test_validation():
    data_dir = 'path/to/data'  # Replace with actual path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EndoscopyClassifier(num_classes=10).to(device)
    _, val_loader, label_encoder = create_dataloaders(data_dir)['val'], create_dataloaders(data_dir)['label_encoder']
    criterion = CombinedLoss(num_classes=10)

    metrics = validate_model(model, val_loader, criterion, device=device)
    display_metrics(metrics)

if __name__ == "__main__":
    test_validation()
