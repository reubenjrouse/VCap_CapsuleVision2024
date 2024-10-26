# testing.py

import torch
import pandas as pd
from typing import List, Tuple
from torch.utils.data import DataLoader
from training import EndoscopyDataset, get_transforms, EndoscopyClassifier

def predict(model: torch.nn.Module, test_loader: DataLoader, device: str = 'cuda') -> Tuple[List[int], List[int]]:
    model.eval()
    binary_preds, multi_preds = [], []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            binary_output, multi_output = model(images, inference=True)
            binary_preds.extend(torch.sigmoid(binary_output).cpu().numpy() > 0.5)
            multi_preds.extend(multi_output.argmax(dim=1).cpu().numpy())

    return binary_preds, multi_preds

def generate_test_loader(data_dir: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    test_dataset = EndoscopyDataset(data_dir=data_dir, split='testing', transform=get_transforms('testing'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader

def save_predictions(binary_preds: List[int], multi_preds: List[int], output_file: str = 'predictions.csv'):
    df = pd.DataFrame({
        'Binary Predictions': binary_preds,
        'Multi-Class Predictions': multi_preds
    })
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def test_testing():
    data_dir = 'path/to/data'  # Replace with actual path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EndoscopyClassifier(num_classes=10).to(device)
    test_loader = generate_test_loader(data_dir)

    binary_preds, multi_preds = predict(model, test_loader, device=device)
    save_predictions(binary_preds, multi_preds, output_file='test_predictions.csv')

if __name__ == "__main__":
    test_testing()
