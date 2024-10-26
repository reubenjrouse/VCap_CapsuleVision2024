
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from PIL import Image
from datetime import datetime


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded file saved at {save_path}")

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped files to {extract_to}")
    os.remove(zip_path)
    print(f"Deleted zip file at {zip_path}")


url = 'https://figshare.com/ndownloader/files/48018562'
download_path = 'downloaded_file.zip'

download_file(url, download_path)
unzip_file(download_path, '/data/mpstme-reuben2/data')

class EndoscopyDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str = 'training',
                 transform=None):
        """
        Args:
            data_dir: Root directory of the dataset
            split: 'training' or 'validation'
            transform: Optional transform to be applied on images
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Get all image paths and labels
        self.image_paths, self.labels = self._build_dataset()

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        # Store class weights for balanced sampling
        self._compute_class_weights()

    def _build_dataset(self) -> Tuple[List[str], List[str]]:
        """Build dataset from directory structure"""
        image_paths = []
        labels = []

        split_dir = self.data_dir / self.split
        for disease_folder in split_dir.iterdir():
            if disease_folder.is_dir():
                disease_name = disease_folder.name
                # Traverse through source datasets
                for source_folder in disease_folder.iterdir():
                    if source_folder.is_dir():
                        for img_path in source_folder.glob('*.jpg'):
                            image_paths.append(str(img_path))
                            labels.append(disease_name)

        return image_paths, labels

    def _compute_class_weights(self):
        """Compute class weights for balanced sampling"""
        class_counts = pd.Series(self.labels).value_counts()
        total_samples = len(self.labels)
        self.class_weights = {
            label: total_samples / (len(class_counts) * count)
            for label, count in class_counts.items()
        }

        # Convert to sample weights
        self.sample_weights = [
            self.class_weights[label] for label in self.labels
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.encoded_labels[idx]

        return image, label

def get_transforms(split: str):
    """Get transforms for training/validation"""
    if split == 'training':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders(data_dir: str,
                      batch_size: int = 32,
                      num_workers: int = 4) -> Dict[str, DataLoader]:
    """Create training and validation dataloaders"""

    # Create datasets
    train_dataset = EndoscopyDataset(
        data_dir=data_dir,
        split='training',
        transform=get_transforms('training')
    )

    val_dataset = EndoscopyDataset(
        data_dir=data_dir,
        split='validation',
        transform=get_transforms('validation')
    )

    # Create weighted sampler for training
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'label_encoder': train_dataset.label_encoder
    }


# Create dataloaders
dataloaders = create_dataloaders(
    data_dir="/data/mpstme-reuben2/data/Dataset",
    batch_size=32,
    num_workers=4
)

# Print some information
for phase in ['train', 'val']:
    loader = dataloaders[phase]
    print(f"{phase} dataset size: {len(loader.dataset)}")
    print(f"{phase} number of batches: {len(loader)}")

# Print class mapping
label_encoder = dataloaders['label_encoder']
print("\nClass mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name}: {i}")



class FeatureAttention(nn.Module):
    """Attention mechanism for feature fusion"""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_models, feature_dim)
        attention_weights = F.softmax(self.attention(x), dim=1)
        weighted_features = x * attention_weights
        return weighted_features.sum(dim=1)

class FeatureFusion(nn.Module):
    """Fusion module for combining features from different models"""
    def __init__(self, feature_dims: Dict[str, int], output_dim: int):
        super().__init__()
        self.dim_match = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in feature_dims.items()
        })
        self.attention = FeatureAttention(output_dim)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        aligned_features = [
            self.dim_match[name](feat)
            for name, feat in features.items()
        ]
        stacked = torch.stack(aligned_features, dim=1)
        return self.attention(stacked)

class EndoscopyClassifier(nn.Module):
    """Two-stage classifier for endoscopy images"""
    def __init__(self,
                 num_classes: int,
                 pretrained: bool = True,
                 abnormal_threshold: float = 0.5):
        super().__init__()

        # Model configuration
        self.num_classes = num_classes
        self.abnormal_threshold = abnormal_threshold

        # Feature extractors
        self.backbone_models = nn.ModuleDict({
            'efficientnet': timm.create_model('resnet50', pretrained=pretrained),
            'vit': timm.create_model('deit_base_patch16_224', pretrained=pretrained),
            'convnext': timm.create_model('mobilenetv3_large_100', pretrained=pretrained)
        })

        # Get feature dimensions
        self.feature_dims = {
            'efficientnet': self.backbone_models['efficientnet'].fc.in_features,
            'vit': self.backbone_models['vit'].head.in_features,
            'convnext': self.backbone_models['convnext'].classifier.in_features
        }

        # Remove original classifiers
        self.backbone_models['efficientnet'].fc = nn.Identity()
        self.backbone_models['vit'].head = nn.Identity()
        self.backbone_models['convnext'].classifier = nn.Identity()

        # Feature fusion
        self.fusion_dim = 512
        self.feature_fusion = FeatureFusion(self.feature_dims, self.fusion_dim)

        # Classifiers
        self.binary_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.multi_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from all backbone models"""
        return {
            name: model(x)
            for name, model in self.backbone_models.items()
        }

    def forward(self,
                x: torch.Tensor,
                inference: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            inference: If True, uses two-stage inference logic
        Returns:
            Tuple of (binary_output, multi_output)
        """
        # Extract and fuse features
        features = self.extract_features(x)
        fused_features = self.feature_fusion(features)

        # Binary classification (normal vs abnormal)
        binary_output = self.binary_classifier(fused_features)

        if inference:
            # During inference, only perform multi-class classification
            # on samples predicted as abnormal
            abnormal_probs = torch.sigmoid(binary_output)
            abnormal_mask = abnormal_probs > self.abnormal_threshold

            # Initialize multi-class output with zeros (class 0 = normal)
            multi_output = torch.zeros(
                (x.size(0), self.num_classes),
                device=x.device
            )

            # Only compute multi-class predictions for abnormal samples
            if torch.any(abnormal_mask):
                abnormal_features = fused_features[abnormal_mask.squeeze()]
                multi_output[abnormal_mask.squeeze()] = self.multi_classifier(abnormal_features)
        else:
            # During training, compute multi-class for all samples
            multi_output = self.multi_classifier(fused_features)

        return binary_output, multi_output

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha: float = 1, gamma: float = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """Combined loss for two-stage classification"""
    def __init__(self,
                 num_classes: int,
                 focal_gamma: float = 2,
                 binary_weight: float = 1.0,
                 multi_weight: float = 1.0):
        super().__init__()
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.multi_loss = FocalLoss(gamma=focal_gamma)
        self.num_classes = num_classes
        self.binary_weight = binary_weight
        self.multi_weight = multi_weight

    def forward(self,
                binary_output: torch.Tensor,
                multi_output: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # Create binary targets (normal vs abnormal)
        binary_targets = (targets > 0).float()

        # Calculate losses
        binary_loss = self.binary_loss(binary_output.squeeze(), binary_targets)
        multi_loss = self.multi_loss(multi_output, targets)

        # Combine losses with weights
        return self.binary_weight * binary_loss + self.multi_weight * multi_loss

import os
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()

    all_binary_preds = []
    all_multi_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        batch_size = images.size(0)

        # Forward pass (training mode)
        binary_output, multi_output = model(images, inference=False)
        loss = criterion(binary_output, multi_output, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update metrics
        losses.update(loss.item(), batch_size)

        # Store predictions
        all_binary_preds.append(torch.sigmoid(binary_output).detach().cpu())
        all_multi_preds.append(torch.softmax(multi_output, dim=1).detach().cpu())
        all_targets.append(targets.cpu())

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    # Calculate metrics
    all_binary_preds = torch.cat(all_binary_preds).numpy()
    all_multi_preds = torch.cat(all_multi_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    binary_targets = (all_targets > 0).astype(float)
    metrics = {
        'train_loss': losses.avg,
        'train_binary_auc': roc_auc_score(binary_targets, all_binary_preds),
        'train_multi_auc': roc_auc_score(all_targets, all_multi_preds, multi_class='ovr'),
        'train_balanced_acc': balanced_accuracy_score(all_targets, all_multi_preds.argmax(axis=1))
    }

    return metrics

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model"""
    model.eval()
    losses = AverageMeter()

    all_binary_preds = []
    all_multi_preds = []
    all_targets = []

    for images, targets in tqdm(val_loader, desc='Validation'):
        images, targets = images.to(device), targets.to(device)

        # Forward pass (inference mode)
        binary_output, multi_output = model(images, inference=True)
        loss = criterion(binary_output, multi_output, targets)

        losses.update(loss.item(), images.size(0))

        all_binary_preds.append(torch.sigmoid(binary_output).cpu())
        all_multi_preds.append(torch.softmax(multi_output, dim=1).cpu())
        all_targets.append(targets.cpu())

    all_binary_preds = torch.cat(all_binary_preds).numpy()
    all_multi_preds = torch.cat(all_multi_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    binary_targets = (all_targets > 0).astype(float)
    metrics = {
        'val_loss': losses.avg,
        'val_binary_auc': roc_auc_score(binary_targets, all_binary_preds),
        'val_multi_auc': roc_auc_score(all_targets, all_multi_preds, multi_class='ovr'),
        'val_balanced_acc': balanced_accuracy_score(all_targets, all_multi_preds.argmax(axis=1))
    }

    return metrics

class Trainer:
    def __init__(self,
                 data_dir: str,
                 model_class: nn.Module,
                 batch_size: int = 32,
                 num_epochs: int = 500,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 0.01,
                 patience: int = 7,
                 num_workers: int = 4,
                 project_name: str = "endoscopy-classification"):
        self.data_dir = data_dir
        self.model_class = model_class
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.num_workers = num_workers
        self.project_name = project_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_frequency = 1
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_checkpoint(self, epoch: int, model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler._LRScheduler,
                        metrics: Dict[str, float]):
        checkpoint_dir = Path('./checkpoints')  # Updated path
        checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics
        }

        filename = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, filename)

    def run(self):
        dataset = EndoscopyDataset(
            data_dir=self.data_dir,
            transform=get_transforms('training')
        )

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        # Create validation dataset and loader
        val_dataset = EndoscopyDataset(
            data_dir=self.data_dir,
            transform=get_transforms('validation')  # Ensure you have a validation transform
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        model = self.model_class(num_classes=10).to(self.device)

        criterion = CombinedLoss(num_classes=10)
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1000
        )

        early_stopping = EarlyStopping(patience=self.patience)

        # Remove loading from checkpoint
        start_epoch = 0
        best_val_auc = 0

        for epoch in range(start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            train_metrics = train_one_epoch(
                model, train_loader, criterion,
                optimizer, scheduler, self.device, epoch
            )

            metrics = {**train_metrics}
            wandb.log(metrics)

            # Print training accuracy
            print(f"Epoch {epoch + 1} - Train Balanced Accuracy: {train_metrics['train_balanced_acc']:.4f}")

            # Validate after training
            val_metrics = validate(model, val_loader, criterion, self.device)
            print(f"Epoch {epoch + 1} - Validation Loss: {val_metrics['val_loss']:.4f}")
            print(f"Epoch {epoch + 1} - Validation Balanced Accuracy: {val_metrics['val_balanced_acc']:.4f}")

            # Save checkpoints
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint(epoch, model, optimizer, scheduler, metrics)

            # Save best model
            if val_metrics['val_balanced_acc'] > best_val_auc:
                best_val_auc = val_metrics['val_balanced_acc']
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_auc': best_val_auc,
                }
                torch.save(best_checkpoint, Path('/data/mpstme-reuben2/checkpoints') / 'best_model_fold_0.pth')

            if early_stopping(train_metrics['train_loss']):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        wandb.finish()

def main():
    CONFIG = {
        'data_dir': '/data/mpstme-reuben2/data/Dataset',
        'batch_size': 128,
        'num_epochs': 500,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'patience': 15,
        'num_workers': 4,
        'project_name': 'endoscopy-classification'
    }

    # Path to the checkpoint file
    checkpoint_path = '/data/mpstme-reuben2/checkpoints/checkpoint_epoch_20.pth'  # Replace with actual path

    # Initialize Trainer and pass checkpoint path
    trainer = Trainer(
        model_class=EndoscopyClassifier,
#         checkpoint_path=checkpoint_path,  # Pass the checkpoint path here
        **CONFIG
    )

    # Run training
    trainer.run()

if __name__ == "__main__":
    main()

