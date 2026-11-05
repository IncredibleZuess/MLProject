"""
Dataset, data transforms, and DataLoader creation for Microcontroller & Component Classification.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from mlproject.config import ProjectConfig, default_config


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Realistic training data augmentations for microcontroller & electronics images:
    - Random rotation (boards can be placed at arbitrary angles on a desk)
    - Horizontal and vertical flips (electronic components are symmetric or reversible on PCB)
    - Color jitter (accommodates varying lighting / shadows on workbench)
    - Random affine (scale and slight shear)
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_test_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Deterministic transforms for validation and test sets (no augmentations, preventing data leakage).
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class MicrocontrollerDataset(Dataset):
    """
    PyTorch Dataset loading images based on manifest CSV containing:
    filepath, label, label_id, [optional: group_id]
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        transform: Optional[Callable] = None,
        base_dir: Optional[Path] = None,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.manifest.iloc[idx]
        raw_path = Path(row["relative_path"] if "relative_path" in row else row["filepath"])
        img_path = raw_path if raw_path.is_absolute() or self.base_dir is None else self.base_dir / raw_path

        image = Image.open(img_path).convert("RGB")
        label_id = int(row["label_id"])

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, label_id, str(img_path)


def compute_class_weights(manifest: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """
    Compute balanced class weights for handling class imbalance using inverse frequency:
    weight[c] = total_samples / (num_classes * count[c])
    """
    counts = manifest["label_id"].value_counts().to_dict()
    total_samples = len(manifest)
    weights = []
    for c in range(num_classes):
        count = counts.get(c, 1)
        w = total_samples / (num_classes * count)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    splits_dir: Optional[Path] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    config: ProjectConfig = default_config,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str], Optional[torch.Tensor]]:
    """
    Create PyTorch DataLoaders for train, validation, and test splits.
    Returns: (train_loader, val_loader, test_loader, class_mapping, class_weights)
    """
    splits_path = splits_dir or config.splits_dir
    b_size = batch_size or config.batch_size
    n_workers = num_workers if num_workers is not None else config.num_workers

    train_df = pd.read_csv(splits_path / "train.csv")
    val_df = pd.read_csv(splits_path / "val.csv")
    test_df = pd.read_csv(splits_path / "test.csv")

    # Extract label mapping
    all_labels = sorted(train_df["label"].unique())
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    train_df["label_id"] = train_df["label"].map(label_to_id)
    val_df["label_id"] = val_df["label"].map(label_to_id)
    test_df["label_id"] = test_df["label"].map(label_to_id)

    train_transform = get_train_transforms(config.image_size, config.mean, config.std)
    eval_transform = get_val_test_transforms(config.image_size, config.mean, config.std)

    train_dataset = MicrocontrollerDataset(train_df, transform=train_transform, base_dir=config.data_dir)
    val_dataset = MicrocontrollerDataset(val_df, transform=eval_transform, base_dir=config.data_dir)
    test_dataset = MicrocontrollerDataset(test_df, transform=eval_transform, base_dir=config.data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=b_size,
        shuffle=True,
        num_workers=n_workers,
        generator=torch.Generator().manual_seed(seed if seed is not None else config.seed),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
    )

    class_weights = compute_class_weights(train_df, len(all_labels))
    return train_loader, val_loader, test_loader, id_to_label, class_weights
