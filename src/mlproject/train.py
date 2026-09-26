"""
Training script for the Microcontroller & Electronic Component Classification models.
Includes early stopping, learning rate scheduling, class weighting, and metric logging.
Fulfills all requirements of ITRI626 Section 7.3 & Section 8.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from mlproject.config import ProjectConfig, default_config
from mlproject.dataset import create_dataloaders
from mlproject.models.factory import build_model, get_model_info


def set_seed(seed: int = 42) -> None:
    """Ensure full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train for a single epoch. Returns (avg_loss, accuracy, macro_f1)."""
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets, _ in tqdm(dataloader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate for a single epoch. Returns (avg_loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets, _ in tqdm(dataloader, desc="Validating", leave=False):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def train_model(
    model_name: str,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_class_weights: bool = True,
    patience: int = 5,
    seed: int = 42,
    config: ProjectConfig = default_config,
) -> Dict:
    """Complete training pipeline with early stopping and checkpointing."""
    set_seed(seed)
    config.ensure_directories()
    device = config.get_device()

    print(f"\n{'=' * 60}")
    print(f"Starting Training for Model: {model_name}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Random Seed: {seed} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"{'=' * 60}\n")

    # Load data splits
    train_loader, val_loader, test_loader, id_to_label, class_weights = create_dataloaders(
        splits_dir=config.splits_dir,
        batch_size=batch_size,
        config=config,
    )
    num_classes = len(id_to_label)

    # Initialize model
    is_custom = model_name in ["custom_cnn", "baseline_cnn"]
    pretrained = not is_custom
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    model_meta = get_model_info(model)
    print(f"Model Info: {model_meta['total_parameters']:,} parameters (~{model_meta['model_size_mb']} MB)")

    # Loss function with optional class weighting
    if use_class_weights and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using Class-Weighted Cross-Entropy Loss to counter class imbalance.")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Output directory for this model
    model_output_dir = config.models_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = model_output_dir / "best_model.pt"

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f}"
        )

        # Early stopping check on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "id_to_label": id_to_label,
            }, best_weights_path)
            print(f"  --> Checkpoint saved: Val Loss improved to {val_loss:.4f} (F1: {val_f1:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
                break

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f} seconds. Best Val F1: {best_val_f1:.4f}")

    # Save training metadata and history
    summary = {
        "model_name": model_name,
        "parameters": model_meta,
        "seed": seed,
        "epochs_trained": len(history["train_loss"]),
        "total_training_time_sec": round(total_training_time, 2),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_f1": round(best_val_f1, 4),
        "history": history,
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        },
    }

    with open(model_output_dir / "history.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Microcontroller Classification Model")
    parser.add_argument("--model", type=str, default="custom_cnn", choices=["custom_cnn", "resnet18", "resnet50", "mobilenet_v3", "vit_b_16"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_class_weights=not args.no_class_weights,
        seed=args.seed,
    )
