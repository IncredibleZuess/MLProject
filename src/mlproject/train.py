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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
    accum_steps: int = 1,
    scaler: Optional["torch.amp.GradScaler"] = None,
) -> Tuple[float, float, float]:
    """Train for one epoch with optional gradient accumulation and mixed precision.
    Returns (avg_loss, accuracy, macro_f1)."""
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    use_amp = scaler is not None
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(dataloader)

    for step, (images, targets, _) in enumerate(tqdm(dataloader, desc="Training", leave=False), start=1):
        images, targets = images.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        loss_to_backprop = loss / accum_steps
        if use_amp:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()
        if step % accum_steps == 0 or step == n_batches:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def environment_info(device: torch.device) -> Dict:
    """Software/hardware details required by ITRI626 Section 7.3."""
    import platform
    import sklearn
    import torchvision
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "device": str(device),
        "gpu_name": "None",
        "gpu_memory_gb": None,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 1)
    return info


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
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    accum_steps: Optional[int] = None,
    weight_decay: Optional[float] = None,
    use_class_weights: bool = True,
    patience: Optional[int] = None,
    seed: Optional[int] = None,
    config: ProjectConfig = default_config,
) -> Dict:
    """Train one model with early stopping on validation loss and checkpointing.
    Any argument left as None is taken from config.model_settings / config."""
    ms = config.model_settings.get(model_name, {})
    epochs = epochs or ms.get("epochs", config.epochs)
    batch_size = batch_size or ms.get("batch_size", config.batch_size)
    lr = lr or ms.get("lr", config.learning_rate)
    accum_steps = accum_steps or ms.get("accum_steps", 1)
    weight_decay = weight_decay if weight_decay is not None else config.weight_decay
    patience = patience or config.early_stopping_patience
    seed = seed if seed is not None else config.seed
    pretrained = ms.get("pretrained", model_name != "custom_cnn")
    if os.environ.get("MLPROJECT_OFFLINE") == "1":  # smoke tests without internet: no weight download
        pretrained = False
    warmup = min(config.warmup_epochs, max(epochs - 1, 0))

    set_seed(seed)
    config.ensure_directories()
    device = config.get_device()
    env = environment_info(device)

    print(f"\n{'=' * 60}")
    print(f"Training {model_name} on {device} ({env['gpu_name']})")
    print(f"seed={seed} epochs={epochs} batch={batch_size}x{accum_steps} lr={lr} wd={weight_decay} pretrained={pretrained}")
    print(f"{'=' * 60}\n")

    train_loader, val_loader, _, id_to_label, class_weights = create_dataloaders(
        splits_dir=config.splits_dir, batch_size=batch_size, config=config, seed=seed,
    )
    num_classes = len(id_to_label)

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained,
                        dropout=config.dropout).to(device)
    model_meta = get_model_info(model)
    print(f"Model: {model_meta['total_parameters']:,} parameters (~{model_meta['model_size_mb']} MB)")

    weight_tensor = class_weights.to(device) if (use_class_weights and class_weights is not None) else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=config.label_smoothing)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # linear warm-up then cosine decay (stepped once per epoch)
    if warmup > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                        CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1), eta_min=1e-6)],
            milestones=[warmup],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    model_output_dir = config.models_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = model_output_dir / "best_model.pt"

    history = {k: [] for k in ["train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1", "lr", "epoch_time_sec"]}
    best_val_loss, best_val_f1, best_epoch = float("inf"), 0.0, 0
    epochs_no_improve = 0
    stopped_early = False
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                          accum_steps=accum_steps, scaler=scaler)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        dt = time.time() - t0

        for k, v in zip(history.keys(), [train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, current_lr, dt]):
            history[k].append(float(v))
        print(f"Epoch {epoch:02d}/{epochs} ({dt:.1f}s) | train loss {train_loss:.4f} acc {train_acc:.3f} F1 {train_f1:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f} F1 {val_f1:.3f} | lr {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss, best_val_f1, best_epoch = val_loss, val_f1, epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "id_to_label": id_to_label,
            }, best_weights_path)
            print(f"  -> checkpoint saved (val loss {val_loss:.4f}, val F1 {val_f1:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stopped_early = True
                print(f"Early stopping after epoch {epoch} (no val-loss improvement for {patience} epochs).")
                break

    total_time = time.time() - start_time
    print(f"Done in {total_time:.1f}s. Best epoch {best_epoch}: val loss {best_val_loss:.4f}, val F1 {best_val_f1:.4f}")

    summary = {
        "model_name": model_name,
        "parameters": model_meta,
        "seed": seed,
        "hyperparameters": {
            "pretrained_weights": "ImageNet-1k (torchvision DEFAULT)" if pretrained else "none (random init)",
            "input_size": list(config.image_size),
            "batch_size": batch_size,
            "accum_steps": accum_steps,
            "effective_batch_size": batch_size * accum_steps,
            "max_epochs": epochs,
            "early_stopping": f"val loss, patience {patience}",
            "optimizer": "AdamW",
            "lr": lr,
            "lr_schedule": f"linear warm-up {warmup} ep + cosine to 1e-6" if warmup else "cosine to 1e-6",
            "weight_decay": weight_decay,
            "loss": "cross-entropy" + (" with inverse-frequency class weights" if weight_tensor is not None else ""),
            "label_smoothing": config.label_smoothing,
            "dropout_head": config.dropout,
            "mixed_precision": scaler is not None,
            "class_weights": [round(float(w), 4) for w in class_weights] if weight_tensor is not None else None,
        },
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "total_training_time_sec": round(total_time, 2),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_f1": round(best_val_f1, 4),
        "history": history,
        "environment": env,
    }
    with open(model_output_dir / "history.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one model")
    parser.add_argument("--model", default="custom_cnn", choices=list(default_config.model_settings))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accum-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-class-weights", action="store_true")
    a = parser.parse_args()
    train_model(a.model, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, accum_steps=a.accum_steps,
                seed=a.seed, use_class_weights=not a.no_class_weights)
