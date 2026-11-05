"""
Evaluation module for the held-out test set.
Computes test accuracy, per-class & macro precision/recall/F1, confusion matrix,
inference latency, and identifies misclassified samples for error analysis.
Meets all requirements of ITRI626 Section 8.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from mlproject.config import ProjectConfig, default_config
from mlproject.dataset import create_dataloaders
from mlproject.models.factory import build_model, get_model_info


@torch.no_grad()
def evaluate_model(
    model_name: str,
    checkpoint_path: Optional[Path] = None,
    config: ProjectConfig = default_config,
) -> Dict:
    """Run full evaluation on held-out test split using saved best checkpoint."""
    config.ensure_directories()
    device = config.get_device()

    # Determine checkpoint
    ckpt_file = checkpoint_path or (config.models_dir / model_name / "best_model.pt")
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}. Please train the model first.")

    checkpoint = torch.load(ckpt_file, map_location=device)
    id_to_label = checkpoint["id_to_label"]
    # Ensure keys are int
    id_to_label = {int(k): v for k, v in id_to_label.items()}
    class_names = [id_to_label[i] for i in range(len(id_to_label))]
    num_classes = len(class_names)

    # Instantiate model and load state
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    model_info = get_model_info(model)

    # Load test dataloader
    _, _, test_loader, _, _ = create_dataloaders(splits_dir=config.splits_dir, batch_size=config.batch_size, config=config)

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_paths: List[str] = []

    # Measure inference latency
    inference_times = []

    # warm-up pass so one-off initialisation is not counted as latency
    for images, _, _ in test_loader:
        model(images[:1].to(device))
        break

    for images, targets, paths in test_loader:
        images = images.to(device)
        start = time.perf_counter()
        outputs = model(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        inference_times.append((end - start) / images.size(0))
        probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = outputs.argmax(dim=1).detach().cpu().numpy()

        all_targets.extend(targets.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_paths.extend(paths)

    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    # Core Metrics
    test_accuracy = float(accuracy_score(targets_arr, preds_arr))
    balanced_acc = float(balanced_accuracy_score(targets_arr, preds_arr))
    macro_precision = float(precision_score(targets_arr, preds_arr, average="macro", zero_division=0))
    macro_recall = float(recall_score(targets_arr, preds_arr, average="macro", zero_division=0))
    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets_arr, preds_arr, average="weighted", zero_division=0))

    # Per-Class Metrics
    per_class_report = classification_report(
        targets_arr,
        preds_arr,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion Matrix
    cm = confusion_matrix(targets_arr, preds_arr, labels=list(range(num_classes)))
    cm_norm = confusion_matrix(targets_arr, preds_arr, labels=list(range(num_classes)), normalize="true")

    # Save Confusion Matrix figure
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"Normalized Confusion Matrix: {model_name} (Acc: {test_accuracy:.2%})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_fig_path = config.figures_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(cm_fig_path, dpi=150)
    plt.close(fig)

    # Error analysis: collect misclassified samples
    misclassified = []
    correct_samples = []
    for idx, (t, p, path, prob) in enumerate(zip(targets_arr, preds_arr, all_paths, probs_arr)):
        entry = {
            "image_path": path,
            "true_class": class_names[t],
            "pred_class": class_names[p],
            "confidence": float(prob[p]),
            "true_class_probability": float(prob[t]),
        }
        if t != p:
            misclassified.append(entry)
        else:
            correct_samples.append(entry)

    # Sort misclassified by model confidence (most confident errors first)
    misclassified = sorted(misclassified, key=lambda x: x["confidence"], reverse=True)
    # correct predictions: one per class (most confident) so every class is shown
    correct_by_class = {}
    for c in sorted(correct_samples, key=lambda x: x["confidence"], reverse=True):
        correct_by_class.setdefault(c["true_class"], c)
    correct_examples = list(correct_by_class.values())

    # full per-image predictions for the notebook / appendix
    pd.DataFrame([
        {"image_path": p_, "true_class": class_names[t], "pred_class": class_names[pr],
         "confidence": float(pb[pr]), "correct": bool(t == pr)}
        for t, pr, p_, pb in zip(targets_arr, preds_arr, all_paths, probs_arr)
    ]).to_csv(config.results_dir / f"{model_name}_test_predictions.csv", index=False)

    # most frequent confusions (true -> predicted, count)
    confusions = [
        {"true_class": class_names[i], "pred_class": class_names[j], "count": int(cm[i, j])}
        for i in range(num_classes) for j in range(num_classes) if i != j and cm[i, j] > 0
    ]
    confusions = sorted(confusions, key=lambda x: x["count"], reverse=True)[:10]

    avg_latency_ms = float(np.mean(inference_times) * 1000)

    results = {
        "model_name": model_name,
        "test_accuracy": round(test_accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "mean_latency_ms_per_image": round(avg_latency_ms, 3),
        "total_test_samples": len(targets_arr),
        "misclassified_count": len(misclassified),
        "parameters": model_info,
        "per_class_report": per_class_report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "top_misclassifications": misclassified[:15],
        "top_correct_predictions": correct_examples,
        "top_confusions": confusions,
        "class_names": class_names,
    }

    results_file = config.results_dir / f"{model_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 65)
    print(f"Test Accuracy:           {test_accuracy:.2%}")
    print(f"Balanced Accuracy:       {balanced_acc:.2%}")
    print(f"Macro F1-Score:          {macro_f1:.4f}")
    print(f"Macro Precision:         {macro_precision:.4f}")
    print(f"Macro Recall:            {macro_recall:.4f}")
    print(f"Mean Latency per sample: {avg_latency_ms:.2f} ms")
    print(f"Total Parameters:        {model_info['total_parameters']:,}")
    print(f"Misclassified samples:   {len(misclassified)} / {len(targets_arr)}")
    print("-" * 65)
    print("Per-class Metrics:")
    for cls in class_names:
        metrics = per_class_report.get(cls, {})
        p = metrics.get("precision", 0.0)
        r = metrics.get("recall", 0.0)
        f = metrics.get("f1-score", 0.0)
        supp = int(metrics.get("support", 0))
        print(f"  {cls:<22} Precision: {p:.3f} | Recall: {r:.3f} | F1: {f:.3f} | Support: {supp}")
    print("=" * 65)
    print(f"Saved confusion matrix plot to: {cm_fig_path}")
    print(f"Saved full evaluation metrics to: {results_file}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Model on Test Split")
    parser.add_argument("--model", type=str, default="custom_cnn", choices=list(default_config.model_settings))
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    evaluate_model(model_name=args.model, checkpoint_path=ckpt)
