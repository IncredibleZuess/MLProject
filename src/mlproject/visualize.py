"""
Visualization module for learning curves, model comparisons, and error analysis.
Meets requirements of ITRI626 Section 8 & Section 12.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from mlproject.config import ProjectConfig, default_config


def plot_learning_curves(
    history_file: Path,
    output_path: Optional[Path] = None,
    config: ProjectConfig = default_config,
) -> Path:
    """
    Plots training & validation loss curves and validation accuracy/F1 curves.
    """
    with open(history_file, "r") as f:
        data = json.load(f)

    model_name = data.get("model_name", "Model")
    hist = data["history"]
    epochs = range(1, len(hist["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, hist["val_loss"], label="Validation Loss", color="#d62728", linewidth=2, linestyle="--")
    axes[0].set_title(f"Loss Curves - {model_name}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)

    # Metric Curve (Accuracy & Macro F1)
    axes[1].plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, hist["val_f1"], label="Val Macro-F1", color="#9467bd", linewidth=2, linestyle="-.")
    axes[1].set_title(f"Validation Metrics - {model_name}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_ylim([0.0, 1.05])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    save_file = output_path or (config.figures_dir / f"learning_curves_{model_name}.png")
    fig.savefig(save_file, dpi=300)
    plt.close(fig)
    return save_file


def plot_model_comparison(
    results_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    config: ProjectConfig = default_config,
) -> Optional[Path]:
    """
    Creates comparative bar charts comparing all trained models across:
    - Test Accuracy
    - Macro F1-score
    - Model Parameters (Millions)
    - Inference Latency (ms)
    """
    res_dir = results_dir or config.results_dir
    result_files = list(res_dir.glob("*_results.json"))
    if not result_files:
        print("No evaluation result files found to compare.")
        return None

    records = []
    for rf in result_files:
        with open(rf, "r") as f:
            d = json.load(f)
            records.append({
                "Model": d["model_name"],
                "Test Accuracy": d["test_accuracy"],
                "Macro F1": d["macro_f1"],
                "Parameters (M)": d["parameters"]["total_parameters"] / 1e6,
                "Latency (ms)": d["mean_latency_ms_per_image"],
            })

    df = pd.DataFrame(records).sort_values("Macro F1", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Test Accuracy
    axes[0, 0].bar(df["Model"], df["Test Accuracy"], color="#1f77b4", width=0.5)
    axes[0, 0].set_title("Test Accuracy", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylim([0, 1.05])
    for i, v in enumerate(df["Test Accuracy"]):
        axes[0, 0].text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

    # Macro F1
    axes[0, 1].bar(df["Model"], df["Macro F1"], color="#2ca02c", width=0.5)
    axes[0, 1].set_title("Macro F1-Score", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylim([0, 1.05])
    for i, v in enumerate(df["Macro F1"]):
        axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # Parameters
    axes[1, 0].bar(df["Model"], df["Parameters (M)"], color="#ff7f0e", width=0.5)
    axes[1, 0].set_title("Total Parameters (Millions)", fontsize=12, fontweight="bold")
    for i, v in enumerate(df["Parameters (M)"]):
        axes[1, 0].text(i, v + 0.05, f"{v:.1f}M", ha="center")

    # Latency
    axes[1, 1].bar(df["Model"], df["Latency (ms)"], color="#d62728", width=0.5)
    axes[1, 1].set_title("Inference Latency per Image (ms)", fontsize=12, fontweight="bold")
    for i, v in enumerate(df["Latency (ms)"]):
        axes[1, 1].text(i, v + 0.1, f"{v:.1f}ms", ha="center")

    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Comparative Evaluation Across Model Families (ITRI626)", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    save_file = output_path or (config.figures_dir / "model_comparison.png")
    fig.savefig(save_file, dpi=300)
    plt.close(fig)
    return save_file


def plot_misclassified_samples(
    results_file: Path,
    num_samples: int = 8,
    output_path: Optional[Path] = None,
    config: ProjectConfig = default_config,
) -> Optional[Path]:
    """
    Renders a grid of misclassified images with true vs predicted labels and confidence.
    Used for qualitative Error Analysis in the report and presentation.
    """
    with open(results_file, "r") as f:
        data = json.load(f)

    errors = data.get("top_misclassifications", [])
    if not errors:
        print("No misclassified samples to plot.")
        return None

    num_samples = min(num_samples, len(errors))
    cols = 4
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        if i < num_samples:
            err = errors[i]
            img_path = Path(err["image_path"])
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                axes[i].imshow(img)
            else:
                axes[i].text(0.5, 0.5, "Image not found", ha="center", va="center")

            title_text = (
                f"True: {err['true_class']}\n"
                f"Pred: {err['pred_class']}\n"
                f"Conf: {err['confidence']:.1%}"
            )
            axes[i].set_title(title_text, color="red", fontsize=10, fontweight="bold")
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    plt.suptitle(f"Top Misclassified Samples - {data.get('model_name')}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_file = output_path or (config.figures_dir / f"misclassified_{data.get('model_name')}.png")
    fig.savefig(save_file, dpi=300)
    plt.close(fig)
    return save_file
