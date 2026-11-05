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
    fig.savefig(save_file, dpi=150)
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
    fig.savefig(save_file, dpi=150)
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
    fig.savefig(save_file, dpi=150)
    plt.close(fig)
    return save_file


# ---------------------------------------------------------------- dataset figures
def plot_class_distribution(config: ProjectConfig = default_config) -> Path:
    """Stacked bar chart of images per class in train / val / test."""
    with open(config.splits_dir / "classes.json") as f:
        meta = json.load(f)
    classes = meta["classes"]
    pc = meta["per_class_counts"]
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(classes))
    for split, color in [("train", "#1f77b4"), ("val", "#ff7f0e"), ("test", "#2ca02c")]:
        vals = np.array([pc[split].get(c, 0) for c in classes])
        ax.bar(classes, vals, bottom=bottom, label=split, color=color)
        bottom += vals
    for i, total in enumerate(bottom):
        ax.text(i, total + max(bottom) * 0.01, f"{int(total)}", ha="center", fontsize=9)
    ax.set_ylabel("Images")
    ax.set_title("Images per class and split (after cleaning)", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = config.figures_dir / "class_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_class_examples(per_class: int = 3, config: ProjectConfig = default_config) -> Path:
    """Grid with `per_class` random training images (rows) for every class (columns)."""
    df = pd.read_csv(config.splits_dir / "train.csv")
    classes = sorted(df["label"].unique())
    fig, axes = plt.subplots(per_class, len(classes), figsize=(1.6 * len(classes), 1.75 * per_class))
    axes = np.array(axes).reshape(per_class, len(classes))
    for col, c in enumerate(classes):
        sub = df[df["label"] == c].sample(n=min(per_class, int((df["label"] == c).sum())), random_state=config.seed)
        for r in range(per_class):
            ax = axes[r, col]
            ax.axis("off")
            if r < len(sub):
                ax.imshow(Image.open(config.data_dir / sub.iloc[r]["relative_path"]).convert("RGB"))
            if r == 0:
                ax.set_title(c.replace("_", " "), fontsize=8, fontweight="bold")
    plt.tight_layout(pad=0.3)
    out = config.figures_dir / "class_examples.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ------------------------------------------------------------- results figures
def plot_correct_samples(results_file: Path, num_samples: int = 8, config: ProjectConfig = default_config) -> Optional[Path]:
    """Grid of correctly classified test images (one per class where possible)."""
    with open(results_file) as f:
        data = json.load(f)
    items = data.get("top_correct_predictions", [])[:num_samples]
    if not items:
        return None
    cols = 4
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(items):
            it = items[i]
            p = Path(it["image_path"])
            if p.exists():
                ax.imshow(Image.open(p).convert("RGB"))
            ax.set_title(f"True/Pred: {it['true_class']}\nConf: {it['confidence']:.1%}", color="green", fontsize=10, fontweight="bold")
    plt.suptitle(f"Correct test predictions - {data['model_name']}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = config.figures_dir / f"correct_{data['model_name']}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_validation_comparison(model_names: List[str], config: ProjectConfig = default_config) -> Optional[Path]:
    """Validation loss and validation macro-F1 of all models on shared axes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    found = False
    for m in model_names:
        hf = config.models_dir / m / "history.json"
        if not hf.exists():
            continue
        found = True
        h = json.load(open(hf))["history"]
        ep = range(1, len(h["val_loss"]) + 1)
        axes[0].plot(ep, h["val_loss"], label=m, linewidth=2)
        axes[1].plot(ep, h["val_f1"], label=m, linewidth=2)
    if not found:
        plt.close(fig)
        return None
    axes[0].set_title("Validation loss", fontweight="bold")
    axes[1].set_title("Validation macro-F1", fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    out = config.figures_dir / "validation_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_per_class_f1(model_names: List[str], config: ProjectConfig = default_config) -> Optional[Path]:
    """Grouped bars: per-class test F1 for every model."""
    data = {}
    for m in model_names:
        rf = config.results_dir / f"{m}_results.json"
        if rf.exists():
            r = json.load(open(rf))
            data[m] = {c: r["per_class_report"][c]["f1-score"] for c in r["class_names"]}
    if not data:
        return None
    classes = list(next(iter(data.values())).keys())
    x = np.arange(len(classes))
    w = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (m, d) in enumerate(data.items()):
        ax.bar(x + i * w - 0.4 + w / 2, [d[c] for c in classes], width=w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test F1")
    ax.set_title("Per-class test F1 by model", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = config.figures_dir / "per_class_f1.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
