"""
Data splitting and validation module for the Microcontroller project.
Implements group-based or stratified splitting to strictly prevent data leakage
between train, validation, and test splits (as mandated by ITRI626 Section 7.1).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from mlproject.config import ProjectConfig, default_config


def scan_dataset_directory(
    data_dir: Path,
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> pd.DataFrame:
    """
    Scans a class-folder directory structure:
      data_dir/
        class_a/
          img1.jpg
          img2.jpg
        class_b/
          ...
    Extracts image filepaths, class labels, and inferred group identifiers
    (e.g., sample ID or batch ID if encoded in filename, else image stem).
    """
    records = []
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    for class_name in classes:
        class_path = data_dir / class_name
        for file_path in class_path.rglob("*"):
            if file_path.suffix.lower() in valid_extensions:
                # Infer group ID from filename prefix if applicable (e.g., 'boardA_01.jpg' -> 'boardA')
                # Otherwise, treat individual file as unique
                stem = file_path.stem
                group_id = stem.split("_")[0] if "_" in stem else stem

                records.append({
                    "filepath": str(file_path.resolve()),
                    "relative_path": str(file_path.relative_to(data_dir.parent)),
                    "label": class_name,
                    "group_id": group_id,
                    "filesize_bytes": file_path.stat().st_size,
                })

    df = pd.DataFrame(records)
    return df


def clean_dataset(df: pd.DataFrame, min_filesize: int = 1024) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Removes corrupted/empty files and duplicate entries.
    Returns cleaned dataframe and dictionary of removed count summary.
    """
    initial_count = len(df)
    
    # Filter 0-byte or corruptly small files
    valid_size = df["filesize_bytes"] >= min_filesize
    removed_corrupt = int((~valid_size).sum())
    cleaned = df[valid_size].copy()

    # Filter duplicate file paths
    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["filepath"])
    removed_duplicates = before_dedup - len(cleaned)

    stats = {
        "initial_samples": initial_count,
        "removed_corrupt": removed_corrupt,
        "removed_duplicates": removed_duplicates,
        "cleaned_samples": len(cleaned),
    }
    return cleaned, stats


def create_splits(
    df: pd.DataFrame,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_aware: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train, validation, and test subsets.
    If group_aware=True, ensures all photos of the same physical board/sample
    are placed in the same split to strictly prevent data leakage.
    Otherwise uses stratified splitting to preserve class proportions.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode integer labels
    unique_labels = sorted(df["label"].unique())
    label_to_id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    df = df.copy()
    df["label_id"] = df["label"].map(label_to_id)

    if group_aware and df["group_id"].nunique() > 3:
        # Group-aware split to prevent data leakage
        test_val_size = val_ratio + test_ratio
        val_fraction_of_remaining = val_ratio / test_val_size

        sgkf = StratifiedGroupKFold(n_splits=int(1.0 / test_val_size), shuffle=True, random_state=seed)
        splits = list(sgkf.split(df, df["label_id"], df["group_id"]))
        train_idx, test_val_idx = splits[0]

        train_df = df.iloc[train_idx].copy()
        test_val_df = df.iloc[test_val_idx].copy()

        # Split test_val into val and test
        sgkf_sub = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        sub_splits = list(sgkf_sub.split(test_val_df, test_val_df["label_id"], test_val_df["group_id"]))
        val_sub_idx, test_sub_idx = sub_splits[0]

        val_df = test_val_df.iloc[val_sub_idx].copy()
        test_df = test_val_df.iloc[test_sub_idx].copy()
    else:
        # Standard Stratified split
        test_val_ratio = val_ratio + test_ratio
        train_df, test_val_df = train_test_split(
            df,
            test_size=test_val_ratio,
            stratify=df["label_id"],
            random_state=seed,
        )

        relative_val_ratio = val_ratio / test_val_ratio
        val_df, test_df = train_test_split(
            test_val_df,
            test_size=(1.0 - relative_val_ratio),
            stratify=test_val_df["label_id"],
            random_state=seed,
        )

    # Save to disk
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Save class metadata
    metadata = {
        "classes": unique_labels,
        "label_to_id": label_to_id,
        "seed": seed,
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(df),
        },
        "per_class_counts": {
            split_name: split_data["label"].value_counts().to_dict()
            for split_name, split_data in [("train", train_df), ("val", val_df), ("test", test_df)]
        },
    }
    with open(output_dir / "classes.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return train_df, val_df, test_df


def print_split_summary(output_dir: Path) -> None:
    """Print readable summary of the generated dataset splits."""
    with open(output_dir / "classes.json", "r") as f:
        meta = json.load(f)

    print("=" * 60)
    print("DATASET SPLIT SUMMARY (ITRI626 Reproducibility Standard)")
    print("=" * 60)
    print(f"Random Seed: {meta['seed']}")
    print(f"Total Samples: {meta['split_counts']['total']}")
    print(f"Train: {meta['split_counts']['train']} ({meta['split_counts']['train'] / meta['split_counts']['total']:.1%})")
    print(f"Validation: {meta['split_counts']['val']} ({meta['split_counts']['val'] / meta['split_counts']['total']:.1%})")
    print(f"Held-out Test: {meta['split_counts']['test']} ({meta['split_counts']['test'] / meta['split_counts']['total']:.1%})")
    print("-" * 60)
    print(f"{'Class':<25} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 60)
    for cls in meta["classes"]:
        t_cnt = meta["per_class_counts"]["train"].get(cls, 0)
        v_cnt = meta["per_class_counts"]["val"].get(cls, 0)
        te_cnt = meta["per_class_counts"]["test"].get(cls, 0)
        print(f"{cls:<25} {t_cnt:<10} {v_cnt:<10} {te_cnt:<10}")
    print("=" * 60)
