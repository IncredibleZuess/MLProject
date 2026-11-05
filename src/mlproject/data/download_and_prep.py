"""
Data ingestion, demo dataset generation, cleaning, and preparation pipeline.
Allows users to either import real microcontroller images or generate a verified
demo dataset to test and reproduce the full training and evaluation workflow immediately.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlproject.config import ProjectConfig, default_config
from mlproject.split import clean_dataset, create_splits, print_split_summary, scan_dataset_directory


DEMO_MARKER = ".SYNTHETIC_DEMO_DATA"


def generate_synthetic_microcontroller_image(
    class_name: str,
    output_path: Path,
    image_size: int = 224,
) -> None:
    """
    Generates a synthetic PCB / microcontroller image with distinct visual patterns
    corresponding to each class for testing and demonstration.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Distinct background PCB colors
    pcb_colors = {
        "arduino_uno": (0, 100, 150),       # Teal/Cyan Arduino blue
        "arduino_nano": (0, 80, 130),       # Darker Arduino blue
        "esp32": (20, 20, 20),              # Matte black PCB
        "esp8266": (10, 40, 90),            # Deep blue PCB
        "raspberry_pi_pico": (0, 120, 50),  # Pico green PCB
        "stm32_bluepill": (20, 70, 160),    # Blue pill royal blue
        "microcontroller_ic": (30, 30, 30), # Black epoxy IC
        "resistor": (180, 160, 120),        # Tan / beige resistor body
        "capacitor": (40, 40, 140),         # Electrolytic blue canister
        "diode_led": (160, 20, 20),         # Red LED body
    }

    base_color = pcb_colors.get(class_name, (50, 50, 50))
    # Add random background variation
    bg_color = (
        max(0, min(255, base_color[0] + random.randint(-15, 15))),
        max(0, min(255, base_color[1] + random.randint(-15, 15))),
        max(0, min(255, base_color[2] + random.randint(-15, 15))),
    )

    img = Image.new("RGB", (image_size, image_size), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw PCB pin headers / metallic tracks
    for y in range(20, image_size - 20, 18):
        # Left pins
        draw.rectangle([10, y, 22, y + 8], fill=(212, 175, 55))  # Gold / brass
        # Right pins
        draw.rectangle([image_size - 22, y, image_size - 10, y + 8], fill=(212, 175, 55))

    # Draw IC chip in the center
    cx, cy = image_size // 2, image_size // 2
    draw.rectangle([cx - 40, cy - 40, cx + 40, cy + 40], fill=(25, 25, 25), outline=(70, 70, 70), width=2)
    # Silver pin 1 dot
    draw.ellipse([cx - 32, cy - 32, cx - 26, cy - 26], fill=(200, 200, 200))

    # Add text label for identification
    text_content = class_name.replace("_", "\n").upper()
    draw.text((cx - 35, cy - 20), text_content, fill=(230, 230, 230))

    # Add noise / random solder pads
    for _ in range(25):
        rx = random.randint(30, image_size - 30)
        ry = random.randint(30, image_size - 30)
        draw.ellipse([rx, ry, rx + 4, ry + 4], fill=(192, 192, 192))

    img.save(output_path, "JPEG", quality=90)


def generate_demo_dataset(
    target_dir: Path,
    classes: Optional[List[str]] = None,
    samples_per_class: int = 40,
) -> None:
    """
    Generates a sample dataset with balanced classes to allow immediate testing
    and full replication of the training, evaluation, and reporting pipeline.
    """
    target_classes = classes or default_config.target_classes
    print(f"Generating demo dataset with {len(target_classes)} classes, {samples_per_class} images per class...")

    for class_name in target_classes:
        class_folder = target_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img_path = class_folder / f"{class_name}_{i:03d}.jpg"
            generate_synthetic_microcontroller_image(class_name, img_path)

    (target_dir / DEMO_MARKER).write_text("Images in this folder are synthetic and must not be reported as results.\n")
    print(f"Successfully generated {len(target_classes) * samples_per_class} sample images in {target_dir}")


def prepare_dataset(
    raw_dir: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
    config: ProjectConfig = default_config,
    generate_if_empty: bool = False,
    group_aware: bool = True,
) -> None:
    """
    Full data preparation pipeline:
    1. Scan data/raw/<class>/...
    2. Clean: corrupt/tiny files, exact duplicates, cross-class near-duplicates
    3. Group-aware stratified 70/15/15 split (seed from config)
    4. Write split CSVs, classes.json and cleaning_report.json to data/splits/
    """
    r_dir = raw_dir or config.raw_data_dir
    s_dir = splits_dir or config.splits_dir
    r_dir.mkdir(parents=True, exist_ok=True)
    s_dir.mkdir(parents=True, exist_ok=True)

    df_raw = scan_dataset_directory(r_dir)
    if len(df_raw) == 0:
        if generate_if_empty:
            print(f"Raw directory '{r_dir}' is empty. Generating SYNTHETIC demo data (pipeline test only)...")
            generate_demo_dataset(r_dir, classes=config.target_classes, samples_per_class=35)
            df_raw = scan_dataset_directory(r_dir)
        else:
            raise FileNotFoundError(
                f"No images found in {r_dir}. Put real images in data/raw/<class_name>/ "
                "(see data/DATASET.md), or run `python -m mlproject prepare --demo` to test the pipeline."
            )

    synthetic = (r_dir / DEMO_MARKER).exists()
    # manual review: images the group judged wrong/unusable (data/manual_exclude.csv)
    manual = []
    excl_file = config.data_dir / "manual_exclude.csv"
    if excl_file.exists():
        import pandas as pd
        excl = pd.read_csv(excl_file)
        reasons = dict(zip(excl["relative_path"], excl["reason"]))
        mask = df_raw["relative_path"].isin(reasons)
        manual = [{"file": r, "label": l, "reason": "manual review: " + reasons[r]}
                  for r, l in zip(df_raw.loc[mask, "relative_path"], df_raw.loc[mask, "label"])]
        print(f"Manual review: excluding {int(mask.sum())} images listed in {excl_file.name}")
        before_manual = df_raw["label"].value_counts().sort_index().to_dict()
        df_raw = df_raw[~mask].reset_index(drop=True)

    cleaned_df, report = clean_dataset(df_raw)
    if manual:
        report["removed_manual_review"] = len(manual)
        report["initial_samples"] += len(manual)
        report["per_class_before"] = {k: int(v) for k, v in before_manual.items()}
        report["removed_files"] = manual + report["removed_files"]
    else:
        report["removed_manual_review"] = 0
    with open(s_dir / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"\nCleaning: {report['initial_samples']} found -> {report['cleaned_samples']} kept "
        f"(corrupt/small: {report['removed_corrupt_or_small']}, exact dups: {report['removed_exact_duplicates']}, "
        f"cross-class near-dups: {report['removed_cross_class_near_duplicates']}, manual review: {report['removed_manual_review']}, "
        f"same-class near-dup pairs grouped: {report['same_class_near_duplicate_pairs_grouped']})"
    )

    create_splits(
        cleaned_df,
        output_dir=s_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        group_aware=group_aware,
        extra_metadata={"synthetic_demo_data": synthetic},
    )
    print_split_summary(s_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and splits")
    parser.add_argument("--generate-demo", action="store_true", help="Generate synthetic demo dataset")
    parser.add_argument("--samples-per-class", type=int, default=40)
    args = parser.parse_args()
    if args.generate_demo:
        generate_demo_dataset(default_config.raw_data_dir, samples_per_class=args.samples_per_class)
    prepare_dataset(config=default_config)
