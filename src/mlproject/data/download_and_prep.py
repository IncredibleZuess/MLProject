"""
Data ingestion, demo dataset generation, cleaning, and preparation pipeline.
Allows users to either import real microcontroller images or generate a verified
demo dataset to test and reproduce the full training and evaluation workflow immediately.
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlproject.config import ProjectConfig, default_config
from mlproject.split import clean_dataset, create_splits, print_split_summary, scan_dataset_directory


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

    print(f"Successfully generated {len(target_classes) * samples_per_class} sample images in {target_dir}")


def prepare_dataset(
    raw_dir: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
    config: ProjectConfig = default_config,
    generate_if_empty: bool = True,
) -> None:
    """
    Full data preparation pipeline:
    1. Scan raw image directory
    2. Clean corrupt files and duplicates
    3. Generate reproducible train/val/test splits (70/15/15)
    4. Save manifest files and summary statistics
    """
    r_dir = raw_dir or config.raw_data_dir
    s_dir = splits_dir or config.splits_dir
    r_dir.mkdir(parents=True, exist_ok=True)
    s_dir.mkdir(parents=True, exist_ok=True)

    df_raw = scan_dataset_directory(r_dir)

    if len(df_raw) == 0:
        if generate_if_empty:
            print(f"Raw directory '{r_dir}' is empty. Generating initial demo dataset for pipeline verification...")
            generate_demo_dataset(r_dir, classes=config.target_classes, samples_per_class=35)
            df_raw = scan_dataset_directory(r_dir)
        else:
            raise FileNotFoundError(f"No images found in raw data directory: {r_dir}")

    # Clean dataset
    cleaned_df, clean_stats = clean_dataset(df_raw)
    print(f"\nData Cleaning Summary: {clean_stats}")

    # Generate splits
    create_splits(
        cleaned_df,
        output_dir=s_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    # Print summary
    print_split_summary(s_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Microcontroller Dataset and Splits")
    parser.add_argument("--generate-demo", action="store_true", help="Generate synthetic demo dataset")
    parser.add_argument("--samples-per-class", type=int, default=40)
    args = parser.parse_args()

    cfg = default_config
    if args.generate_demo:
        generate_demo_dataset(cfg.raw_data_dir, samples_per_class=args.samples_per_class)
    prepare_dataset(config=cfg)
