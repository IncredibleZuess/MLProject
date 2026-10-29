"""
Build data/raw/ from two public GitHub datasets (reproducible).

Sources
-------
1. Boards: Gilbert Tanner, "Tensorflow-Object-Detection-API-Train-Model" (MIT licence)
   https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model
   149 photos with Pascal-VOC boxes for Arduino_Nano, ESP8266, Heltec_ESP32_Lora,
   Raspberry_Pi_3. Every box is cropped (10% padding) to one classification image.
   All crops from one photo share a group id, so they stay in the same split.
2. Components: P. Faraji, "Electronic-Components-Classification" (no licence file;
   images crawled from Digi-Key and the Kaggle dataset aryaminus/electronic-components)
   https://github.com/pouria-faraji/Electronic-Components-Classification
   Folders Capacitor, Diode, IC, Inductor, Resistor, Transformer.

Usage
-----
    git clone --depth 1 https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model.git ext/boards
    git clone --depth 1 https://github.com/pouria-faraji/Electronic-Components-Classification.git ext/components
    python -m mlproject.data.build_public_dataset --boards ext/boards --components ext/components

Large component classes are randomly capped at --cap images (seed 42) to limit the
class imbalance (boards have ~45 images each, IC has ~1500).
"""

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from mlproject.config import default_config

BOARD_MAP = {
    "Arduino_Nano": "arduino_nano",
    "ESP8266": "esp8266",
    "Heltec_ESP32_Lora": "esp32_heltec_lora",
    "Raspberry_Pi_3": "raspberry_pi_3",
}
COMPONENT_MAP = {
    "Capacitor": "capacitor",
    "Diode": "diode",
    "IC": "ic",
    "Inductor": "inductor",
    "Resistor": "resistor",
    "Transformer": "transformer",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crop_boards(src: Path, out: Path, pad: float = 0.10, max_side: int = 640) -> dict:
    counts = {}
    for xml_file in sorted(src.glob("images/*/*.xml")):
        root = ET.parse(xml_file).getroot()
        img_path = xml_file.with_suffix(".jpg")
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        photo_id = img_path.stem  # group id: one physical photo
        for k, obj in enumerate(root.findall("object")):
            name = obj.find("name").text
            cls = BOARD_MAP.get(name)
            if cls is None:
                continue
            bb = obj.find("bndbox")
            x1, y1, x2, y2 = (int(float(bb.find(t).text)) for t in ("xmin", "ymin", "xmax", "ymax"))
            dw, dh = (x2 - x1) * pad, (y2 - y1) * pad
            box = (max(0, int(x1 - dw)), max(0, int(y1 - dh)), min(W, int(x2 + dw)), min(H, int(y2 + dh)))
            crop = img.crop(box)
            crop.thumbnail((max_side, max_side))
            d = out / cls
            d.mkdir(parents=True, exist_ok=True)
            crop.save(d / f"tanner{photo_id}__{k}.jpg", quality=92)
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def copy_components(src: Path, out: Path, cap: int, seed: int) -> dict:
    rng = random.Random(seed)
    counts = {}
    for folder, cls in COMPONENT_MAP.items():
        files = sorted(p for p in (src / "Dataset" / folder).iterdir() if p.suffix.lower() in IMG_EXT)
        if cap and len(files) > cap:
            files = sorted(rng.sample(files, cap))
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in files:
            try:
                im = Image.open(f).convert("RGB")
            except Exception:
                continue  # unreadable files are also caught again by the cleaner
            im.save(d / f"faraji_{f.stem}.jpg", quality=92)
            n += 1
        counts[cls] = n
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boards", type=Path, required=True)
    ap.add_argument("--components", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=default_config.raw_data_dir)
    ap.add_argument("--cap", type=int, default=400, help="max images per component class (0 = no cap)")
    ap.add_argument("--seed", type=int, default=default_config.seed)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    marker = a.out / ".SYNTHETIC_DEMO_DATA"
    if marker.exists():
        raise SystemExit(f"{a.out} contains synthetic demo data. Delete data/raw first.")
    boards = crop_boards(a.boards, a.out)
    comps = copy_components(a.components, a.out, a.cap, a.seed)
    summary = {
        "sources": {
            "boards": "https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model (MIT)",
            "components": "https://github.com/pouria-faraji/Electronic-Components-Classification (no licence file)",
        },
        "component_cap": a.cap,
        "seed": a.seed,
        "counts": {**boards, **comps},
    }
    (a.out / "SOURCES.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
