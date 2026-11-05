"""Add your own / newly downloaded images to data/raw/<class>/ in the format the pipeline expects.

Example:
    python -m mlproject.data.ingest_images --src ~/Downloads/uno_photos --label arduino_uno \
        --prefix uno_karl --source "own photos, NWU lab" --group-size 5

--group-size N: consecutive files (sorted by name) are treated as photos of ONE physical object
and share a group id (prefix + object number), so the split cannot leak them across train/test.
Use --group-size 1 if every image is a different object.
Every added file is recorded in data/raw/SOURCES_added.csv (needed for the report's licence section).
"""
import argparse, csv, shutil
from pathlib import Path
from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--label", required=True, help="class name = folder name (must be in config.target_classes)")
    ap.add_argument("--prefix", required=True, help="short unique prefix, e.g. uno_karl")
    ap.add_argument("--source", required=True, help="where the images came from + licence")
    ap.add_argument("--group-size", type=int, default=1)
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    a = ap.parse_args()

    out = a.raw / a.label
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in a.src.rglob("*") if p.suffix.lower() in EXT)
    log = a.raw / "SOURCES_added.csv"
    new = not log.exists()
    added = skipped = 0
    with open(log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["file", "label", "source", "original_name"])
        for i, p in enumerate(files):
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    if min(im.size) < a.min_side:
                        raise ValueError("too small")
                    rgb = im.convert("RGB")
            except Exception as e:
                print(f"skip {p.name}: {e}")
                skipped += 1
                continue
            obj, k = divmod(i, a.group_size)
            name = f"{a.prefix}{obj:03d}__{k:02d}.jpg"
            rgb.save(out / name, quality=95)
            w.writerow([f"raw/{a.label}/{name}", a.label, a.source, p.name])
            added += 1
    print(f"added {added}, skipped {skipped} -> {out}")
    print("Next: python -m mlproject run-all   (then update every number in report/slides/oral_prep)")


if __name__ == "__main__":
    main()
