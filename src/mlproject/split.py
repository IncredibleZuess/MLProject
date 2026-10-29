"""
Data cleaning and leakage-free splitting (ITRI626 Sections 5 and 7.1).

Pipeline
--------
1. scan_dataset_directory  - find every image under data/raw/<class>/...
2. clean_dataset           - remove unreadable/corrupt/tiny files, exact duplicates
                             (MD5) and cross-class near-duplicates (uncertain labels);
                             merge same-class near-duplicates into one group
3. create_splits           - group-aware stratified 70/15/15 split so that photos of
                             the same physical object never straddle train/val/test
4. verify_no_leakage       - asserts that no group id and no image hash is shared

Group ids
---------
Photos of the *same physical board/component* must share a group id. Encode it in
the file name with a DOUBLE underscore:   uno01__0001.jpg, uno01__0002.jpg, ...
or put them in a sub-folder:              data/raw/arduino_uno/uno01/0001.jpg
Files with neither convention are treated as their own group, but near-duplicate
images (perceptual hash) are still merged into one group automatically.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- scan
def _infer_group(file_path: Path, class_path: Path) -> str:
    rel = file_path.relative_to(class_path)
    if len(rel.parts) > 1:  # sub-folder convention
        return f"{class_path.name}/{rel.parts[0]}"
    stem = file_path.stem
    if "__" in stem:  # double-underscore convention
        return f"{class_path.name}/{stem.split('__')[0]}"
    return f"{class_path.name}/{stem}"  # singleton group


def scan_dataset_directory(data_dir: Path) -> pd.DataFrame:
    """Scan data_dir/<class_name>/**/<image> into a dataframe."""
    records = []
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    for class_name in classes:
        class_path = data_dir / class_name
        for file_path in sorted(class_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                records.append({
                    "filepath": str(file_path.resolve()),
                    "relative_path": file_path.relative_to(data_dir.parent).as_posix(),
                    "label": class_name,
                    "group_id": _infer_group(file_path, class_path),
                    "filesize_bytes": file_path.stat().st_size,
                })
    return pd.DataFrame(records, columns=["filepath", "relative_path", "label", "group_id", "filesize_bytes"])


# -------------------------------------------------------------------------- clean
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash(img: Image.Image, size: int = 16) -> int:
    """256-bit difference hash: robust to resizing / recompression / small colour changes."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def clean_dataset(
    df: pd.DataFrame,
    min_filesize: int = 1024,
    min_side: int = 32,
    near_dup_threshold: int = 10,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove corrupt / unreadable / tiny images, exact duplicates and cross-class
    near-duplicates. Same-class near-duplicates are kept but forced into one group.
    Returns (cleaned_df, report) where report holds per-class before/after counts
    and a log of every removed file with the reason.
    """
    removed: List[Dict] = []
    before_counts = df["label"].value_counts().sort_index().to_dict()

    # 1) corrupt / too small
    keep_rows, hashes, md5s = [], [], []
    for row in df.itertuples(index=False):
        reason = None
        if row.filesize_bytes < min_filesize:
            reason = f"file smaller than {min_filesize} bytes"
        else:
            try:
                with Image.open(row.filepath) as im:
                    im.verify()
                with Image.open(row.filepath) as im:
                    im.load()
                    if min(im.size) < min_side:
                        reason = f"resolution below {min_side}px"
                    else:
                        hashes.append(_dhash(im))
            except (UnidentifiedImageError, OSError, SyntaxError) as e:
                reason = f"corrupt/unreadable ({type(e).__name__})"
        if reason:
            removed.append({"file": row.relative_path, "label": row.label, "reason": reason})
        else:
            keep_rows.append(row._asdict())
            md5s.append(_md5(row.filepath))
    clean = pd.DataFrame(keep_rows, columns=df.columns)
    clean["md5"] = md5s
    clean["dhash"] = hashes if hashes else []

    # 2) exact duplicates (byte-identical). Keep first; if copies carry different
    #    labels the label is uncertain, so drop every copy.
    exact_removed = 0
    keep_mask = np.ones(len(clean), dtype=bool)
    for _, grp in clean.groupby("md5"):
        if len(grp) == 1:
            continue
        if grp["label"].nunique() > 1:
            idxs = list(grp.index)
            reason = "exact duplicate with conflicting labels"
        else:
            idxs = list(grp.index[1:])
            reason = f"exact duplicate of {grp.iloc[0]['relative_path']}"
        for i in idxs:
            keep_mask[i] = False
            exact_removed += 1
            removed.append({"file": clean.at[i, "relative_path"], "label": clean.at[i, "label"], "reason": reason})
    clean = clean[keep_mask].reset_index(drop=True)

    # 3) near duplicates via perceptual hash
    n = len(clean)
    uf = _UnionFind(n)
    # images that already share a group (same physical object) stay together
    for _, idxs in clean.groupby("group_id").indices.items():
        for j in idxs[1:]:
            uf.union(int(idxs[0]), int(j))
    # 256-bit hashes stored as 4 x uint64 per image
    hs = np.array([[(h >> (64 * k)) & 0xFFFFFFFFFFFFFFFF for k in range(4)] for h in clean["dhash"]],
                  dtype=np.uint64).reshape(-1, 4)
    labels = clean["label"].to_numpy()
    conflict = set()
    near_pairs_same = 0
    for i in range(n - 1):
        x = np.bitwise_xor(hs[i + 1:], hs[i])
        dist = np.unpackbits(x.view(np.uint8).reshape(len(x), -1), axis=1).sum(axis=1)
        for off in np.nonzero(dist <= near_dup_threshold)[0]:
            j = i + 1 + int(off)
            if labels[i] == labels[j]:
                uf.union(i, j)
                near_pairs_same += 1
            else:
                conflict.update((i, j))
    for i in sorted(conflict):
        removed.append({"file": clean.at[i, "relative_path"], "label": clean.at[i, "label"],
                        "reason": "near-duplicate of an image in a different class (uncertain label)"})
    # merge same-class near duplicates into one group (keeps them in the same split)
    root_group: Dict[int, str] = {}
    new_groups = []
    for i in range(n):
        r = uf.find(i)
        root_group.setdefault(r, clean.at[i, "group_id"])
        new_groups.append(root_group[r])
    clean["group_id"] = new_groups
    clean = clean.drop(index=sorted(conflict)).reset_index(drop=True)

    after_counts = clean["label"].value_counts().sort_index().to_dict()
    report = {
        "initial_samples": int(len(df)),
        "removed_corrupt_or_small": int(sum(1 for r in removed if r["reason"].startswith(("file smaller", "resolution", "corrupt")))),
        "removed_exact_duplicates": int(exact_removed),
        "removed_cross_class_near_duplicates": int(len(conflict)),
        "same_class_near_duplicate_pairs_grouped": int(near_pairs_same),
        "cleaned_samples": int(len(clean)),
        "num_groups": int(clean["group_id"].nunique()),
        "near_dup_threshold_bits": near_dup_threshold,
        "per_class_before": {k: int(v) for k, v in before_counts.items()},
        "per_class_after": {k: int(v) for k, v in after_counts.items()},
        "removed_files": removed,
    }
    return clean, report


# -------------------------------------------------------------------------- split
def create_splits(
    df: pd.DataFrame,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_aware: bool = True,
    extra_metadata: Dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Group-aware stratified split. With k = round(1/test_ratio) folds of
    StratifiedGroupKFold, one fold becomes test, one becomes validation and the
    rest training (~70/15/15 for k=7). Falls back to plain stratified splitting
    only when group_aware=False.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(df["label"].unique())
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    df = df.copy().reset_index(drop=True)
    df["label_id"] = df["label"].map(label_to_id)

    if group_aware:
        k = max(3, int(round(1.0 / test_ratio)))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = [test_idx for _, test_idx in sgkf.split(df, df["label_id"], df["group_id"])]
        test_idx, val_idx = folds[0], folds[1]
        train_idx = np.setdiff1d(np.arange(len(df)), np.concatenate([test_idx, val_idx]))
        train_df, val_df, test_df = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
        method = f"StratifiedGroupKFold (k={k}): fold 0 = test, fold 1 = validation, remaining folds = train"
    else:
        train_df, tv = train_test_split(df, test_size=val_ratio + test_ratio, stratify=df["label_id"], random_state=seed)
        val_df, test_df = train_test_split(tv, test_size=test_ratio / (val_ratio + test_ratio),
                                           stratify=tv["label_id"], random_state=seed)
        method = "Stratified random split (NOT group-aware)"

    verify_no_leakage(train_df, val_df, test_df)

    cols = ["relative_path", "label", "label_id", "group_id", "md5"]
    cols = [c for c in cols if c in df.columns]
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part[cols].sort_values("relative_path").to_csv(output_dir / f"{name}.csv", index=False)

    metadata = {
        "classes": unique_labels,
        "label_to_id": label_to_id,
        "seed": seed,
        "split_method": method,
        "group_aware": group_aware,
        "split_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df), "total": len(df)},
        "group_counts": {n: int(p["group_id"].nunique()) for n, p in
                         [("train", train_df), ("val", val_df), ("test", test_df)]},
        "per_class_counts": {
            n: {c: int(p["label"].eq(c).sum()) for c in unique_labels}
            for n, p in [("train", train_df), ("val", val_df), ("test", test_df)]
        },
        "leakage_check": "passed: no group id or file hash shared between splits",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with open(output_dir / "classes.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return train_df, val_df, test_df


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    parts = {"train": train_df, "val": val_df, "test": test_df}
    for key in ["group_id", "md5"]:
        if key not in train_df.columns:
            continue
        sets = {n: set(p[key]) for n, p in parts.items()}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise AssertionError(f"Data leakage: {len(overlap)} {key} values shared between {a} and {b}")


def print_split_summary(output_dir: Path) -> None:
    with open(output_dir / "classes.json") as f:
        meta = json.load(f)
    sc = meta["split_counts"]
    print("=" * 66)
    print("DATASET SPLIT SUMMARY")
    print("=" * 66)
    if meta.get("synthetic_demo_data"):
        print("WARNING: SYNTHETIC DEMO DATA - results are NOT valid for the report")
    print(f"Method: {meta['split_method']}")
    print(f"Seed: {meta['seed']} | Leakage check: {meta['leakage_check']}")
    for s in ["train", "val", "test"]:
        print(f"{s:>5}: {sc[s]:5d} images ({sc[s] / sc['total']:.1%}), {meta['group_counts'][s]} groups")
    print("-" * 66)
    print(f"{'Class':<24}{'Train':>8}{'Val':>8}{'Test':>8}{'Total':>8}")
    for c in meta["classes"]:
        t, v, te = (meta["per_class_counts"][s][c] for s in ["train", "val", "test"])
        print(f"{c:<24}{t:>8}{v:>8}{te:>8}{t + v + te:>8}")
    print("=" * 66)
