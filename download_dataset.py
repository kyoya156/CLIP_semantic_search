"""
download_dataset.py — Download a sample image dataset for testing.

Usage:
    python download_dataset.py --dataset cifar100 --count 1000 #max 60000
    python download_dataset.py --dataset food101 --count 10000 #max 101000
    python download_dataset.py --dataset oxford_pets --count 1000 #max 7,390
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

DATASETS = {
    "cifar100": {
        "hf_name": "cifar100",
        "split": "train",
        "image_key": "img",
        "label_key": "fine_label",
        "label_names": None,  # loaded from dataset info
        "description": "100 object classes, clean photos",
    },
    "food101": {
        "hf_name": "food101",
        "split": "train",
        "image_key": "image",
        "label_key": "label",
        "label_names": None,
        "description": "101 food categories, great for visual search",
    },
    "oxford_pets": {
        "hf_name": "pcuenq/oxford-pets",
        "split": "train",
        "image_key": "image",
        "label_key": "label",
        "label_names": None,
        "description": "37 cat & dog breeds",
    },
}


def download_images(dataset_key: str, count: int, out_dir: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Run: pip install datasets")

    cfg = DATASETS[dataset_key]
    print(f"Dataset   : {dataset_key} — {cfg['description']}")
    print(f"Saving to : {out_dir}")
    print(f"Count     : {count}\n")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Stream so we don't download the full dataset upfront
    ds = load_dataset(cfg["hf_name"], split=cfg["split"], streaming=True, trust_remote_code=True)

    # Grab label names if available
    try:
        ds_info = load_dataset(cfg["hf_name"], split=cfg["split"]).features
        label_names = ds_info[cfg["label_key"]].names if cfg["label_key"] else None
    except Exception:
        label_names = None

    saved = 0
    for item in tqdm(ds, total=count, desc="Downloading"):
        if saved >= count:
            break

        img = item[cfg["image_key"]]
        label_id = item.get(cfg["label_key"], 0)

        # Use label name in filename if available
        if label_names:
            label_str = label_names[label_id].replace(" ", "_").replace("/", "-")
        else:
            label_str = str(label_id)

        filename = f"{label_str}_{saved:05d}.jpg"
        filepath = os.path.join(out_dir, filename)

        if not os.path.exists(filepath):
            img.convert("RGB").save(filepath, "JPEG", quality=90)

        saved += 1

    print(f"\n✅ Saved {saved} images to '{out_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a test image dataset.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="food101",
        help="Which dataset to download",
    )
    parser.add_argument("--count", type=int, default=1000, help="Number of images to save")
    parser.add_argument("--out", default="./images", help="Output folder")
    args = parser.parse_args()

    download_images(args.dataset, args.count, args.out)