"""Download and materialize a real change-detection dataset into local data/.

Source: Hugging Face dataset `blanchon/LEVIR_CDPlus`.

Exports to:
  data/train/{before,after,labels}
  data/test/{before,after,labels}
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import load_dataset


def _clear_split_dirs(data_dir: Path) -> None:
    for split in ("train", "test"):
        for folder in ("before", "after", "labels"):
            target = data_dir / split / folder
            if not target.exists():
                continue
            for item in target.iterdir():
                if item.is_file():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)


def export_split(dataset_split, out_dir: Path, prefix: str) -> None:
    before_dir = out_dir / "before"
    after_dir = out_dir / "after"
    labels_dir = out_dir / "labels"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(dataset_split):
        name = f"{prefix}_{index:04d}.png"
        sample["image1"].convert("RGB").save(before_dir / name)
        sample["image2"].convert("RGB").save(after_dir / name)
        sample["mask"].convert("L").save(labels_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and export LEVIR-CD+ dataset")
    parser.add_argument("--data-dir", default="data", help="Target data directory")
    parser.add_argument("--replace", action="store_true", help="Delete existing data-dir first")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.replace and data_dir.exists():
        _clear_split_dirs(data_dir)

    dataset = load_dataset("blanchon/LEVIR_CDPlus")
    export_split(dataset["train"], data_dir / "train", "train")
    export_split(dataset["test"], data_dir / "test", "test")

    print(f"Export complete under: {data_dir}")
    print(f"train samples: {len(dataset['train'])}")
    print(f"test samples: {len(dataset['test'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
