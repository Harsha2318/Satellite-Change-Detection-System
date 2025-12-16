"""
Data preparation utilities for setting up the change detection dataset
"""
import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_data_structure(root_dir):
    """
    Create the required data directory structure.
    
    Args:
        root_dir: Root directory for data
    """
    root = Path(root_dir)
    
    # Create directories
    dirs = [
        root / "train" / "before",
        root / "train" / "after",
        root / "train" / "labels",
        root / "val" / "before",
        root / "val" / "after",
        root / "val" / "labels",
        root / "test" / "before",
        root / "test" / "after",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")


def organize_image_pairs(source_dir, output_dir, train_split=0.7, val_split=0.15):
    """
    Organize image pairs into train/val/test directories.
    
    Expects directory structure:
    source_dir/
    ├── image1_before.tif
    ├── image1_after.tif
    ├── image1_label.tif
    ├── image2_before.tif
    ├── image2_after.tif
    ├── image2_label.tif
    
    Args:
        source_dir: Source directory with image pairs
        output_dir: Output directory
        train_split: Fraction for training (0.7 = 70%)
        val_split: Fraction for validation (0.15 = 15%)
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    # Create data structure
    create_data_structure(output_dir)
    
    # Find all unique image IDs
    before_images = list(source.glob("*_before.tif"))
    image_ids = set()
    
    for img in before_images:
        name = img.stem.replace("_before", "")
        image_ids.add(name)
    
    image_ids = sorted(list(image_ids))
    n_images = len(image_ids)
    
    # Calculate split indices
    train_count = int(n_images * train_split)
    val_count = int(n_images * val_split)
    
    train_ids = image_ids[:train_count]
    val_ids = image_ids[train_count:train_count + val_count]
    test_ids = image_ids[train_count + val_count:]
    
    # Copy files
    for img_id in train_ids:
        _copy_image_pair(source, output / "train", img_id)
    
    for img_id in val_ids:
        _copy_image_pair(source, output / "val", img_id)
    
    for img_id in test_ids:
        _copy_image_pair(source, output / "test", img_id)
    
    logger.info(f"Organized {n_images} images:")
    logger.info(f"  Train: {len(train_ids)}")
    logger.info(f"  Val: {len(val_ids)}")
    logger.info(f"  Test: {len(test_ids)}")


def _copy_image_pair(source_dir, output_dir, image_id):
    """Copy an image pair to the output directory"""
    source = Path(source_dir)
    output = Path(output_dir)
    
    # Copy before image
    before_src = source / f"{image_id}_before.tif"
    before_dst = output / "before" / f"{image_id}.tif"
    if before_src.exists():
        shutil.copy(before_src, before_dst)
    
    # Copy after image
    after_src = source / f"{image_id}_after.tif"
    after_dst = output / "after" / f"{image_id}.tif"
    if after_src.exists():
        shutil.copy(after_src, after_dst)
    
    # Copy label image
    label_src = source / f"{image_id}_label.tif"
    label_dst = output / "labels" / f"{image_id}.tif"
    if label_src.exists():
        shutil.copy(label_src, label_dst)


def verify_data_structure(data_dir):
    """
    Verify the data directory structure is correct.
    
    Args:
        data_dir: Data directory to verify
        
    Returns:
        bool: True if structure is valid
    """
    data = Path(data_dir)
    required_dirs = [
        "train/before", "train/after", "train/labels",
        "val/before", "val/after", "val/labels",
        "test/before", "test/after"
    ]
    
    for d in required_dirs:
        dir_path = data / d
        if not dir_path.exists():
            logger.error(f"Missing directory: {dir_path}")
            return False
    
    # Count files
    train_before = len(list((data / "train" / "before").glob("*.tif")))
    train_after = len(list((data / "train" / "after").glob("*.tif")))
    train_labels = len(list((data / "train" / "labels").glob("*.tif")))
    
    if train_before == 0 or train_after == 0 or train_labels == 0:
        logger.warning("No files found in training directory")
    
    logger.info(f"Data structure verification:")
    logger.info(f"  Train images: {train_before} before, {train_after} after, {train_labels} labels")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_prep.py <source_dir> [output_dir]")
        sys.exit(1)
    
    source = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "data"
    
    logging.basicConfig(level=logging.INFO)
    
    organize_image_pairs(source, output)
    verify_data_structure(output)
