"""
Compare just a single image pair in detail to check for visible changes
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path

# Set the directories
PAIRS_DIR = Path("changedetect/data/processed/train_pairs")

def compare_images(sample_name="0_10"):
    """Compare before and after images to see if there are visible changes."""
    t1_path = PAIRS_DIR / f"{sample_name}_t1.tif"
    t2_path = PAIRS_DIR / f"{sample_name}_t2.tif"
    
    if not t1_path.exists() or not t2_path.exists():
        print(f"Sample {sample_name} files not found.")
        return
    
    # Open the images
    t1 = rasterio.open(t1_path)
    t2 = rasterio.open(t2_path)
    
    # Get image arrays (RGB)
    t1_rgb = t1.read([1, 2, 3])
    t2_rgb = t2.read([1, 2, 3])
    
    # Print statistics
    print(f"T1 (before) statistics:")
    print(f"  Shape: {t1_rgb.shape}")
    print(f"  Value ranges: R({t1_rgb[0].min()}-{t1_rgb[0].max()}), "
          f"G({t1_rgb[1].min()}-{t1_rgb[1].max()}), B({t1_rgb[2].min()}-{t1_rgb[2].max()})")
    print(f"  Mean values: R({t1_rgb[0].mean():.2f}), G({t1_rgb[1].mean():.2f}), B({t1_rgb[2].mean():.2f})")
    
    print(f"T2 (after) statistics:")
    print(f"  Shape: {t2_rgb.shape}")
    print(f"  Value ranges: R({t2_rgb[0].min()}-{t2_rgb[0].max()}), "
          f"G({t2_rgb[1].min()}-{t2_rgb[1].max()}), B({t2_rgb[2].min()}-{t2_rgb[2].max()})")
    print(f"  Mean values: R({t2_rgb[0].mean():.2f}), G({t2_rgb[1].mean():.2f}), B({t2_rgb[2].mean():.2f})")
    
    # Calculate difference
    diff = np.abs(t1_rgb.astype(np.float32) - t2_rgb.astype(np.float32))
    print(f"Difference statistics:")
    print(f"  Max absolute difference per channel: R({diff[0].max():.2f}), "
          f"G({diff[1].max():.2f}), B({diff[2].max():.2f})")
    print(f"  Mean absolute difference per channel: R({diff[0].mean():.2f}), "
          f"G({diff[1].mean():.2f}), B({diff[2].mean():.2f})")
    print(f"  Total pixels with significant difference (>50): {(diff > 50).any(axis=0).sum()}")
    
    # ASCII visualization of differences
    diff_sum = diff.sum(axis=0)
    threshold = 100
    diff_significant = diff_sum > threshold
    
    print(f"\nASCII visualization of differences (threshold > {threshold}):")
    height, width = diff_sum.shape
    scale_h, scale_w = max(1, height // 40), max(1, width // 80)
    
    ascii_map = ''
    for y in range(0, height, scale_h):
        line = ''
        for x in range(0, width, scale_w):
            window = diff_significant[y:y+scale_h, x:x+scale_w]
            if window.any():
                line += '#'
            else:
                line += '.'
        ascii_map += line + '\n'
    
    print(ascii_map)

if __name__ == "__main__":
    print("\nComparing sample 0_10:")
    compare_images("0_10")
    
    print("\nComparing sample 0_11:")
    compare_images("0_11")