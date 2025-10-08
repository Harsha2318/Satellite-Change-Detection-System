"""
Visualize satellite image change detection results.

This script loads a pair of satellite images (T1 and T2), the predicted change mask,
the ground truth mask (if available), and the vectorized changes. It displays them side by side
to help assess the quality of the change detection.

Usage:
    python visualize_changes.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geopandas as gpd

# Sample to visualize (change as needed)
SAMPLE_NAME = "0_12"  # e.g., "0_0", "0_1", "0_10", etc.

# Paths
PRED_DIR = Path("changedetect/predictions_run1_small")
PAIRS_DIR = Path("changedetect/data/processed/train_pairs")
MASK_DIR = Path("changedetect/data/processed/masks_small")

def load_and_visualize(sample_name):
    """Load and visualize a sample's results."""
    # Construct paths
    t1_path = PAIRS_DIR / f"{sample_name}_t1.tif"
    t2_path = PAIRS_DIR / f"{sample_name}_t2.tif"
    pred_mask_path = PRED_DIR / f"{sample_name}_change_mask.tif"
    gt_mask_path = MASK_DIR / f"{sample_name}_mask.tif"
    vector_path = PRED_DIR / f"{sample_name}_change_vectors.shp"
    
    # Check file existence
    if not t1_path.exists() or not t2_path.exists():
        print(f"Error: T1/T2 pair {sample_name} not found.")
        return
    
    if not pred_mask_path.exists():
        print(f"Error: Predicted mask for {sample_name} not found.")
        return
    
    has_gt = gt_mask_path.exists()
    has_vector = vector_path.exists()
    
    # Load files
    t1 = rasterio.open(t1_path)
    t2 = rasterio.open(t2_path)
    pred_mask = rasterio.open(pred_mask_path)
    
    if has_gt:
        gt_mask = rasterio.open(gt_mask_path)
    
    if has_vector:
        vectors = gpd.read_file(vector_path)
    
    # Create visualizations
    n_plots = 4 if has_gt else 3
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 5, 5))
    
    # Normalize visualization to improve contrast
    def normalize_img(img_array):
        """Normalize image for better visualization."""
        # If it's single channel, don't normalize (likely a mask)
        if img_array.ndim == 2 or img_array.shape[0] == 1:
            return img_array
        
        # For RGB/multi-band, perform band-wise normalization
        result = np.zeros_like(img_array, dtype=np.float32)
        for i in range(img_array.shape[0]):
            band = img_array[i].astype(np.float32)
            p2 = np.percentile(band, 2)
            p98 = np.percentile(band, 98)
            if p98 > p2:  # Avoid division by zero
                result[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                result[i] = band / band.max() if band.max() > 0 else band
        return result
    
    # Display images
    t1_array = t1.read([1, 2, 3])  # RGB bands
    t2_array = t2.read([1, 2, 3])  # RGB bands
    
    t1_norm = normalize_img(t1_array)
    t2_norm = normalize_img(t2_array)
    
    # Create figure
    axs[0].set_title("Before (T1)")
    axs[0].imshow(np.transpose(t1_norm, (1, 2, 0)))
    
    axs[1].set_title("After (T2)")
    axs[1].imshow(np.transpose(t2_norm, (1, 2, 0)))
    
    # Show predicted mask
    pred_array = pred_mask.read(1)
    axs[2].set_title("Predicted Changes")
    axs[2].imshow(np.transpose(t2_norm, (1, 2, 0)))  # Background image
    
    # Create a red overlay for changes
    pred_overlay = np.zeros((t2_array.shape[1], t2_array.shape[2], 4))
    pred_overlay[..., 0] = 1.0  # Red channel
    pred_overlay[..., 3] = (pred_array > 0) * 0.5  # Alpha for detected changes
    axs[2].imshow(pred_overlay)
    
    # If vectors are available, also show them
    if has_vector:
        vectors.boundary.plot(ax=axs[2], color='yellow', linewidth=0.8)
    
    # Show ground truth mask if available
    if has_gt:
        gt_array = gt_mask.read(1)
        axs[3].set_title("Ground Truth")
        axs[3].imshow(np.transpose(t2_norm, (1, 2, 0)))  # Background image
        
        # Green overlay for ground truth
        gt_overlay = np.zeros((t2_array.shape[1], t2_array.shape[2], 4))
        gt_overlay[..., 1] = 1.0  # Green channel
        gt_overlay[..., 3] = (gt_array > 0) * 0.5  # Alpha
        axs[3].imshow(gt_overlay)
    
    # Turn off axis ticks for all subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{sample_name}_visualization.png", dpi=200, bbox_inches='tight')
    print(f"Saved visualization to visualizations/{sample_name}_visualization.png")
    
    # Show figure if in interactive mode
    plt.show()
    
    # Print some statistics about the changes
    pred_area = (pred_array > 0).sum()
    pred_percent = pred_area / (pred_array.shape[0] * pred_array.shape[1]) * 100
    
    print(f"\nAnalysis for sample {sample_name}:")
    print(f"- Image size: {pred_array.shape[1]} x {pred_array.shape[0]} pixels")
    print(f"- Detected change area: {pred_area} pixels ({pred_percent:.2f}% of image)")
    
    if has_vector:
        print(f"- Detected objects: {len(vectors)} polygons")
        areas = vectors.area.describe()
        print(f"- Object sizes: min={areas['min']:.1f}, mean={areas['mean']:.1f}, max={areas['max']:.1f} map units")
    
    if has_gt:
        gt_area = (gt_array > 0).sum()
        gt_percent = gt_area / (gt_array.shape[0] * gt_array.shape[1]) * 100
        print(f"- Ground truth area: {gt_area} pixels ({gt_percent:.2f}% of image)")
        
        # Calculate simple IoU
        intersection = ((pred_array > 0) & (gt_array > 0)).sum()
        union = ((pred_array > 0) | (gt_array > 0)).sum()
        iou = intersection / union if union > 0 else 0
        
        print(f"- Intersection over Union (IoU): {iou:.4f}")
        print(f"- True Positives: {intersection} pixels")
        print(f"- False Positives: {(pred_array > 0).sum() - intersection} pixels")
        print(f"- False Negatives: {(gt_array > 0).sum() - intersection} pixels")

def check_for_changes():
    """Check all prediction files for ones with actual detected changes."""
    pred_dir = Path("changedetect/predictions_run1_small")
    change_files = list(pred_dir.glob("*_change_mask.tif"))
    
    print(f"Checking {len(change_files)} prediction files for changes...")
    
    samples_with_changes = []
    
    for pred_file in change_files:
        sample_name = pred_file.stem.replace("_change_mask", "")
        
        # Open and check if there are any changes
        with rasterio.open(pred_file) as src:
            pred = src.read(1)
            changes = (pred > 0).sum()
            
            if changes > 0:
                percent = changes / (pred.shape[0] * pred.shape[1]) * 100
                samples_with_changes.append((sample_name, changes, percent))
    
    # Sort by number of changes (descending)
    samples_with_changes.sort(key=lambda x: x[1], reverse=True)
    
    if not samples_with_changes:
        print("No samples with detected changes found.")
        return []
    
    print("\nTop samples with detected changes:")
    for sample, changes, percent in samples_with_changes[:10]:
        print(f"- {sample}: {changes} pixels ({percent:.2f}% of image)")
    
    return [s[0] for s in samples_with_changes[:5]]  # Return the top 5 samples

def main():
    """Main entry point."""
    # Create directory for visualizations
    os.makedirs("visualizations", exist_ok=True)
    
    # First check for samples with actual changes
    samples_with_changes = check_for_changes()
    
    if samples_with_changes:
        print("\nVisualizing samples with detected changes...")
        for sample in samples_with_changes[:3]:  # Show top 3 only
            print(f"\n------- Processing sample {sample} -------")
            load_and_visualize(sample)
    else:
        # If no samples with changes, visualize a couple default samples
        default_samples = ["0_0", "0_10", "0_20"]
        print("\nVisualizing default samples (no changes detected in any samples)...")
        for sample in default_samples:
            print(f"\n------- Processing sample {sample} -------")
            load_and_visualize(sample)

if __name__ == "__main__":
    main()