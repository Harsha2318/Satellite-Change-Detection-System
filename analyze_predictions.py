"""
Analyze satellite image pairs and model predictions.

This script:
1. Visualizes before/after satellite images to see if there are visible changes
2. Analyzes the prediction files to understand the actual values in them
3. Checks if the model is detecting changes correctly
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from PIL import Image
import glob

# Set the directories
PAIRS_DIR = Path("changedetect/data/processed/train_pairs")
PRED_DIR = Path("changedetect/predictions_run1_small")
MASK_DIR = Path("changedetect/data/processed/masks_small")

def analyze_raw_images(sample_name=None, num_samples=5):
    """
    Analyze raw image pairs to check for visible changes.
    If sample_name is provided, only that sample is shown.
    Otherwise, shows the first num_samples image pairs.
    """
    # Create directory for visualizations
    os.makedirs("visualizations", exist_ok=True)
    
    if sample_name:
        # Process specific sample
        samples = [sample_name]
    else:
        # Find all available sample pairs
        t1_files = sorted(list(PAIRS_DIR.glob("*_t1.tif")))
        samples = [f.stem.replace("_t1", "") for f in t1_files[:num_samples]]
    
    for sample in samples:
        t1_path = PAIRS_DIR / f"{sample}_t1.tif"
        t2_path = PAIRS_DIR / f"{sample}_t2.tif"
        
        if not t1_path.exists() or not t2_path.exists():
            print(f"Sample {sample} files not found. Skipping.")
            continue
        
        print(f"\n----- Analyzing raw images for sample {sample} -----")
        
        # Open the images
        t1 = rasterio.open(t1_path)
        t2 = rasterio.open(t2_path)
        
        # Get image arrays (RGB)
        t1_rgb = t1.read([1, 2, 3])
        t2_rgb = t2.read([1, 2, 3])
        
        # Basic image statistics
        for idx, (name, img) in enumerate([("T1 (before)", t1_rgb), ("T2 (after)", t2_rgb)]):
            print(f"{name}:")
            print(f"  Shape: {img.shape}")
            print(f"  Min values: {[img[b].min() for b in range(3)]}")
            print(f"  Max values: {[img[b].max() for b in range(3)]}")
            print(f"  Mean values: {[img[b].mean() for b in range(3)]}")
        
        # Compute simple difference to highlight changes
        diff = np.abs(t1_rgb.astype(np.float32) - t2_rgb.astype(np.float32))
        diff_normalized = diff / diff.max() if diff.max() > 0 else diff
        
        # Create a false-color difference image (red = big difference)
        diff_vis = np.zeros((diff.shape[1], diff.shape[2], 3))
        diff_intensity = np.sum(diff_normalized, axis=0) / 3
        diff_vis[..., 0] = diff_intensity  # Red channel
        
        # Create visualization
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Normalize images for better visualization
        def normalize_for_display(img):
            result = np.zeros_like(img, dtype=np.float32)
            for b in range(img.shape[0]):
                band = img[b].astype(np.float32)
                p2 = np.percentile(band, 2)
                p98 = np.percentile(band, 98)
                if p98 > p2:
                    result[b] = np.clip((band - p2) / (p98 - p2), 0, 1)
                else:
                    result[b] = band / band.max() if band.max() > 0 else band
            return np.transpose(result, (1, 2, 0))
        
        t1_vis = normalize_for_display(t1_rgb)
        t2_vis = normalize_for_display(t2_rgb)
        
        # Plot images
        axs[0, 0].set_title("T1 (Before)")
        axs[0, 0].imshow(t1_vis)
        
        axs[0, 1].set_title("T2 (After)")
        axs[0, 1].imshow(t2_vis)
        
        # Compute and show a "difference" visualization
        axs[1, 0].set_title("Difference (Red = changes)")
        axs[1, 0].imshow(diff_vis)
        
        # Create an overlay where T1 is in red channel and T2 is in green channel
        # This creates a yellow color where pixels are the same and red/green where they differ
        overlay = np.zeros((t1_rgb.shape[1], t1_rgb.shape[2], 3))
        
        for i in range(3):  # For each RGB channel
            # Get normalized bands
            t1_band = t1_rgb[i].astype(float)
            t2_band = t2_rgb[i].astype(float)
            
            # Normalize to 0-1
            if t1_band.max() > t1_band.min():
                t1_band = (t1_band - t1_band.min()) / (t1_band.max() - t1_band.min())
            if t2_band.max() > t2_band.min():
                t2_band = (t2_band - t2_band.min()) / (t2_band.max() - t2_band.min())
            
            # Add to overlay
            overlay[..., 0] += t1_band / 3  # Red channel = T1
            overlay[..., 1] += t2_band / 3  # Green channel = T2
            overlay[..., 2] += (t1_band + t2_band) / 6  # Blue channel = average
        
        # Clip to valid range
        overlay = np.clip(overlay, 0, 1)
        
        axs[1, 1].set_title("Red=T1, Green=T2 (Yellow=Same)")
        axs[1, 1].imshow(overlay)
        
        # Remove ticks
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f"visualizations/{sample}_raw_analysis.png", dpi=200, bbox_inches='tight')
        print(f"Saved raw image analysis to visualizations/{sample}_raw_analysis.png")

def analyze_prediction_files(num_files=10):
    """
    Analyze the prediction files to understand the actual values in them.
    """
    pred_files = list(PRED_DIR.glob("*_change_mask.tif"))
    print(f"\n----- Analyzing {min(num_files, len(pred_files))} prediction files -----")
    
    # Process each prediction file
    for i, pred_file in enumerate(pred_files[:num_files]):
        sample_name = pred_file.stem.replace("_change_mask", "")
        
        # Open the prediction file
        with rasterio.open(pred_file) as src:
            pred = src.read(1)
            profile = src.profile
            
            # Get basic statistics
            min_val = pred.min()
            max_val = pred.max()
            mean_val = pred.mean()
            unique_vals = np.unique(pred)
            
            print(f"\nPrediction file: {pred_file.name}")
            print(f"  Format: {profile['dtype']}, shape: {pred.shape}")
            print(f"  Min: {min_val}, Max: {max_val}, Mean: {mean_val:.6f}")
            print(f"  Unique values: {unique_vals}")
            
            # Count pixels with different values
            pixel_counts = [(val, (pred == val).sum()) for val in unique_vals]
            for val, count in pixel_counts:
                percentage = (count / (pred.shape[0] * pred.shape[1])) * 100
                print(f"    Value {val}: {count} pixels ({percentage:.2f}%)")

def analyze_vector_files():
    """
    Analyze the vector shapefile outputs.
    """
    vector_files = list(PRED_DIR.glob("*_change_vectors.shp"))
    print(f"\n----- Analyzing {len(vector_files)} vector files -----")
    
    # Try to import geopandas
    try:
        import geopandas as gpd
        
        # Count vectors with features
        vectors_with_features = 0
        
        for vector_file in vector_files:
            try:
                gdf = gpd.read_file(vector_file)
                feature_count = len(gdf)
                
                if feature_count > 0:
                    vectors_with_features += 1
                    print(f"Vector file with features: {vector_file.name} - {feature_count} features")
            except Exception as e:
                print(f"Error reading {vector_file.name}: {str(e)}")
        
        print(f"\nFound {vectors_with_features} out of {len(vector_files)} vector files with features.")
        
    except ImportError:
        print("Geopandas not available. Skipping vector analysis.")

def check_threshold_values(threshold_values=[0, 1, 10, 50, 100, 127, 255]):
    """
    Check prediction files against different threshold values to see if any changes are detected.
    """
    pred_files = list(PRED_DIR.glob("*_change_mask.tif"))
    if not pred_files:
        print("No prediction files found.")
        return
    
    print(f"\n----- Checking {len(pred_files)} prediction files against multiple thresholds -----")
    
    # Results structure: {threshold: [(sample_name, change_pixels, percentage), ...]}
    results = {t: [] for t in threshold_values}
    
    for pred_file in pred_files:
        sample_name = pred_file.stem.replace("_change_mask", "")
        
        with rasterio.open(pred_file) as src:
            pred = src.read(1)
            
            # Check each threshold
            for threshold in threshold_values:
                changes = (pred > threshold).sum()
                if changes > 0:
                    percentage = (changes / (pred.shape[0] * pred.shape[1])) * 100
                    results[threshold].append((sample_name, changes, percentage))
    
    # Report findings
    for threshold in threshold_values:
        samples_with_changes = results[threshold]
        if samples_with_changes:
            samples_with_changes.sort(key=lambda x: x[1], reverse=True)
            top_samples = samples_with_changes[:5]
            
            print(f"\nThreshold > {threshold}: {len(samples_with_changes)} samples with changes")
            print("  Top samples:")
            for name, changes, percentage in top_samples:
                print(f"    - {name}: {changes} pixels ({percentage:.2f}%)")
        else:
            print(f"\nThreshold > {threshold}: No samples with changes")

def main():
    """Main entry point."""
    # Create directory for visualizations
    os.makedirs("visualizations", exist_ok=True)
    
    # First analyze a few raw image pairs to see if there are visible changes
    analyze_raw_images(num_samples=3)
    
    # Then analyze the prediction files to understand their values
    analyze_prediction_files(num_files=5)
    
    # Check vector files
    analyze_vector_files()
    
    # Check different threshold values
    check_threshold_values()
    
    print("\nAnalysis complete. Check the visualizations directory for output images.")

if __name__ == "__main__":
    main()