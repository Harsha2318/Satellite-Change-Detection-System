#!/usr/bin/env python3
"""
XBoson AI - PS-10 Change Detection Visualization
Overlay shapefile change detection results on original RGB satellite images
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import rasterio
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_rgb_image(image_path):
    """Load and prepare RGB image for visualization"""
    with rasterio.open(image_path) as src:
        # Read RGB bands (assuming bands 1,2,3 are RGB or similar)
        bands = src.read([1, 2, 3])
        
        # Convert to format suitable for matplotlib (H, W, C)
        rgb = np.transpose(bands, (1, 2, 0))
        
        # Normalize values to 0-1 range for display
        rgb = rgb.astype(np.float32)
        
        # Handle different value ranges
        if rgb.max() > 1.0:
            if rgb.max() > 255:
                rgb = rgb / rgb.max()  # Normalize to 0-1
            else:
                rgb = rgb / 255.0  # Assume 0-255 range
        
        # Enhance contrast for better visualization
        rgb = np.clip(rgb, 0, 1)
        
        return rgb, src.transform, src.crs

def visualize_change_overlay(rgb_image_path, shapefile_path, output_path=None, title="Change Detection Overlay"):
    """
    Create visualization with shapefile overlay on RGB image
    """
    print(f"Processing: {rgb_image_path}")
    print(f"Shapefile: {shapefile_path}")
    
    try:
        # Load RGB image
        rgb, transform, crs = load_rgb_image(rgb_image_path)
        
        # Load shapefile
        if os.path.exists(shapefile_path) and os.path.getsize(shapefile_path) > 100:
            gdf = gpd.read_file(shapefile_path)
            print(f"Loaded {len(gdf)} change polygons")
        else:
            print("No change polygons found in shapefile")
            gdf = None
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Original RGB image
        axes[0].imshow(rgb, extent=[transform[2], transform[2] + transform[0] * rgb.shape[1],
                                   transform[5] + transform[4] * rgb.shape[0], transform[5]])
        axes[0].set_title("Original RGB Image")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        
        # Plot 2: RGB with change overlay
        axes[1].imshow(rgb, extent=[transform[2], transform[2] + transform[0] * rgb.shape[1],
                                   transform[5] + transform[4] * rgb.shape[0], transform[5]])
        
        # Overlay change polygons if they exist
        if gdf is not None and len(gdf) > 0:
            gdf.plot(ax=axes[1], color='red', alpha=0.7, edgecolor='darkred', linewidth=2)
            axes[1].set_title(f"Changes Detected ({len(gdf)} polygons)")
        else:
            axes[1].set_title("No Changes Detected")
        
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"Error processing {rgb_image_path}: {e}")
        import traceback
        traceback.print_exc()

def create_before_after_overlay(before_image, after_image, shapefile_path, output_path=None):
    """
    Create before/after comparison with change overlay
    """
    print(f"Creating before/after comparison...")
    
    try:
        # Load both images
        rgb_before, transform_before, crs_before = load_rgb_image(before_image)
        rgb_after, transform_after, crs_after = load_rgb_image(after_image)
        
        # Load shapefile
        if os.path.exists(shapefile_path) and os.path.getsize(shapefile_path) > 100:
            gdf = gpd.read_file(shapefile_path)
        else:
            gdf = None
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Before image
        axes[0].imshow(rgb_before, extent=[transform_before[2], transform_before[2] + transform_before[0] * rgb_before.shape[1],
                                         transform_before[5] + transform_before[4] * rgb_before.shape[0], transform_before[5]])
        axes[0].set_title("Before (T1)")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        
        # After image
        axes[1].imshow(rgb_after, extent=[transform_after[2], transform_after[2] + transform_after[0] * rgb_after.shape[1],
                                        transform_after[5] + transform_after[4] * rgb_after.shape[0], transform_after[5]])
        axes[1].set_title("After (T2)")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        
        # After image with change overlay
        axes[2].imshow(rgb_after, extent=[transform_after[2], transform_after[2] + transform_after[0] * rgb_after.shape[1],
                                        transform_after[5] + transform_after[4] * rgb_after.shape[0], transform_after[5]])
        
        if gdf is not None and len(gdf) > 0:
            gdf.plot(ax=axes[2], color='red', alpha=0.7, edgecolor='darkred', linewidth=2)
            axes[2].set_title(f"Changes Detected ({len(gdf)} areas)")
        else:
            axes[2].set_title("No Changes Detected")
        
        axes[2].set_xlabel("Longitude")
        axes[2].set_ylabel("Latitude")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved before/after visualization to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"Error creating before/after comparison: {e}")
        import traceback
        traceback.print_exc()

def visualize_ps10_sample_data():
    """
    Visualize PS-10 sample data with change detection overlay
    """
    print("=== PS-10 Sample Data Visualization ===")
    
    # PS-10 sample data paths
    sample_dir = Path("PS10_data/Sample_Set/LISS-4/Sample")
    old_image = sample_dir / "Old_Image_MX_Band_2_3_4.tif"
    new_image = sample_dir / "New_Image_MX_Band_2_3_4.tif"
    
    # Check if sample data exists
    if not old_image.exists() or not new_image.exists():
        print("PS-10 sample data not found. Please ensure data is in PS10_data/Sample_Set/LISS-4/Sample/")
        return
    
    # Find corresponding shapefile (use the first one as example)
    results_dir = Path("PS10_submission_results")
    if results_dir.exists():
        shapefiles = list(results_dir.glob("Change_Mask_*.shp"))
        if shapefiles:
            shapefile = shapefiles[0]  # Use first shapefile as example
            
            # Create visualizations
            output_dir = Path("visualizations/overlays")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Single image overlay
            visualize_change_overlay(
                new_image, 
                shapefile, 
                output_dir / "ps10_sample_overlay.png",
                "PS-10 Sample: Change Detection Overlay"
            )
            
            # Before/after comparison
            create_before_after_overlay(
                old_image,
                new_image, 
                shapefile,
                output_dir / "ps10_sample_before_after.png"
            )
            
        else:
            print("No shapefiles found in PS10_submission_results")
    else:
        print("PS10_submission_results directory not found")

def visualize_all_predictions():
    """
    Create overlay visualizations for all prediction results
    """
    print("=== Creating Overlay Visualizations for All Predictions ===")
    
    # Paths
    predictions_dir = Path("predictions_final")
    results_dir = Path("PS10_submission_results") 
    test_data_dir = Path("changedetect/data/processed/train_pairs_small")
    output_dir = Path("visualizations/all_overlays")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all shapefiles
    if results_dir.exists():
        shapefiles = list(results_dir.glob("Change_Mask_*.shp"))
        print(f"Found {len(shapefiles)} shapefiles to visualize")
        
        for i, shapefile in enumerate(shapefiles[:5]):  # Limit to first 5 for demo
            # Extract coordinate info from filename
            coord_part = shapefile.stem.replace("Change_Mask_", "")
            
            # Find corresponding test images (use first available as example)
            test_images = list(test_data_dir.glob("*_t2.tif"))  # Use t2 (after) images
            
            if test_images and i < len(test_images):
                test_image = test_images[i]
                
                output_path = output_dir / f"overlay_{coord_part}.png"
                
                visualize_change_overlay(
                    test_image,
                    shapefile,
                    output_path,
                    f"Change Detection: {coord_part}"
                )
            
    print(f"Overlay visualizations saved to: {output_dir}")

def main():
    """Main function for visualization"""
    print("=== Change Detection Overlay Visualization ===")
    
    choice = input("""
Choose visualization option:
1. PS-10 sample data (recommended)
2. All prediction overlays
3. Custom image and shapefile paths
4. Create before/after comparison for sample data

Enter choice (1-4): """).strip()
    
    if choice == "1":
        visualize_ps10_sample_data()
        
    elif choice == "2":
        visualize_all_predictions()
        
    elif choice == "3":
        image_path = input("Enter RGB image path: ").strip()
        shapefile_path = input("Enter shapefile path: ").strip()
        
        if os.path.exists(image_path) and os.path.exists(shapefile_path):
            visualize_change_overlay(image_path, shapefile_path)
        else:
            print("One or both files not found!")
            
    elif choice == "4":
        sample_dir = Path("PS10_data/Sample_Set/LISS-4/Sample")
        old_image = sample_dir / "Old_Image_MX_Band_2_3_4.tif"
        new_image = sample_dir / "New_Image_MX_Band_2_3_4.tif"
        
        results_dir = Path("PS10_submission_results")
        shapefiles = list(results_dir.glob("Change_Mask_*.shp"))
        
        if old_image.exists() and new_image.exists() and shapefiles:
            create_before_after_overlay(old_image, new_image, shapefiles[0])
        else:
            print("Required files not found!")
            
    else:
        print("Invalid choice. Running PS-10 sample visualization...")
        visualize_ps10_sample_data()
    
    print("\nVisualization complete!")
    print("Check the 'visualizations/overlays' directory for output images.")

if __name__ == "__main__":
    main()