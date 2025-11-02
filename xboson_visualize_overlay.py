#!/usr/bin/env python3
"""
XBoson AI - PS-10 Change Detection Enhanced Visualization
Overlay shapefile change detection results on original RGB satellite images with advanced visualization
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
import rasterio
import geopandas as gpd
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def load_rgb_image(image_path):
    """Load and prepare RGB image for visualization with enhanced contrast"""
    try:
        with rasterio.open(image_path) as src:
            # Read RGB bands (assuming bands 1,2,3 are RGB or similar)
            if src.count >= 3:
                # Read first 3 bands for RGB visualization
                r = src.read(1)
                g = src.read(2) 
                b = src.read(3)
                
                # Stack into RGB
                rgb = np.stack([r, g, b], axis=2)
                
                # Normalize with enhanced contrast using percentile clipping
                rgb_norm = np.zeros_like(rgb, dtype=np.float32)
                
                for i in range(3):
                    band = rgb[:,:,i].astype(np.float32)
                    # Use percentile clipping for better contrast
                    p2 = np.percentile(band, 2)
                    p98 = np.percentile(band, 98)
                    if p98 > p2:
                        rgb_norm[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
                    else:
                        rgb_norm[:,:,i] = (band - band.min()) / (band.max() - band.min()) if band.max() > band.min() else band
                
                return rgb_norm, src.transform, src.crs
            else:
                # Handle grayscale image - convert to RGB
                band = src.read(1)
                # Normalize with enhanced contrast
                band_norm = np.clip((band - np.percentile(band, 2)) / (np.percentile(band, 98) - np.percentile(band, 2)), 0, 1) \
                            if np.percentile(band, 98) > np.percentile(band, 2) else \
                            (band - band.min()) / (band.max() - band.min()) if band.max() > band.min() else band
                
                rgb_norm = np.stack([band_norm, band_norm, band_norm], axis=2)
                return rgb_norm, src.transform, src.crs
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None

def visualize_change_overlay(rgb_image_path, shapefile_path, output_path=None, title="XBoson AI - Man-made Change Detection Overlay"):
    """
    Create visualization with shapefile overlay on RGB image with enhanced styling,
    specifically highlighting man-made changes as required by PS-10
    """
    print(f"Processing: {rgb_image_path}")
    print(f"Shapefile: {shapefile_path}")
    
    try:
        # Load RGB image
        rgb, transform, crs = load_rgb_image(rgb_image_path)
        if rgb is None:
            print("Failed to load RGB image")
            return
            
        # Load shapefile
        if os.path.exists(shapefile_path) and os.path.getsize(shapefile_path) > 100:
            gdf = gpd.read_file(shapefile_path)
            print(f"Loaded {len(gdf)} change polygons")
        else:
            print("No change polygons found in shapefile")
            gdf = None
        
        # Create figure with enhanced styling
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
        
        # Set background color for professional look
        fig.patch.set_facecolor('#F0F0F0')
        
        # Plot 1: Original RGB image with enhanced styling
        axes[0].imshow(rgb)
        axes[0].set_title("Original RGB Image", fontsize=14, fontweight='bold', pad=10)
        axes[0].set_xlabel("X (Image Coordinates)", fontsize=10)
        axes[0].set_ylabel("Y (Image Coordinates)", fontsize=10)
        axes[0].grid(False)
        
        # Plot 2: RGB with professional change overlay
        axes[1].imshow(rgb)
        
        # Overlay change polygons if they exist with enhanced styling
        if gdf is not None and len(gdf) > 0:
            # Create more professional-looking overlay
            gdf.plot(ax=axes[1], color='red', alpha=0.65, edgecolor='yellow', linewidth=1.5)
            axes[1].set_title(f"Man-made Changes Detected ({len(gdf)} features)", fontsize=14, fontweight='bold', pad=10)
        else:
            axes[1].set_title("No Man-made Changes Detected", fontsize=14, fontweight='bold', pad=10)
        
        axes[1].set_xlabel("X (Image Coordinates)", fontsize=10)
        axes[1].set_ylabel("Y (Image Coordinates)", fontsize=10)
        axes[1].grid(False)
        
        # Add legend with enhanced styling
        if gdf is not None and len(gdf) > 0:
            change_patch = patches.Patch(color='red', alpha=0.65, label='Detected Man-made Changes')
            axes[1].legend(handles=[change_patch], loc='upper right', frameon=True, 
                          framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Add figure title with enhanced styling
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Add PS-10 compliance note
        fig.text(0.5, 0.01, 'PS-10: Man-made Change Detection in Satellite Imagery', 
                fontsize=9, color='navy', ha='center', fontweight='bold')
        
        # Add branding and timestamp
        fig.text(0.01, 0.01, f'XBoson AI • {datetime.now().strftime("%Y-%m-%d")}', 
                fontsize=9, color='dimgray', ha='left')
        
        # Add technical info with coordinate system verification
        # PS-10 requires georeferenced output
        if crs:
            coord_info = f"Georeferenced: {crs.to_string()}"
            coord_color = 'darkgreen'
        else:
            coord_info = "WARNING: Missing coordinate reference system"
            coord_color = 'darkred'
            
        fig.text(0.99, 0.01, coord_info, fontsize=8, color=coord_color, ha='right')
        
        plt.tight_layout()
        
        # Save or show with enhanced quality
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"Error processing visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_before_after_comparison(before_image, after_image, shapefile_path, output_path=None, title="XBoson AI - Before/After Change Detection"):
    """
    Create professional before/after comparison with change overlay and advanced layout
    """
    print(f"Creating before/after comparison...")
    
    try:
        # Load both images
        rgb_before, transform_before, crs_before = load_rgb_image(before_image)
        rgb_after, transform_after, crs_after = load_rgb_image(after_image)
        
        if rgb_before is None or rgb_after is None:
            print("Failed to load images")
            return
        
        # Load shapefile
        if os.path.exists(shapefile_path) and os.path.getsize(shapefile_path) > 100:
            gdf = gpd.read_file(shapefile_path)
            has_changes = len(gdf) > 0
        else:
            gdf = None
            has_changes = False
        
        # Create figure with advanced layout
        fig = plt.figure(figsize=(20, 10), dpi=100, facecolor='white')
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.05], width_ratios=[1, 1, 1])
        
        # Before image
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(rgb_before)
        ax1.set_title("Before (T1) Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # After image
        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(rgb_after)
        ax2.set_title("After (T2) Image", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # After image with change overlay
        ax3 = plt.subplot(gs[0, 2])
        ax3.imshow(rgb_after)
        
        if has_changes:
            gdf.plot(ax=ax3, color='red', alpha=0.65, edgecolor='yellow', linewidth=1.5)
            ax3.set_title(f"Detected Changes ({len(gdf)} features)", fontsize=14, fontweight='bold')
        else:
            ax3.set_title("No Changes Detected", fontsize=14, fontweight='bold')
            
            # Add text for no changes case
            ax3.text(0.5, 0.5, 'No changes detected in this image pair', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12, color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='navy', alpha=0.7))
        
        ax3.axis('off')
        
        # Add colorbar section
        if has_changes:
            cax = plt.subplot(gs[1, 1])
            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), cax=cax, orientation='horizontal')
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(['Low Confidence', 'Medium', 'High Confidence'])
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('Change Detection Confidence (XBoson AI)', fontsize=12)
        
        # Add overall title with enhanced styling
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Add branding and information
        fig.text(0.01, 0.02, f'XBoson AI • PS-10 Change Detection Solution', 
                fontsize=10, color='navy', ha='left', weight='bold')
        
        fig.text(0.99, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                fontsize=8, color='gray', ha='right')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save or show with high resolution
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved before/after comparison to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"Error creating before/after comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_multi_band_visualization(image_path, shapefile_path=None, output_path=None, title="XBoson AI - Multi-band Visualization"):
    """
    Create visualization showing different band combinations with optional change overlay
    """
    print(f"Creating multi-band visualization for: {image_path}")
    
    try:
        # Load image with all bands
        with rasterio.open(image_path) as src:
            if src.count < 3:
                print("Image doesn't have enough bands for multi-band visualization")
                return False
            
            # Read bands
            band_count = min(src.count, 6)  # Limit to 6 bands for visualization
            bands = [src.read(i) for i in range(1, band_count + 1)]
            
            # Get metadata
            transform = src.transform
            crs = src.crs
        
        # Load shapefile if provided
        if shapefile_path and os.path.exists(shapefile_path) and os.path.getsize(shapefile_path) > 100:
            gdf = gpd.read_file(shapefile_path)
            print(f"Loaded {len(gdf)} change polygons")
            has_changes = len(gdf) > 0
        else:
            gdf = None
            has_changes = False
        
        # Create figure for multi-band visualization
        fig = plt.figure(figsize=(15, 10), dpi=100)
        
        # Determine layouts based on available bands
        if band_count >= 3:
            # Create True Color (RGB) - Bands 1,2,3
            ax1 = fig.add_subplot(2, 2, 1)
            rgb = np.stack([bands[0], bands[1], bands[2]], axis=2)
            rgb_norm = np.zeros_like(rgb, dtype=np.float32)
            
            for i in range(3):
                band = rgb[:,:,i].astype(np.float32)
                p2 = np.percentile(band, 2)
                p98 = np.percentile(band, 98)
                if p98 > p2:
                    rgb_norm[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
                else:
                    rgb_norm[:,:,i] = (band - band.min()) / (band.max() - band.min()) if band.max() > band.min() else band
            
            ax1.imshow(rgb_norm)
            ax1.set_title("True Color (RGB)", fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # Add change overlay to the RGB image if available
            if has_changes:
                gdf.plot(ax=ax1, color='yellow', alpha=0.5, edgecolor='red', linewidth=1)
            
            # If we have more than 3 bands, create false color composites
            if band_count >= 4:
                # False Color 1 (near IR, Red, Green) - typically bands 4,1,2
                ax2 = fig.add_subplot(2, 2, 2)
                
                ir_idx = min(3, band_count-1)  # Use 4th band if available, otherwise 3rd
                nir_rg = np.stack([bands[ir_idx], bands[0], bands[1]], axis=2)
                nir_rg_norm = np.zeros_like(nir_rg, dtype=np.float32)
                
                for i in range(3):
                    band = nir_rg[:,:,i].astype(np.float32)
                    p2 = np.percentile(band, 2)
                    p98 = np.percentile(band, 98)
                    if p98 > p2:
                        nir_rg_norm[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
                    else:
                        nir_rg_norm[:,:,i] = (band - band.min()) / (band.max() - band.min()) if band.max() > band.min() else band
                
                ax2.imshow(nir_rg_norm)
                ax2.set_title(f"False Color (NIR-R-G)", fontsize=12, fontweight='bold')
                ax2.axis('off')
            
            # Add NDVI if we have Red and NIR bands
            if band_count >= 4:
                ax3 = fig.add_subplot(2, 2, 3)
                
                # Compute simple NDVI (using band 4 as NIR and band 1 as Red)
                red = bands[0].astype(np.float32)
                nir = bands[3].astype(np.float32)
                
                # Avoid division by zero
                denominator = nir + red
                ndvi = np.zeros_like(red)
                valid_idx = denominator > 0
                ndvi[valid_idx] = (nir[valid_idx] - red[valid_idx]) / denominator[valid_idx]
                
                # Display NDVI
                im3 = ax3.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                ax3.set_title("NDVI (Vegetation Index)", fontsize=12, fontweight='bold')
                ax3.axis('off')
                
                # Add colorbar
                cbar = plt.colorbar(im3, ax=ax3, orientation='vertical', shrink=0.8)
                cbar.set_label('NDVI Value', fontsize=10)
            
            # Add change detection overlay on RGB with legend
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.imshow(rgb_norm)
            ax4.set_title("XBoson AI Change Detection", fontsize=12, fontweight='bold')
            ax4.axis('off')
            
            if has_changes:
                gdf.plot(ax=ax4, color='red', alpha=0.7, edgecolor='yellow', linewidth=1.5)
                
                # Add legend
                change_patch = patches.Patch(color='red', alpha=0.7, label='Detected Changes')
                ax4.legend(handles=[change_patch], loc='upper right')
            else:
                ax4.text(0.5, 0.5, 'No changes detected', 
                        transform=ax4.transAxes, ha='center', va='center', fontsize=12, 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.7))
        
        # Add title and branding
        plt.suptitle(title, fontsize=16, fontweight='bold')
        fig.text(0.01, 0.01, f'XBoson AI • {datetime.now().strftime("%Y-%m-%d")}',
                fontsize=9, color='dimgray', ha='left')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save or show with high resolution
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-band visualization to: {output_path}")
        else:
            plt.show()
            
        plt.close()
        return True
    
    except Exception as e:
        print(f"Error creating multi-band visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_change_dashboard(before_image, after_image, mask_path, shapefile_path, output_path=None, title="XBoson AI - Change Detection Dashboard"):
    """
    Create comprehensive change detection dashboard with multiple visualization components
    """
    print("Creating comprehensive change detection dashboard...")
    
    try:
        # Load before and after images
        rgb_before, transform_before, crs_before = load_rgb_image(before_image)
        rgb_after, transform_after, crs_after = load_rgb_image(after_image)
        
        if rgb_before is None or rgb_after is None:
            print("Failed to load before/after images")
            return False
        
        # Load change mask if available
        if mask_path and os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                change_mask = src.read(1)
                has_mask = True
        else:
            has_mask = False
            change_mask = None
        
        # Load shapefile if available
        if shapefile_path and os.path.exists(shapefile_path) and os.path.getsize(shapefile_path) > 100:
            gdf = gpd.read_file(shapefile_path)
            has_changes = len(gdf) > 0
        else:
            gdf = None
            has_changes = False
        
        # Create dashboard layout
        fig = plt.figure(figsize=(18, 12), dpi=100, facecolor='white')
        gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.1])
        
        # Before image (large)
        ax1 = plt.subplot(gs[0, 0:2])
        ax1.imshow(rgb_before)
        ax1.set_title("Before (T1) Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # After image (large)
        ax2 = plt.subplot(gs[0, 2:4])
        ax2.imshow(rgb_after)
        ax2.set_title("After (T2) Image", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Image difference visualization
        ax3 = plt.subplot(gs[1, 0])
        
        # Simple difference visualization (red channel)
        diff_r = np.abs(rgb_after[:,:,0] - rgb_before[:,:,0])
        diff_g = np.abs(rgb_after[:,:,1] - rgb_before[:,:,1])
        diff_b = np.abs(rgb_after[:,:,2] - rgb_before[:,:,2])
        
        # Combine differences
        diff_rgb = np.stack([diff_r, diff_g, diff_b], axis=2)
        # Enhance contrast
        p98 = np.percentile(diff_rgb, 98)
        diff_rgb_norm = np.clip(diff_rgb / p98 if p98 > 0 else diff_rgb, 0, 1)
        
        ax3.imshow(diff_rgb_norm)
        ax3.set_title("Image Difference", fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Change mask visualization
        ax4 = plt.subplot(gs[1, 1])
        ax4.imshow(rgb_after)  # Background image
        
        if has_mask:
            # Create a red overlay for changes
            mask_overlay = np.zeros((*change_mask.shape, 4))
            mask_overlay[..., 0] = 1.0  # Red
            mask_overlay[..., 3] = (change_mask > 0) * 0.6  # Alpha for changes
            ax4.imshow(mask_overlay)
            ax4.set_title("Change Mask", fontsize=12, fontweight='bold')
        else:
            ax4.set_title("Change Mask (Not Available)", fontsize=12, fontweight='bold')
        
        ax4.axis('off')
        
        # Vector overlay visualization
        ax5 = plt.subplot(gs[1, 2])
        ax5.imshow(rgb_after)
        
        if has_changes:
            gdf.plot(ax=ax5, color='red', alpha=0.6, edgecolor='yellow', linewidth=1)
            ax5.set_title(f"Vector Overlay ({len(gdf)} features)", fontsize=12, fontweight='bold')
        else:
            ax5.set_title("Vector Overlay (No Changes)", fontsize=12, fontweight='bold')
        
        ax5.axis('off')
        
        # Combined visualization
        ax6 = plt.subplot(gs[1, 3])
        
        # Create a side-by-side comparison in one panel
        # Split the image in half
        h, w = rgb_before.shape[0], rgb_before.shape[1]
        combined = np.zeros_like(rgb_before)
        combined[:, :w//2, :] = rgb_before[:, :w//2, :]
        combined[:, w//2:, :] = rgb_after[:, w//2:, :]
        
        # Draw dividing line
        ax6.imshow(combined)
        ax6.axvline(x=w//2, color='white', linewidth=2)
        
        # Add text labels
        ax6.text(w//4, 20, "Before", color='white', fontsize=12, ha='center',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
        ax6.text(3*w//4, 20, "After", color='white', fontsize=12, ha='center',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
        
        ax6.set_title("Before/After Split View", fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # Add legend in the bottom row
        legend_ax = plt.subplot(gs[2, 1:3])
        legend_ax.axis('off')
        
        # Create legend elements
        if has_changes:
            change_patch = patches.Patch(color='red', alpha=0.7, label='Detected Changes')
            legend_elements = [change_patch]
            
            # Add statistics if we have them
            if has_mask:
                change_percent = (change_mask > 0).sum() / change_mask.size * 100
                stats_text = f"Changes: {change_percent:.2f}% of image area"
            else:
                stats_text = f"Changes: {len(gdf)} vector features detected"
                
            legend_ax.text(0.5, 0.7, stats_text, ha='center', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        else:
            legend_ax.text(0.5, 0.5, 'No changes detected in this image pair', 
                        ha='center', va='center', fontsize=14, fontweight='bold')
        
        if has_changes:
            # Add legend
            legend = legend_ax.legend(handles=legend_elements, loc='center', 
                                    fontsize=12, frameon=True, framealpha=0.9,
                                    title="XBoson AI Change Detection", title_fontsize=14)
        
        # Add title and branding
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Add footer with technical details and branding
        footer_text = f"XBoson AI • PS-10 Change Detection Solution • Generated: {datetime.now().strftime('%Y-%m-%d')}"
        fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10, color='dimgray')
        
        # Add coordinate system info if available
        if crs_after:
            crs_text = f"Coordinate System: {crs_after.to_string()}"
            fig.text(0.01, 0.01, crs_text, ha='left', fontsize=8, color='dimgray')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save or show with high resolution
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved comprehensive dashboard to: {output_path}")
        else:
            plt.show()
            
        plt.close()
        return True
    
    except Exception as e:
        print(f"Error creating change dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_ps10_sample_data():
    """
    Visualize PS-10 sample data with change detection overlay
    """
    print("\n=== XBoson AI - PS-10 Sample Data Visualization ===")
    
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
            
            # Create visualizations directory
            output_dir = Path("XBoson_AI_Visualizations")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Single image overlay
            visualize_change_overlay(
                new_image, 
                shapefile, 
                output_dir / "XBoson_AI_ps10_sample_overlay.png",
                "XBoson AI - PS-10 Sample: Change Detection Overlay"
            )
            
            # Before/after comparison
            create_before_after_comparison(
                old_image,
                new_image, 
                shapefile,
                output_dir / "XBoson_AI_ps10_sample_before_after.png",
                "XBoson AI - PS-10 Sample: Before/After Comparison"
            )
            
            # Multi-band visualization
            create_multi_band_visualization(
                new_image,
                shapefile,
                output_dir / "XBoson_AI_ps10_sample_multi_band.png",
                "XBoson AI - PS-10 Sample: Multi-band Analysis"
            )
            
            # Comprehensive dashboard
            mask_file = results_dir / shapefiles[0].name.replace('.shp', '.tif')
            create_change_dashboard(
                old_image,
                new_image,
                mask_file if mask_file.exists() else None,
                shapefile,
                output_dir / "XBoson_AI_ps10_sample_dashboard.png",
                "XBoson AI - PS-10 Change Detection Dashboard"
            )
            
            print(f"\n✅ Created PS-10 sample visualizations in {output_dir}")
        else:
            print("No shapefiles found in PS10_submission_results")
    else:
        print("PS10_submission_results directory not found")

def visualize_all_predictions():
    """
    Create overlay visualizations for all prediction results
    """
    print("\n=== XBoson AI - Creating Overlay Visualizations for All Predictions ===")
    
    # Paths
    predictions_dir = Path("predictions_final")
    results_dir = Path("PS10_submission_results") 
    test_data_dir = Path("changedetect/data/processed/train_pairs_small")
    output_dir = Path("XBoson_AI_Visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all shapefiles
    if results_dir.exists():
        shapefiles = list(results_dir.glob("Change_Mask_*.shp"))
        print(f"Found {len(shapefiles)} shapefiles to visualize")
        
        # Process a limited number for demo purposes
        max_visualizations = 5
        for i, shapefile in enumerate(shapefiles[:max_visualizations]):
            # Extract coordinate info from filename
            coord_part = shapefile.stem.replace("Change_Mask_", "")
            
            # Find corresponding test images
            test_images_t1 = list(test_data_dir.glob("*_t1.tif"))
            test_images_t2 = list(test_data_dir.glob("*_t2.tif"))
            
            if test_images_t1 and test_images_t2 and i < len(test_images_t1):
                t1_image = test_images_t1[i]
                t2_image = test_images_t2[i]
                
                if not t1_image.exists() or not t2_image.exists():
                    continue
                
                print(f"\n🔍 Processing location {coord_part}...")
                
                # Create standard overlay
                output_path = output_dir / f"XBoson_AI_overlay_{coord_part}.png"
                visualize_change_overlay(
                    t2_image,
                    shapefile,
                    output_path,
                    f"XBoson AI - Change Detection at {coord_part}"
                )
                
                # Create before/after comparison
                output_path = output_dir / f"XBoson_AI_before_after_{coord_part}.png"
                create_before_after_comparison(
                    t1_image,
                    t2_image,
                    shapefile,
                    output_path,
                    f"XBoson AI - Change Comparison at {coord_part}"
                )
        
        # Create summary visualization
        fig, axes = plt.subplots(1, len(shapefiles[:max_visualizations]), figsize=(5*max_visualizations, 6))
        
        if max_visualizations == 1:
            axes = [axes]  # Make it iterable when there's only one
            
        for i, (ax, shapefile) in enumerate(zip(axes, shapefiles[:max_visualizations])):
            # Extract coordinate info
            coord_part = shapefile.stem.replace("Change_Mask_", "")
            
            # Load a sample visualization if available
            vis_path = output_dir / f"XBoson_AI_overlay_{coord_part}.png"
            if vis_path.exists():
                try:
                    img = plt.imread(str(vis_path))
                    ax.imshow(img)
                    ax.set_title(f"Location {coord_part}", fontsize=12)
                except:
                    ax.text(0.5, 0.5, f"Location\n{coord_part}", 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=14, bbox=dict(facecolor='lightblue', alpha=0.5))
            else:
                ax.text(0.5, 0.5, f"Location\n{coord_part}", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, bbox=dict(facecolor='lightblue', alpha=0.5))
            
            ax.axis('off')
        
        plt.suptitle("XBoson AI - PS-10 Change Detection Summary", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save summary visualization
        summary_path = output_dir / "XBoson_AI_detection_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Created summary visualization: {summary_path}")
        print(f"✅ All overlay visualizations saved to: {output_dir}")
    else:
        print("❌ Results directory not found")

def run_demo_overlay():
    """Create a demo overlay using predictions_threshold_0.1 data"""
    print("\n=== XBoson AI - Running Quick Demo Overlay ===")
    
    # Use the threshold_0.1 data as an example
    pred_dir = Path("predictions_threshold_0.1")
    if not pred_dir.exists():
        print("❌ No prediction data found")
        return
    
    # Find a sample change mask and corresponding shapefile
    change_masks = list(pred_dir.glob("*_change_mask.tif"))
    
    if not change_masks:
        print("❌ No change mask files found")
        return
    
    # Use the first sample
    sample_mask = change_masks[0]
    sample_base = sample_mask.stem.replace("_change_mask", "")
    sample_shapefile = pred_dir / f"{sample_base}_change_vectors.shp"
    
    # Try to find the corresponding original images
    test_dir = Path("changedetect/data/processed/train_pairs")
    sample_t1 = test_dir / f"{sample_base}_t1.tif"
    sample_t2 = test_dir / f"{sample_base}_t2.tif"
    
    if not sample_t1.exists() or not sample_t2.exists() or not sample_shapefile.exists():
        print("❌ Could not find matching image pairs or shapefile")
        return
    
    # Create output directory
    output_dir = Path("XBoson_AI_Demo")
    output_dir.mkdir(exist_ok=True)
    
    # Create overlay visualization
    print(f"Creating overlay visualization for sample {sample_base}...")
    visualize_change_overlay(
        sample_t2,
        sample_shapefile,
        output_dir / "XBoson_AI_demo_overlay.png",
        f"XBoson AI - Demo Change Detection Overlay"
    )
    
    # Create before/after visualization
    create_before_after_comparison(
        sample_t1,
        sample_t2,
        sample_shapefile,
        output_dir / "XBoson_AI_demo_before_after.png",
        f"XBoson AI - Demo Before/After Comparison"
    )
    
    print(f"\n✅ Demo visualizations saved to: {output_dir}")

def main():
    """Main function for visualization"""
    print("\n" + "="*70)
    print("  XBOSON AI - PS-10 CHANGE DETECTION VISUALIZATION TOOLKIT  ".center(70))
    print("="*70)
    
    choice = input("""
Choose visualization option:
1. PS-10 sample data visualization (recommended)
2. All prediction overlays
3. Quick demo using existing predictions
4. Custom image and shapefile paths
5. Run full XBoson AI implementation

Enter choice (1-5): """).strip()
    
    if choice == "1":
        visualize_ps10_sample_data()
        
    elif choice == "2":
        visualize_all_predictions()
        
    elif choice == "3":
        run_demo_overlay()
        
    elif choice == "4":
        image_path = input("Enter RGB image path: ").strip()
        shapefile_path = input("Enter shapefile path: ").strip()
        output_path = input("Enter output image path (optional): ").strip() or None
        
        if os.path.exists(image_path) and os.path.exists(shapefile_path):
            visualize_change_overlay(image_path, shapefile_path, output_path)
        else:
            print("❌ One or both files not found!")
            
    elif choice == "5":
        # Execute the full implementation
        try:
            from run_xboson_ai import run_xboson_full_implementation
            run_xboson_full_implementation()
        except Exception as e:
            print(f"❌ Error running full implementation: {e}")
            print("Try running directly: python run_xboson_ai.py")
            
    else:
        print("Invalid choice. Running PS-10 sample visualization...")
        visualize_ps10_sample_data()
    
    print("\n" + "="*70)
    print("  XBOSON AI - VISUALIZATION COMPLETE  ".center(70))
    print("="*70)

if __name__ == "__main__":
    main()