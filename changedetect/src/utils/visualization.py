"""
Visualization utilities for satellite image change detection
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
import rasterio
from rasterio.plot import show
import geopandas as gpd
import folium
from folium import Map, GeoJson, LayerControl, plugins
import webbrowser
import io
import base64
from PIL import Image
import logging
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_image_pair(image1_path, image2_path, figsize=(12, 6), title=None, bands=(3, 2, 1)):
    """
    Plot a pair of satellite images side by side.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        figsize: Size of the figure
        title: Title for the figure
        bands: RGB band indices (1-indexed for display)
    
    Returns:
        Figure and axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Convert to 0-indexed for array access
    band_indices = [i-1 for i in bands]
    
    # Load and plot first image
    with rasterio.open(image1_path) as src:
        # Get RGB bands
        if src.count >= 3:
            rgb = np.zeros((src.height, src.width, 3), dtype=np.float32)
            for i, band_idx in enumerate(band_indices):
                if band_idx < src.count:
                    band = src.read(band_idx + 1)
                    # Normalize band for display
                    band_min, band_max = np.percentile(band, (2, 98))
                    rgb[:, :, i] = np.clip((band - band_min) / (band_max - band_min), 0, 1)
            axes[0].imshow(rgb)
        else:
            # Show single band image
            band = src.read(1)
            axes[0].imshow(band, cmap='gray')
        
        axes[0].set_title(f"Image 1 - {os.path.basename(image1_path)}")
        axes[0].axis('off')
    
    # Load and plot second image
    with rasterio.open(image2_path) as src:
        # Get RGB bands
        if src.count >= 3:
            rgb = np.zeros((src.height, src.width, 3), dtype=np.float32)
            for i, band_idx in enumerate(band_indices):
                if band_idx < src.count:
                    band = src.read(band_idx + 1)
                    # Normalize band for display
                    band_min, band_max = np.percentile(band, (2, 98))
                    rgb[:, :, i] = np.clip((band - band_min) / (band_max - band_min), 0, 1)
            axes[1].imshow(rgb)
        else:
            # Show single band image
            band = src.read(1)
            axes[1].imshow(band, cmap='gray')
            
        axes[1].set_title(f"Image 2 - {os.path.basename(image2_path)}")
        axes[1].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    fig.tight_layout()
    
    return fig, axes

def plot_change_detection_results(image1_path, image2_path, prediction_path, ground_truth_path=None, figsize=(15, 10), alpha=0.7):
    """
    Plot change detection results with comparison to ground truth if available.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        prediction_path: Path to the prediction mask
        ground_truth_path: Path to the ground truth mask (optional)
        figsize: Size of the figure
        alpha: Transparency of the overlaid masks
    
    Returns:
        Figure and axes objects
    """
    if ground_truth_path:
        # If ground truth is available, create a 2x2 plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
    else:
        # If no ground truth, create a 1x3 plot
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Load images
    with rasterio.open(image1_path) as src:
        img1 = src.read()
        if img1.shape[0] >= 3:
            # True color composite (RGB)
            rgb1 = np.dstack((img1[0], img1[1], img1[2]))
            # Normalize for display
            rgb1 = np.clip((rgb1 - np.percentile(rgb1, 2)) / (np.percentile(rgb1, 98) - np.percentile(rgb1, 2)), 0, 1)
            axes[0].imshow(rgb1)
        else:
            # Single band image
            axes[0].imshow(img1[0], cmap='gray')
        axes[0].set_title("Image 1")
        axes[0].axis('off')
    
    with rasterio.open(image2_path) as src:
        img2 = src.read()
        if img2.shape[0] >= 3:
            # True color composite (RGB)
            rgb2 = np.dstack((img2[0], img2[1], img2[2]))
            # Normalize for display
            rgb2 = np.clip((rgb2 - np.percentile(rgb2, 2)) / (np.percentile(rgb2, 98) - np.percentile(rgb2, 2)), 0, 1)
            axes[1].imshow(rgb2)
        else:
            # Single band image
            axes[1].imshow(img2[0], cmap='gray')
        axes[1].set_title("Image 2")
        axes[1].axis('off')
    
    # Load prediction mask
    with rasterio.open(prediction_path) as src:
        pred = src.read(1)
        
    # Create a color overlay for prediction
    with rasterio.open(image2_path) as src:
        img2 = src.read()
        if img2.shape[0] >= 3:
            rgb2 = np.dstack((img2[0], img2[1], img2[2]))
            # Normalize for display
            rgb2 = np.clip((rgb2 - np.percentile(rgb2, 2)) / (np.percentile(rgb2, 98) - np.percentile(rgb2, 2)), 0, 1)
        else:
            # Convert single band to RGB for overlay
            rgb2 = np.dstack([img2[0]]*3)
            # Normalize for display
            rgb2 = np.clip((rgb2 - np.percentile(rgb2, 2)) / (np.percentile(rgb2, 98) - np.percentile(rgb2, 2)), 0, 1)
        
        # Create mask overlay
        mask_rgb = np.zeros_like(rgb2)
        mask_rgb[:, :, 0] = (pred > 0) * 1.0  # Red channel
        
        # Plot prediction overlay
        axes[2].imshow(rgb2)
        axes[2].imshow(mask_rgb, alpha=alpha)
        axes[2].set_title("Predicted Changes")
        axes[2].axis('off')
    
    # If ground truth is available, add it to the plot
    if ground_truth_path:
        with rasterio.open(ground_truth_path) as src:
            gt = src.read(1)
        
        # Create a color overlay for ground truth
        with rasterio.open(image2_path) as src:
            img2 = src.read()
            if img2.shape[0] >= 3:
                rgb2 = np.dstack((img2[0], img2[1], img2[2]))
                # Normalize for display
                rgb2 = np.clip((rgb2 - np.percentile(rgb2, 2)) / (np.percentile(rgb2, 98) - np.percentile(rgb2, 2)), 0, 1)
            else:
                # Convert single band to RGB for overlay
                rgb2 = np.dstack([img2[0]]*3)
                # Normalize for display
                rgb2 = np.clip((rgb2 - np.percentile(rgb2, 2)) / (np.percentile(rgb2, 98) - np.percentile(rgb2, 2)), 0, 1)
            
            # Create ground truth overlay
            gt_rgb = np.zeros_like(rgb2)
            gt_rgb[:, :, 1] = (gt > 0) * 1.0  # Green channel
            
            # Plot ground truth overlay
            axes[3].imshow(rgb2)
            axes[3].imshow(gt_rgb, alpha=alpha)
            axes[3].set_title("Ground Truth")
            axes[3].axis('off')
    
    fig.tight_layout()
    return fig, axes

def plot_confusion_matrix(pred_mask, gt_mask, title="Confusion Matrix"):
    """
    Plot confusion matrix for binary change detection.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        title: Title for the plot
    
    Returns:
        Figure object
    """
    # Flatten masks
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Calculate confusion matrix
    cm = confusion_matrix(gt_flat, pred_flat)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Change", "Change"])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    
    # Add percentages to the plot
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]} ({cm[i, j]/total:.1%})",
                   ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
    
    ax.set_title(title)
    
    return fig

def plot_metrics_comparison(metrics_list, model_names=None, figsize=(12, 8)):
    """
    Plot comparison of evaluation metrics for multiple models.
    
    Args:
        metrics_list: List of dictionaries containing metrics
        model_names: List of model names (optional)
        figsize: Size of the figure
    
    Returns:
        Figure object
    """
    # Define metrics to plot
    plot_metrics = ['iou', 'precision', 'recall', 'f1', 'accuracy']
    
    # Set model names if not provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(metrics_list))]
    
    # Create figure
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=figsize)
    
    # Plot each metric
    for i, metric_name in enumerate(plot_metrics):
        ax = axes[i]
        
        # Extract values for this metric
        values = [metrics.get(metric_name, 0) for metrics in metrics_list]
        
        # Create bar chart
        bars = ax.bar(model_names, values, color=sns.color_palette("muted", len(values)))
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Set labels and title
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()} Comparison")
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    return fig

def create_interactive_map(image_path, prediction_path=None, ground_truth_path=None, vector_path=None, output_path=None):
    """
    Create an interactive map for visualizing change detection results.
    
    Args:
        image_path: Path to the satellite image
        prediction_path: Path to the prediction mask (optional)
        ground_truth_path: Path to the ground truth mask (optional)
        vector_path: Path to the vector file (optional)
        output_path: Path to save the HTML map (optional)
    
    Returns:
        Folium map object
    """
    try:
        # Get image geospatial information
        with rasterio.open(image_path) as src:
            bounds = src.bounds
            crs = src.crs
            
            # Get center coordinates for the map
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
        
        # Create base map
        m = Map(location=[center_lat, center_lon], zoom_start=14, 
                tiles='OpenStreetMap')
        
        # Add satellite layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False
        ).add_to(m)
        
        # Add vector data if available
        if vector_path and os.path.exists(vector_path):
            # Read vector file
            gdf = gpd.read_file(vector_path)
            
            # Convert to WGS84 if needed
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Add to map
            geojson_data = gdf.to_json()
            
            # Create GeoJSON layer
            folium.GeoJson(
                geojson_data,
                name='Detected Changes',
                style_function=lambda x: {
                    'fillColor': 'red',
                    'color': 'red',
                    'weight': 2,
                    'fillOpacity': 0.6
                }
            ).add_to(m)
            
            # Add feature info popup
            folium.GeoJson(
                geojson_data,
                name='Change Info',
                style_function=lambda x: {'fillOpacity': 0},
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['area_m2'],
                    aliases=['Area (m²):'],
                    localize=True
                )
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Save to HTML if path provided
        if output_path:
            m.save(output_path)
            logger.info(f"Interactive map saved to {output_path}")
            
            # Open in browser
            webbrowser.open('file://' + os.path.abspath(output_path))
        
        return m
    
    except Exception as e:
        logger.error(f"Error creating interactive map: {e}")
        logger.exception(e)
        return None

def plot_training_history(history_path, figsize=(12, 5)):
    """
    Plot training history from a saved JSON file.
    
    Args:
        history_path: Path to the training history JSON file
        figsize: Size of the figure
    
    Returns:
        Figure object
    """
    import json
    
    try:
        # Load training history
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot training and validation loss
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot validation metrics
        if 'val_iou' in history:
            axes[1].plot(epochs, history['val_iou'], 'g-', label='IoU')
        if 'val_dice' in history:
            axes[1].plot(epochs, history['val_dice'], 'y-', label='Dice')
        if 'val_f1' in history:
            axes[1].plot(epochs, history['val_f1'], 'm-', label='F1')
        
        axes[1].set_title('Validation Metrics')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].grid(True)
        axes[1].legend()
        
        fig.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting training history: {e}")
        logger.exception(e)
        return None

def visualize_model_predictions(model, image1_path, image2_path, output_path=None, tile_size=256, device='cuda'):
    """
    Visualize model predictions for a pair of images.
    
    Args:
        model: Trained change detection model
        image1_path: Path to the first image
        image2_path: Path to the second image
        output_path: Path to save the visualization (optional)
        tile_size: Size of tiles for prediction
        device: Device to run prediction on
        
    Returns:
        Figure object and prediction mask
    """
    import torch
    from changedetect.src.inference import predict_large_image
    
    try:
        # Make prediction
        if output_path is None:
            output_path = os.path.join(os.path.dirname(image1_path), "prediction_temp.tif")
        
        # Run inference
        predict_large_image(model, image1_path, image2_path, output_path, tile_size=tile_size, device=device)
        
        # Load the prediction
        with rasterio.open(output_path) as src:
            prediction = src.read(1)
        
        # Create visualization
        fig, axes = plot_change_detection_results(image1_path, image2_path, output_path)
        
        return fig, prediction
    
    except Exception as e:
        logger.error(f"Error visualizing model predictions: {e}")
        logger.exception(e)
        return None, None