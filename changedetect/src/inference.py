"""
Inference script for satellite image change detection
"""
import os
import sys
import time
import logging
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
import geopandas as gpd
from shapely.geometry import shape, mapping

from changedetect.src.models.siamese_unet import get_change_detection_model
from changedetect.src.data.tile import generate_tile_windows
from changedetect.src.data.preprocess import preprocess_pair, normalize_image
from changedetect.src.utils.geoutils import read_geotiff, write_geotiff, raster_to_vector
from changedetect.src.utils.md5_utils import compute_md5, verify_md5

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path, model_type, in_channels, device, features=64, bilinear=False):
    """
    Load a trained change detection model.
    
    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model to load
        in_channels: Number of input channels
        device: Device to load the model on
        features: Number of features in the first layer
        bilinear: Whether to use bilinear interpolation or transposed convolutions
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    # Create model architecture
    model = get_change_detection_model(
        model_type=model_type,
        in_channels=in_channels,
        out_channels=1,
        features=features,
        bilinear=bilinear
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded with IoU: {checkpoint.get('val_iou', 'N/A')}")
    
    return model


def predict_tile(model, t1_tile, t2_tile, device):
    """
    Make a prediction for a single tile.
    
    Args:
        model: Trained model
        t1_tile: Image tile at time 1 (np.ndarray)
        t2_tile: Image tile at time 2 (np.ndarray)
        device: Device to run inference on
        
    Returns:
        Prediction mask (np.ndarray)
    """
    # Preprocess tiles
    t1_tile, t2_tile = preprocess_pair(t1_tile, t2_tile)
    
    # Convert to PyTorch tensors
    t1_tensor = torch.from_numpy(t1_tile).unsqueeze(0).to(device).float()
    t2_tensor = torch.from_numpy(t2_tile).unsqueeze(0).to(device).float()
    
    # Make prediction
    with torch.no_grad():
        output = model(t1_tensor, t2_tensor)
        prediction = torch.sigmoid(output) > 0.5
        
    # Convert to numpy array
    prediction = prediction.squeeze().cpu().numpy().astype(np.uint8) * 255
    
    return prediction


def merge_tile_predictions(tile_predictions, output_shape, overlap=0):
    """
    Merge tile predictions into a single image with averaging in overlapping regions.
    
    Args:
        tile_predictions: List of dicts with 'prediction' and 'window' keys
        output_shape: (height, width) of the output image
        overlap: Overlap between tiles in pixels
        
    Returns:
        Merged prediction as numpy array
    """
    height, width = output_shape
    merged = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    
    for tile_pred in tile_predictions:
        prediction = tile_pred['prediction']
        window = tile_pred['window']
        
        # Get window bounds
        row_off = int(window.row_off)
        col_off = int(window.col_off)
        win_height = int(window.height)
        win_width = int(window.width)
        
        # Ensure prediction matches window size
        if prediction.shape != (win_height, win_width):
            # Prediction might be 2D already, if not squeeze it
            if prediction.ndim > 2:
                prediction = prediction.squeeze()
        
        # Add to merged image
        merged[row_off:row_off+win_height, col_off:col_off+win_width] += prediction
        counts[row_off:row_off+win_height, col_off:col_off+win_width] += 1
    
    # Average overlapping regions
    counts[counts == 0] = 1  # Avoid division by zero
    merged = merged / counts
    
    return merged.astype(np.uint8)


def predict_large_image(model, t1_path, t2_path, output_path, tile_size=256, overlap=64, device='cuda'):
    """
    Make predictions for a large image by tiling.
    
    Args:
        model: Trained model
        t1_path: Path to image at time 1
        t2_path: Path to image at time 2
        output_path: Path to save the prediction mask
        tile_size: Size of tiles for inference
        overlap: Overlap between tiles
        device: Device to run inference on
        
    Returns:
        Path to the saved prediction mask
    """
    # Read images
    with rasterio.open(t1_path) as src:
        t1_profile = src.profile.copy()
        height, width = src.height, src.width
    
    # Create tiles (in-memory windows)
    tiles = generate_tile_windows(
        (height, width),
        tile_size=tile_size,
        overlap=overlap,
        src_transform=t1_profile.get('transform') if 'transform' in t1_profile else None
    )
    
    # Make predictions for each tile
    tile_predictions = []
    
    for i, tile in enumerate(tiles):
        logger.info(f"Processing tile {i+1}/{len(tiles)}")
        
        # Read tile data
        with rasterio.open(t1_path) as src:
            t1_tile = src.read(window=tile['window'])
        
        with rasterio.open(t2_path) as src:
            t2_tile = src.read(window=tile['window'])
        
        # Make prediction
        prediction = predict_tile(model, t1_tile, t2_tile, device)
        
        # Store prediction with window information
        tile_predictions.append({
            'prediction': prediction,
            'window': tile['window'],
            'transform': tile['transform']
        })
    
    # Merge tiles into a single image
    merged_prediction = merge_tile_predictions(
        tile_predictions,
        (height, width),
        overlap=overlap
    )
    
    # Set output profile
    out_profile = t1_profile.copy()
    out_profile.update({
        'count': 1,
        'dtype': 'uint8',
        'compress': 'lzw',
        'nodata': 0
    })
    
    # Write merged prediction to file
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(merged_prediction.astype(np.uint8), 1)
    
    logger.info(f"Saved prediction to {output_path}")
    
    return output_path


def create_vector_output(raster_path, vector_path, min_area=10):
    """
    Convert a raster change detection mask to a vector shapefile.
    
    Args:
        raster_path: Path to the raster change detection mask
        vector_path: Path to save the vector output
        min_area: Minimum area in pixels for a feature to be included
        
    Returns:
        Path to the saved vector output
    """
    logger.info(f"Converting raster {raster_path} to vector {vector_path}")

    try:
        # raster_to_vector expects (raster_path, vector_path, value=1, min_area=0)
        raster_to_vector(raster_path, vector_path, value=1, min_area=min_area)
        logger.info(f"Saved vector output to {vector_path}")
        return vector_path
    except Exception as e:
        logger.error(f"Failed to create vector output for {raster_path}: {e}")
        raise


def batch_inference(model, image_pairs, output_dir, tile_size=256, overlap=64, 
                   device='cuda', create_vector=True, min_area=10):
    """
    Run inference on multiple image pairs.
    
    Args:
        model: Trained model
        image_pairs: List of dictionaries with 't1', 't2', and 'name' keys
        output_dir: Directory to save outputs
        tile_size: Size of tiles for inference
        overlap: Overlap between tiles
        device: Device to run inference on
        create_vector: Whether to create vector outputs
        min_area: Minimum area for vector features
        
    Returns:
        List of output paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = []
    
    for pair in image_pairs:
        t1_path = pair['t1']
        t2_path = pair['t2']
        name = pair['name']
        
        logger.info(f"Processing image pair: {name}")
        
        # Create output paths
        raster_output = os.path.join(output_dir, f"{name}_change_mask.tif")
        vector_output = os.path.join(output_dir, f"{name}_change_vectors.shp")
        
        # Run inference
        predict_large_image(
            model,
            t1_path,
            t2_path,
            raster_output,
            tile_size=tile_size,
            overlap=overlap,
            device=device
        )
        
        # Create vector output if requested
        if create_vector:
            create_vector_output(
                raster_output,
                vector_output,
                min_area=min_area
            )
        
        output_paths.append({
            'name': name,
            'raster': raster_output,
            'vector': vector_output if create_vector else None
        })
    
    return output_paths


def get_image_pairs(image_dir):
    """
    Get list of image pair files.
    
    Args:
        image_dir: Directory containing image pairs
        
    Returns:
        List of dictionaries containing t1, t2, and name file paths
    """
    image_dir = Path(image_dir)
    image_pairs = []
    
    # List all t1 images
    t1_images = sorted(list(image_dir.glob("*_t1.tif")))
    
    for t1_path in t1_images:
        # Get corresponding t2 image
        name = t1_path.stem.replace("_t1", "")
        t2_path = image_dir / f"{name}_t2.tif"
        
        if not t2_path.exists():
            logger.warning(f"Missing t2 image for {name}")
            continue
        
        image_pairs.append({
            "name": name,
            "t1": str(t1_path),
            "t2": str(t2_path),
        })
    
    return image_pairs


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run inference for satellite image change detection")
    
    # Input parameters
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing image pairs (t1 and t2)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./predictions",
                       help="Directory to save predictions")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="siamese_unet",
                       choices=["siamese_unet", "siamese_diff", "fcn_diff"],
                       help="Type of change detection model")
    parser.add_argument("--in_channels", type=int, default=3,
                       help="Number of input channels per image")
    parser.add_argument("--features", type=int, default=64,
                       help="Number of features in the first layer")
    parser.add_argument("--bilinear", action="store_true", default=False,
                       help="Use bilinear interpolation instead of transposed convolutions (default: False, use transposed convolutions)")
    
    # Inference parameters
    parser.add_argument("--tile_size", type=int, default=256,
                       help="Size of image tiles for inference")
    parser.add_argument("--overlap", type=int, default=64,
                       help="Overlap between tiles in pixels")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to run inference on")
    parser.add_argument("--no_vector", action="store_true",
                       help="Skip vector output generation")
    parser.add_argument("--min_area", type=int, default=10,
                       help="Minimum area in pixels for vector features")
    parser.add_argument("--verify_md5", action="store_true",
                       help="Verify MD5 hash of the model")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Verify model MD5 if requested
    if args.verify_md5:
        hash_file = f"{args.model_path}.md5"
        if verify_md5(args.model_path, hash_file=hash_file):
            logger.info("Model MD5 hash verification passed")
        else:
            logger.error("Model MD5 hash verification failed")
            return 1
    
    # Load model
    model = load_model(args.model_path, args.model_type, args.in_channels, device, args.features, args.bilinear)
    
    # Get image pairs
    image_pairs = get_image_pairs(args.image_dir)
    logger.info(f"Found {len(image_pairs)} image pairs")
    
    if not image_pairs:
        logger.error("No image pairs found")
        return 1
    
    # Run batch inference
    output_paths = batch_inference(
        model,
        image_pairs,
        args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=device,
        create_vector=not args.no_vector,
        min_area=args.min_area
    )
    
    # Save summary of outputs
    summary_path = os.path.join(args.output_dir, "inference_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(output_paths, f, indent=4)
    
    logger.info(f"Inference completed for {len(image_pairs)} image pairs")
    logger.info(f"Results saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())