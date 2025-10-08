"""
Postprocessing utilities for change detection masks
"""
import numpy as np
import cv2
from skimage import morphology
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon
import geopandas as gpd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_morphological_operations(mask, operation='opening', kernel_size=3):
    """
    Apply morphological operations to a binary mask.
    
    Args:
        mask: Binary mask (numpy array)
        operation: Type of operation ('opening', 'closing', 'dilation', 'erosion')
        kernel_size: Size of the kernel for morphological operations
        
    Returns:
        Processed mask
    """
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply morphological operation
    if operation == 'opening':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'dilation':
        return cv2.dilate(mask, kernel)
    elif operation == 'erosion':
        return cv2.erode(mask, kernel)
    else:
        raise ValueError(f"Unknown morphological operation: {operation}")


def remove_small_objects(mask, min_size=10):
    """
    Remove small objects from a binary mask.
    
    Args:
        mask: Binary mask (numpy array)
        min_size: Minimum size of objects to keep
        
    Returns:
        Cleaned mask
    """
    # Convert to bool for skimage
    binary_mask = mask.astype(bool)
    
    # Remove small objects
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
    
    return cleaned_mask.astype(mask.dtype)


def remove_small_holes(mask, min_size=10):
    """
    Remove small holes from a binary mask.
    
    Args:
        mask: Binary mask (numpy array)
        min_size: Minimum size of holes to keep
        
    Returns:
        Filled mask
    """
    # Convert to bool for skimage
    binary_mask = mask.astype(bool)
    
    # Remove small holes
    filled_mask = morphology.remove_small_holes(binary_mask, area_threshold=min_size)
    
    return filled_mask.astype(mask.dtype)


def apply_threshold(mask, threshold=0.5):
    """
    Apply threshold to a probability mask.
    
    Args:
        mask: Probability mask (numpy array)
        threshold: Threshold value
        
    Returns:
        Binary mask
    """
    return (mask > threshold).astype(np.uint8)


def smooth_boundaries(mask, sigma=1.0):
    """
    Smooth mask boundaries using Gaussian blur.
    
    Args:
        mask: Binary mask (numpy array)
        sigma: Standard deviation for Gaussian blur
        
    Returns:
        Smoothed mask
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma)
    
    # Re-threshold to get binary mask
    return (blurred > 0.5).astype(np.uint8)


def filter_by_confidence(mask, confidence, threshold=0.8):
    """
    Filter mask by confidence values.
    
    Args:
        mask: Binary mask (numpy array)
        confidence: Confidence values (numpy array)
        threshold: Confidence threshold
        
    Returns:
        Filtered mask
    """
    # Apply confidence threshold
    high_confidence = confidence > threshold
    
    # Only keep high confidence pixels
    filtered_mask = mask.copy()
    filtered_mask[~high_confidence] = 0
    
    return filtered_mask


def process_change_detection_mask(mask_path, output_path, 
                                min_size=10, apply_opening=True, 
                                apply_closing=True, fill_holes=True):
    """
    Process a change detection mask with multiple operations.
    
    Args:
        mask_path: Path to input mask
        output_path: Path to save processed mask
        min_size: Minimum size of objects to keep
        apply_opening: Whether to apply morphological opening
        apply_closing: Whether to apply morphological closing
        fill_holes: Whether to fill small holes
        
    Returns:
        Path to processed mask
    """
    # Read the mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        profile = src.profile.copy()
    
    # Ensure binary mask
    binary_mask = (mask > 0).astype(np.uint8) * 255
    
    # Apply processing steps
    processed_mask = binary_mask.copy()
    
    if apply_opening:
        processed_mask = apply_morphological_operations(processed_mask, 'opening', kernel_size=3)
    
    if apply_closing:
        processed_mask = apply_morphological_operations(processed_mask, 'closing', kernel_size=3)
    
    if min_size > 0:
        processed_mask = remove_small_objects(processed_mask, min_size=min_size)
    
    if fill_holes:
        processed_mask = remove_small_holes(processed_mask, min_size=min_size)
    
    # Write processed mask
    profile.update({
        'dtype': 'uint8',
        'compress': 'lzw'
    })
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(processed_mask, 1)
    
    logger.info(f"Processed mask saved to {output_path}")
    
    return output_path


def simplify_vector_geometries(vector_path, output_path, tolerance=1.0):
    """
    Simplify vector geometries.
    
    Args:
        vector_path: Path to input vector file
        output_path: Path to save simplified vector
        tolerance: Simplification tolerance
        
    Returns:
        Path to simplified vector
    """
    # Read the vector file
    gdf = gpd.read_file(vector_path)
    
    # Simplify geometries
    gdf['geometry'] = gdf.geometry.simplify(tolerance, preserve_topology=True)
    
    # Save simplified vector
    gdf.to_file(output_path)
    
    logger.info(f"Simplified vector saved to {output_path}")
    
    return output_path


def filter_vectors_by_area(vector_path, output_path, min_area=10):
    """
    Filter vector features by area.
    
    Args:
        vector_path: Path to input vector file
        output_path: Path to save filtered vector
        min_area: Minimum area in square meters
        
    Returns:
        Path to filtered vector
    """
    # Read the vector file
    gdf = gpd.read_file(vector_path)
    
    # Calculate areas if not already present
    if 'area' not in gdf.columns:
        gdf['area'] = gdf.geometry.area
    
    # Filter by area
    filtered_gdf = gdf[gdf['area'] >= min_area]
    
    # Save filtered vector
    filtered_gdf.to_file(output_path)
    
    logger.info(f"Filtered vector saved to {output_path} (removed {len(gdf) - len(filtered_gdf)} features)")
    
    return output_path


def mask_to_polygons(mask, transform, min_area=10):
    """
    Convert a binary mask to polygons.
    
    Args:
        mask: Binary mask (numpy array)
        transform: Affine transform
        min_area: Minimum area for polygons
        
    Returns:
        List of polygon geometries
    """
    # Get shapes from mask
    results = (
        {'geometry': shape(geom), 'value': value}
        for geom, value in shapes(mask.astype(np.int16), mask=mask > 0, transform=transform)
    )
    
    # Filter by value and area
    polygons = []
    for result in results:
        if result['value'] > 0 and result['geometry'].area >= min_area:
            polygons.append(result['geometry'])
    
    return polygons


def compute_change_statistics(mask_path):
    """
    Compute statistics for a change detection mask.
    
    Args:
        mask_path: Path to change detection mask
        
    Returns:
        Dictionary of statistics
    """
    # Read the mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        total_pixels = mask.size
    
    # Compute statistics
    change_pixels = (mask > 0).sum()
    change_percentage = (change_pixels / total_pixels) * 100
    
    # Get polygon features
    polygons = mask_to_polygons(mask, transform)
    
    # Compute polygon statistics
    num_polygons = len(polygons)
    
    if num_polygons > 0:
        areas = [p.area for p in polygons]
        avg_area = sum(areas) / len(areas)
        max_area = max(areas)
        min_area = min(areas)
    else:
        avg_area = 0
        max_area = 0
        min_area = 0
    
    # Return statistics
    stats = {
        'total_pixels': int(total_pixels),
        'change_pixels': int(change_pixels),
        'change_percentage': float(change_percentage),
        'num_change_polygons': num_polygons,
        'avg_polygon_area': float(avg_area),
        'max_polygon_area': float(max_area),
        'min_polygon_area': float(min_area)
    }
    
    return stats


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Process change detection masks")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to input mask")
    parser.add_argument("--output", type=str, required=True,
                      help="Path to save processed mask")
    parser.add_argument("--min_size", type=int, default=10,
                      help="Minimum size of objects to keep")
    parser.add_argument("--no_opening", action="store_true",
                      help="Disable morphological opening")
    parser.add_argument("--no_closing", action="store_true",
                      help="Disable morphological closing")
    parser.add_argument("--no_fill_holes", action="store_true",
                      help="Disable filling of small holes")
    parser.add_argument("--stats", action="store_true",
                      help="Compute and display statistics")
    
    args = parser.parse_args()
    
    # Process mask
    output_path = process_change_detection_mask(
        args.input,
        args.output,
        min_size=args.min_size,
        apply_opening=not args.no_opening,
        apply_closing=not args.no_closing,
        fill_holes=not args.no_fill_holes
    )
    
    # Compute statistics if requested
    if args.stats:
        stats = compute_change_statistics(output_path)
        print(f"Change detection statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    sys.exit(0)