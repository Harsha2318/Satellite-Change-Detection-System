"""
Generate synthetic test data for testing the change detection pipeline
"""
import os
import sys
import numpy as np
from pathlib import Path
import random
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import rasterio
    from rasterio.transform import from_origin
    import cv2
except ImportError:
    logger.error("Required packages (rasterio, opencv-python) not installed.")
    logger.error("Install them with: pip install rasterio opencv-python")
    sys.exit(1)


def create_synthetic_landcover(width=512, height=512):
    """
    Create a synthetic landcover image with roads, buildings, and vegetation.
    
    Args:
        width (int): Width of the image
        height (int): Height of the image
        
    Returns:
        numpy.ndarray: Synthetic RGB image
    """
    # Create base image with vegetation-like texture
    base = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add ground texture (brownish-green)
    for i in range(3):
        base[:, :, i] = np.random.randint(0, 50, (height, width))
    
    base[:, :, 1] += np.random.randint(50, 100, (height, width))  # More green
    
    # Add some texture using perlin-like noise
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            c = np.random.randint(-20, 20)
            y_end = min(y + 4, height)
            x_end = min(x + 4, width)
            base[y:y_end, x:x_end, 1] = np.clip(base[y:y_end, x:x_end, 1] + c, 0, 255)
    
    # Add roads (gray lines)
    # Main roads
    thickness = width // 25
    cv2.line(base, (width//4, 0), (width//4, height), (100, 100, 100), thickness)
    cv2.line(base, (3*width//4, 0), (3*width//4, height), (100, 100, 100), thickness)
    cv2.line(base, (0, height//3), (width, height//3), (100, 100, 100), thickness)
    cv2.line(base, (0, 2*height//3), (width, 2*height//3), (100, 100, 100), thickness)
    
    # Secondary roads
    thickness = width // 50
    for i in range(3):
        x = np.random.randint(0, width)
        cv2.line(base, (x, 0), (x, height), (120, 120, 120), thickness)
        
    for i in range(3):
        y = np.random.randint(0, height)
        cv2.line(base, (0, y), (width, y), (120, 120, 120), thickness)
    
    # Add buildings (rectangular blocks)
    num_buildings = width * height // 10000
    for _ in range(num_buildings):
        bw = np.random.randint(10, width // 10)
        bh = np.random.randint(10, height // 10)
        bx = np.random.randint(0, width - bw)
        by = np.random.randint(0, height - bh)
        
        color = np.random.randint(150, 200, 3)
        cv2.rectangle(base, (bx, by), (bx + bw, by + bh), color.tolist(), -1)
        
        # Add building shadows
        shadow_color = color // 2
        shadow_offset = 5
        shadow_points = np.array([
            [bx + shadow_offset, by + shadow_offset],
            [bx + bw + shadow_offset, by + shadow_offset],
            [bx + bw + shadow_offset, by + bh + shadow_offset],
            [bx + shadow_offset, by + bh + shadow_offset]
        ])
        cv2.fillPoly(base, [shadow_points], shadow_color.tolist())
    
    # Add some random vegetation patches (bright green)
    num_patches = width * height // 5000
    for _ in range(num_patches):
        px = np.random.randint(0, width)
        py = np.random.randint(0, height)
        radius = np.random.randint(5, 30)
        color = (np.random.randint(0, 30), np.random.randint(180, 230), np.random.randint(0, 30))
        cv2.circle(base, (px, py), radius, color, -1)
    
    return base


def create_change_mask(image1, image2, threshold=30):
    """
    Create a binary change mask by comparing two images.
    
    Args:
        image1 (numpy.ndarray): First image
        image2 (numpy.ndarray): Second image
        threshold (int): Threshold for change detection
        
    Returns:
        numpy.ndarray: Binary change mask
    """
    # Compute absolute difference
    diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
    
    # Convert to grayscale if the images are RGB
    if len(diff.shape) == 3:
        diff_gray = np.mean(diff, axis=2)
    else:
        diff_gray = diff
    
    # Create binary mask
    mask = (diff_gray > threshold).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def add_changes(image, change_type='development'):
    """
    Add changes to an image to simulate development or natural changes.
    
    Args:
        image (numpy.ndarray): Base image to modify
        change_type (str): Type of change to add ('development' or 'natural')
        
    Returns:
        numpy.ndarray: Modified image with changes
    """
    height, width = image.shape[:2]
    modified = image.copy()
    
    if change_type == 'development':
        # Add new buildings
        num_new_buildings = width * height // 20000
        for _ in range(num_new_buildings):
            bw = np.random.randint(15, width // 8)
            bh = np.random.randint(15, height // 8)
            bx = np.random.randint(0, width - bw)
            by = np.random.randint(0, height - bh)
            
            color = np.random.randint(150, 220, 3)
            cv2.rectangle(modified, (bx, by), (bx + bw, by + bh), color.tolist(), -1)
            
            # Add shadow
            shadow_color = color // 2
            shadow_offset = 5
            shadow_points = np.array([
                [bx + shadow_offset, by + shadow_offset],
                [bx + bw + shadow_offset, by + shadow_offset],
                [bx + bw + shadow_offset, by + bh + shadow_offset],
                [bx + shadow_offset, by + bh + shadow_offset]
            ])
            cv2.fillPoly(modified, [shadow_points], shadow_color.tolist())
        
        # Add new roads
        num_new_roads = np.random.randint(1, 4)
        thickness = width // 40
        for _ in range(num_new_roads):
            if np.random.random() > 0.5:
                # Horizontal road
                y = np.random.randint(0, height)
                cv2.line(modified, (0, y), (width, y), (110, 110, 110), thickness)
            else:
                # Vertical road
                x = np.random.randint(0, width)
                cv2.line(modified, (x, 0), (x, height), (110, 110, 110), thickness)
        
    elif change_type == 'natural':
        # Add patches of different vegetation or water
        num_patches = width * height // 15000
        for _ in range(num_patches):
            px = np.random.randint(0, width)
            py = np.random.randint(0, height)
            radius = np.random.randint(10, 50)
            
            if np.random.random() > 0.7:
                # Water-like (blue)
                color = (np.random.randint(100, 150), np.random.randint(50, 100), np.random.randint(0, 50))
            else:
                # New vegetation (different green)
                color = (np.random.randint(0, 30), np.random.randint(100, 200), np.random.randint(0, 30))
                
            cv2.circle(modified, (px, py), radius, color, -1)
    
    return modified


def save_geotiff(image, output_path, transform=None, crs=None):
    """
    Save an image as a GeoTIFF file.
    
    Args:
        image (numpy.ndarray): Image to save
        output_path (str): Path to save the GeoTIFF file
        transform (affine.Affine): Affine transform for the GeoTIFF
        crs (rasterio.crs.CRS): Coordinate reference system
        
    Returns:
        str: Path to the saved file
    """
    # Handle RGB images
    if len(image.shape) == 3:
        height, width, channels = image.shape
        count = channels
    else:
        height, width = image.shape
        count = 1
    
    # Create default transform if not provided
    if transform is None:
        transform = from_origin(0, 0, 1, 1)
    
    # Create default CRS if not provided
    if crs is None:
        crs = {'init': 'epsg:4326'}
    
    # Prepare image for writing
    if count == 1 and len(image.shape) > 2:
        image = image[..., 0]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=image.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        if count == 1:
            dst.write(image, 1)
        else:
            for i in range(count):
                dst.write(image[..., i], i+1)
    
    logger.info(f"Saved GeoTIFF to {output_path}")
    return output_path


def generate_image_pair(output_dir, width=512, height=512, change_type='development'):
    """
    Generate a synthetic image pair with changes and corresponding change mask.
    
    Args:
        output_dir (str): Directory to save the generated images
        width (int): Width of the images
        height (int): Height of the images
        change_type (str): Type of change to add ('development' or 'natural')
        
    Returns:
        tuple: Paths to the before image, after image, and change mask
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base image
    image1 = create_synthetic_landcover(width, height)
    
    # Add changes to create the after image
    image2 = add_changes(image1, change_type)
    
    # Create change mask
    mask = create_change_mask(image1, image2)
    
    # Save images and mask
    image1_path = os.path.join(output_dir, "image1.tif")
    image2_path = os.path.join(output_dir, "image2.tif")
    mask_path = os.path.join(output_dir, "mask.tif")
    
    save_geotiff(image1, image1_path)
    save_geotiff(image2, image2_path)
    save_geotiff(mask, mask_path)
    
    return image1_path, image2_path, mask_path


def generate_test_dataset(output_dir, num_pairs=5, width=512, height=512):
    """
    Generate a synthetic test dataset with multiple image pairs.
    
    Args:
        output_dir (str): Directory to save the dataset
        num_pairs (int): Number of image pairs to generate
        width (int): Width of the images
        height (int): Height of the images
        
    Returns:
        list: List of tuples (before_path, after_path, mask_path) for each generated pair
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pairs = []
    for i in range(num_pairs):
        # Alternate between development and natural changes
        change_type = 'development' if i % 2 == 0 else 'natural'
        
        pair_dir = os.path.join(output_dir, f"pair_{i+1}")
        os.makedirs(pair_dir, exist_ok=True)
        
        logger.info(f"Generating pair {i+1}/{num_pairs} ({change_type} changes)")
        
        before_path, after_path, mask_path = generate_image_pair(
            pair_dir, width, height, change_type
        )
        
        pairs.append((before_path, after_path, mask_path))
    
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic test data for change detection")
    parser.add_argument("--output", type=str, default="tests/test_data/synthetic",
                      help="Output directory for the synthetic dataset")
    parser.add_argument("--num_pairs", type=int, default=5,
                      help="Number of image pairs to generate")
    parser.add_argument("--width", type=int, default=512,
                      help="Width of the generated images")
    parser.add_argument("--height", type=int, default=512,
                      help="Height of the generated images")
    
    args = parser.parse_args()
    
    try:
        generate_test_dataset(args.output, args.num_pairs, args.width, args.height)
        logger.info(f"Successfully generated {args.num_pairs} image pairs in {args.output}")
    except Exception as e:
        logger.error(f"Error generating test dataset: {str(e)}")
        sys.exit(1)