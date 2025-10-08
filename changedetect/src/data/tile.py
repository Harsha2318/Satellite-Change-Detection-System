"""
Utilities for tiling large satellite images into manageable chunks.
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from ..utils.geoutils import read_image_window, write_geotiff

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tiles(image_path, output_dir, tile_size=1024, overlap=128, bands=None, prefix=None):
    """
    Tile a large image into smaller chunks with optional overlap.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save the tiles
        tile_size (int): Size of the tiles in pixels
        overlap (int): Overlap between tiles in pixels
        bands (list, optional): List of bands to include (1-based)
        prefix (str, optional): Prefix for tile filenames
        
    Returns:
        list: List of paths to created tiles
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set prefix to basename of input file if not provided
    if prefix is None:
        prefix = Path(image_path).stem
    
    try:
        with rasterio.open(image_path) as src:
            # Get image dimensions
            height = src.height
            width = src.width
            
            # Generate tile windows
            effective_size = tile_size - overlap
            
            # Calculate number of tiles in each dimension
            n_tiles_height = max(1, int(np.ceil(height / effective_size)))
            n_tiles_width = max(1, int(np.ceil(width / effective_size)))
            
            logger.info(f"Creating {n_tiles_height * n_tiles_width} tiles for image {image_path}")
            
            tile_paths = []
            
            for i in range(n_tiles_height):
                for j in range(n_tiles_width):
                    # Calculate tile coordinates
                    row_start = min(i * effective_size, height - tile_size)
                    col_start = min(j * effective_size, width - tile_size)
                    row_start = max(0, row_start)  # Ensure non-negative
                    col_start = max(0, col_start)  # Ensure non-negative
                    
                    # Handle edge cases
                    row_end = min(row_start + tile_size, height)
                    col_end = min(col_start + tile_size, width)
                    
                    # Create window
                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    
                    # Read data
                    if bands is None:
                        bands_to_read = range(1, src.count + 1)
                    else:
                        bands_to_read = bands
                    
                    data = src.read(bands_to_read, window=window)
                    
                    # Update metadata
                    meta = src.meta.copy()
                    meta.update({
                        'height': window.height,
                        'width': window.width,
                        'transform': rasterio.windows.transform(window, src.transform),
                        'count': len(bands_to_read)
                    })
                    
                    # Generate output path
                    tile_path = os.path.join(output_dir, f"{prefix}_tile_{i}_{j}.tif")
                    
                    # Write tile
                    with rasterio.open(tile_path, 'w', **meta) as dst:
                        dst.write(data)
                    
                    tile_paths.append(tile_path)
            
            return tile_paths
    except Exception as e:
        logger.error(f"Error creating tiles for {image_path}: {e}")
        raise


def generate_tile_windows(image_size, tile_size=1024, overlap=128, src_transform=None):
    """
    Generate tile window definitions for an image shape without writing files.

    Args:
        image_size (tuple): (height, width) of the image
        tile_size (int): Tile size in pixels
        overlap (int): Overlap between tiles in pixels
        src_transform (Affine, optional): Source transform to compute per-tile transform

    Returns:
        list: List of dicts with keys 'id', 'window', and 'transform'
    """
    height, width = image_size

    effective_size = tile_size - overlap
    n_tiles_height = max(1, int(np.ceil(height / effective_size)))
    n_tiles_width = max(1, int(np.ceil(width / effective_size)))

    windows = []

    for i in range(n_tiles_height):
        for j in range(n_tiles_width):
            # Calculate tile coordinates
            row_start = min(i * effective_size, height - tile_size)
            col_start = min(j * effective_size, width - tile_size)
            row_start = max(0, row_start)
            col_start = max(0, col_start)

            row_end = min(row_start + tile_size, height)
            col_end = min(col_start + tile_size, width)

            window = Window(col_start, row_start, col_end - col_start, row_end - row_start)

            if src_transform is not None:
                transform = rasterio.windows.transform(window, src_transform)
            else:
                transform = None

            windows.append({
                'id': f"{i}_{j}",
                'window': window,
                'transform': transform
            })

    return windows

def create_tile_pairs(pre_image, post_image, output_dir, tile_size=1024, overlap=128, 
                     bands=None, prefix=None, mask=None):
    """
    Create corresponding tiles from pre and post images, and optionally a mask.
    
    Args:
        pre_image (str): Path to pre-change image
        post_image (str): Path to post-change image
        output_dir (str): Directory to save tile pairs
        tile_size (int): Size of tiles in pixels
        overlap (int): Overlap between tiles
        bands (list, optional): Bands to include (1-based)
        prefix (str, optional): Prefix for tile filenames
        mask (str, optional): Path to ground truth mask
        
    Returns:
        dict: Dictionary mapping tile indices to paths of pre, post, and mask tiles
    """
    # Ensure output directories exist
    pre_dir = os.path.join(output_dir, "pre")
    post_dir = os.path.join(output_dir, "post")
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)
    
    # Set prefix if not provided
    if prefix is None:
        prefix = f"{Path(pre_image).stem}_{Path(post_image).stem}"
    
    # Create mask directory if needed
    mask_dir = None
    if mask is not None:
        mask_dir = os.path.join(output_dir, "mask")
        os.makedirs(mask_dir, exist_ok=True)
    
    try:
        # Open all input images
        with rasterio.open(pre_image) as pre_src, \
             rasterio.open(post_image) as post_src:
            
            # Verify images have the same dimensions
            if pre_src.height != post_src.height or pre_src.width != post_src.width:
                logger.error("Pre and post images must have the same dimensions")
                raise ValueError("Image dimensions do not match")
            
            # Open mask if provided
            mask_src = None
            if mask is not None:
                mask_src = rasterio.open(mask)
                if pre_src.height != mask_src.height or pre_src.width != mask_src.width:
                    logger.error("Mask dimensions do not match image dimensions")
                    raise ValueError("Mask dimensions do not match image dimensions")
            
            # Get image dimensions
            height = pre_src.height
            width = pre_src.width
            
            # Generate tile windows
            effective_size = tile_size - overlap
            
            # Calculate number of tiles in each dimension
            n_tiles_height = max(1, int(np.ceil(height / effective_size)))
            n_tiles_width = max(1, int(np.ceil(width / effective_size)))
            
            logger.info(f"Creating {n_tiles_height * n_tiles_width} tile pairs")
            
            # Dictionary to store tile paths
            tile_pairs = {}
            
            for i in range(n_tiles_height):
                for j in range(n_tiles_width):
                    # Calculate tile coordinates
                    row_start = min(i * effective_size, height - tile_size)
                    col_start = min(j * effective_size, width - tile_size)
                    row_start = max(0, row_start)  # Ensure non-negative
                    col_start = max(0, col_start)  # Ensure non-negative
                    
                    # Handle edge cases
                    row_end = min(row_start + tile_size, height)
                    col_end = min(col_start + tile_size, width)
                    
                    # Create window
                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    
                    # Process pre-image tile
                    pre_data, pre_meta = read_image_window(pre_image, window, bands)
                    pre_tile_path = os.path.join(pre_dir, f"{prefix}_pre_tile_{i}_{j}.tif")
                    write_geotiff(pre_tile_path, pre_data, pre_meta)
                    
                    # Process post-image tile
                    post_data, post_meta = read_image_window(post_image, window, bands)
                    post_tile_path = os.path.join(post_dir, f"{prefix}_post_tile_{i}_{j}.tif")
                    write_geotiff(post_tile_path, post_data, post_meta)
                    
                    # Process mask tile if provided
                    mask_tile_path = None
                    if mask is not None:
                        mask_data, mask_meta = read_image_window(mask, window, [1])
                        mask_tile_path = os.path.join(mask_dir, f"{prefix}_mask_tile_{i}_{j}.tif")
                        write_geotiff(mask_tile_path, mask_data, mask_meta, dtype='uint8')
                    
                    # Store tile paths
                    tile_pairs[(i, j)] = {
                        'pre': pre_tile_path,
                        'post': post_tile_path,
                        'mask': mask_tile_path
                    }
            
            # Close mask if opened
            if mask_src is not None:
                mask_src.close()
                
            return tile_pairs
            
    except Exception as e:
        logger.error(f"Error creating tile pairs: {e}")
        raise

def merge_tiles(tile_dir, output_path, pattern="*.tif", mode="mean"):
    """
    Merge tiles back into a single image.
    
    Args:
        tile_dir (str): Directory containing tiles
        output_path (str): Path to save the merged image
        pattern (str): Glob pattern to match tile files
        mode (str): How to handle overlapping areas: 'mean', 'max', or 'min'
        
    Returns:
        str: Path to the merged image
    """
    # TODO: Implement tile merging logic
    # This is complex and would involve:
    # 1. Reading each tile and its position
    # 2. Creating a merged array of appropriate size
    # 3. Handling overlaps according to the specified mode
    # 4. Writing the final merged array with correct georeference
    
    logger.warning("merge_tiles functionality not fully implemented")
    return output_path