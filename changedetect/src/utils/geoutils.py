"""
Geospatial utilities for handling GeoTIFF and other geospatial data formats.
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_geotiff(filepath):
    """
    Read a GeoTIFF file and return data as numpy array along with metadata.
    
    Args:
        filepath (str): Path to the GeoTIFF file
        
    Returns:
        tuple: (numpy array, metadata dictionary)
    """
    try:
        # Use rasterio.Env to prefer EPSG parameters from the registry over GeoTIFF keys
        # when PROJ installations conflict (common on Windows with multiple PROJ installs).
        with rasterio.Env(GTIFF_SRS_SOURCE='EPSG'):
            with rasterio.open(filepath) as src:
                data = src.read()
                meta = src.meta.copy()

                # If single band, remove the band dimension for easier handling
                if data.shape[0] == 1:
                    data = data[0]

                return data, meta
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        raise

def write_geotiff(filepath, data, metadata, dtype=None):
    """
    Write a numpy array to a GeoTIFF file with proper metadata.
    
    Args:
        filepath (str): Output file path
        data (numpy.ndarray): Data to write
        metadata (dict): Metadata dictionary with rasterio profile
        dtype (rasterio.dtype, optional): Data type for the output file
    """
    meta = metadata.copy()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare data if it's not 3D
    if len(data.shape) == 2:
        data = data[np.newaxis, :, :]  # Add band dimension
    
    # Update metadata
    meta.update({
        'count': data.shape[0],
        'height': data.shape[1],
        'width': data.shape[2],
        'dtype': dtype if dtype else data.dtype
    })
    
    try:
        # Use rasterio.Env to ensure GTIFF_SRS_SOURCE is set to EPSG during write
        with rasterio.Env(GTIFF_SRS_SOURCE='EPSG'):
            with rasterio.open(filepath, 'w', **meta) as dst:
                dst.write(data)
        logger.info(f"Written GeoTIFF to {filepath}")
    except Exception as e:
        logger.error(f"Error writing to {filepath}: {e}")
        raise

def reproject_raster(src_path, dst_path, dst_crs):
    """
    Reproject a raster to a new coordinate reference system.
    
    Args:
        src_path (str): Source raster path
        dst_path (str): Destination raster path
        dst_crs (str): Target CRS in any format accepted by rasterio
    """
    try:
        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            with rasterio.open(dst_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )
        logger.info(f"Reprojected {src_path} to {dst_path} in CRS {dst_crs}")
    except Exception as e:
        logger.error(f"Error reprojecting raster: {e}")
        raise

def raster_to_vector(raster_path, vector_path, value=1, min_area=0):
    """
    Convert a binary raster to a vector shapefile.
    
    Args:
        raster_path (str): Path to input GeoTIFF
        vector_path (str): Path to output shapefile
        value (int): The pixel value to vectorize (default: 1, for binary masks where 1=change)
        min_area (float): Minimum area in map units to keep (default: 0, keep all)
    """
    try:
        with rasterio.open(raster_path) as src:
            # Read the binary mask
            if src.count == 1:
                # Single band raster
                mask = src.read(1)
            else:
                # Multi-band, use first band
                logger.warning(f"Raster has {src.count} bands, using first band only for vectorization")
                mask = src.read(1)
            
            # Get shapes from the mask
            results = (
                {'properties': {'value': v}, 'geometry': s}
                for s, v in shapes(mask, mask=(mask == value), transform=src.transform)
            )
            
            # Convert to GeoDataFrame
            geoms = list(results)
            if not geoms:
                logger.warning(f"No geometries found in {raster_path} with value={value}")
                # Create an empty GeoDataFrame with correct CRS
                gdf = gpd.GeoDataFrame(geometry=[], crs=src.crs)
            else:
                gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)
                
                # Filter by minimum area if specified
                if min_area > 0:
                    initial_count = len(gdf)
                    gdf = gdf[gdf.geometry.area >= min_area]
                    filtered_count = initial_count - len(gdf)
                    logger.info(f"Filtered out {filtered_count} polygons smaller than {min_area} map units")
            
            # Save to shapefile
            gdf.to_file(vector_path)
            logger.info(f"Saved {len(gdf)} polygons to {vector_path}")
    except Exception as e:
        logger.error(f"Error converting raster to vector: {e}")
        raise

def read_image_window(filepath, window, bands=None):
    """
    Read a subset of a GeoTIFF file.
    
    Args:
        filepath (str): Path to the GeoTIFF file
        window (rasterio.windows.Window): Window object defining the region to read
        bands (list, optional): List of band indices to read (1-based). If None, read all bands.
    
    Returns:
        tuple: (numpy array, updated metadata dictionary)
    """
    try:
        with rasterio.open(filepath) as src:
            if bands is None:
                bands = range(1, src.count + 1)
            
            # Read specified bands within window
            data = src.read(bands, window=window)
            
            # Update metadata with new dimensions
            meta = src.meta.copy()
            meta.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, src.transform)
            })
            
            return data, meta
    except Exception as e:
        logger.error(f"Error reading window from {filepath}: {e}")
        raise

def get_image_tiles(height, width, tile_size=1024, overlap=0):
    """
    Generate a list of tile windows for processing large images.
    
    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels
        tile_size (int): Size of tiles in pixels
        overlap (int): Overlap between tiles in pixels
    
    Returns:
        list: List of rasterio.windows.Window objects
    """
    effective_size = tile_size - overlap
    
    # Calculate number of tiles in each dimension
    n_tiles_height = max(1, int(np.ceil(height / effective_size)))
    n_tiles_width = max(1, int(np.ceil(width / effective_size)))
    
    windows = []
    
    for i in range(n_tiles_height):
        for j in range(n_tiles_width):
            # Calculate tile coordinates
            row_start = i * effective_size
            col_start = j * effective_size
            
            # Handle edge cases
            row_end = min(row_start + tile_size, height)
            col_end = min(col_start + tile_size, width)
            
            # Create window
            windows.append(
                Window(col_start, row_start, col_end - col_start, row_end - row_start)
            )
    
    return windows