"""
Preprocessing functions for satellite imagery.
"""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from skimage import exposure
import cv2
from scipy import ndimage
import logging
from pathlib import Path

from ..utils.geoutils import read_geotiff, write_geotiff

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_image(img, method='minmax', percentiles=(2, 98)):
    """
    Normalize image values using various methods.
    
    Args:
        img (numpy.ndarray): Input image
        method (str): Normalization method: 'minmax', 'percentile', or 'histogram'
        percentiles (tuple): Lower and upper percentiles for percentile normalization
        
    Returns:
        numpy.ndarray: Normalized image
    """
    if method == 'minmax':
        # Simple min-max normalization to [0, 1]
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val > min_val:
            normalized = (img - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(img)
    
    elif method == 'percentile':
        # Clip values to percentiles then normalize
        p_low, p_high = np.percentile(img, percentiles)
        if p_high <= p_low:
            # Image is constant or percentiles collapsed; return zeros to avoid NaNs
            normalized = np.zeros_like(img, dtype=np.float32)
        else:
            normalized = np.clip(img, p_low, p_high)
            normalized = (normalized - p_low) / (p_high - p_low)
    
    elif method == 'histogram':
        # Histogram equalization
        normalized = exposure.equalize_hist(img)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def cloud_mask_sentinel2(image_data, bands, threshold=0.3):
    """
    Simple cloud detection for Sentinel-2 imagery.
    
    Args:
        image_data (numpy.ndarray): Image data with bands as first dimension
        bands (dict): Dictionary mapping band names to indices
        threshold (float): Cloud detection threshold
        
    Returns:
        numpy.ndarray: Binary cloud mask (1=cloud, 0=clear)
    """
    # Simple thresholding approach using visible and NIR bands
    # More sophisticated cloud masking would use Sentinel-2 SCL band if available
    
    # Use blue band for cloud detection
    if 'blue' in bands:
        blue_idx = bands['blue']
        blue_band = image_data[blue_idx]
        
        # Threshold the blue band (clouds are bright in blue)
        cloud_mask = blue_band > threshold
        
        # Optionally apply morphological operations to clean up the mask
        cloud_mask = ndimage.binary_opening(cloud_mask, structure=np.ones((5, 5)))
        cloud_mask = ndimage.binary_closing(cloud_mask, structure=np.ones((5, 5)))
        
        return cloud_mask.astype(np.uint8)
    else:
        logger.warning("Blue band not found, cannot create cloud mask")
        return np.zeros_like(image_data[0], dtype=np.uint8)

def coregister_images(reference_path, target_path, output_path, max_shift=20, method='phase'):
    """
    Co-register target image to reference image using phase correlation or feature matching.
    
    Args:
        reference_path (str): Path to reference image
        target_path (str): Path to target image
        output_path (str): Path to save co-registered image
        max_shift (int): Maximum allowed shift in pixels
        method (str): Method for co-registration: 'phase' or 'feature'
        
    Returns:
        tuple: (output_path, translation vector)
    """
    # Read images
    reference, ref_meta = read_geotiff(reference_path)
    target, target_meta = read_geotiff(target_path)
    
    # Handle multi-band images
    if len(reference.shape) > 2 and reference.shape[0] > 1:
        # Use first band for co-registration
        reference_band = reference[0]
        target_band = target[0]
    else:
        reference_band = reference
        target_band = target
    
    # Normalize images to improve matching
    ref_norm = normalize_image(reference_band, method='percentile')
    target_norm = normalize_image(target_band, method='percentile')
    
    # Convert to uint8 for OpenCV
    ref_uint8 = (ref_norm * 255).astype(np.uint8)
    target_uint8 = (target_norm * 255).astype(np.uint8)
    
    if method == 'phase':
        # Phase correlation method
        shift = cv2.phaseCorrelate(ref_uint8.astype(np.float32), target_uint8.astype(np.float32))
        dx, dy = shift[0]
        
        # Limit shifts to max_shift
        dx = np.clip(dx, -max_shift, max_shift)
        dy = np.clip(dy, -max_shift, max_shift)
        
        logger.info(f"Detected shift: dx={dx}, dy={dy}")
        
        # Apply shift using affine transformation
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        
    elif method == 'feature':
        # Feature-based matching using ORB features
        orb = cv2.ORB_create()
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(ref_uint8, None)
        kp2, des2 = orb.detectAndCompute(target_uint8, None)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take top matches
        good_matches = matches[:min(50, len(matches))]
        
        # Extract locations of matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Extract translation components
        dx = M[0, 2]
        dy = M[1, 2]
        
        # Limit shifts to max_shift
        dx = np.clip(dx, -max_shift, max_shift)
        dy = np.clip(dy, -max_shift, max_shift)
        
        logger.info(f"Detected shift: dx={dx}, dy={dy}")
        
        # Create translation matrix
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    
    else:
        raise ValueError(f"Unknown co-registration method: {method}")
    
    # Apply the transformation to each band
    coregistered = np.zeros_like(target)
    
    for i in range(target.shape[0]):
        coregistered[i] = cv2.warpAffine(
            target[i], 
            translation_matrix, 
            (target.shape[2], target.shape[1]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
    
    # Update geotransform in metadata
    # (In a real implementation, we would update the geotransform properly)
    
    # Write co-registered image
    write_geotiff(output_path, coregistered, target_meta)
    
    return output_path, (dx, dy)

def preprocess_pair(pre_image, post_image, output_dir=None, 
                   normalize=True, coregister=True, create_composite=True):
    """
    Preprocess a pair of images for change detection.

    This function accepts either file paths (strings) or in-memory numpy arrays.
    - If provided paths and output_dir is set, it will write preprocessed files and return paths.
    - If provided numpy arrays and output_dir is None, it will return processed numpy arrays.

    Args:
        pre_image (str or np.ndarray): Path to pre-change image or in-memory array
        post_image (str or np.ndarray): Path to post-change image or in-memory array
        output_dir (str, optional): Directory to save preprocessed outputs (if writing files)
        normalize (bool): Whether to normalize images
        coregister (bool): Whether to co-register post image to pre image (only when using file paths)
        create_composite (bool): Whether to create a composite of pre and post images (file mode)

    Returns:
        If output_dir is provided and inputs are paths: dict with 'pre','post','composite' paths.
        If inputs are arrays and output_dir is None: tuple (pre_array, post_array)
    """

    # If inputs are numpy arrays (in-memory path), perform simple normalization and return arrays
    if not isinstance(pre_image, (str, Path)) and not isinstance(post_image, (str, Path)) and output_dir is None:
        pre_img = pre_image
        post_img = post_image
        # Ensure arrays are in (C,H,W)
        if pre_img.ndim == 2:
            pre_img = pre_img[np.newaxis, ...]
        if post_img.ndim == 2:
            post_img = post_img[np.newaxis, ...]

        # Normalize if requested
        if normalize:
            for i in range(pre_img.shape[0]):
                pre_img[i] = normalize_image(pre_img[i], method='percentile')
            for i in range(post_img.shape[0]):
                post_img[i] = normalize_image(post_img[i], method='percentile')

        # Coregistration not performed on arrays (requires georeference)
        return pre_img, post_img

    # Otherwise fall back to file-path based processing and writing outputs
    pre_image_path = pre_image
    post_image_path = post_image
    os.makedirs(output_dir, exist_ok=True)

    # Paths for outputs
    pre_basename = Path(pre_image_path).stem
    post_basename = Path(post_image_path).stem

    pre_output = os.path.join(output_dir, f"{pre_basename}_preprocessed.tif")
    post_output = os.path.join(output_dir, f"{post_basename}_preprocessed.tif")
    composite_output = os.path.join(output_dir, f"{pre_basename}_{post_basename}_composite.tif")

    # Read input images
    pre_img, pre_meta = read_geotiff(pre_image_path)
    post_img, post_meta = read_geotiff(post_image_path)
    
    # Ensure images are 3D arrays with bands as first dimension
    if len(pre_img.shape) == 2:
        pre_img = pre_img[np.newaxis, :, :]
    if len(post_img.shape) == 2:
        post_img = post_img[np.newaxis, :, :]
    
    # Normalize images if requested
    if normalize:
        logger.info("Normalizing images...")
        for i in range(pre_img.shape[0]):
            pre_img[i] = normalize_image(pre_img[i], method='percentile')
        
        for i in range(post_img.shape[0]):
            post_img[i] = normalize_image(post_img[i], method='percentile')
    
    # Co-register post image to pre image if requested
    if coregister:
        logger.info("Co-registering post image to pre image...")
        # Write temporary normalized images for co-registration
        temp_pre = os.path.join(output_dir, "temp_pre.tif")
        temp_post = os.path.join(output_dir, "temp_post.tif")
        
        write_geotiff(temp_pre, pre_img, pre_meta)
        write_geotiff(temp_post, post_img, post_meta)
        
        # Co-register
        coregister_images(temp_pre, temp_post, post_output, max_shift=20, method='phase')
        
        # Re-read the co-registered post image
        post_img, post_meta = read_geotiff(post_output)
        
        # Clean up temporary files
        os.remove(temp_pre)
        os.remove(temp_post)
    else:
        # Write post image if not co-registered
        write_geotiff(post_output, post_img, post_meta)
    
    # Write pre image
    write_geotiff(pre_output, pre_img, pre_meta)
    
    # Create composite if requested
    if create_composite:
        logger.info("Creating composite image...")
        # Simple stacking of bands for visualization
        # In practice, you might want more sophisticated compositing
        
        # Get minimum number of bands from both images
        min_bands = min(pre_img.shape[0], post_img.shape[0])
        
        # Create a composite with interleaved bands
        composite = np.zeros((min_bands * 2, pre_img.shape[1], pre_img.shape[2]), dtype=pre_img.dtype)
        
        for i in range(min_bands):
            composite[i*2] = pre_img[i]
            composite[i*2 + 1] = post_img[i]
        
        # Write composite
        composite_meta = pre_meta.copy()
        composite_meta.update({'count': min_bands * 2})
        write_geotiff(composite_output, composite, composite_meta)
    else:
        composite_output = None
    
    return {
        'pre': pre_output,
        'post': post_output,
        'composite': composite_output
    }