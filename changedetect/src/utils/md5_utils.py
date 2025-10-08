"""
Utilities for generating and verifying MD5 hashes of model files.
"""
import os
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_md5(filepath):
    """
    Compute MD5 hash of a file.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        str: MD5 hash as a hexadecimal string
    """
    md5_hash = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error computing MD5 hash for {filepath}: {e}")
        raise

def save_md5_hash(model_path, output_path=None):
    """
    Compute and save MD5 hash of a model file.
    
    Args:
        model_path (str): Path to the model file
        output_path (str, optional): Path to save the hash. If None, uses model_path + '.md5'
        
    Returns:
        str: Path to the saved MD5 hash file
    """
    if output_path is None:
        output_path = f"{model_path}.md5"
    
    md5_hash = compute_md5(model_path)
    
    try:
        with open(output_path, "w") as f:
            f.write(md5_hash)
        logger.info(f"Saved MD5 hash for {model_path} to {output_path}: {md5_hash}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving MD5 hash to {output_path}: {e}")
        raise

def verify_md5(filepath, expected_hash=None, hash_file=None):
    """
    Verify the MD5 hash of a file against an expected hash or a hash file.
    
    Args:
        filepath (str): Path to the file to verify
        expected_hash (str, optional): Expected MD5 hash to compare against
        hash_file (str, optional): Path to a file containing the expected MD5 hash
        
    Returns:
        bool: True if verification succeeds, False otherwise
    """
    if expected_hash is None and hash_file is None:
        raise ValueError("Either expected_hash or hash_file must be provided")
    
    # If hash_file is provided, read the expected hash from it
    if hash_file is not None:
        try:
            with open(hash_file, "r") as f:
                expected_hash = f.read().strip()
        except Exception as e:
            logger.error(f"Error reading hash file {hash_file}: {e}")
            return False
    
    # Compute the actual hash
    actual_hash = compute_md5(filepath)
    
    # Compare hashes
    if actual_hash == expected_hash:
        logger.info(f"MD5 hash verification succeeded for {filepath}")
        return True
    else:
        logger.warning(f"MD5 hash verification failed for {filepath}")
        logger.warning(f"Expected: {expected_hash}")
        logger.warning(f"Actual: {actual_hash}")
        return False