"""
Real Data Inference for PS-10
October 31, 2025

Runs change detection on the prepared Sentinel-2 and LISS4 image pairs.
"""

import os
import sys
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PS-10 CHANGE DETECTION INFERENCE")
print("="*80)

# Paths
BASE_DIR = Path(r"C:\Users\harsh\PS-10")
INPUT_DIR = BASE_DIR / "ps10_predictions"
OUTPUT_DIR = BASE_DIR / "ps10_predictions_formatted"
MODEL_PATH = BASE_DIR / "model" / "model.h5"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Configuration:")
print(f"   Input: {INPUT_DIR}")
print(f"   Output: {OUTPUT_DIR}")
print(f"   Model: {MODEL_PATH}")

# Check model exists
if not MODEL_PATH.exists():
    print(f"\n❌ Model not found: {MODEL_PATH}")
    sys.exit(1)

try:
    import rasterio
    from rasterio.features import shapes
    import fiona
    from fiona.crs import from_epsg
    from shapely.geometry import shape, mapping
    print("✓ Geospatial libraries loaded")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    sys.exit(1)

# Try to load model (check if it's Keras or another format)
print("\n🔄 Loading model...")

try:
    # Try Keras/TensorFlow first
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        from tensorflow import keras
        
        model = keras.models.load_model(str(MODEL_PATH), compile=False)
        print(f"✓ Loaded Keras model: {model_path.name}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        MODEL_TYPE = "keras"
    except Exception as ke:
        print(f"⚠️ Not a Keras model: {ke}")
        # Try PyTorch
        try:
            import torch
            model = torch.load(str(MODEL_PATH))
            print(f"✓ Loaded PyTorch model")
            MODEL_TYPE = "pytorch"
        except:
            print("❌ Could not load model as Keras or PyTorch")
            print("   Will use simple threshold-based change detection")
            model = None
            MODEL_TYPE = "simple"

except Exception as e:
    print(f"⚠️ Model load error: {e}")
    print("   Using simple change detection")
    model = None
    MODEL_TYPE = "simple"

def simple_change_detection(img1, img2):
    """Simple threshold-based change detection"""
    print("   Using simple change detection (image differencing)")
    
    # Ensure same shape
    if img1.shape != img2.shape:
        print(f"   ⚠️ Shape mismatch: {img1.shape} vs {img2.shape}")
        return None
    
    # Calculate absolute difference
    diff = np.abs(img1.astype(float) - img2.astype(float))
    
    # Average across bands
    if len(diff.shape) == 3:
        diff = np.mean(diff, axis=2)
    
    # Normalize
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-10)
    
    # Threshold
    threshold = 0.15
    change_mask = (diff > threshold).astype(np.uint8)
    
    print(f"   Detected changes: {change_mask.sum()} pixels ({change_mask.sum()/change_mask.size*100:.2f}%)")
    
    return change_mask

def keras_change_detection(model, img1, img2, img_size=256):
    """Run Keras model inference"""
    print("   Using Keras model inference")
    
    # Get original dimensions
    orig_h, orig_w = img1.shape[:2]
    
    # Resize to model input size if needed
    if img1.shape[:2] != (img_size, img_size):
        from skimage.transform import resize
        img1_resized = resize(img1, (img_size, img_size), preserve_range=True).astype(np.uint8)
        img2_resized = resize(img2, (img_size, img_size), preserve_range=True).astype(np.uint8)
    else:
        img1_resized = img1
        img2_resized = img2
    
    # Normalize
    img1_norm = img1_resized.astype(np.float32) / 255.0
    img2_norm = img2_resized.astype(np.float32) / 255.0
    
    # Ensure 3 channels
    if len(img1_norm.shape) == 2:
        img1_norm = np.stack([img1_norm]*3, axis=-1)
        img2_norm = np.stack([img2_norm]*3, axis=-1)
    
    # Stack as input (batch of 2 images or concatenate)
    if 'siamese' in str(model.input_shape).lower() or len(model.input_shape) > 1:
        # Siamese network - two inputs
        input_data = [np.expand_dims(img1_norm, axis=0), np.expand_dims(img2_norm, axis=0)]
    else:
        # Single input - concatenate
        input_data = np.expand_dims(np.concatenate([img1_norm, img2_norm], axis=-1), axis=0)
    
    # Predict
    prediction = model.predict(input_data, verbose=0)
    
    # Get change mask
    if len(prediction.shape) == 4:
        change_mask = prediction[0, :, :, 0]
    else:
        change_mask = prediction[0]
    
    # Resize back to original
    if change_mask.shape != (orig_h, orig_w):
        from skimage.transform import resize
        change_mask = resize(change_mask, (orig_h, orig_w), preserve_range=True)
    
    # Threshold
    change_mask = (change_mask > 0.5).astype(np.uint8)
    
    print(f"   Detected changes: {change_mask.sum()} pixels ({change_mask.sum()/(orig_h*orig_w)*100:.2f}%)")
    
    return change_mask

def process_image_pair(t1_path, t2_path, output_prefix):
    """Process a single image pair"""
    print(f"\n🔄 Processing: {output_prefix}")
    print(f"   T1: {t1_path.name}")
    print(f"   T2: {t2_path.name}")
    
    # Read images
    with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
        img1 = src1.read()  # Shape: (bands, height, width)
        img2 = src2.read()
        
        # Transpose to (height, width, bands)
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        
        print(f"   Image size: {img1.shape}")
        
        # Run change detection
        if MODEL_TYPE == "keras" and model is not None:
            change_mask = keras_change_detection(model, img1, img2)
        else:
            change_mask = simple_change_detection(img1, img2)
        
        if change_mask is None:
            print("   ❌ Change detection failed")
            return False
        
        # Save GeoTIFF
        tif_output = OUTPUT_DIR / f"{output_prefix}_change_mask.tif"
        profile = src1.profile.copy()
        profile.update({
            'count': 1,
            'dtype': 'uint8',
            'compress': 'lzw'
        })
        
        with rasterio.open(tif_output, 'w', **profile) as dst:
            dst.write(change_mask, 1)
        
        print(f"   ✅ Saved: {tif_output.name}")
        
        # Create shapefile
        shp_output = OUTPUT_DIR / f"{output_prefix}_change_vectors.shp"
        
        # Vectorize
        shapes_gen = shapes(change_mask.astype(np.int16), transform=src1.transform)
        geometries = []
        
        for geom, value in shapes_gen:
            if value == 1:  # Only changed areas
                geometries.append({
                    'geometry': geom,
                    'properties': {'change': 1}
                })
        
        if geometries:
            # Save shapefile
            schema = {
                'geometry': 'Polygon',
                'properties': {'change': 'int'}
            }
            
            with fiona.open(str(shp_output), 'w', 
                          driver='ESRI Shapefile',
                          crs=src1.crs,
                          schema=schema) as dst:
                dst.writerecords(geometries)
            
            print(f"   ✅ Saved: {shp_output.name} ({len(geometries)} polygons)")
        else:
            print(f"   ⚠️ No changes detected, skipping shapefile")
        
        return True

def main():
    print("\n" + "="*80)
    print("RUNNING INFERENCE ON IMAGE PAIRS")
    print("="*80)
    
    # Find image pairs
    pairs = []
    
    # Sentinel-2
    s2_t1 = INPUT_DIR / "Sentinel2_t1.tif"
    s2_t2 = INPUT_DIR / "Sentinel2_t2.tif"
    if s2_t1.exists() and s2_t2.exists():
        pairs.append((s2_t1, s2_t2, "Sentinel2_20200328_20250307"))
    
    # LISS4
    liss_t1 = INPUT_DIR / "LISS4_t1.tif"
    liss_t2 = INPUT_DIR / "LISS4_t2.tif"
    if liss_t1.exists() and liss_t2.exists():
        pairs.append((liss_t1, liss_t2, "LISS4_20200318_20250127"))
    
    print(f"\n📊 Found {len(pairs)} image pair(s) to process")
    
    if not pairs:
        print("❌ No image pairs found!")
        return False
    
    # Process each pair
    success_count = 0
    for t1, t2, prefix in pairs:
        if process_image_pair(t1, t2, prefix):
            success_count += 1
    
    print("\n" + "="*80)
    print(f"✅ INFERENCE COMPLETE: {success_count}/{len(pairs)} pairs processed")
    print("="*80)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
    print(f"\n🚀 Next step: Create submission package")
    print(f"   Command: python prepare_ps10_submission.py")
    
    return success_count == len(pairs)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
