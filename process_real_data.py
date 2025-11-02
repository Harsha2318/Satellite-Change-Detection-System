"""
PS-10 Change Detection - Real Data Processing
Date: October 31, 2025

This script processes the actual Sentinel-2 and LISS4 data for change detection.
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import rasterio
try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.merge import merge
except ImportError:
    print("Installing required packages...")
    os.system("pip install rasterio")
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.merge import merge

# TensorFlow not needed for data preparation
# Will be used later for inference

print("="*80)
print("PS-10 REAL DATA PROCESSING")
print("="*80)

# Paths
BASE_DIR = Path(r"C:\Users\harsh\PS-10")
INPUT_DIR = BASE_DIR / "PS10_Input_Data"
SENTINEL_DIR = INPUT_DIR / "Sentinel2_L1C"
LISS_DIR = INPUT_DIR / "LISS4_L2"
PROCESSED_DIR = BASE_DIR / "changedetect" / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "ps10_predictions"
FORMATTED_DIR = BASE_DIR / "ps10_predictions_formatted"

# Create output directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FORMATTED_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Working directories:")
print(f"   Input: {INPUT_DIR}")
print(f"   Processed: {PROCESSED_DIR}")
print(f"   Output: {OUTPUT_DIR}")

def extract_sentinel2_bands(safe_folder):
    """Extract and merge Sentinel-2 bands from SAFE format."""
    print(f"\n🛰️ Processing Sentinel-2: {safe_folder.name}")
    
    granule_dir = safe_folder / "GRANULE"
    granule_folders = list(granule_dir.glob("L1C_*"))
    
    if not granule_folders:
        print("   ⚠️ No granule folder found")
        return None
    
    img_data_dir = granule_folders[0] / "IMG_DATA"
    
    # Get the date from folder name
    date_str = safe_folder.name.split("_")[2].split("T")[0]
    
    # Find RGB bands (B02=Blue, B03=Green, B04=Red, B08=NIR)
    bands = {
        'B02': None,  # Blue
        'B03': None,  # Green
        'B04': None,  # Red
        'B08': None   # NIR
    }
    
    for band_name in bands.keys():
        band_files = list(img_data_dir.glob(f"*_{band_name}.jp2"))
        if band_files:
            bands[band_name] = band_files[0]
            print(f"   ✓ Found {band_name}: {band_files[0].name}")
    
    # Create RGB composite
    if all(bands.values()):
        output_file = PROCESSED_DIR / f"S2_{date_str}_composite.tif"
        
        print(f"   📸 Creating RGB-NIR composite...")
        
        # Read and stack bands
        with rasterio.open(bands['B04']) as src_r, \
             rasterio.open(bands['B03']) as src_g, \
             rasterio.open(bands['B02']) as src_b, \
             rasterio.open(bands['B08']) as src_nir:
            
            # Read data
            red = src_r.read(1)
            green = src_g.read(1)
            blue = src_b.read(1)
            nir = src_nir.read(1)
            
            # Get metadata from red band
            profile = src_r.profile.copy()
            
            # Sentinel-2 JP2 files may not have CRS, set manually
            # UTM Zone 43N (based on T43RCM tile)
            if not profile.get('crs'):
                from rasterio.crs import CRS
                # Use WKT to avoid PROJ database issues
                wkt_43n = 'PROJCS["WGS 84 / UTM zone 43N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",75],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]'
                profile['crs'] = CRS.from_wkt(wkt_43n)
            
            profile.update({
                'count': 4,
                'dtype': 'uint16',
                'driver': 'GTiff',
                'compress': 'lzw'
            })
            
            # Write composite
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(red, 1)
                dst.write(green, 2)
                dst.write(blue, 3)
                dst.write(nir, 4)
        
        print(f"   ✅ Created: {output_file.name}")
        return output_file
    
    return None

def process_liss4_bands(liss_folder):
    """Process LISS4 multi-band data."""
    print(f"\n🛰️ Processing LISS4: {liss_folder.name}")
    
    # Get date from folder name
    date_str = liss_folder.name.split("MAR" if "MAR" in liss_folder.name else "JAN")[1][:8]
    
    # Find band files
    band_files = sorted(liss_folder.glob("BAND*.tif"))
    
    if len(band_files) < 3:
        print("   ⚠️ Not enough bands found")
        return None
    
    print(f"   ✓ Found {len(band_files)} bands")
    for bf in band_files:
        print(f"      - {bf.name}")
    
    # Create composite from available bands
    output_file = PROCESSED_DIR / f"LISS_{date_str}_composite.tif"
    
    print(f"   📸 Creating multi-band composite...")
    
    # Read first band to get metadata
    with rasterio.open(band_files[0]) as src:
        profile = src.profile.copy()
        profile.update({
            'count': len(band_files),
            'dtype': 'uint16',
            'driver': 'GTiff',
            'compress': 'lzw'
        })
    
    # Stack all bands
    with rasterio.open(output_file, 'w', **profile) as dst:
        for idx, band_file in enumerate(band_files, start=1):
            with rasterio.open(band_file) as src:
                data = src.read(1)
                dst.write(data, idx)
    
    print(f"   ✅ Created: {output_file.name}")
    return output_file

def normalize_and_align_images(img1_path, img2_path, output_prefix):
    """Normalize two images to same resolution and extent."""
    print(f"\n🔄 Aligning image pair: {output_prefix}")
    
    with rasterio.open(img1_path) as src1, rasterio.open(img2_path) as src2:
        # Get common bounds
        bounds1 = src1.bounds
        bounds2 = src2.bounds
        
        print(f"   Image 1: {src1.width}x{src1.height} @ {src1.res}")
        print(f"   Image 2: {src2.width}x{src2.height} @ {src2.res}")
        
        # Use higher resolution
        target_res = min(src1.res[0], src2.res[0])
        print(f"   Target resolution: {target_res}m")
        
        # Determine overlapping bounds
        min_x = max(bounds1.left, bounds2.left)
        max_x = min(bounds1.right, bounds2.right)
        min_y = max(bounds1.bottom, bounds2.bottom)
        max_y = min(bounds1.top, bounds2.top)
        
        if min_x >= max_x or min_y >= max_y:
            print("   ⚠️ No overlap between images!")
            # Use image 1 bounds as fallback
            min_x, min_y, max_x, max_y = bounds1.left, bounds1.bottom, bounds1.right, bounds1.top
        
        print(f"   Overlap: ({min_x:.2f}, {min_y:.2f}) to ({max_x:.2f}, {max_y:.2f})")
        
        # Calculate output dimensions
        width = int((max_x - min_x) / target_res)
        height = int((max_y - min_y) / target_res)
        
        # Ensure dimensions are reasonable
        if width > 10000 or height > 10000:
            print(f"   ⚠️ Output too large ({width}x{height}), resampling...")
            scale = 10000 / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
            target_res = target_res / scale
        
        print(f"   Output size: {width}x{height}")
        
        # Create output profile
        profile = src1.profile.copy()
        profile.update({
            'width': width,
            'height': height,
            'transform': rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, width, height),
            'crs': src1.crs,  # Ensure CRS is set
            'driver': 'GTiff',
            'compress': 'lzw'
        })
        
        # Process each image
        output_files = []
        for idx, (src_path, src) in enumerate([(img1_path, src1), (img2_path, src2)], start=1):
            output_file = PROCESSED_DIR / f"{output_prefix}_t{idx}.tif"
            
            # Update profile for this specific source
            out_profile = profile.copy()
            out_profile['crs'] = src.crs
            out_profile['count'] = src.count
            out_profile['dtype'] = src.dtypes[0]
            
            # Read and resample
            data = np.zeros((src.count, height, width), dtype=src.dtypes[0])
            
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=data[band_idx-1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=out_profile['transform'],
                    dst_crs=out_profile['crs'],
                    resampling=Resampling.bilinear
                )
            
            # Write output
            with rasterio.open(output_file, 'w', **out_profile) as dst:
                dst.write(data)
            
            print(f"   ✅ Created t{idx}: {output_file.name}")
            output_files.append(output_file)
    
    return output_files

def prepare_for_model(img_t1, img_t2, output_prefix):
    """Prepare image pair for model inference."""
    print(f"\n🎨 Preparing images for model: {output_prefix}")
    
    with rasterio.open(img_t1) as src1, rasterio.open(img_t2) as src2:
        # Read first 3 bands (RGB)
        img1_data = np.stack([src1.read(i) for i in range(1, min(4, src1.count + 1))])
        img2_data = np.stack([src2.read(i) for i in range(1, min(4, src2.count + 1))])
        
        # Normalize to 0-255
        for img in [img1_data, img2_data]:
            for band_idx in range(img.shape[0]):
                band = img[band_idx]
                p2, p98 = np.percentile(band[band > 0], [2, 98])
                band = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                img[band_idx] = band.astype(np.uint8)
        
        # Ensure 3 bands
        if img1_data.shape[0] < 3:
            img1_data = np.repeat(img1_data, 3, axis=0)[:3]
        if img2_data.shape[0] < 3:
            img2_data = np.repeat(img2_data, 3, axis=0)[:3]
        
        # Save as RGB
        profile = src1.profile.copy()
        profile.update({
            'count': 3,
            'dtype': 'uint8',
            'driver': 'GTiff',
            'compress': 'lzw'
        })
        
        output_t1 = OUTPUT_DIR / f"{output_prefix}_t1.tif"
        output_t2 = OUTPUT_DIR / f"{output_prefix}_t2.tif"
        
        with rasterio.open(output_t1, 'w', **profile) as dst:
            dst.write(img1_data[:3])
        
        with rasterio.open(output_t2, 'w', **profile) as dst:
            dst.write(img2_data[:3])
        
        print(f"   ✅ Created: {output_t1.name}")
        print(f"   ✅ Created: {output_t2.name}")
        
        return output_t1, output_t2

def main():
    print("\n" + "="*80)
    print("STEP 1: EXTRACT AND PROCESS RAW DATA")
    print("="*80)
    
    # Process Sentinel-2
    sentinel_folders = sorted(SENTINEL_DIR.glob("S2*.SAFE"))
    sentinel_images = []
    
    for sf in sentinel_folders:
        img = extract_sentinel2_bands(sf)
        if img:
            sentinel_images.append(img)
    
    # Process LISS4
    liss_folders = sorted(LISS_DIR.glob("R2F*"))
    liss_images = []
    
    for lf in liss_folders:
        img = process_liss4_bands(lf)
        if img:
            liss_images.append(img)
    
    print("\n" + "="*80)
    print("STEP 2: ALIGN IMAGE PAIRS")
    print("="*80)
    
    aligned_pairs = []
    
    # Align Sentinel-2 pair
    if len(sentinel_images) >= 2:
        s2_pair = normalize_and_align_images(
            sentinel_images[0], 
            sentinel_images[1],
            "Sentinel2_pair"
        )
        aligned_pairs.append(("Sentinel2", s2_pair))
    
    # Align LISS4 pair
    if len(liss_images) >= 2:
        liss_pair = normalize_and_align_images(
            liss_images[0],
            liss_images[1],
            "LISS4_pair"
        )
        aligned_pairs.append(("LISS4", liss_pair))
    
    print("\n" + "="*80)
    print("STEP 3: PREPARE FOR MODEL INFERENCE")
    print("="*80)
    
    model_ready_pairs = []
    
    for name, (t1, t2) in aligned_pairs:
        pair = prepare_for_model(t1, t2, name)
        model_ready_pairs.append((name, pair))
    
    print("\n" + "="*80)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   - Processed Sentinel-2 images: {len(sentinel_images)}")
    print(f"   - Processed LISS4 images: {len(liss_images)}")
    print(f"   - Aligned pairs ready: {len(aligned_pairs)}")
    print(f"   - Model-ready pairs: {len(model_ready_pairs)}")
    print(f"\n📁 Output location: {OUTPUT_DIR}")
    print(f"\n🚀 Next step: Run inference with the model")
    print(f"   Command: python oct31_rapid_inference.py")
    
    return model_ready_pairs

if __name__ == "__main__":
    try:
        pairs = main()
        print("\n✅ SUCCESS! Ready for inference.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
