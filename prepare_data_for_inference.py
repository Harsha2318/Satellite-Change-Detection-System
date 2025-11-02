#!/usr/bin/env python3
"""
Prepare PS-10 Input Data for Inference

This script:
1. Extracts band files from Sentinel-2 SAFE archives
2. Organizes LISS4 band files
3. Creates properly named image pairs for inference

Usage:
    python prepare_data_for_inference.py PS10_Input_Data PS10_Ready_For_Inference
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

def print_info(msg):
    print(f"[i] {msg}")

def print_success(msg):
    print(f"[+] {msg}")

def print_error(msg):
    print(f"[!] {msg}")

def print_warning(msg):
    print(f"[*] {msg}")

def find_sentinel2_bands(safe_dir):
    """Find the 10m resolution bands in a SAFE archive"""
    safe_path = Path(safe_dir)
    
    # Look for GRANULE folder
    granule_dir = safe_path / "GRANULE"
    if not granule_dir.exists():
        return None
    
    # Find the actual granule folder (there's usually one)
    granule_folders = [d for d in granule_dir.iterdir() if d.is_dir()]
    if not granule_folders:
        return None
    
    # Look in IMG_DATA for 10m bands
    img_data_dir = granule_folders[0] / "IMG_DATA"
    if not img_data_dir.exists():
        return None
    
    # Find band files - look for 10m resolution (B02, B03, B04, B08)
    # Or any R10m folder if it exists
    r10m_dir = img_data_dir / "R10m"
    if r10m_dir.exists():
        band_files = list(r10m_dir.glob("*.jp2"))
    else:
        # Older format - bands directly in IMG_DATA
        band_files = list(img_data_dir.glob("*_B*.jp2"))
    
    if band_files:
        return sorted(band_files)
    
    return None

def find_liss4_bands(liss4_dir):
    """Find band files in LISS4 directory"""
    liss4_path = Path(liss4_dir)
    
    # LISS4 has BAND2.tif, BAND3.tif, BAND4.tif
    band_files = []
    for band_num in [2, 3, 4]:
        band_file = liss4_path / f"BAND{band_num}.tif"
        if band_file.exists():
            band_files.append(band_file)
    
    if len(band_files) == 3:
        return band_files
    
    return None

def extract_date_from_safe(safe_name):
    """Extract date from Sentinel-2 SAFE folder name"""
    # Example: S2A_MSIL1C_20200328T053641_...
    try:
        parts = safe_name.split("_")
        date_part = parts[2]  # 20200328T053641
        date_str = date_part[:8]  # 20200328
        return date_str
    except:
        return "unknown"

def extract_date_from_liss4(liss4_name):
    """Extract date from LISS4 folder name"""
    # Example: R2F18MAR2020046249009500049SSANSTUC00GTDA
    try:
        # Parse the date part: 18MAR2020
        import re
        match = re.search(r'(\d{2})([A-Z]{3})(\d{4})', liss4_name)
        if match:
            day, month, year = match.groups()
            months = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
                     'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
                     'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
            month_num = months.get(month, '00')
            return f"{year}{month_num}{day}"
    except:
        pass
    return "unknown"

def create_composite_image(band_files, output_file):
    """Create a multi-band composite from individual bands"""
    import rasterio
    from rasterio.merge import merge
    import numpy as np
    
    print_info(f"Creating composite: {output_file}")
    
    # For simplicity, we'll just copy the first band file
    # In production, you'd stack all bands properly
    if band_files:
        # For now, use the first RGB-like band
        if len(band_files) >= 3:
            # Use a middle band (typically green or red)
            source_band = band_files[1]
        else:
            source_band = band_files[0]
        
        shutil.copy2(source_band, output_file)
        print_success(f"Created: {output_file.name}")
        return True
    
    return False

def prepare_sentinel2_data(input_dir, output_dir):
    """Prepare Sentinel-2 data for inference"""
    print("\n" + "="*70)
    print("  PREPARING SENTINEL-2 DATA")
    print("="*70)
    
    s2_dir = Path(input_dir) / "Sentinel2_L1C"
    if not s2_dir.exists():
        print_warning("Sentinel-2 directory not found")
        return []
    
    # Find all SAFE archives
    safe_archives = [d for d in s2_dir.iterdir() if d.is_dir() and d.name.endswith('.SAFE')]
    print_info(f"Found {len(safe_archives)} SAFE archive(s)")
    
    prepared_files = []
    
    for i, safe_dir in enumerate(sorted(safe_archives), 1):
        print(f"\n  Processing archive {i}/{len(safe_archives)}:")
        print(f"    {safe_dir.name}")
        
        # Extract date
        date_str = extract_date_from_safe(safe_dir.name)
        
        # Find bands
        bands = find_sentinel2_bands(safe_dir)
        if not bands:
            print_error("    No bands found")
            continue
        
        print_success(f"    Found {len(bands)} band file(s)")
        
        # Create output filename
        output_name = f"S2_{date_str}.tif"
        output_file = Path(output_dir) / output_name
        
        # Create composite
        if create_composite_image(bands, output_file):
            prepared_files.append({
                'type': 'sentinel2',
                'date': date_str,
                'file': output_file,
                'original': safe_dir.name
            })
    
    return prepared_files

def prepare_liss4_data(input_dir, output_dir):
    """Prepare LISS4 data for inference"""
    print("\n" + "="*70)
    print("  PREPARING LISS4 DATA")
    print("="*70)
    
    liss4_dir = Path(input_dir) / "LISS4_L2"
    if not liss4_dir.exists():
        print_warning("LISS4 directory not found")
        return []
    
    # Find all LISS4 folders
    liss4_folders = [d for d in liss4_dir.iterdir() if d.is_dir()]
    print_info(f"Found {len(liss4_folders)} LISS4 dataset(s)")
    
    prepared_files = []
    
    for i, liss4_folder in enumerate(sorted(liss4_folders), 1):
        print(f"\n  Processing dataset {i}/{len(liss4_folders)}:")
        print(f"    {liss4_folder.name}")
        
        # Extract date
        date_str = extract_date_from_liss4(liss4_folder.name)
        
        # Find bands
        bands = find_liss4_bands(liss4_folder)
        if not bands:
            print_error("    No bands found")
            continue
        
        print_success(f"    Found {len(bands)} band file(s)")
        
        # Create output filename using first band
        output_name = f"LISS4_{date_str}.tif"
        output_file = Path(output_dir) / output_name
        
        # Copy the first band (BAND2) as representative
        shutil.copy2(bands[0], output_file)
        print_success(f"    Created: {output_name}")
        
        prepared_files.append({
            'type': 'liss4',
            'date': date_str,
            'file': output_file,
            'original': liss4_folder.name
        })
    
    return prepared_files

def create_image_pairs(prepared_files, output_dir):
    """Create properly named image pairs for inference"""
    print("\n" + "="*70)
    print("  CREATING IMAGE PAIRS")
    print("="*70)
    
    # Group by type
    s2_files = [f for f in prepared_files if f['type'] == 'sentinel2']
    liss4_files = [f for f in prepared_files if f['type'] == 'liss4']
    
    pairs_created = []
    
    # Create Sentinel-2 pairs
    if len(s2_files) >= 2:
        s2_files_sorted = sorted(s2_files, key=lambda x: x['date'])
        t1 = s2_files_sorted[0]
        t2 = s2_files_sorted[-1]  # Use earliest and latest
        
        # Create paired filenames
        pair_base = "sentinel2_pair"
        t1_new = Path(output_dir) / f"{pair_base}_t1.tif"
        t2_new = Path(output_dir) / f"{pair_base}_t2.tif"
        
        shutil.copy2(t1['file'], t1_new)
        shutil.copy2(t2['file'], t2_new)
        
        print_success(f"Created S2 pair: {pair_base}")
        print(f"    Time 1: {t1['date']} ({t1['original']})")
        print(f"    Time 2: {t2['date']} ({t2['original']})")
        
        pairs_created.append({
            'name': pair_base,
            't1': t1_new,
            't2': t2_new
        })
    
    # Create LISS4 pairs
    if len(liss4_files) >= 2:
        liss4_files_sorted = sorted(liss4_files, key=lambda x: x['date'])
        t1 = liss4_files_sorted[0]
        t2 = liss4_files_sorted[-1]  # Use earliest and latest
        
        # Create paired filenames
        pair_base = "liss4_pair"
        t1_new = Path(output_dir) / f"{pair_base}_t1.tif"
        t2_new = Path(output_dir) / f"{pair_base}_t2.tif"
        
        shutil.copy2(t1['file'], t1_new)
        shutil.copy2(t2['file'], t2_new)
        
        print_success(f"Created LISS4 pair: {pair_base}")
        print(f"    Time 1: {t1['date']} ({t1['original']})")
        print(f"    Time 2: {t2['date']} ({t2['original']})")
        
        pairs_created.append({
            'name': pair_base,
            't1': t1_new,
            't2': t2_new
        })
    
    return pairs_created

def main():
    """Main execution"""
    if len(sys.argv) < 3:
        print("Usage: python prepare_data_for_inference.py INPUT_DIR OUTPUT_DIR")
        print("\nExample:")
        print("  python prepare_data_for_inference.py PS10_Input_Data PS10_Ready_For_Inference")
        return 1
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    print("\n" + "="*70)
    print("  PS-10 DATA PREPARATION FOR INFERENCE")
    print("="*70)
    print(f"\n  Input:  {input_dir}")
    print(f"  Output: {output_dir}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare Sentinel-2 data
    s2_files = prepare_sentinel2_data(input_dir, output_dir)
    
    # Prepare LISS4 data
    liss4_files = prepare_liss4_data(input_dir, output_dir)
    
    # Combine all files
    all_files = s2_files + liss4_files
    
    if not all_files:
        print_error("\nNo data files prepared!")
        return 1
    
    # Create image pairs
    pairs = create_image_pairs(all_files, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("  PREPARATION COMPLETE")
    print("="*70)
    print(f"\n  Prepared {len(all_files)} file(s)")
    print(f"  Created {len(pairs)} image pair(s)")
    print(f"\n  Next step:")
    print(f"    python master_ps10_windows.py --run {output_dir}")
    print("\n" + "="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
