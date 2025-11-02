#!/usr/bin/env python3
"""
PS-10 Submission Verification Tool

This script verifies that all files in a submission package meet the PS-10
requirements for man-made change detection:

1. Raster files:
   - Named "Change_Mask_Lat_Long.tif"
   - Binary mask with values 0 (no change) and 1 (change)
   - Properly georeferenced

2. Vector files:
   - Named "Change_Mask_Lat_Long.shp" (etc.)
   - Complete with all required components (.shx, .dbf, .prj)
   - Properly georeferenced

3. MD5 hash:
   - Valid model_md5.txt file with proper format

4. Package structure:
   - ZIP name follows PS10_[DD-MMM-YYYY]_[Startup].zip format
"""
import os
import sys
import re
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime

try:
    import rasterio
    import numpy as np
    import geopandas as gpd
    DEPENDENCIES_INSTALLED = True
except ImportError:
    print("Warning: Some dependencies are missing. Full validation is not available.")
    print("Install required packages with: pip install rasterio numpy geopandas")
    DEPENDENCIES_INSTALLED = False


def verify_raster_compliance(raster_path):
    """Verify that a raster file meets PS-10 requirements"""
    issues = []
    
    if not DEPENDENCIES_INSTALLED:
        return ["Rasterio/numpy not installed - skipping detailed validation"]
    
    try:
        with rasterio.open(raster_path) as src:
            # Check if properly georeferenced
            if src.crs is None:
                issues.append("Missing coordinate reference system (CRS)")
            
            # Check if transform is valid
            if src.transform[0] == 0 or src.transform[4] == 0:
                issues.append("Invalid geotransform (missing pixel size)")
            
            # Check if values are 0 and 1 (man-made change detection requirement)
            data = src.read(1)
            unique_vals = np.unique(data)
            
            if not set(unique_vals).issubset({0, 1}):
                issues.append(f"Contains values other than 0 and 1: {unique_vals}")
                
            # Check if file contains any changes (not all zeros)
            if set(unique_vals) == {0}:
                issues.append("No changes detected (all zeros)")
                
    except Exception as e:
        issues.append(f"Error reading raster: {str(e)}")
    
    return issues


def verify_vector_compliance(vector_path):
    """Verify that a vector file meets PS-10 requirements"""
    issues = []
    
    if not DEPENDENCIES_INSTALLED:
        return ["Geopandas not installed - skipping detailed validation"]
    
    try:
        # Check if file exists and has content
        gdf = gpd.read_file(vector_path)
        
        # Check if geodataframe has geometry
        if len(gdf) == 0:
            issues.append("Empty shapefile (no features)")
            
        # Check if geodataframe has a valid CRS
        if not gdf.crs:
            issues.append("Missing coordinate reference system (CRS)")
            
    except Exception as e:
        issues.append(f"Error reading vector: {str(e)}")
    
    return issues


def verify_md5_format(md5_content):
    """Verify that an MD5 hash is properly formatted"""
    # MD5 should be 32 hexadecimal characters
    if not re.match(r'^[0-9a-f]{32}$', md5_content.lower()):
        return False
    return True


def verify_submission_dir(submission_dir):
    """Verify that a submission directory meets all PS-10 requirements"""
    print(f"Verifying PS-10 compliance for: {submission_dir}\n")
    
    submission_path = Path(submission_dir)
    if not submission_path.exists() or not submission_path.is_dir():
        print(f"❌ Error: {submission_dir} is not a valid directory")
        return False
    
    # Check for required files
    raster_files = list(submission_path.glob("Change_Mask_*.tif"))
    if not raster_files:
        print("❌ Error: No raster files found with required naming pattern 'Change_Mask_*.tif'")
        return False
    
    vector_files = list(submission_path.glob("Change_Mask_*.shp"))
    if not vector_files:
        print("❌ Error: No vector files found with required naming pattern 'Change_Mask_*.shp'")
        return False
    
    md5_file = submission_path / "model_md5.txt"
    if not md5_file.exists():
        print("❌ Error: Missing model_md5.txt file")
        return False
    
    # Verify MD5 hash format
    with open(md5_file, 'r') as f:
        md5_content = f.read().strip()
        if not verify_md5_format(md5_content):
            print(f"❌ Error: Invalid MD5 hash format in model_md5.txt: {md5_content}")
            print("   MD5 hash should be 32 hexadecimal characters")
            return False
    
    print("✅ Found required files with correct naming patterns")
    print(f"   • {len(raster_files)} GeoTIFF files")
    print(f"   • {len(vector_files)} Shapefile primary files")
    print(f"   • model_md5.txt with valid format")
    
    # Track issues
    has_issues = False
    processed_locations = set()
    
    # Verify each raster file
    print("\nVerifying raster files (Change_Mask_*.tif):")
    for raster_file in raster_files:
        # Extract location from filename
        match = re.search(r'Change_Mask_(.+?)\.tif$', raster_file.name)
        if match:
            location = match.group(1)
            processed_locations.add(location)
            
            # Verify raster content
            if DEPENDENCIES_INSTALLED:
                issues = verify_raster_compliance(raster_file)
                if issues:
                    has_issues = True
                    print(f"  ❌ {raster_file.name}:")
                    for issue in issues:
                        print(f"     - {issue}")
                else:
                    print(f"  ✅ {raster_file.name}: Valid")
    
    # Verify each vector file
    print("\nVerifying vector files (Change_Mask_*.shp):")
    for vector_file in vector_files:
        # Extract location from filename
        match = re.search(r'Change_Mask_(.+?)\.shp$', vector_file.name)
        if match:
            location = match.group(1)
            
            # Check if corresponding raster exists
            if location not in processed_locations:
                has_issues = True
                print(f"  ❌ {vector_file.name}: No corresponding raster file")
                continue
            
            # Check for required shapefile components
            missing_components = []
            for ext in [".shx", ".dbf", ".prj"]:
                component = submission_path / f"Change_Mask_{location}{ext}"
                if not component.exists():
                    missing_components.append(ext)
            
            if missing_components:
                has_issues = True
                print(f"  ❌ {vector_file.name}: Missing required components: {', '.join(missing_components)}")
                continue
            
            # Verify vector content
            if DEPENDENCIES_INSTALLED:
                issues = verify_vector_compliance(vector_file)
                if issues:
                    has_issues = True
                    print(f"  ❌ {vector_file.name}:")
                    for issue in issues:
                        print(f"     - {issue}")
                else:
                    print(f"  ✅ {vector_file.name}: Valid")
    
    # Check for rasters without vectors
    for location in processed_locations:
        vector_file = submission_path / f"Change_Mask_{location}.shp"
        if not vector_file.exists():
            has_issues = True
            print(f"  ❌ Missing vector file for location: {location}")
    
    # Final summary
    print("\nSummary:")
    if has_issues:
        print("⚠️ Submission has issues that should be fixed before submitting")
    else:
        print("✅ Submission appears to meet PS-10 requirements")
        print("✅ Man-made change detection outputs properly formatted")
    
    return not has_issues


def verify_zip_package(zip_path):
    """Verify that a ZIP package meets PS-10 requirements"""
    print(f"Verifying PS-10 compliance for ZIP package: {zip_path}\n")
    
    if not Path(zip_path).exists():
        print(f"❌ Error: {zip_path} does not exist")
        return False
    
    # Verify filename format: PS10_[DD-MMM-YYYY]_[Startup].zip
    filename = os.path.basename(zip_path)
    if not re.match(r'^PS10_\d{2}-[A-Za-z]{3}-\d{4}_[A-Za-z0-9]+\.zip$', filename):
        print(f"❌ Error: ZIP filename does not follow PS-10 format: {filename}")
        print("   Required format: PS10_[DD-MMM-YYYY]_[Startup].zip")
        print("   Example: PS10_31-Oct-2025_XBoson.zip")
        return False
    
    # Extract and verify contents
    try:
        # Create temporary directory for extraction
        temp_dir = Path("temp_verification")
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        # Extract ZIP contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Verify extracted contents
        result = verify_submission_dir(temp_dir)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        return result
    
    except Exception as e:
        print(f"❌ Error verifying ZIP package: {str(e)}")
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python verify_ps10_submission.py <submission_directory_or_zip>")
        print("\nExamples:")
        print("  python verify_ps10_submission.py PS10_09-Oct-2025_XBoson.zip")
        print("  python verify_ps10_submission.py PS10_submission_results")
        return 1
    
    path = sys.argv[1]
    
    # Check if using placeholder date in filename
    if "DD-MMM-YYYY" in path:
        # Replace placeholder with actual date
        today = datetime.now()
        date_str = today.strftime("%d-%b-%Y")  # Format: DD-MMM-YYYY (e.g., 09-Oct-2025)
        actual_path = path.replace("DD-MMM-YYYY", date_str)
        print(f"Note: Replacing placeholder date with today's date: {date_str}")
        print(f"Looking for: {actual_path}")
        path = actual_path
    
    if os.path.isdir(path):
        verify_submission_dir(path)
    elif os.path.isfile(path) and path.endswith('.zip'):
        verify_zip_package(path)
    else:
        print(f"❌ Error: {path} is not a valid directory or ZIP file")
        print("\nTo create a submission package first, run:")
        print("  python create_ps10_submission.py predictions_threshold_0.1 <model_file> XBoson")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())