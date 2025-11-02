#!/usr/bin/env python
"""
PS-10 Compliance Validator

This script validates if the output files and directory structure meet 
the requirements specified in PS-10 guidelines:

1. Output file naming: 'Change_Mask_Lat_Long.tif' (Raster) and 'Change_Mask_Lat_Long.shp' (Vector)
2. Verify that each raster has corresponding vector files
3. Verify that model MD5 hash is properly generated
4. Check GeoTIFF georeferencing and value ranges (0-1 for changes)
"""

import os
import glob
import re
import sys
import hashlib
from pathlib import Path
import rasterio
import geopandas as gpd
from osgeo import gdal

def validate_raster_content(raster_path):
    """Validate the content and format of a GeoTIFF raster file"""
    issues = []
    
    try:
        # Open raster with rasterio
        with rasterio.open(raster_path) as src:
            # Check if georeferenced
            if not src.crs:
                issues.append(f"{os.path.basename(raster_path)}: Missing coordinate reference system (CRS)")
            
            # Check if values are 0 and 1
            data = src.read(1)
            unique_vals = set(data.flatten())
            if not unique_vals.issubset({0, 1}):
                issues.append(f"{os.path.basename(raster_path)}: Contains values other than 0 and 1 ({unique_vals})")
            
            # Check if file is empty (all zeros)
            if set(unique_vals) == {0}:
                issues.append(f"{os.path.basename(raster_path)}: No changes detected (all zeros)")
            
            # Check if transform is valid
            if not src.transform or src.transform == rasterio.transform.Affine(1, 0, 0, 0, 1, 0):
                issues.append(f"{os.path.basename(raster_path)}: Invalid or missing geotransform")
            
    except Exception as e:
        issues.append(f"{os.path.basename(raster_path)}: Failed to open or read file - {str(e)}")
    
    return issues

def validate_vector_file(vector_path):
    """Validate the content and format of a shapefile"""
    issues = []
    
    try:
        # Check if file exists and has content
        gdf = gpd.read_file(vector_path)
        
        # Check if geodataframe has geometry
        if len(gdf) == 0:
            issues.append(f"{os.path.basename(vector_path)}: Empty shapefile (no features)")
            
        # Check if geodataframe has a valid CRS
        if not gdf.crs:
            issues.append(f"{os.path.basename(vector_path)}: Missing coordinate reference system (CRS)")
            
    except Exception as e:
        issues.append(f"{os.path.basename(vector_path)}: Failed to open or read file - {str(e)}")
    
    return issues

def validate_output_directory(output_dir):
    """Validate PS-10 compliance for all files in the output directory"""
    print(f"Validating PS-10 compliance for files in: {output_dir}\n")
    
    output_dir = Path(output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        print(f"❌ Error: Directory '{output_dir}' does not exist or is not a directory")
        return False
    
    # Find all raster files with correct naming
    raster_pattern = "Change_Mask_*.tif"
    raster_files = list(output_dir.glob(raster_pattern))
    
    if not raster_files:
        print("❌ Error: No raster files found with the required naming pattern 'Change_Mask_*.tif'")
        return False
    
    issues_found = []
    processed_locations = []
    
    print(f"Found {len(raster_files)} raster files with correct naming convention")
    
    # Validate each raster and its corresponding vector files
    for raster_file in raster_files:
        # Extract the location part (e.g., "28.5_77.2")
        match = re.search(r'Change_Mask_(.+?)\.tif$', raster_file.name)
        if match:
            location = match.group(1)
            processed_locations.append(location)
            
            # Validate raster content
            raster_issues = validate_raster_content(raster_file)
            if raster_issues:
                issues_found.extend(raster_issues)
            
            # Check for all required shapefile components
            shapefile = output_dir / f"Change_Mask_{location}.shp"
            if not shapefile.exists():
                issues_found.append(f"Missing shapefile for location {location}")
                continue
                
            # Validate vector content
            vector_issues = validate_vector_file(shapefile)
            if vector_issues:
                issues_found.extend(vector_issues)
                
            # Check for all shapefile components
            missing_components = []
            for ext in [".shx", ".dbf", ".prj"]:  # CPG is optional
                component = output_dir / f"Change_Mask_{location}{ext}"
                if not component.exists():
                    missing_components.append(ext)
            
            if missing_components:
                issues_found.append(f"Location {location}: Missing shapefile components: {', '.join(missing_components)}")
    
    # Check for MD5 hash file
    md5_file = output_dir / "model_md5.txt"
    if not md5_file.exists():
        issues_found.append("Missing model_md5.txt file required for PS-10 submission")
    else:
        # Verify MD5 hash content
        with open(md5_file, 'r') as f:
            md5_content = f.read().strip()
            if not re.match(r'^[a-f0-9]{32}$', md5_content):
                issues_found.append("model_md5.txt does not contain a valid MD5 hash")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  • Processed {len(processed_locations)} location(s)")
    print(f"  • Found {len(raster_files)} GeoTIFF file(s)")
    print(f"  • Found {len(list(output_dir.glob('*.shp')))} Shapefile(s)")
    
    if issues_found:
        print(f"\n⚠️ Found {len(issues_found)} issue(s) that need to be addressed:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        print("\nPlease fix these issues before submitting your PS-10 package.")
        return False
    else:
        print("\n✅ All files are compliant with PS-10 requirements!")
        print("✅ Your submission is ready for packaging and upload.")
        return True

def validate_ps10_package(package_path):
    """Validate a PS-10 submission ZIP package"""
    import zipfile
    
    if not os.path.exists(package_path):
        print(f"❌ Error: Package file '{package_path}' does not exist")
        return False
    
    # Check if filename follows PS-10 naming convention
    filename = os.path.basename(package_path)
    if not re.match(r'^PS10_\d{2}-[A-Za-z]{3}-\d{4}_[A-Za-z0-9]+\.zip$', filename):
        print(f"❌ Error: Package filename '{filename}' does not follow PS-10 naming convention")
        print("   Required format: PS10_[DD-MMM-YYYY]_[Startup/Group Name without Space].zip")
    
    # Check contents of the ZIP file
    try:
        with zipfile.ZipFile(package_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Check for required files
            raster_files = [f for f in file_list if f.endswith('.tif') and f.startswith('Change_Mask_')]
            shp_files = [f for f in file_list if f.endswith('.shp') and f.startswith('Change_Mask_')]
            
            if not raster_files:
                print("❌ Error: No GeoTIFF files found in the package")
                return False
            
            if not shp_files:
                print("❌ Error: No shapefile (.shp) files found in the package")
                return False
            
            if "model_md5.txt" not in file_list:
                print("❌ Error: Missing model_md5.txt file in the package")
                return False
            
            # Extract the locations from raster files
            locations = []
            for raster in raster_files:
                match = re.search(r'Change_Mask_(.+?)\.tif$', raster)
                if match:
                    locations.append(match.group(1))
            
            # Check if all required vector files exist for each location
            missing_components = []
            for loc in locations:
                for ext in [".shp", ".shx", ".dbf", ".prj"]:
                    component = f"Change_Mask_{loc}{ext}"
                    if component not in file_list:
                        missing_components.append(component)
            
            if missing_components:
                print("❌ Error: Missing required shapefile components:")
                for component in missing_components:
                    print(f"   • {component}")
                return False
            
            print(f"\n✅ Package structure validation passed for {package_path}")
            print(f"   • {len(locations)} location(s)")
            print(f"   • {len(raster_files)} GeoTIFF file(s)")
            print(f"   • {len(shp_files)} Shapefile(s)")
            
            return True
            
    except zipfile.BadZipFile:
        print(f"❌ Error: '{package_path}' is not a valid ZIP file")
        return False
    except Exception as e:
        print(f"❌ Error validating package: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_ps10_compliance.py <output_directory_or_zip>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        validate_output_directory(path)
    elif os.path.isfile(path) and path.endswith('.zip'):
        validate_ps10_package(path)
    else:
        print("❌ Error: Provided path is not a valid directory or ZIP file")
        sys.exit(1)