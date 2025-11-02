#!/usr/bin/env python3
"""
PS-10: Man-made Change Detection Submission Package Creator

This script creates a submission package that strictly follows the PS-10 requirements:
1. Output file naming: 'Change_Mask_Lat_Long.tif' (Raster) and 'Change_Mask_Lat_Long.shp' (Vector)
2. Submission folder/zip: PS10_[DD-MMM-YYYY]_[Startup/Group Name without Space].zip
3. Must include MD5 hash value of the model

As specified in PS-10 documentation, the solution:
- Detects man-made changes from satellite imagery pairs
- Outputs a raster mask where pixel value 1 represents change and pixel value 0 represents no change
- Provides corresponding vector files in shapefile format
- All outputs are georeferenced
"""
import os
import sys
import shutil
import hashlib
from pathlib import Path
import zipfile
from datetime import datetime

try:
    import rasterio
    import numpy as np
    import geopandas as gpd
except ImportError:
    print("Warning: Some dependencies are missing. Install with: pip install rasterio numpy geopandas")
    print("Continuing with limited functionality...")
    rasterio = None
    np = None
    gpd = None

def create_test_pairs_from_sample():
    """Create proper test image pairs from the PS10 sample data"""
    print("Creating test pairs from PS10 sample data...")
    
    # Source paths
    sample_dir = Path("PS10_data/Sample_Set/LISS-4/Sample")
    old_image = sample_dir / "Old_Image_MX_Band_2_3_4.tif"
    new_image = sample_dir / "New_Image_MX_Band_2_3_4.tif"
    
    # Create output directory for test pairs
    test_pairs_dir = Path("PS10_test_data")
    test_pairs_dir.mkdir(exist_ok=True)
    
    # Mock coordinates from the sample (you'd get real ones from the dataset listing)
    test_locations = [
        ("28.5", "77.2"),  # Delhi area coordinates as example
        # Add more as needed for your actual test data
    ]
    
    for i, (lat, lon) in enumerate(test_locations):
        # Copy and rename files for each test location
        t1_name = f"test_{lat}_{lon}_t1.tif"
        t2_name = f"test_{lat}_{lon}_t2.tif"
        
        shutil.copy2(old_image, test_pairs_dir / t1_name)
        shutil.copy2(new_image, test_pairs_dir / t2_name)
        
        print(f"Created test pair: {t1_name}, {t2_name}")
    
    return test_pairs_dir

def get_ps10_location_mapping():
    """
    Return mapping from PS-10 documentation section 6.i
    These are the specific terrain types required by PS-10
    """
    return {
        "0_10": "34.0531_74.3909",  # Snow
        "0_11": "13.3143_77.6157",  # Plain
        "0_12": "31.2834_76.7904",  # Hill
        "0_13": "26.9027_70.9543",  # Desert
        "0_14": "23.7380_84.2129",  # Forest
        "0_15": "28.1740_77.6126",  # Urban
        # Add more mappings from the actual dataset as needed
    }

def calculate_model_md5(model_path):
    """Calculate MD5 hash of model file as required by PS-10"""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        print("Creating a placeholder MD5 hash. Replace this with your actual model hash before submission!")
        # Generate placeholder MD5 hash (not for actual submission)
        placeholder = hashlib.md5(f"placeholder_for_{os.path.basename(model_path)}".encode()).hexdigest()
        return placeholder
        
    try:
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating MD5: {str(e)}")
        print("Creating a placeholder MD5 hash. Replace this with your actual model hash before submission!")
        # Generate placeholder MD5 hash (not for actual submission)
        placeholder = hashlib.md5(f"placeholder_for_{os.path.basename(model_path)}".encode()).hexdigest()
        return placeholder

def fix_tif_values(tif_file):
    """
    Convert TIF values from 0/255 to 0/1 as required by PS-10
    
    PS-10 requires pixel value 1 for change and 0 for no change
    """
    if rasterio is None or np is None:
        print("Warning: rasterio or numpy not available. Skipping TIF value conversion.")
        return False
        
    try:
        with rasterio.open(tif_file, "r+") as src:
            # Read the data
            data = src.read(1)
            
            # Check if conversion is needed
            unique_vals = np.unique(data)
            if set(unique_vals).issubset({0, 1}):
                print(f"  ✓ Values already compliant: {unique_vals}")
                return True
                
            if 255 in unique_vals:
                # Convert 255 to 1
                data = np.where(data == 255, 1, data)
                
                # Write the modified data back
                src.write(data, 1)
                
                # Verify the change
                updated_data = src.read(1)
                updated_vals = np.unique(updated_data)
                print(f"  ✓ Converted values from {unique_vals} to {updated_vals}")
                return True
            else:
                print(f"  ⚠️ TIF has unexpected values: {unique_vals}")
                return False
    except Exception as e:
        print(f"  ❌ Error fixing TIF values: {str(e)}")
        return False

def rename_predictions_to_ps10_format(predictions_dir, output_dir):
    """
    Rename prediction files to PS-10 submission format
    
    PS-10 requires:
    - Raster mask in GeoTIFF format named 'Change_Mask_Lat_Long.tif'
    - Vector files in shapefile format named 'Change_Mask_Lat_Long.shp' (etc.)
    - Each change mask should have pixel value 1 for change and 0 for no change
    """
    print(f"Converting predictions from {predictions_dir} to PS-10 format...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get official PS-10 coordinate mapping
    coord_mapping = get_ps10_location_mapping()
    
    pred_path = Path(predictions_dir)
    
    # Find all change mask files
    change_mask_files = list(pred_path.glob("*_change_mask.tif"))
    if not change_mask_files:
        print(f"No change mask files found in {predictions_dir}")
        return False
    
    print(f"Found {len(change_mask_files)} change mask files to process")
    
    # Track results
    successful_conversions = 0
    missing_components = 0
    fixed_values = 0
    
    # Process each prediction file
    for change_mask_file in change_mask_files:
        # Extract the image pair ID (e.g., "0_10")
        file_stem = change_mask_file.stem  # e.g., "0_10_change_mask"
        match = file_stem.split("_change_mask")[0]  # e.g., "0_10"
        
        if match in coord_mapping:
            lat_long = coord_mapping[match]
            print(f"Processing {match} → {lat_long} (from PS-10 reference)")
        else:
            # If not in mapping, use the original ID as fallback
            lat_long = match
            print(f"Processing {match} → {lat_long} (fallback)")
        
        # Process TIF files (change masks)
        new_tif = output_path / f"Change_Mask_{lat_long}.tif"
        shutil.copy2(change_mask_file, new_tif)
        print(f"✓ Created {new_tif.name}")
        
        # Fix TIF values (convert from 0/255 to 0/1)
        print(f"  Fixing TIF values to comply with PS-10 requirements...")
        if fix_tif_values(new_tif):
            fixed_values += 1
        
        # Process shapefile components
        shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
        shapefile_base = file_stem.replace("_change_mask", "_change_vectors")
        
        # Track component completeness
        components_found = 0
        missing_exts = []
        
        for ext in shapefile_extensions:
            old_shp = pred_path / f"{shapefile_base}{ext}"
            if old_shp.exists():
                new_shp = output_path / f"Change_Mask_{lat_long}{ext}"
                shutil.copy2(old_shp, new_shp)
                print(f"✓ Created {new_shp.name}")
                components_found += 1
            else:
                missing_exts.append(ext)
        
        if components_found == len(shapefile_extensions):
            successful_conversions += 1
        else:
            missing_components += 1
            if missing_exts:
                print(f"⚠️ Warning: Missing vector components for {lat_long}: {', '.join(missing_exts)}")
    
    print(f"\n✓ Successfully converted {successful_conversions} locations with complete components")
    if missing_components > 0:
        print(f"⚠️ Found {missing_components} locations with missing components")
    if fixed_values > 0:
        print(f"✓ Fixed TIF values for {fixed_values} files (converted 0/255 to 0/1 for PS-10 compliance)")
    
    return successful_conversions > 0

def create_submission_package(results_dir, model_path, startup_name="XBoson"):
    """
    Create final submission package exactly according to PS-10 requirements
    
    PS-10 requires:
    - Folder/zip name: PS10_[DD-MMM-YYYY]_[StartupName without spaces].zip
    - Raster mask named Change_Mask_Lat_Long.tif
    - Vector files named Change_Mask_Lat_Long.shp (with supporting components)
    - MD5 hash of model file
    """
    # Verify that results directory exists and contains required files
    results_path = Path(results_dir)
    if not results_path.exists() or not results_path.is_dir():
        print(f"Error: Results directory '{results_dir}' does not exist")
        return None
        
    # Check for required files
    tif_files = list(results_path.glob("Change_Mask_*.tif"))
    if not tif_files:
        print(f"Error: No properly named Change_Mask_*.tif files found in {results_dir}")
        print("Please ensure files follow PS-10 naming conventions")
        return None
    
    # Create submission folder name exactly as specified in PS-10:
    # PS10_[DD-MMM-YYYY]_[StartupName without spaces]
    today = datetime.now()
    date_str = today.strftime("%d-%b-%Y")  # Format: DD-MMM-YYYY (e.g., 09-Oct-2025)
    folder_name = f"PS10_{date_str}_{startup_name}"
    
    # Create submission directory
    submission_dir = Path(folder_name)
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(exist_ok=True)
    
    # Copy all results with proper verification
    copied_files = 0
    tif_count = 0
    shp_count = 0
    
    # Track locations to verify completeness
    processed_locations = []
    
    print(f"\nAdding files to PS-10 submission package:")
    
    for tif_file in tif_files:
        # Extract the location part (e.g., "28.5_77.2")
        location = tif_file.stem.replace("Change_Mask_", "")
        processed_locations.append(location)
        
        # Copy the TIF file
        shutil.copy2(tif_file, submission_dir / tif_file.name)
        tif_count += 1
        print(f"✓ Added raster: {tif_file.name}")
        
        # Copy all shapefile components for each location
        shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
        for ext in shapefile_extensions:
            shp_component = results_path / f"Change_Mask_{location}{ext}"
            if shp_component.exists():
                shutil.copy2(shp_component, submission_dir / shp_component.name)
                if ext == '.shp':
                    shp_count += 1
                print(f"✓ Added vector: {shp_component.name}")
                copied_files += 1
    
    # Calculate and save the model MD5 hash as required by PS-10
    model_md5 = calculate_model_md5(model_path)
    if model_md5:
        with open(submission_dir / "model_md5.txt", "w") as f:
            f.write(f"{model_md5}")
        print(f"✓ Added model MD5 hash: {model_md5}")
        copied_files += 1
    else:
        print("⚠️ Warning: Failed to generate model MD5 hash")
    
    print(f"\nSummary: Added {copied_files} files to submission")
    print(f"- {tif_count} GeoTIFF raster masks for man-made change detection")
    print(f"- {shp_count} Shapefile vector files (with supporting files)")
    
    # Create zip file according to PS-10 requirements
    zip_path = Path(f"{folder_name}.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in submission_dir.glob("*"):
            zipf.write(file, file.name)
            print(f"Zipped: {file.name}")
    
    print(f"\n✅ Created PS-10 submission package: {zip_path}")
    print(f"   • {len(processed_locations)} location(s)")
    print(f"   • {tif_count} GeoTIFF file(s)")
    print(f"   • {shp_count} Shapefile(s)")
    print(f"   • Total size: {zip_path.stat().st_size / 1024:.1f} KB")
    
    # Verify that the submission meets PS-10 requirements
    verify_ps10_compliance(submission_dir, processed_locations)
    
    return zip_path


def verify_ps10_compliance(submission_dir, processed_locations):
    """Verify that the submission meets PS-10 requirements"""
    print("\n🔍 Verifying PS-10 compliance:")
    
    # Check for required files
    all_compliant = True
    
    # Check for model_md5.txt
    if not (submission_dir / "model_md5.txt").exists():
        print("❌ Missing model_md5.txt file")
        all_compliant = False
    
    # Check for all required files for each location
    for location in processed_locations:
        # Check for raster file
        raster_file = submission_dir / f"Change_Mask_{location}.tif"
        if not raster_file.exists():
            print(f"❌ Missing raster file: {raster_file.name}")
            all_compliant = False
        
        # Check for all shapefile components
        for ext in [".shp", ".shx", ".dbf", ".prj"]:  # CPG is optional
            vector_file = submission_dir / f"Change_Mask_{location}{ext}"
            if not vector_file.exists():
                print(f"❌ Missing vector file: {vector_file.name}")
                all_compliant = False
    
    if all_compliant:
        print("✅ All required files are present")
        print("✅ Naming conventions follow PS-10 guidelines")
        print("✅ Submission package is ready for upload")
    else:
        print("⚠️ Some issues found with the submission package")
        print("   Please address them before submitting")
        
    return all_compliant

def verify_output_format(results_dir):
    """
    Verify that output files match PS-10 requirements for man-made change detection
    
    PS-10 requires:
    - GeoTIFF raster where pixel value 1 represents change and 0 represents no change
    - Vector files in shapefile format
    - All outputs must be properly georeferenced
    """
    print("Verifying output format for PS-10 compliance...")
    
    results_path = Path(results_dir)
    
    # Check for required files
    tif_files = list(results_path.glob("Change_Mask_*.tif"))
    shp_files = list(results_path.glob("Change_Mask_*.shp"))
    
    print(f"Found {len(tif_files)} TIF files")
    print(f"Found {len(shp_files)} SHP files")
    
    # Verify TIF format if rasterio is available
    if rasterio is not None and np is not None:
        for tif_file in tif_files:
            try:
                with rasterio.open(tif_file) as src:
                    # Check that it's georeferenced
                    if src.crs is None:
                        print(f"WARNING: {tif_file.name} has no CRS!")
                    else:
                        print(f"✓ {tif_file.name}: CRS = {src.crs}")
                    
                    # Check data values (should be 0 or 1)
                    data = src.read(1)
                    unique_vals = np.unique(data)
                    if set(unique_vals).issubset({0, 1}):
                        print(f"✓ {tif_file.name}: Values = {unique_vals} (Valid for PS-10)")
                    else:
                        print(f"⚠️ WARNING: {tif_file.name} has invalid values: {unique_vals}")
                        print("   PS-10 requires pixel value 1 for change and 0 for no change")
                    
            except Exception as e:
                print(f"ERROR reading {tif_file.name}: {e}")
    else:
        print("Note: rasterio or numpy not available. Skipping detailed raster validation.")
        print("Install with: pip install rasterio numpy")
    
    # Check shapefile completeness
    for shp_file in shp_files:
        base_name = shp_file.stem
        required_extensions = ['.shx', '.dbf', '.prj']
        
        missing = []
        for ext in required_extensions:
            companion_file = results_path / f"{base_name}{ext}"
            if not companion_file.exists():
                missing.append(ext)
        
        if missing:
            print(f"⚠️ WARNING: {shp_file.name} missing components: {', '.join(missing)}")
            print("   PS-10 requires complete shapefiles with all components")
        else:
            print(f"✓ {shp_file.name}: Complete shapefile")

def main():
    """Main function to create PS-10 submission for man-made change detection"""
    print("=== PS-10 Man-made Change Detection Submission Creator ===")
    
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python create_ps10_submission.py <predictions_dir> <model_path> [startup_name]")
        print("\nWhere:")
        print("  <predictions_dir> is the directory containing your change detection outputs")
        print("  <model_path> is the path to your model file (for MD5 hash calculation)")
        print("  [startup_name] is your startup/group name (without spaces, default: XBoson)")
        return 1
    
    # Get command line arguments
    predictions_dir = sys.argv[1]
    model_path = sys.argv[2]
    startup_name = sys.argv[3] if len(sys.argv) > 3 else "XBoson"
    
    # Verify inputs
    if not Path(predictions_dir).exists():
        print(f"Error: Predictions directory '{predictions_dir}' does not exist")
        return 1
        
    if not Path(model_path).exists():
        print(f"Warning: Model file '{model_path}' not found")
        print("Continuing with a placeholder model hash. Replace with your actual model hash before final submission.")
        
        # If model_path contains directory that doesn't exist, create it
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created directory: {model_dir}")
            
        # Create an empty placeholder model file for development/testing
        try:
            with open(model_path, 'w') as f:
                f.write("This is a placeholder model file for testing submission package creation.\n")
                f.write("Replace this with your actual model before submission.\n")
            print(f"Created placeholder model file: {model_path}")
        except Exception as e:
            print(f"Warning: Could not create placeholder model file: {str(e)}")
            print("Continuing with MD5 hash generation only...")
    
    ps10_results_dir = "PS10_submission_results"
    
    # Step 1: Rename existing predictions to PS-10 format
    print("\nStep 1: Converting predictions to PS-10 format...")
    if not rename_predictions_to_ps10_format(predictions_dir, ps10_results_dir):
        print(f"Error: Failed to convert predictions from {predictions_dir}")
        return 1
    
    # Step 2: Verify output format
    print("\nStep 2: Verifying output format...")
    verify_output_format(ps10_results_dir)
    
    # Step 3: Create submission package
    print("\nStep 3: Creating PS-10 submission package...")
    zip_file = create_submission_package(ps10_results_dir, model_path, startup_name)
    
    if not zip_file:
        print("\n❌ Failed to create submission package")
        return 1
    
    print(f"\n=== PS-10 SUBMISSION READY ===")
    print(f"✅ Results directory: {ps10_results_dir}")
    print(f"✅ Submission package: {zip_file}")
    print("\nSubmission Checklist:")
    print("1. Verify that all outputs follow PS-10 naming conventions")
    print("2. Ensure all files are properly georeferenced")
    print("3. Confirm that raster masks have correct pixel values (1=change, 0=no change)")
    print("4. Verify that vector files have complete components (.shp, .shx, .dbf, .prj)")
    print("5. Check that model MD5 hash is correctly generated")
    print("\nSubmission steps:")
    print(f"1. Upload {zip_file}")
    print("2. Submit the hash value from model_md5.txt on the submission platform")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

if __name__ == "__main__":
    main()