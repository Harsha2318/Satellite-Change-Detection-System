#!/usr/bin/env python3
"""
PS-10 Master Execution Script - WINDOWS POWERSHELL COMPATIBLE

NO UNICODE/EMOJI - Pure ASCII output for Windows compatibility

This script:
1. Tests your complete setup
2. Runs inference when ready
3. Corrects filenames (coordinates from GeoTIFF metadata)
4. Creates submission package
5. Validates everything

Usage:
    # Test everything first
    python master_ps10_windows.py --test
    
    # On Oct 31 after downloading data  
    python master_ps10_windows.py --run PS10_shortlisting_data
    
    # Emergency quick run (skip tests)
    python master_ps10_windows.py --quick PS10_shortlisting_data
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime

# Set UTF-8 encoding for Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

# Try to import geospatial packages
try:
    import rasterio
    import numpy as np
    HAS_GEO = True
except ImportError:
    HAS_GEO = False


class PS10Master:
    def __init__(self):
        self.start_time = time.time()
        self.model_path = "models/xboson_change_detector.pt"
        self.team_name = "XBoson AI"
        
    def banner(self, text):
        """Print fancy banner (ASCII only)"""
        print("\n" + "="*70)
        print(f"{text:^70}")
        print("="*70 + "\n")
    
    def log(self, message, level="INFO"):
        """Print timestamped log message (ASCII safe)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {"INFO": "[i]", "SUCCESS": "[+]", "ERROR": "[!]", "WARNING": "[*]"}
        symbol = symbols.get(level, "[.]")
        print(f"[{timestamp}] {symbol} {message}")
    
    def extract_lat_long_from_tif(self, tif_path):
        """Extract latitude and longitude from GeoTIFF bounds"""
        if not HAS_GEO:
            self.log("rasterio not available, using default naming", "WARNING")
            return None, None
            
        try:
            with rasterio.open(tif_path) as src:
                bounds = src.bounds
                center_lon = round((bounds.left + bounds.right) / 2, 4)
                center_lat = round((bounds.bottom + bounds.top) / 2, 4)
                return center_lat, center_lon
        except Exception as e:
            self.log(f"Could not extract coordinates from {Path(tif_path).name}: {e}", "WARNING")
            return None, None
    
    def fix_submission_format(self, input_dir, output_dir):
        """CRITICAL: Fix filenames to PS-10 format with decimal coordinates"""
        
        self.banner("STEP 1: CORRECTING FILENAMES TO PS-10 FORMAT")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            self.log(f"Input directory not found: {input_dir}", "ERROR")
            return False
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all TIF files
        tif_files = list(input_path.glob("*change_mask.tif")) + list(input_path.glob("*.tif"))
        tif_files = sorted(list(set(tif_files)))
        
        if not tif_files:
            self.log(f"No TIF files found in {input_dir}", "ERROR")
            return False
        
        self.log(f"Found {len(tif_files)} TIF files to rename")
        
        processed_files = set()
        renamed_count = 0
        
        for tif_file in tif_files:
            if str(tif_file) in processed_files:
                continue
            
            # Extract coordinates from GeoTIFF metadata
            lat, lon = self.extract_lat_long_from_tif(str(tif_file))
            
            if lat is None or lon is None:
                # Fallback: use original name
                new_tif_name = f"Change_Mask_{tif_file.stem}.tif"
                self.log(f"Using fallback name: {new_tif_name}", "WARNING")
            else:
                new_tif_name = f"Change_Mask_{lat}_{lon}.tif"
            
            new_tif_path = output_path / new_tif_name
            
            try:
                shutil.copy2(tif_file, new_tif_path)
                self.log(f"{tif_file.name} --> {new_tif_name}", "SUCCESS")
                processed_files.add(str(tif_file))
                renamed_count += 1
            except Exception as e:
                self.log(f"Failed to copy {tif_file.name}: {e}", "ERROR")
                continue
            
            # Handle associated shapefiles
            for shp_candidate in input_path.glob("*change_vectors.shp"):
                candidate_stem = str(shp_candidate.stem).replace("_change_vectors", "")
                tif_stem = str(tif_file.stem).replace("_change_mask", "")
                
                if candidate_stem == tif_stem:
                    if lat is None or lon is None:
                        new_shp_base = f"Change_Mask_{tif_file.stem}"
                    else:
                        new_shp_base = f"Change_Mask_{lat}_{lon}"
                    
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        old_file = input_path / f"{shp_candidate.stem}{ext}"
                        if old_file.exists():
                            new_file = output_path / f"{new_shp_base}{ext}"
                            try:
                                shutil.copy2(old_file, new_file)
                            except Exception as e:
                                self.log(f"Could not copy {ext} component: {e}", "WARNING")
                    break
        
        self.log(f"[+] Renamed {renamed_count} files with correct PS-10 format", "SUCCESS")
        return True
    
    def run_inference(self, input_dir, output_dir):
        """Run inference on image pairs"""
        
        self.banner("STEP 2: RUNNING INFERENCE")
        
        cmd = [
            sys.executable, "simple_ps10_inference.py",
            input_dir, output_dir
        ]
        
        self.log(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            self.log("Inference completed successfully", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Inference failed with code {e.returncode}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error running inference: {e}", "ERROR")
            return False
    
    def run_tests(self):
        """Run comprehensive setup tests"""
        
        self.banner("RUNNING SETUP TESTS")
        
        cmd = [sys.executable, "test_complete_workflow_windows.py"]
        
        self.log(f"Executing tests...")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            self.log("All tests passed!", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Tests failed with code {e.returncode}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error running tests: {e}", "ERROR")
            return False
    
    def create_zip_package(self, formatted_dir):
        """Create final submission ZIP"""
        
        self.banner("STEP 3: CREATING SUBMISSION PACKAGE")
        
        formatted_path = Path(formatted_dir)
        
        if not formatted_path.exists():
            self.log(f"Formatted directory not found: {formatted_dir}", "ERROR")
            return None
        
        if not os.path.exists(self.model_path):
            self.log(f"Model file not found: {self.model_path}", "ERROR")
            return None
        
        # Calculate model MD5
        self.log("Calculating model MD5 hash...")
        try:
            hash_md5 = hashlib.md5()
            with open(self.model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            model_hash = hash_md5.hexdigest()
            self.log(f"Model hash: {model_hash}", "SUCCESS")
        except Exception as e:
            self.log(f"Error calculating hash: {e}", "ERROR")
            return None
        
        # Create ZIP
        today = datetime.now()
        date_str = today.strftime("%d-%b-%Y")
        team_clean = self.team_name.replace(" ", "")
        zip_name = f"PS10_{date_str}_{team_clean}.zip"
        
        self.log(f"Creating: {zip_name}")
        
        try:
            with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
                file_count = 0
                for file_path in sorted(formatted_path.glob("*")):
                    if file_path.is_file():
                        zf.write(file_path, file_path.name)
                        file_count += 1
                
                # Add model hash
                hash_file = "model_md5.txt"
                with open(hash_file, 'w') as f:
                    f.write(f"{model_hash}\n")
                zf.write(hash_file, hash_file)
                file_count += 1
            
            self.log(f"[+] Added {file_count} files to ZIP", "SUCCESS")
            self.log(f"ZIP file created: {zip_name}", "SUCCESS")
            return zip_name
        
        except Exception as e:
            self.log(f"Error creating ZIP: {e}", "ERROR")
            return None
    
    def run_validation(self, zip_file):
        """Validate submission package"""
        
        self.banner("STEP 4: VALIDATING SUBMISSION")
        
        cmd = [sys.executable, "validate_ps10_compliance.py", zip_file]
        
        self.log(f"Validating: {zip_file}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            self.log("[+] Validation passed!", "SUCCESS")
            return True
        except Exception as e:
            self.log(f"Validation warning: {e}", "WARNING")
            return False
    
    def test_mode(self):
        """Run tests only"""
        self.banner("TEST MODE: VERIFYING SETUP")
        
        success = self.run_tests()
        
        if success:
            self.banner("[+] SETUP VERIFICATION COMPLETE")
            self.log("Your system is ready for October 31!", "SUCCESS")
        else:
            self.banner("[!] SETUP VERIFICATION FAILED")
            self.log("Please fix the issues above", "ERROR")
        
        return success
    
    def quick_mode(self, input_dir):
        """Quick run: skip tests, format-fix + inference + package"""
        
        self.banner("QUICK MODE: FORMAT-FIX + INFERENCE + PACKAGE")
        
        output_dir = "ps10_predictions"
        formatted_dir = f"{output_dir}_formatted"
        
        # Step 1: Run inference
        if not self.run_inference(input_dir, output_dir):
            self.log("Inference failed", "ERROR")
            return False
        
        # Step 2: Fix format
        if not self.fix_submission_format(output_dir, formatted_dir):
            self.log("Format fix failed", "ERROR")
            return False
        
        # Step 3: Create package
        zip_file = self.create_zip_package(formatted_dir)
        if not zip_file:
            self.log("Package creation failed", "ERROR")
            return False
        
        # Step 4: Validate
        self.run_validation(zip_file)
        
        self.banner("[+] QUICK MODE COMPLETE")
        self.log(f"Submission ready: {zip_file}", "SUCCESS")
        
        return True
    
    def run_mode(self, input_dir):
        """Full run: tests --> inference --> format-fix --> package --> validate"""
        
        self.banner("FULL MODE: COMPLETE WORKFLOW")
        
        # Step 0: Tests
        if not self.run_tests():
            self.log("Setup tests failed, aborting", "ERROR")
            return False
        
        output_dir = "ps10_predictions"
        formatted_dir = f"{output_dir}_formatted"
        
        # Step 1: Run inference
        if not self.run_inference(input_dir, output_dir):
            self.log("Inference failed", "ERROR")
            return False
        
        # Step 2: Fix format (CRITICAL!)
        if not self.fix_submission_format(output_dir, formatted_dir):
            self.log("Format fix failed", "ERROR")
            return False
        
        # Step 3: Create package
        zip_file = self.create_zip_package(formatted_dir)
        if not zip_file:
            self.log("Package creation failed", "ERROR")
            return False
        
        # Step 4: Validate
        self.run_validation(zip_file)
        
        self.banner("[+] FULL WORKFLOW COMPLETE")
        self.log(f"[+] Submission ready: {zip_file}", "SUCCESS")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="PS-10 Master Execution Script (Windows Compatible)")
    parser.add_argument("--test", action="store_true", help="Test setup only")
    parser.add_argument("--run", metavar="INPUT_DIR", help="Full workflow on INPUT_DIR")
    parser.add_argument("--quick", metavar="INPUT_DIR", help="Quick workflow on INPUT_DIR (skip tests)")
    
    args = parser.parse_args()
    
    master = PS10Master()
    
    if args.test:
        success = master.test_mode()
    elif args.run:
        success = master.run_mode(args.run)
    elif args.quick:
        success = master.quick_mode(args.quick)
    else:
        parser.print_help()
        return 1
    
    elapsed = time.time() - master.start_time
    master.log(f"Total execution time: {elapsed:.1f} seconds")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
