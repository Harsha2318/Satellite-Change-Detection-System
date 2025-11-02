#!/usr/bin/env python3
"""
PS-10 Rapid Inference Script for October 31, 2025

This script is optimized for the 4-hour submission window:
- Quick setup and execution
- Clear progress tracking
- Automatic error recovery
- Time-optimized processing

Usage on Oct 31:
    python oct31_rapid_inference.py PS10_shortlisting_data PS10_final_predictions
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

def print_banner():
    """Print startup banner"""
    print("\n" + "="*70)
    print(" "*15 + "PS-10 RAPID INFERENCE RUNNER")
    print(" "*10 + "Optimized for October 31, 2025 Submission")
    print("="*70 + "\n")

def print_progress(message, level="INFO"):
    """Print progress message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbols = {
        "INFO": "ℹ",
        "SUCCESS": "✓",
        "ERROR": "✗",
        "WARNING": "⚠"
    }
    symbol = symbols.get(level, "•")
    print(f"[{timestamp}] {symbol} {message}")

def check_prerequisites():
    """Check if all prerequisites are met"""
    print_progress("Checking prerequisites...", "INFO")
    
    issues = []
    
    # Check model file
    model_path = "models/xboson_change_detector.pt"
    if not os.path.exists(model_path):
        issues.append(f"Model file not found: {model_path}")
    else:
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print_progress(f"Model found: {model_path} ({size_mb:.1f} MB)", "SUCCESS")
    
    # Check inference script
    inference_script = "changedetect/src/inference.py"
    if not os.path.exists(inference_script):
        issues.append(f"Inference script not found: {inference_script}")
    else:
        print_progress(f"Inference script found", "SUCCESS")
    
    # Check required packages
    try:
        import torch
        print_progress(f"PyTorch {torch.__version__} available", "SUCCESS")
    except ImportError:
        issues.append("PyTorch not installed")
    
    try:
        import rasterio
        print_progress("rasterio available", "SUCCESS")
    except ImportError:
        issues.append("rasterio not installed")
    
    try:
        import geopandas
        print_progress("geopandas available", "SUCCESS")
    except ImportError:
        issues.append("geopandas not installed")
    
    if issues:
        print_progress("Prerequisites check FAILED:", "ERROR")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print_progress("All prerequisites met!", "SUCCESS")
    return True

def find_image_pairs(data_dir):
    """Find image pairs in the data directory"""
    print_progress(f"Scanning for image pairs in {data_dir}...", "INFO")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print_progress(f"Directory not found: {data_dir}", "ERROR")
        return []
    
    # Look for common patterns
    patterns = ["*_t1.tif", "*_t1.jp2", "*_time1.tif", "*_T1.tif"]
    
    pairs = []
    for pattern in patterns:
        t1_files = list(data_path.glob(pattern))
        for t1_file in t1_files:
            # Try to find corresponding t2 file
            base_name = str(t1_file.stem)
            
            # Try different naming patterns
            t2_patterns = [
                base_name.replace("_t1", "_t2"),
                base_name.replace("_T1", "_T2"),
                base_name.replace("_time1", "_time2"),
                base_name.replace("time1", "time2")
            ]
            
            for t2_base in t2_patterns:
                for ext in [".tif", ".jp2", ".TIF", ".JP2"]:
                    t2_file = data_path / (t2_base + ext)
                    if t2_file.exists():
                        pairs.append({
                            "name": base_name.replace("_t1", "").replace("_T1", ""),
                            "t1": str(t1_file),
                            "t2": str(t2_file)
                        })
                        print_progress(f"Found pair: {pairs[-1]['name']}", "SUCCESS")
                        break
                if pairs and pairs[-1]["name"] == base_name.replace("_t1", "").replace("_T1", ""):
                    break
    
    if not pairs:
        print_progress("No image pairs found!", "WARNING")
        print("  Expected file naming:")
        print("    - *_t1.tif and *_t2.tif")
        print("    - *_time1.tif and *_time2.tif")
        print("  Or similar patterns")
    
    return pairs

def run_inference(data_dir, output_dir, model_path=None, device="cuda"):
    """Run inference on all image pairs"""
    
    if model_path is None:
        model_path = "models/xboson_change_detector.pt"
    
    print_progress(f"Starting inference...", "INFO")
    print_progress(f"  Input: {data_dir}", "INFO")
    print_progress(f"  Output: {output_dir}", "INFO")
    print_progress(f"  Model: {model_path}", "INFO")
    print_progress(f"  Device: {device}", "INFO")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "changedetect/src/inference.py",
        "--image_dir", data_dir,
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--device", device,
        "--tile_size", "512",
        "--overlap", "64"
    ]
    
    print_progress("Executing inference command...", "INFO")
    print(f"  Command: {' '.join(cmd)}")
    
    # Run inference
    import subprocess
    
    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            line = line.strip()
            if line:
                # Show important messages
                if "Processing" in line or "Saved" in line or "Error" in line:
                    print(f"    {line}")
        
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print_progress(f"Inference completed in {elapsed/60:.1f} minutes", "SUCCESS")
            return True
        else:
            print_progress(f"Inference failed with return code {process.returncode}", "ERROR")
            return False
            
    except Exception as e:
        print_progress(f"Inference error: {str(e)}", "ERROR")
        return False

def verify_outputs(output_dir):
    """Verify that outputs were generated correctly"""
    print_progress("Verifying outputs...", "INFO")
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print_progress(f"Output directory not found: {output_dir}", "ERROR")
        return False
    
    # Check for TIF files
    tif_files = list(output_path.glob("*.tif"))
    print_progress(f"Found {len(tif_files)} TIF files", "INFO")
    
    # Check for shapefiles
    shp_files = list(output_path.glob("*.shp"))
    print_progress(f"Found {len(shp_files)} shapefile sets", "INFO")
    
    # Verify each shapefile has all components
    complete_shapefiles = 0
    for shp_file in shp_files:
        base_name = str(shp_file).replace(".shp", "")
        required = [".shp", ".shx", ".dbf", ".prj"]
        
        if all(os.path.exists(base_name + ext) for ext in required):
            complete_shapefiles += 1
    
    print_progress(f"{complete_shapefiles}/{len(shp_files)} shapefiles complete", "INFO")
    
    if len(tif_files) > 0 and complete_shapefiles > 0:
        print_progress("Output verification PASSED", "SUCCESS")
        return True
    else:
        print_progress("Output verification FAILED", "ERROR")
        return False

def estimate_time(num_pairs):
    """Estimate processing time"""
    time_per_pair = 20  # minutes (conservative estimate)
    total_minutes = num_pairs * time_per_pair
    
    hours = int(total_minutes / 60)
    minutes = int(total_minutes % 60)
    
    return hours, minutes

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="PS-10 Rapid Inference for October 31 submission"
    )
    parser.add_argument("input_dir", help="Directory with image pairs")
    parser.add_argument("output_dir", help="Directory for outputs")
    parser.add_argument("--model", default="models/xboson_change_detector.pt",
                       help="Path to model file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip prerequisite checks (not recommended)")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            print_progress("Please fix issues before running inference", "ERROR")
            return 1
    
    # Find image pairs
    pairs = find_image_pairs(args.input_dir)
    if not pairs:
        print_progress("No image pairs found in input directory", "ERROR")
        return 1
    
    print(f"\n{'─'*70}")
    print(f"  Found {len(pairs)} image pair(s) to process")
    
    # Estimate time
    hours, minutes = estimate_time(len(pairs))
    if hours > 0:
        time_str = f"{hours}h {minutes}m"
    else:
        time_str = f"{minutes}m"
    
    print(f"  Estimated processing time: {time_str}")
    print(f"{'─'*70}\n")
    
    # Confirm
    response = input("  Proceed with inference? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print_progress("Inference cancelled by user", "WARNING")
        return 0
    
    print()
    
    # Run inference
    overall_start = time.time()
    
    success = run_inference(
        args.input_dir,
        args.output_dir,
        args.model,
        args.device
    )
    
    if not success:
        print_progress("Inference failed!", "ERROR")
        return 1
    
    # Verify outputs
    if not verify_outputs(args.output_dir):
        print_progress("Output verification failed!", "ERROR")
        return 1
    
    # Summary
    total_time = time.time() - overall_start
    print("\n" + "="*70)
    print("  INFERENCE COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Outputs saved to: {args.output_dir}")
    print("="*70)
    
    print("\n  Next steps:")
    print("    1. Create submission package:")
    print(f"       python prepare_ps10_final.py {args.output_dir} models/xboson_change_detector.pt \"XBoson AI\"")
    print("    2. Validate package:")
    print("       python validate_ps10_compliance.py PS10_*.zip")
    print("    3. Submit before 16:00!\n")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
