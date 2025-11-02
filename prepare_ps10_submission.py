#!/usr/bin/env python3
"""
PS-10 Submission Helper

This script helps prepare a compliant PS-10 submission package by:
1. Fixing TIF values to be 0/1 instead of 0/255
2. Creating a properly named submission package
3. Verifying compliance with PS-10 requirements

Usage:
  python prepare_ps10_submission.py <predictions_dir> [startup_name]

Where:
  <predictions_dir> is the directory containing prediction files (e.g. predictions_threshold_0.1)
  [startup_name] is your startup/group name without spaces (default: XBoson)
"""

import os
import sys
from pathlib import Path
import subprocess

def main():
    """Main function to prepare a PS-10 submission"""
    print("=== PS-10 Submission Helper ===")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python prepare_ps10_submission.py <predictions_dir> [startup_name]")
        print("\nWhere:")
        print("  <predictions_dir> is the directory containing prediction files")
        print("  [startup_name] is your startup/group name without spaces (default: XBoson)")
        return 1
    
    predictions_dir = sys.argv[1]
    startup_name = sys.argv[2] if len(sys.argv) > 2 else "XBoson"
    
    # Check if predictions directory exists
    if not Path(predictions_dir).exists():
        print(f"Error: Predictions directory '{predictions_dir}' does not exist")
        return 1
    
    # Placeholder model file
    model_path = "model/model.h5"
    
    # Create model directory if it doesn't exist
    model_dir = Path("model")
    if not model_dir.exists():
        model_dir.mkdir(exist_ok=True)
        print(f"Created directory: {model_dir}")
    
    # Create placeholder model file if it doesn't exist
    if not Path(model_path).exists():
        try:
            with open(model_path, 'w') as f:
                f.write("This is a placeholder model file for testing submission package creation.\n")
                f.write("Replace this with your actual model before submission.\n")
            print(f"Created placeholder model file: {model_path}")
        except Exception as e:
            print(f"Warning: Could not create placeholder model file: {str(e)}")
    
    # Run the create_ps10_submission.py script
    print("\nRunning PS-10 submission creator...")
    cmd = [sys.executable, "create_ps10_submission.py", predictions_dir, model_path, startup_name]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running create_ps10_submission.py: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    
    print("\n=== PS-10 Submission Process Complete ===")
    print("Next steps:")
    print("1. Check the PS10_submission_results directory to verify outputs")
    print("2. Check the generated ZIP file for final submission")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())