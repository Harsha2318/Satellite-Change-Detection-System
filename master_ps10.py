#!/usr/bin/env python3
"""
PS-10 Master Execution Script

One script to rule them all - comprehensive execution for October 31.

This script:
1. Tests your complete setup
2. Runs inference when ready
3. Creates submission package
4. Validates everything
5. Gives you final submission files

Usage:
    # Test everything first
    python master_ps10.py --test
    
    # On Oct 31 after downloading data
    python master_ps10.py --run PS10_shortlisting_data
    
    # Emergency quick run (skip tests)
    python master_ps10.py --quick PS10_shortlisting_data
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

class PS10Master:
    def __init__(self):
        self.start_time = time.time()
        self.model_path = "models/xboson_change_detector.pt"
        self.team_name = "XBoson AI"
        
    def banner(self, text):
        """Print fancy banner"""
        print("\n" + "="*70)
        print(f"{text:^70}")
        print("="*70 + "\n")
    
    def log(self, message, level="INFO"):
        """Print timestamped log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {"INFO": "ℹ", "SUCCESS": "✓", "ERROR": "✗", "WARNING": "⚠"}
        symbol = symbols.get(level, "•")
        print(f"[{timestamp}] {symbol} {message}")
    
    def run_command(self, cmd, description):
        """Run a command and report results"""
        self.log(f"{description}...", "INFO")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                self.log(f"{description} completed", "SUCCESS")
                return True, result.stdout
            else:
                self.log(f"{description} failed", "ERROR")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False, result.stderr
        except subprocess.TimeoutExpired:
            self.log(f"{description} timed out", "ERROR")
            return False, "Timeout"
        except Exception as e:
            self.log(f"{description} error: {str(e)}", "ERROR")
            return False, str(e)
    
    def test_setup(self):
        """Run complete setup test"""
        self.banner("TESTING COMPLETE SETUP")
        
        cmd = [sys.executable, "test_complete_workflow.py"]
        success, output = self.run_command(cmd, "Running comprehensive tests")
        
        if success:
            self.log("Setup test PASSED ✓", "SUCCESS")
        else:
            self.log("Setup test FAILED ✗", "ERROR")
            self.log("Please fix issues before proceeding", "WARNING")
        
        return success
    
    def run_inference(self, input_dir, output_dir="PS10_final_predictions"):
        """Run inference on input data"""
        self.banner(f"RUNNING INFERENCE: {input_dir}")
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            self.log(f"Input directory not found: {input_dir}", "ERROR")
            return False
        
        # Count files
        tif_files = list(Path(input_dir).glob("*.tif")) + list(Path(input_dir).glob("*.jp2"))
        self.log(f"Found {len(tif_files)} image files", "INFO")
        
        # Run rapid inference
        cmd = [
            sys.executable,
            "oct31_rapid_inference.py",
            input_dir,
            output_dir,
            "--model", self.model_path,
            "--device", "cuda"
        ]
        
        self.log("Starting inference (this may take 1-2 hours)...", "INFO")
        
        # Run with streaming output
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            
            if process.returncode == 0:
                self.log("Inference completed successfully", "SUCCESS")
                return True, output_dir
            else:
                self.log("Inference failed", "ERROR")
                return False, None
        except Exception as e:
            self.log(f"Inference error: {str(e)}", "ERROR")
            return False, None
    
    def create_submission(self, predictions_dir):
        """Create PS-10 submission package"""
        self.banner("CREATING SUBMISSION PACKAGE")
        
        cmd = [
            sys.executable,
            "prepare_ps10_final.py",
            predictions_dir,
            self.model_path,
            self.team_name
        ]
        
        success, output = self.run_command(cmd, "Creating submission package")
        
        if success:
            # Find created ZIP file
            import glob
            zip_files = glob.glob("PS10_*_XBosonAI.zip")
            if zip_files:
                zip_file = zip_files[0]
                size_mb = os.path.getsize(zip_file) / (1024*1024)
                self.log(f"Package created: {zip_file} ({size_mb:.2f} MB)", "SUCCESS")
                return True, zip_file
        
        return False, None
    
    def validate_submission(self, package_path):
        """Validate submission package"""
        self.banner("VALIDATING SUBMISSION")
        
        cmd = [
            sys.executable,
            "validate_ps10_compliance.py",
            package_path
        ]
        
        success, output = self.run_command(cmd, "Validating submission")
        
        if success:
            self.log("Validation PASSED ✓", "SUCCESS")
        else:
            self.log("Validation FAILED ✗", "ERROR")
        
        return success
    
    def final_summary(self, zip_file):
        """Print final summary and instructions"""
        elapsed = time.time() - self.start_time
        
        self.banner("SUBMISSION READY!")
        
        print(f"  Total time elapsed: {elapsed/60:.1f} minutes\n")
        print(f"  📦 Submission package: {zip_file}")
        print(f"  📄 Model hash file: model_md5.txt\n")
        
        print("  🎯 FINAL STEPS:")
        print("     1. Locate your submission files:")
        print(f"        - {zip_file}")
        print(f"        - model_md5.txt (inside ZIP)")
        print()
        print("     2. Go to PS-10 submission portal")
        print()
        print("     3. Upload the ZIP file")
        print()
        print("     4. Copy model hash from model_md5.txt and submit")
        print()
        print("     5. Verify confirmation received")
        print()
        print("     6. SUBMIT BEFORE 16:00 ⏰")
        print()
        print("  ✨ Good luck! You're ready!")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="PS-10 Master Execution Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test your complete setup
  python master_ps10.py --test
  
  # Full run on Oct 31 (recommended)
  python master_ps10.py --run PS10_shortlisting_data
  
  # Quick run without testing (emergency only)
  python master_ps10.py --quick PS10_shortlisting_data
        """
    )
    
    parser.add_argument("--test", action="store_true",
                       help="Run complete setup test only")
    parser.add_argument("--run", metavar="INPUT_DIR",
                       help="Run full workflow (test + inference + package)")
    parser.add_argument("--quick", metavar="INPUT_DIR",
                       help="Quick run without testing (emergency)")
    parser.add_argument("--output", default="PS10_final_predictions",
                       help="Output directory for predictions")
    
    args = parser.parse_args()
    
    master = PS10Master()
    
    # Display header
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "PS-10 MASTER SCRIPT" + " "*29 + "║")
    print("║" + " "*15 + "Complete Workflow Automation" + " "*24 + "║")
    print("║" + " "*18 + "October 31, 2025 Ready" + " "*27 + "║")
    print("╚" + "="*68 + "╝\n")
    
    # Mode selection
    if args.test:
        # Test only mode
        master.test_setup()
        
    elif args.run:
        # Full workflow mode
        master.log("Starting FULL WORKFLOW mode", "INFO")
        master.log("This will: test → inference → package → validate", "INFO")
        print()
        
        # Step 1: Test
        if not master.test_setup():
            master.log("Test failed. Fix issues before continuing.", "ERROR")
            return 1
        
        # Step 2: Inference
        success, predictions_dir = master.run_inference(args.run, args.output)
        if not success:
            master.log("Inference failed", "ERROR")
            return 1
        
        # Step 3: Create package
        success, zip_file = master.create_submission(predictions_dir)
        if not success:
            master.log("Package creation failed", "ERROR")
            return 1
        
        # Step 4: Validate
        if not master.validate_submission(zip_file):
            master.log("Validation failed - review outputs", "WARNING")
            # Don't fail - package might still be usable
        
        # Step 5: Final summary
        master.final_summary(zip_file)
        
    elif args.quick:
        # Quick mode (skip tests)
        master.log("Starting QUICK mode (skipping tests)", "WARNING")
        master.log("Use this only if you're short on time!", "WARNING")
        print()
        
        # Inference
        success, predictions_dir = master.run_inference(args.quick, args.output)
        if not success:
            return 1
        
        # Create package
        success, zip_file = master.create_submission(predictions_dir)
        if not success:
            return 1
        
        # Validate
        master.validate_submission(zip_file)
        
        # Summary
        master.final_summary(zip_file)
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
