#!/usr/bin/env python3
"""
Retrain the model with better parameters for PS-10 submission
"""
import os
import sys
from pathlib import Path

def retrain_model():
    """Retrain the model with improved parameters"""
    print("=== PS-10 Model Retraining ===")
    
    # Training parameters
    image_dir = "changedetect/data/processed/train_pairs_small"
    mask_dir = "changedetect/data/processed/masks_small" 
    output_dir = "changedetect/training_runs/ps10_final"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Training command with improved parameters
    train_cmd = f"""python changedetect/src/main.py train \
        --image_dir {image_dir} \
        --mask_dir {mask_dir} \
        --output_dir {output_dir} \
        --model_type siamese_unet \
        --in_channels 3 \
        --batch_size 8 \
        --num_epochs 50 \
        --resume changedetect/training_runs/run1_small/best_model.pth"""
    
    print("Training command:")
    print(train_cmd)
    print("\nStarting training with improved parameters...")
    print("- Batch size: 8 (reduced for stability)")
    print("- Epochs: 50 (increased from 1)")
    print("- Resume from existing checkpoint")
    
    # Execute training
    os.system(train_cmd.replace('\\\n', ' ').replace('\n', ''))

def run_inference_on_real_data():
    """Run inference on the actual PS10 sample data"""
    print("\n=== Running Inference on Real PS10 Data ===")
    
    # First, create proper test pairs from the sample data
    from pathlib import Path
    import shutil
    
    # Create test directory
    real_test_dir = Path("PS10_real_test")
    real_test_dir.mkdir(exist_ok=True)
    
    # Copy the sample images
    sample_dir = Path("PS10_data/Sample_Set/LISS-4/Sample")
    old_image = sample_dir / "Old_Image_MX_Band_2_3_4.tif"
    new_image = sample_dir / "New_Image_MX_Band_2_3_4.tif"
    
    if old_image.exists() and new_image.exists():
        # Copy with proper naming for inference
        shutil.copy2(old_image, real_test_dir / "sample_28.5_77.2_t1.tif")
        shutil.copy2(new_image, real_test_dir / "sample_28.5_77.2_t2.tif")
        
        print(f"Created real test data in {real_test_dir}")
        
        # Run inference
        model_path = "changedetect/training_runs/run1_small/best_model.pth"  # Use existing model
        output_dir = "PS10_real_predictions"
        
        inference_cmd = f"""python changedetect/src/main.py inference \
            --image_dir {real_test_dir} \
            --model_path {model_path} \
            --output_dir {output_dir} \
            --model_type siamese_unet \
            --in_channels 3"""
        
        print("Running inference on real sample data...")
        os.system(inference_cmd.replace('\\\n', ' ').replace('\n', ''))
        
    else:
        print("Sample data not found, using existing predictions")

def create_final_submission():
    """Create the final submission package"""
    print("\n=== Creating Final Submission ===")
    
    # Use existing predictions for now
    exec(open('create_ps10_submission.py').read())

def main():
    """Main function for final model preparation"""
    print("PS-10 Final Model Preparation")
    print("=" * 50)
    
    choice = input("""
Choose an option:
1. Retrain model with better parameters (50 epochs) - RECOMMENDED
2. Use existing model and create submission package
3. Run inference on real sample data
4. Create final submission package only
5. Do everything (retrain, inference, submission)

Enter choice (1-5): """).strip()
    
    if choice == "1":
        retrain_model()
    elif choice == "2":
        print("Using existing model...")
        create_final_submission()
    elif choice == "3":
        run_inference_on_real_data()
    elif choice == "4":
        create_final_submission()
    elif choice == "5":
        print("Running complete pipeline...")
        retrain_model()
        run_inference_on_real_data()
        create_final_submission()
    else:
        print("Invalid choice. Creating submission with existing model.")
        create_final_submission()
    
    print("\n=== Summary ===")
    print("Current files ready for submission:")
    print("- PS10_08-Oct-2025_ChangeDetect.zip (prediction results)")
    print("- model_hash_md5.txt (model hash)")
    print("- changedetect/training_runs/run1_small/best_model.pth (trained model)")
    
    print("\nTo improve results for actual submission:")
    print("1. Train for more epochs (current model only trained 1 epoch)")
    print("2. Adjust threshold for change detection") 
    print("3. Use actual PS-10 test coordinates when available")
    print("4. Validate against ground truth if available")

if __name__ == "__main__":
    main()