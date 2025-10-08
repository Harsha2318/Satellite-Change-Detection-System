import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Open the change mask
with rasterio.open('predictions_threshold_0.1/7_10_change_mask.tif') as src:
    mask = src.read(1)
    profile = src.profile

print(f'Shape: {mask.shape}')
print(f'Unique values: {np.unique(mask)}')
print(f'Percentage of change: {(mask > 0).sum() / mask.size * 100:.2f}%')

# Let's look at the before/after images
img_names = ["7_10_t1.tif", "7_10_t2.tif"]
imgs = []

for img_name in img_names:
    try:
        with rasterio.open(f"changedetect/data/processed/train_pairs/{img_name}") as src:
            img = src.read()
            imgs.append(img)
            print(f"Successfully loaded {img_name}, shape: {img.shape}")
    except Exception as e:
        print(f"Error loading {img_name}: {e}")

# Let's examine the model weights
import torch
from changedetect.src.models.siamese_unet import get_change_detection_model

# Path to model weights
model_path = 'changedetect/training_runs/run1_small/best_model.pth'

try:
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Print checkpoint keys
    print("\nCheckpoint keys:", checkpoint.keys())
    
    # Print training info if available
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    
    if 'val_iou' in checkpoint:
        print(f"Validation IoU: {checkpoint['val_iou']}")
    
    # Look at some parameter statistics
    if 'model_state_dict' in checkpoint:
        print("\nModel parameters:")
        for name, param in checkpoint['model_state_dict'].items():
            if param.requires_grad:
                print(f"{name}: shape {param.shape}, mean {param.mean().item():.6f}, std {param.std().item():.6f}")
                
                # Only print a few parameters
                if name.startswith("encoder1"):
                    break
except Exception as e:
    print(f"Error loading model: {e}")