import torch
import numpy as np
from changedetect.src.models.siamese_unet import get_change_detection_model

# Path to model weights
model_path = 'changedetect/training_runs/run1_small/best_model.pth'

# Create model
print('Creating model...')
model = get_change_detection_model(
    model_type='siamese_unet',
    in_channels=3,
    out_channels=1,
    features=64,
    bilinear=False
)

# Load model weights
print('Loading weights...')
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get training details
print(f"\nTraining details:")
print(f"Epochs trained: {checkpoint.get('epoch', 'N/A')}")
print(f"Validation IoU: {checkpoint.get('val_iou', 'N/A')}")
print(f"Validation Dice: {checkpoint.get('val_dice', 'N/A')}")
print(f"Validation F1: {checkpoint.get('val_f1', 'N/A')}")
print(f"Validation Loss: {checkpoint.get('val_loss', 'N/A')}")

# Create random input tensors to simulate image pairs
print('\nGenerating random input tensors...')
t1_tensor = torch.rand(1, 3, 512, 512)
t2_tensor = torch.rand(1, 3, 512, 512)

# Run inference
print('Running inference...')
with torch.no_grad():
    output = model(t1_tensor, t2_tensor)
    raw_output = output.squeeze().cpu().numpy()
    sigmoid_output = torch.sigmoid(output).squeeze().cpu().numpy()

# Analyze raw output
print('\nRaw model output (logits):')
print(f"Min: {raw_output.min():.6f}")
print(f"Max: {raw_output.max():.6f}")
print(f"Mean: {raw_output.mean():.6f}")
print(f"Std: {raw_output.std():.6f}")
print(f"25th percentile: {np.percentile(raw_output, 25):.6f}")
print(f"50th percentile: {np.percentile(raw_output, 50):.6f}")
print(f"75th percentile: {np.percentile(raw_output, 75):.6f}")

# Analyze sigmoid output (probabilities)
print('\nSigmoid output (probabilities):')
print(f"Min: {sigmoid_output.min():.6f}")
print(f"Max: {sigmoid_output.max():.6f}")
print(f"Mean: {sigmoid_output.mean():.6f}")
print(f"Std: {sigmoid_output.std():.6f}")
print(f"25th percentile: {np.percentile(sigmoid_output, 25):.6f}")
print(f"50th percentile: {np.percentile(sigmoid_output, 50):.6f}")
print(f"75th percentile: {np.percentile(sigmoid_output, 75):.6f}")

# Analyze threshold effects
print('\nPercentage of pixels above thresholds:')
thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for threshold in thresholds:
    above_threshold = (sigmoid_output > threshold).sum() / sigmoid_output.size * 100
    print(f"Threshold {threshold:.2f}: {above_threshold:.2f}% of pixels classified as change")

# Try multiple random inputs to see if output range is consistent
print("\nTesting 5 different random inputs to check consistency:")
for i in range(5):
    # Generate new random inputs
    t1 = torch.rand(1, 3, 512, 512)
    t2 = torch.rand(1, 3, 512, 512)
    
    # Run inference
    with torch.no_grad():
        output = model(t1, t2)
        sigmoid = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Print statistics
    print(f"Run {i+1}: Min={sigmoid.min():.6f}, Max={sigmoid.max():.6f}, Mean={sigmoid.mean():.6f}")