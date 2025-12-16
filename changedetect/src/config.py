"""
Default configuration for change detection training
"""

# Model configuration
MODEL_CONFIG = {
    'type': 'siamese_unet',          # Model architecture
    'in_channels': 3,                 # Input channels (RGB)
    'out_channels': 1,                # Output channels (binary change)
    'features': 64,                   # Features in first layer
    'bilinear': True,                 # Use bilinear upsampling
    'dropout': 0.2,                   # Dropout rate
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 100,                    # Number of training epochs
    'batch_size': 32,                 # Batch size
    'learning_rate': 0.001,           # Initial learning rate
    'weight_decay': 0.0001,           # Weight decay (L2 regularization)
    'optimizer': 'adam',              # Optimizer type
    'scheduler': 'cosine',            # Learning rate scheduler
    'warmup_epochs': 5,               # Warmup epochs
    'gradient_clip': 1.0,             # Gradient clipping norm
}

# Data configuration
DATA_CONFIG = {
    'tile_size': 256,                 # Size of image tiles
    'overlap': 32,                    # Overlap between tiles
    'num_workers': 4,                 # DataLoader workers
    'pin_memory': True,               # Pin memory for DataLoader
    'augment': True,                  # Use data augmentation
    'normalize': True,                # Normalize images
    'use_rasterio': True,             # Use rasterio for GeoTIFF
}

# Loss configuration
LOSS_CONFIG = {
    'type': 'dice_bce',               # Loss function type
    'dice_weight': 0.5,               # Weight for Dice loss
    'bce_weight': 0.5,                # Weight for BCE loss
    'smooth': 1.0,                    # Smoothing factor
    'pos_weight': 1.0,                # Positive class weight
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.5,
    'rotation': 0.3,
    'scale': (0.8, 1.2),
    'elastic': 0.3,
    'perspective': 0.2,
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.1,
}

# Evaluation configuration
EVAL_CONFIG = {
    'metrics': ['iou', 'dice', 'precision', 'recall', 'f1'],
    'threshold': 0.5,                 # Change detection threshold
    'save_visualizations': True,
    'save_predictions': True,
}

# Default paths
DEFAULT_PATHS = {
    'data_dir': 'data',
    'output_dir': 'outputs',
    'model_dir': 'models',
    'log_dir': 'logs',
    'config_dir': 'configs',
}

# Device configuration
DEVICE_CONFIG = {
    'device': 'cuda',                 # 'cuda' or 'cpu'
    'num_gpus': 1,
    'distributed': False,
}
