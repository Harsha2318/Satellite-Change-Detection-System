# Satellite Image Change Detection

An end-to-end solution for detecting man-made changes in satellite imagery pairs.

## Problem Overview

This solution addresses the PS-10 Change Detection challenge which requires:
- Comparing two satellite images from the same sensor and location at different time periods
- Detecting man-made changes between the image pairs
- Generating change masks in both GeoTIFF and vector (shapefile) formats
- Working with different terrain types (snow, plains, hills, desert, forest, urban)
- Processing multi-spectral satellite data

## Solution Architecture

Our solution employs a Siamese U-Net architecture for detecting changes between satellite image pairs. The pipeline consists of several integrated modules:

### 1. Data Processing Pipeline

- **Download Module**: Utilities for downloading satellite imagery from Copernicus and other sources
- **Preprocessing Module**: 
  - Image co-registration using phase correlation and feature-based methods
  - Tiling large images with overlap for efficient processing
  - Cloud masking for Sentinel-2 imagery
  - Radiometric normalization to handle different lighting conditions
  - Image fusion for multi-sensor data (Stage 2/3)

### 2. Deep Learning Model Architecture

- **Siamese Encoder Network**: 
  - Shared weights for processing both temporal images
  - Feature extraction at multiple scales
  - Support for multi-spectral input (RGB, NIR, etc.)

- **Change Detection Decoder**:
  - Skip connections to preserve spatial details
  - Multi-scale feature fusion
  - Attention mechanisms to focus on change areas
  - Binary segmentation head for change/no-change classification

- **Model Variants**:
  - `siamese_unet`: Full Siamese architecture with shared encoder and difference-based skip connections
  - `siamese_diff`: Early fusion by concatenating both images
  - `fcn_diff`: Lightweight FCN model for faster inference

### 3. Post-processing Module

- Morphological operations to remove noise
- Small area filtering to eliminate false positives
- Boundary smoothing for more natural change boundaries
- Vectorization to create shapefiles with attribute information
- Simplification of vector geometries for efficient storage

### 4. Evaluation System

- IoU (Intersection over Union) calculation
- Precision, recall, and F1-score metrics
- Vector-based accuracy assessment
- Confusion matrix generation
- Runtime and memory profiling

## Installation

### Method 1: Using pip

```bash
# Create a Python 3.10+ environment
conda create -n changedetect python=3.10
conda activate changedetect

# Clone this repository
git clone https://github.com/yourusername/changedetect.git
cd changedetect

# Install dependencies
pip install -r requirements.txt

# For GDAL/rasterio on Windows, use conda
conda install -c conda-forge gdal rasterio
```

### Method 2: Using Docker

```bash
# Build the Docker image
docker build -t ps10-changedetect:latest .

# Or use docker-compose to build
docker-compose build
```

## Project Structure

```
changedetect/
├── src/
│   ├── data/                # Data handling modules
│   │   ├── download.py      # Satellite imagery download utilities
│   │   ├── preprocess.py    # Image preprocessing functions
│   │   ├── dataset.py       # PyTorch dataset classes
│   │   └── tile.py          # Image tiling functions
│   │
│   ├── models/              # Deep learning models
│   │   ├── unet.py          # Base U-Net implementation
│   │   └── siamese_unet.py  # Siamese U-Net for change detection
│   │
│   ├── utils/               # Utility functions
│   │   ├── geoutils.py      # Geospatial utilities
│   │   ├── md5_utils.py     # MD5 hash utilities for model verification
│   │   ├── metrics.py       # Evaluation metrics
│   │   └── postprocessing.py # Post-processing functions
│   │
│   ├── train.py             # Training script
│   ├── inference.py         # Inference script
│   ├── evaluate.py          # Evaluation script
│   └── main.py              # Main entry point
│
├── data/                    # Data directory (not in git)
│   ├── raw/                 # Raw satellite images
│   ├── processed/           # Preprocessed images
│   ├── train/               # Training data
│   └── test/                # Test data
│
├── outputs/                 # Output directory (not in git)
│   ├── models/              # Trained models
│   ├── predictions/         # Prediction results
│   └── evaluation/          # Evaluation results
│
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Usage

### Using the Main CLI Interface

Our solution provides a unified command-line interface (CLI) for all operations:

```bash
# Get help
python -m changedetect.src.main --help

# List available commands
python -m changedetect.src.main
```

### Data Download and Preparation

```bash
# List sample locations from the PS-10 problem statement
python -m changedetect.src.main download --list-locations

# Download Sentinel-2 imagery
python -m changedetect.src.main download \
  --source sentinel2 \
  --output ./data/raw \
  --lat 28.1740 --lon 77.6126 \
  --start-date 2023-01-01 --end-date 2023-02-01 \
  --cloud-cover 10
```

### Model Training

```bash
# Train the Siamese U-Net model
python -m changedetect.src.main train \
  --image_dir ./data/train/images \
  --mask_dir ./data/train/masks \
  --output_dir ./outputs/models \
  --model_type siamese_unet \
  --in_channels 3 \
  --batch_size 16 \
  --num_epochs 100

# Resume training from a checkpoint
python -m changedetect.src.main train \
  --image_dir ./data/train/images \
  --mask_dir ./data/train/masks \
  --output_dir ./outputs/models \
  --resume ./outputs/models/checkpoint_epoch_50.pth
```

### Inference

```bash
# Run inference with a trained model
python -m changedetect.src.main inference \
  --image_dir ./data/test/images \
  --model_path ./outputs/models/best_model.pth \
  --output_dir ./outputs/predictions \
  --model_type siamese_unet \
  --in_channels 3
  
# Run inference without vector output
python -m changedetect.src.main inference \
  --image_dir ./data/test/images \
  --model_path ./outputs/models/best_model.pth \
  --output_dir ./outputs/predictions \
  --no_vector
```

### Evaluation

```bash
# Evaluate predictions against ground truth
python -m changedetect.src.main evaluate \
  --pred_dir ./outputs/predictions \
  --gt_dir ./data/test/masks \
  --output_dir ./outputs/evaluation

# Include vector evaluation
python -m changedetect.src.main evaluate \
  --pred_dir ./outputs/predictions \
  --gt_dir ./data/test/masks \
  --output_dir ./outputs/evaluation \
  --vector
```

### Using Docker

```bash
# Run training with Docker
docker-compose run train

# Run inference with Docker
docker-compose run inference

# Run evaluation with Docker
docker-compose run evaluate
```

## Solution Technical Details

### Model Architecture

Our Siamese U-Net architecture consists of:
- Shared encoder: Modified ResNet or U-Net encoder (configurable)
- Skip connections: Using difference or concatenation operations
- Feature fusion: Bottleneck fusion module for temporal feature integration
- Output head: Segmentation layer for binary change detection

### Training Strategy

- Loss function: Combined BCE and Dice Loss
- Optimizer: Adam with learning rate scheduling
- Data augmentation: Rotation, flipping, brightness adjustments
- Early stopping: Based on validation IoU
- Checkpointing: Save best model by IoU score
- Mixed precision training for efficiency

### Performance Metrics

- IoU (Jaccard Index): Target >0.75 on test set
- F1-Score: Target >0.80 on test set
- Average inference time: ~3-5 seconds per 1024x1024 tile on GPU
- Memory usage: ~4GB for inference on 2048x2048 images

### Hardware Requirements

- Training: CUDA-capable GPU with 8GB+ VRAM
- Inference: CUDA-capable GPU with 4GB+ VRAM
- CPU-only inference is supported but significantly slower

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.