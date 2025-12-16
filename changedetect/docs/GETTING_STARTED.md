# Getting Started with Change Detection

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Harsha2318/Satellite-Change-Detection-System.git
cd Satellite-Change-Detection-System

# Create environment
conda create -n changedetect python=3.9 -y
conda activate changedetect

# Install package
pip install -e changedetect/
```

### 2. Data Preparation

Place your satellite imagery in the following structure:

```
data/
├── train/
│   ├── before/
│   │   ├── image1.tif
│   │   └── image2.tif
│   ├── after/
│   │   ├── image1.tif
│   │   └── image2.tif
│   └── labels/
│       ├── image1_mask.tif
│       └── image2_mask.tif
├── val/
│   └── (same structure)
└── test/
    ├── before/
    └── after/
```

### 3. Training

```bash
cd changedetect

python -m src.main train \
  --image_dir ../data/train \
  --mask_dir ../data/train/labels \
  --output_dir models \
  --num_epochs 100 \
  --batch_size 32
```

### 4. Inference

```bash
python -m src.main inference \
  --image_dir ../data/test \
  --model_path models/best_model.pth \
  --output_dir predictions
```

### 5. Evaluation

```bash
python -m src.main evaluate \
  --pred_dir predictions/ \
  --gt_dir ../data/test/labels/ \
  --output_dir evaluation
```

## Advanced Usage

### Command Line Interface

```bash
python -m src.main --help
```

Available commands:
- `train`: Train a change detection model
- `inference`: Run inference on image pairs
- `evaluate`: Evaluate predictions against ground truth
- `visualize`: Create visualization overlays

### Docker Usage

```bash
cd changedetect

# Build image
docker build -t changedetect:latest .

# Run training
docker run -v $(pwd)/data:/workspace/data \
           -v $(pwd)/models:/workspace/models \
           changedetect:latest \
           python -m src.main train --image_dir data/train --mask_dir data/train/labels --output_dir models

# Or use docker-compose
docker-compose up train
```

## Configuration

Create a `config.yaml` file to customize training parameters:

```yaml
model:
  type: siamese_unet
  in_channels: 3
  out_channels: 1
  features: 64
  bilinear: true
  dropout: 0.2

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adam

data:
  tile_size: 256
  overlap: 32
  augment: true

loss:
  type: dice_bce
  dice_weight: 0.5
  bce_weight: 0.5
```

Then run with config:

```bash
python -m src.main train --config config.yaml
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size`
- Use smaller `tile_size`
- Enable gradient checkpointing

### Poor Results
- Increase `epochs` (try 100-200)
- Check data preprocessing
- Verify coordinate system (CRS) alignment
- Ensure temporal consistency between image pairs

### ImportError for rasterio/GDAL
```bash
conda install -c conda-forge rasterio gdal
```

## Model Architecture

The Siamese U-Net processes two images independently through a shared encoder, then fuses the feature maps and decodes to produce a pixel-wise change probability map.

```
Image 1 ──┐
          ├─→ Shared Encoder ──→ Fusion ──→ Decoder ──→ Change Map
Image 2 ──┘
```

## Performance Tips

1. **Preprocessing**: Ensure consistent radiometric correction between images
2. **Augmentation**: Use albumentations for robust training
3. **Normalization**: Apply consistent normalization (e.g., Z-score)
4. **Validation**: Monitor validation metrics throughout training
5. **Threshold Selection**: Adjust decision threshold based on precision/recall needs

## Next Steps

- See `notebooks/change_detection_demo.ipynb` for interactive examples
- Check `docs/visualization.md` for visualization techniques
- Review test cases in `tests/` for usage examples

## Support

For issues or questions, open an issue on GitHub:
https://github.com/Harsha2318/Satellite-Change-Detection-System/issues
