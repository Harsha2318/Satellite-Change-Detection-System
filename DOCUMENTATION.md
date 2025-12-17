# Satellite Image Change Detection System - Full Documentation

## Overview

This project implements a deep learning-based system for detecting changes in satellite imagery using a Siamese U-Net architecture. It supports multi-spectral satellite data and includes end-to-end workflows for training, inference, and evaluation.

## Quick Start (Lightweight)

If you're on Windows or want to avoid installing heavy geospatial dependencies, use the lightweight scripts at repository root:

```powershell
# Train
python simple_train.py

# Inference (batch)
python simple_inference.py batch

# Evaluate
python simple_evaluate.py
```

These scripts use `Pillow` and `NumPy` only and are the recommended way to try the system quickly.

> Note: The full CLI under `changedetect/src/` provides more features (rasterio, skimage, Docker), but may require extra packages on Windows.

## System Architecture

### 1. Model Architecture: Siamese U-Net

The system uses a Siamese U-Net that processes two satellite images independently through a shared encoder before detecting changes:

```
Before Image ──────┐
                   ├─→ Shared Encoder ──→ Feature Fusion ──→ Decoder ──→ Change Map
After Image ───────┘
                        (Shared Weights)
```

**Components:**
- **Shared Encoder**: Extracts features from both images using identical weights
- **Fusion Module**: Concatenates feature maps from both paths
- **Decoder**: Upsamples fused features to original resolution
- **Output Layer**: Produces pixel-wise change probability map

### 2. Key Features

- **Multi-resolution Processing**: Handles 256×256 to large tiles with overlapping
- **Data Augmentation**: Rotation, flipping, elastic deformation, brightness/contrast
- **Loss Functions**: Dice Loss + Binary Cross-Entropy for balanced training
- **Metrics**: IoU, Dice, Precision, Recall, F1-Score
- **Geospatial Support**: CRS-aware processing with rasterio/GDAL
- **Distributed Training**: Multi-GPU support via PyTorch
- **Visualization**: Change maps, uncertainty maps, temporal overlays

## Installation & Setup

### Quick Start (Conda)

```bash
git clone https://github.com/Harsha2318/Satellite-Change-Detection-System.git
cd Satellite-Change-Detection-System

conda create -n changedetect python=3.9 -y
conda activate changedetect

pip install -e changedetect/
# Or: pip install -r requirements.txt
```

### Docker Setup

```bash
cd changedetect
docker build -t changedetect:latest .
docker-compose up --build train
```

### Conda from Scratch (MacOS/Linux)

```bash
conda env create -f environment.yml
conda activate changedetect
```

## Data Preparation

### Required Directory Structure

```
data/
├── train/
│   ├── before/           # Time 1 images (RGB or multispectral)
│   │   ├── location1.tif
│   │   └── location2.tif
│   ├── after/            # Time 2 images (same locations, later time)
│   │   ├── location1.tif
│   │   └── location2.tif
│   └── labels/           # Change labels (binary masks)
│       ├── location1.tif
│       └── location2.tif
├── val/
│   └── (same structure)
└── test/
    ├── before/
    └── after/
```

### Data Preparation Script

```bash
python data_prep.py /path/to/raw/images data/
```

This script:
1. Organizes image pairs into train/val/test
2. Validates data structure
3. Creates required directories

### Image Format Requirements

- **Format**: GeoTIFF (.tif) recommended for geospatial data, PNG/JPEG for standard imagery
- **Size**: Minimum 256×256 pixels
- **Channels**: 
  - RGB: 3 channels
  - Multispectral: 4+ channels
  - Grayscale: 1 channel
- **Data Type**: uint8 (0-255) or float32 (0.0-1.0)
- **Labels**: Binary (0=no change, 1=change)

## Training

### Basic Training

```bash
cd changedetect

python -m src.main train \
  --image_dir ../data/train \
  --mask_dir ../data/train/labels \
  --output_dir models/ \
  --num_epochs 100 \
  --batch_size 32
```

### Advanced Training with Config

Create `config.yaml`:

```yaml
model:
  type: siamese_unet
  in_channels: 3
  features: 64
  dropout: 0.2

training:
  epochs: 200
  batch_size: 16
  learning_rate: 0.0005
  weight_decay: 0.0001
  
data:
  tile_size: 256
  overlap: 32
  augment: true
```

Then:
```bash
python -m src.main train --config config.yaml
```

### Training Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `num_epochs` | 100 | 50-500 | Increase for better accuracy |
| `batch_size` | 16 | 8-128 | Reduce if GPU memory issues |
| `learning_rate` | 0.001 | 0.0001-0.01 | Lower for fine-tuning |
| `weight_decay` | 0.0001 | 0-0.001 | L2 regularization strength |
| `tile_size` | 256 | 128-512 | Adjust based on GPU memory |
| `dropout` | 0.2 | 0-0.5 | Increase for better regularization |

### Training Outputs

```
models/
├── best_model.pth          # Best model checkpoint
├── last_model.pth          # Last epoch checkpoint
├── training_history.json   # Loss/metric history
└── config.json            # Training configuration
```

## Inference

### Single Image Directory

```bash
python -m src.main inference \
  --image_dir ../data/test \
  --model_path models/best_model.pth \
  --output_dir predictions/
```

### Batch Inference (Multiple Locations)

```bash
python -m src.main inference \
  --image_dir ../data/test \
  --model_path models/best_model.pth \
  --output_dir predictions/ \
  --batch_size 8
```

### Output Interpretation

- **Output Range**: 0.0 to 1.0 (probability of change)
- **Threshold**: Default 0.5 (adjust based on precision/recall needs)
- **Binary Mask**: Pixels ≥ threshold marked as change
- **Confidence Map**: Optional probabilistic output map

## Evaluation

### Metrics Computation

```bash
python -m src.main evaluate \
  --pred_dir predictions/ \
  --gt_dir ../data/test/labels/ \
  --output_dir evaluation/
```

### Available Metrics

- **Intersection over Union (IoU)**: Ratio of overlap to union
- **Dice Score**: 2×TP/(2×TP+FP+FN)
- **Precision**: TP/(TP+FP)
- **Recall**: TP/(TP+FN)
- **F1-Score**: 2×(Precision×Recall)/(Precision+Recall)
- **Confusion Matrix**: TP, TN, FP, FN
- **AUC-ROC**: Receiver Operating Characteristic curve

### Interpreting Results

```json
{
  "overall": {
    "iou": 0.823,
    "dice": 0.902,
    "precision": 0.915,
    "recall": 0.890,
    "f1": 0.902
  },
  "per_image": [...]
}
```

**Good Performance Indicators:**
- IoU > 0.80
- Dice > 0.85
- F1 > 0.85
- Balanced Precision/Recall

## Visualization

The inference outputs can be visualized using the output files. Use external tools like QGIS or Rasterio for visualization.
2. **Overlay**: Change map overlay on satellite imagery
3. **Comparison**: Side-by-side before/after with change highlights
4. **Uncertainty**: Pixel-wise uncertainty/confidence maps

## Project Structure

```
changedetect/
├── src/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── config.py            # Configuration defaults
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset class
│   │   ├── preprocess.py    # Preprocessing utilities
│   │   └── tile.py          # Tiling utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── siamese_unet.py  # Siamese U-Net
│   │   └── unet.py          # Base U-Net
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py       # Evaluation metrics
│   │   ├── geoutils.py      # Geospatial utilities
│   │   ├── visualization.py # Visualization
│   │   └── postprocessing.py # Post-processing
│   │
│   ├── train.py             # Training script
│   ├── inference.py         # Inference script
│   └── evaluate.py          # Evaluation script
│
├── notebooks/
│   └── change_detection_demo.ipynb
│
├── tests/
│   ├── __init__.py
│   └── test_models.py
│
├── docs/
│   ├── GETTING_STARTED.md
│   └── visualization.md
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### Multi-GPU Training

```bash
cd changedetect
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  -m src.main train \
  --image_dir ../data/train \
  --mask_dir ../data/train/labels
```

### Custom Loss Functions

Edit `src/train.py` to implement custom losses:

```python
class CustomLoss(nn.Module):
    def forward(self, pred, target):
        # Custom loss implementation
        return loss
```

### Model Export

```python
# Convert to ONNX
torch.onnx.export(model, (x1, x2), "model.onnx")

# Convert to TorchScript
scripted = torch.jit.script(model)
scripted.save("model.pt")
```

### Fine-tuning Pretrained Model

```bash
cd changedetect
python -m src.main train \
  --image_dir ../data/train \
  --mask_dir ../data/train/labels \
  --model_path models/base_model.pth
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` or `tile_size` |
| ImportError (rasterio) | `conda install -c conda-forge rasterio` |
| Poor accuracy | Increase `epochs`, check data quality |
| Slow training | Use GPU, increase `num_workers` |
| CRS mismatch | Ensure consistent coordinate systems |

### Debug Mode

```bash
python -m src.main train --debug --verbose
```

## Performance Optimization

### Training Speed

1. **Increase Batch Size**: 32 → 64 (if GPU memory allows)
2. **Mixed Precision**: Add `--mixed_precision` flag
3. **DataLoader Workers**: Increase `num_workers` to 4-8
4. **Pin Memory**: Set `pin_memory=True`

### Model Accuracy

1. **Ensemble**: Train multiple models and average predictions
2. **Augmentation**: Increase augmentation diversity
3. **Loss Tuning**: Adjust dice_weight vs bce_weight
4. **Data**: Use more training samples
5. **Epochs**: Train longer (100 → 200+)

### Inference Speed

1. **Batch Processing**: Process multiple images simultaneously
2. **TorchScript**: Compile model with `torch.jit`
3. **Quantization**: Use INT8 quantization
4. **Model Pruning**: Remove less important connections

## Citation

If you use this project in research, please cite:

```bibtex
@software{changedetection2024,
  title={Satellite Image Change Detection System},
  author={Harsha},
  url={https://github.com/Harsha2318/Satellite-Change-Detection-System},
  year={2024}
}
```

## References

- U-Net: [Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Siamese Networks: [Learning a Similarity Metric Discriminatively](https://ieeexplore.ieee.org/document/1640964)
- Change Detection: [A Survey on Change Detection for Optical Imagery](https://ieeexplore.ieee.org/document/8451652)

## License

MIT License - See LICENSE file for details

## Support

- **Documentation**: See `docs/` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: harsha@example.com (replace with actual contact)

## Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## Acknowledgments

- PyTorch team for deep learning framework
- Rasterio team for geospatial I/O
- GeoPandas for vector operations
- OpenCV for image processing
