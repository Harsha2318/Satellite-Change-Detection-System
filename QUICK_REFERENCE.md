# Quick Reference Guide

## Installation

```powershell
git clone https://github.com/Harsha2318/Satellite-Change-Detection-System.git
cd Satellite-Change-Detection-System
conda create -n changedetect python=3.9 -y
conda activate changedetect
# For full/production features (geospatial libs)
pip install -r requirements.txt
# Or (editable install for development):
pip install -e changedetect/
```

## Basic Commands

### Training
```powershell
# Quick (recommended on Windows): run the lightweight script from repository root
python simple_train.py

# Full/production (requires dependencies):
cd changedetect
python -m src.main train \
  --image_dir /path/to/images \
  --mask_dir /path/to/masks \
  --output_dir models \
  --num_epochs 100 \
  --batch_size 16
```

### Inference
```powershell
# Quick (recommended):
python simple_inference.py batch

# Full/production:
python -m src.main inference \
  --image_dir /path/to/test/images \
  --model_path models/best_model.pth \
  --output_dir predictions/
```

### Evaluation
```bash
python -m src.main evaluate \
  --pred_dir predictions/ \
  --gt_dir /path/to/ground_truth/ \
  --output_dir evaluation/
```

## Data Structure

```
data/
├── train/
│   ├── before/
│   ├── after/
│   └── labels/
├── val/
│   ├── before/
│   ├── after/
│   └── labels/
└── test/
    ├── before/
    └── after/
```

## Docker

```powershell
# Docker: Dockerfile is located inside the `changedetect/` folder. Example:
docker build -t changedetect:latest changedetect/

# Run (mount data directory):
docker run -v ${PWD}/data:/data changedetect:latest \
  python -m src.main train --image_dir /data/train --mask_dir /data/train/labels --output_dir /data/models

# Docker Compose (from changedetect/)
cd changedetect
docker-compose up --build
```

## Configuration

Edit `changedetect/src/config.py` to change default parameters:

- `TRAINING_CONFIG`: Learning rate, epochs, batch size
- `MODEL_CONFIG`: Architecture parameters
- `DATA_CONFIG`: Data loading parameters
- `LOSS_CONFIG`: Loss function parameters

## File Structure

```
changedetect/
├── src/
│   ├── data/         # Dataset and preprocessing
│   ├── models/       # Neural network architectures
│   ├── utils/        # Utilities (metrics, visualization, etc.)
│   ├── train.py      # Training script
│   ├── inference.py  # Inference script
│   ├── evaluate.py   # Evaluation script
│   ├── main.py       # CLI entry point
│   └── config.py     # Configuration
├── notebooks/        # Jupyter notebooks
├── tests/           # Unit tests
├── docs/            # Documentation
├── requirements.txt # Python dependencies
├── Dockerfile       # Docker configuration
└── docker-compose.yml
```

## Model Architecture

**Siamese U-Net**: 
- Shared encoder processes both images
- Feature fusion module concatenates encodings
- Decoder produces change probability maps
- Output: (B, 1, H, W) binary change map

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Number of training epochs |
| `batch_size` | 32 | Batch size for training |
| `learning_rate` | 0.001 | Initial learning rate |
| `tile_size` | 256 | Size of image tiles |
| `overlap` | 32 | Tile overlap in pixels |
| `features` | 64 | Initial feature channels |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size or tile_size |
| Poor accuracy | Increase epochs, check data preprocessing |
| Import errors | `pip install -e changedetect/` |
| GDAL errors | `conda install -c conda-forge gdal` |
| Slow training | Use GPU, increase num_workers |

## Performance Tips

1. **Data**: Ensure consistent radiometric correction
2. **Augmentation**: Use albumentations for better generalization
3. **Normalization**: Apply Z-score or min-max normalization
4. **Loss**: Use Dice + BCE for balanced results
5. **Threshold**: Adjust based on precision/recall needs

## Resources

- **Documentation**: `changedetect/docs/GETTING_STARTED.md`
- **Demo**: `changedetect/notebooks/change_detection_demo.ipynb`
- **Visualization**: `changedetect/docs/visualization.md`
- **GitHub**: https://github.com/Harsha2318/Satellite-Change-Detection-System
