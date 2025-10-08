# Satellite Image Change Detection

A deep learning-based project for detecting changes between satellite imagery pairs using a Siamese U-Net architecture.

## Project Structure

```
changedetect/
│
├── data/                       # Data directory
│   ├── processed/              # Processed data ready for training/inference
│   └── raw/                    # Raw satellite imagery
│
├── docker/                     # Docker configuration
│   ├── docker-compose.yml
│   └── Dockerfile
│
├── docs/                       # Documentation
│   └── visualization.md
│
├── experiments/                # Experiment tracking and configs
│
├── notebooks/                  # Jupyter notebooks
│   └── change_detection_demo.ipynb
│
├── outputs/                    # Output directory for visualizations and results
│
├── predictions/                # Change detection prediction outputs
│
├── scripts/                    # Utility scripts
│
├── src/                        # Source code
│   ├── data/
│   │   ├── dataset.py          # Dataset creation and loading
│   │   ├── download.py         # Satellite imagery download utilities
│   │   └── preprocess.py       # Data preprocessing
│   │
│   ├── models/
│   │   └── siamese_unet.py     # Siamese U-Net model architecture
│   │
│   ├── utils/
│   │   ├── geoutils.py         # Geospatial utilities
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── postprocessing.py   # Result post-processing
│   │
│   ├── inference.py            # Inference script
│   └── train.py                # Training script
│
└── training_runs/              # Model checkpoints and training logs
    └── run1_small/             # Example training run
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/satellite-change-detection.git
cd satellite-change-detection

# Create and activate a conda environment
conda create -n changedetect python=3.9
conda activate changedetect

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python -m changedetect.src.train --config configs/train_config.json
```

### Inference

```bash
python -m changedetect.src.inference --image_dir data/processed/test_pairs --output_dir predictions
```

### Demo Notebook

See `notebooks/change_detection_demo.ipynb` for an interactive demonstration of the change detection pipeline.

## Model Analysis

Recent model analysis revealed some issues with the current model:

1. The model outputs probabilities in a very narrow range (0.097-0.206)
2. Training was insufficient (only 1 epoch)
3. The model shows poor discrimination ability

We recommend retraining the model with more epochs (50-100) for better performance.

## License

[MIT License](LICENSE)

## Acknowledgements

- [U-Net Architecture](https://arxiv.org/abs/1505.04597)
- [Siamese Networks for Change Detection](https://ieeexplore.ieee.org/document/8451652)