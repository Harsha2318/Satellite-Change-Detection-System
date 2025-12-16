# Satellite Image Change Detection

A deep learning-based project for detecting changes between satellite imagery pairs using a Siamese U-Net architecture. This system leverages convolutional neural networks to identify and map temporal changes in satellite imagery with high precision.

## Features

- **Siamese U-Net Architecture**: Dual-path encoder-decoder network for change detection
- **Multi-spectral Support**: Handles multi-band satellite imagery (RGB, Multispectral, SAR)
- **Batch Processing**: Efficient inference on large imagery datasets
- **Geospatial Integration**: CRS-aware processing with coordinate system support
- **Visualization Tools**: Change map generation and overlay visualization
- **Docker Support**: Containerized deployment for reproducibility

## Project Structure

```
changedetect/
├── src/
│   ├── data/                   # Data handling modules
│   │   ├── dataset.py          # Dataset creation and loading
│   │   └── preprocess.py       # Data preprocessing utilities
│   │
│   ├── models/                 # Model architectures
│   │   └── siamese_unet.py     # Siamese U-Net implementation
│   │
│   ├── utils/                  # Utility functions
│   │   ├── geoutils.py         # Geospatial utilities
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── visualization.py    # Visualization utilities
│   │
│   ├── train.py                # Training script
│   ├── inference.py            # Inference script
│   ├── evaluate.py             # Model evaluation
│   └── main.py                 # Entry point
│
├── notebooks/                  # Jupyter notebooks
│   └── change_detection_demo.ipynb
│
├── tests/                      # Unit tests
│
├── training_runs/              # Model checkpoints and logs
│
├── docs/                       # Documentation
│   └── visualization.md
│
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/Harsha2318/Satellite-Change-Detection-System.git
cd Satellite-Change-Detection-System

# Create conda environment
conda create -n changedetect python=3.9 -y
conda activate changedetect

# Install dependencies
pip install -r changedetect/requirements.txt
```

### Docker Setup

```bash
# Build and run with Docker
cd changedetect
docker-compose up --build
```

## Usage

### Training

```bash
cd changedetect
python -m src.main train \
  --image_dir ../data/train \
  --mask_dir ../data/train/labels \
  --num_epochs 100 \
  --batch_size 16
```

### Inference

```bash
python -m src.main inference \
  --image_dir ../data/test \
  --model_path models/best_model.pth \
  --output_dir predictions/
```

### Evaluation

```bash
python -m src.main evaluate \
  --pred_dir predictions/ \
  --gt_dir ../data/test/labels/ \
  --output_dir evaluation/
```

### Interactive Demo

```bash
cd changedetect
jupyter notebook notebooks/change_detection_demo.ipynb
```

## Model Architecture

The Siamese U-Net architecture consists of:

1. **Shared Encoder**: Feature extraction from both image timesteps
2. **Skip Connections**: Multi-scale feature preservation
3. **Decoder**: Progressive upsampling with change detection
4. **Output Layer**: Pixel-wise change probability maps

## Key Technologies

- **Deep Learning**: PyTorch
- **Geospatial**: Rasterio, GDAL, Geopandas
- **Processing**: NumPy, Pandas, Scikit-image
- **Visualization**: Matplotlib, Folium
- **Containerization**: Docker, Docker Compose

## Performance

The model achieves strong performance on change detection tasks:
- Precision/Recall balanced for false positive minimization
- Handles various temporal scales
- Robust to illumination and seasonal variations

## Citation

If you use this project in your research, please cite:

```bibtex
@software{changedetection2024,
  title={Satellite Image Change Detection System},
  author={Harsha},
  url={https://github.com/Harsha2318/Satellite-Change-Detection-System},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please follow the standard GitHub workflow for PRs.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Acknowledgements

Built with PyTorch, Rasterio, and the open-source geospatial community.

- [U-Net Architecture](https://arxiv.org/abs/1505.04597)
- [Siamese Networks for Change Detection](https://ieeexplore.ieee.org/document/8451652)