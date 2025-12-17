# Satellite Image Change Detection

Lightweight, dependency-tolerant satellite change detection system with two usage paths:

- Quick path: `simple_train.py`, `simple_inference.py`, `simple_evaluate.py` (works on Windows without heavy geospatial deps)
- Full path: `changedetect/src/` production code (Siamese U-Net, Docker support, geospatial utilities)

This repository contains an end-to-end pipeline for training, inference, and evaluation of pixel-wise change detection models.

## Quick Start (recommended for most users)

1. Prepare your data under `data/`:

```
data/
├── train/
│   ├── before/
│   ├── after/
│   └── labels/
└── test/
    ├── before/
    ├── after/
    └── labels/
```

2. Train with the simple script (no heavy deps required):

```powershell
python simple_train.py
```

3. Run batch inference:

```powershell
python simple_inference.py batch
```

4. Evaluate results:

```powershell
python simple_evaluate.py
```

These `simple_*` scripts use PIL + NumPy only and are the fastest way to get started on Windows.

## Full/Production Path

If you need geospatial features, Docker, or the full Siamese U-Net, use the `changedetect/` package:

Install dependencies (recommended in a conda env):

```powershell
conda create -n changedetect python=3.9 -y
conda activate changedetect
pip install -r requirements.txt
# or: pip install -e changedetect/
```

Notes:
- The `Dockerfile` and `docker-compose.yml` used for building containers are in the `changedetect/` folder. To build with Docker run:

```powershell
# from repository root
docker build -t changedetect:latest changedetect/
docker run -v ${PWD}/data:/data changedetect:latest python -m src.main train --image_dir /data/train --mask_dir /data/train/labels
```

- Docker must be installed and available in PATH. On Windows, use Docker Desktop.
- The full CLI (`python -m src.main`) may require additional geospatial packages (rasterio, scikit-image). If you see import errors, prefer the `simple_*` scripts.

## Project layout (high level)

```
.
├── changedetect/            # Production package (models, full CLI, Dockerfiles)
├── data/                    # Training / test data
├── predictions/             # Inference outputs
├── evaluation/              # Evaluation outputs
├── simple_train.py          # Lightweight training script (PIL + NumPy)
├── simple_inference.py      # Lightweight inference
├── simple_evaluate.py       # Lightweight evaluation
├── requirements.txt         # Optional: full project dependencies
└── README.md
```

## Troubleshooting & Notes

- If `docker build` fails with "no such file or directory", ensure you run the build from the repository root and use the `changedetect/` path (see command above).
- If `python -m src.main ...` raises import errors for `skimage` or `rasterio`, install the packages or use `simple_*` scripts instead.
- For Windows emoji/unicode issues the CLI uses plain ASCII output.

## License

MIT