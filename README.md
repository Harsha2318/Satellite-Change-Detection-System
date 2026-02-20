# Satellite Change Detection System

Recommended runtime: lightweight baseline CLI.

This repository now has one official beginner-friendly execution path that runs end-to-end with minimal dependencies:

- prepare sample dataset
- train model
- run inference
- run evaluation

## Recommended way to run the project

Use `simple_main.py`.

## Python version

Use Python 3.10 to 3.12 (also works with newer versions if `numpy` and `Pillow` wheels are available).

## Install

```powershell
python -m pip install -r requirements.txt
```

## Quick Start (fully working)

### Option A: One-command end-to-end run

```powershell
python simple_main.py run-all
```

This generates sample data, trains a baseline model, runs inference, and evaluates results.

### Option B: Step-by-step

```powershell
python simple_main.py prepare-sample
python simple_main.py train
python simple_main.py infer
python simple_main.py evaluate
```

## Outputs

After running, you will have:

- Trained model: `outputs/models/baseline_model.json`
- Prediction masks and probability maps: `outputs/predictions/`
- Metrics JSON: `outputs/evaluation/evaluation_results.json`

## Using your own dataset

Expected structure:

```text
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

Image names must match across `before`, `after`, and `labels` within each split.

Then run:

```powershell
python simple_main.py run-all-real --data-dir data
```

If your folders are not perfectly aligned and you still want to run on overlapping filenames only, keep default non-strict mode. To enforce exact matching, add `--strict`.

### Download real dataset (LEVIR-CD+) and run fully

If you want the project to fetch and export the Hugging Face dataset first:

```powershell
python -m pip install datasets
python simple_main.py run-all-real --download --data-dir data
```

Or only export dataset files:

```powershell
python simple_main.py prepare-real --data-dir data
```

## Optional advanced path

The `changedetect/src/` stack is kept for advanced geospatial/deep-learning experiments and is not the default supported path.

If you need it, install:

```powershell
python -m pip install -r requirements-full.txt
```

## Backward compatibility

Legacy scripts are still present and redirect to the recommended CLI:

- `simple_train.py`
- `simple_inference.py`
- `simple_evaluate.py`

## License

MIT
