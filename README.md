# Satellite Change Detection System

Official runtime path: `simple_main.py` (lightweight baseline pipeline).

This project supports a complete end-to-end flow:

- dataset preparation (sample or real)
- training
- inference
- evaluation

## Requirements

- Python 3.10+
- Recommended: virtual environment

Install base dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Full Pipeline (your dataset already available)

If your dataset is already present in `data/`, run everything with one command:

```powershell
python simple_main.py run-all-real --data-dir data
```

Use strict filename validation when needed:

```powershell
python simple_main.py run-all-real --data-dir data --strict
```

## Expected Dataset Layout

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

File names should align across `before`, `after`, and `labels` within each split.

## Real Dataset Download (optional)

If you want the CLI to download/export LEVIR-CD+ first:

```powershell
python -m pip install datasets
python simple_main.py run-all-real --download --data-dir data
```

Only download/export the real dataset (without train/infer/eval):

```powershell
python simple_main.py prepare-real --data-dir data
```

## Sample Dataset Pipeline

Run fully on generated synthetic data:

```powershell
python simple_main.py run-all
```

Or step-by-step:

```powershell
python simple_main.py prepare-sample
python simple_main.py train
python simple_main.py infer
python simple_main.py evaluate
```

## Outputs

Default output locations:

- Model: `outputs/models/baseline_model.json`
- Predictions: `outputs/predictions/`
- Evaluation JSON: `outputs/evaluation/evaluation_results.json`

## Command Summary

- `prepare-sample` – generate synthetic train/test data
- `train` – train threshold baseline model
- `infer` – run predictions on test split
- `evaluate` – compute metrics from predictions vs labels
- `prepare-real` – download/export LEVIR-CD+
- `run-all` – full sample-data pipeline
- `run-all-real` – full real-data pipeline

## Optional Advanced Stack

The `changedetect/src/` workflow remains available for advanced experimentation.

Install full dependencies:

```powershell
python -m pip install -r requirements-full.txt
```

## License

MIT
