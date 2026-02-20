"""Official lightweight CLI for end-to-end change detection workflow.

Recommended command interface for first-time users.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image

from baseline_pipeline import train_threshold_model, run_inference, evaluate_predictions


DEFAULT_MODEL_PATH = "outputs/models/baseline_model.json"
DEFAULT_PRED_DIR = "outputs/predictions"
DEFAULT_EVAL_PATH = "outputs/evaluation/evaluation_results.json"


def _mkdirs(base: Path) -> None:
    for rel in (
        "train/before",
        "train/after",
        "train/labels",
        "test/before",
        "test/after",
        "test/labels",
    ):
        (base / rel).mkdir(parents=True, exist_ok=True)


def _make_sample_pair(size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before = np.zeros((size, size, 3), dtype=np.uint8)
    before[:] = [rng.randint(60, 110), rng.randint(90, 140), rng.randint(60, 110)]
    noise = np.random.default_rng(rng.randint(0, 10_000)).integers(-20, 21, size=(size, size, 3), dtype=np.int16)
    before = np.clip(before.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    after = before.copy()
    mask = np.zeros((size, size), dtype=np.uint8)

    num_changes = rng.randint(1, 4)
    for _ in range(num_changes):
        x1 = rng.randint(0, size // 2)
        y1 = rng.randint(0, size // 2)
        w = rng.randint(size // 10, size // 4)
        h = rng.randint(size // 10, size // 4)
        x2 = min(size, x1 + w)
        y2 = min(size, y1 + h)
        color = np.array([rng.randint(120, 255), rng.randint(120, 255), rng.randint(120, 255)], dtype=np.uint8)
        after[y1:y2, x1:x2] = color
        mask[y1:y2, x1:x2] = 255

    return before, after, mask


def generate_sample_data(base_dir: Path, train_count: int, test_count: int, size: int, seed: int) -> None:
    rng = random.Random(seed)
    _mkdirs(base_dir)
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            before, after, mask = _make_sample_pair(size, rng)
            name = f"img_{index:03d}.png"
            Image.fromarray(before).save(base_dir / split / "before" / name)
            Image.fromarray(after).save(base_dir / split / "after" / name)
            Image.fromarray(mask).save(base_dir / split / "labels" / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight satellite change detection CLI")
    sub = parser.add_subparsers(dest="command")

    prep = sub.add_parser("prepare-sample", help="Generate synthetic sample dataset")
    prep.add_argument("--data-dir", default="data")
    prep.add_argument("--train-count", type=int, default=24)
    prep.add_argument("--test-count", type=int, default=8)
    prep.add_argument("--size", type=int, default=128)
    prep.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train", help="Train baseline model")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    train.add_argument("--strict", action="store_true", help="Require exact filename matching across before/after/labels")

    infer = sub.add_parser("infer", help="Run inference on test set")
    infer.add_argument("--data-dir", default="data")
    infer.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    infer.add_argument("--output-dir", default=DEFAULT_PRED_DIR)
    infer.add_argument("--strict", action="store_true", help="Require exact filename matching across before/after")

    evaluate = sub.add_parser("evaluate", help="Evaluate predictions against labels")
    evaluate.add_argument("--pred-dir", default=DEFAULT_PRED_DIR)
    evaluate.add_argument("--labels-dir", default="data/test/labels")
    evaluate.add_argument("--output-path", default=DEFAULT_EVAL_PATH)
    evaluate.add_argument("--strict", action="store_true", help="Require a label for every predicted mask")

    prep_real = sub.add_parser("prepare-real", help="Download and export LEVIR-CD+ into data directory")
    prep_real.add_argument("--data-dir", default="data")
    prep_real.add_argument("--replace", action="store_true")

    run_real = sub.add_parser("run-all-real", help="Run full pipeline on real dataset already in data/ or downloaded from HF")
    run_real.add_argument("--data-dir", default="data")
    run_real.add_argument("--download", action="store_true", help="Download LEVIR-CD+ before running")
    run_real.add_argument("--replace", action="store_true", help="When used with --download, clear existing split files first")
    run_real.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    run_real.add_argument("--output-dir", default=DEFAULT_PRED_DIR)
    run_real.add_argument("--eval-path", default=DEFAULT_EVAL_PATH)
    run_real.add_argument("--strict", action="store_true", help="Require perfect filename alignment")

    run_all = sub.add_parser("run-all", help="Prepare sample data and run full pipeline")
    run_all.add_argument("--data-dir", default="data")
    run_all.add_argument("--train-count", type=int, default=24)
    run_all.add_argument("--test-count", type=int, default=8)
    run_all.add_argument("--size", type=int, default=128)
    run_all.add_argument("--seed", type=int, default=42)
    run_all.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    run_all.add_argument("--output-dir", default=DEFAULT_PRED_DIR)
    run_all.add_argument("--eval-path", default=DEFAULT_EVAL_PATH)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "prepare-sample":
        generate_sample_data(Path(args.data_dir), args.train_count, args.test_count, args.size, args.seed)
        print(f"Sample dataset generated under: {args.data_dir}")
        return 0

    if args.command == "train":
        payload = train_threshold_model(
            Path(args.data_dir) / "train" / "before",
            Path(args.data_dir) / "train" / "after",
            Path(args.data_dir) / "train" / "labels",
            Path(args.model_path),
            strict=args.strict,
        )
        print(f"Model saved: {args.model_path}")
        print(f"Selected threshold: {payload['threshold']:.2f}")
        return 0

    if args.command == "infer":
        summary = run_inference(
            Path(args.model_path),
            Path(args.data_dir) / "test" / "before",
            Path(args.data_dir) / "test" / "after",
            Path(args.output_dir),
            strict=args.strict,
        )
        print(f"Inference done for {summary['total_processed']} images")
        print(f"Outputs: {args.output_dir}")
        return 0

    if args.command == "evaluate":
        metrics = evaluate_predictions(
            Path(args.pred_dir),
            Path(args.labels_dir),
            Path(args.output_path),
            strict=args.strict,
        )
        print(f"Evaluation saved: {args.output_path}")
        print(f"F1: {metrics['average_metrics']['f1']:.4f}, IoU: {metrics['average_metrics']['iou']:.4f}")
        return 0

    if args.command == "prepare-real":
        cmd = [
            sys.executable,
            "download_real_dataset.py",
            "--data-dir",
            str(args.data_dir),
        ]
        if args.replace:
            cmd.append("--replace")
        completed = subprocess.run(cmd, check=False)
        return completed.returncode

    if args.command == "run-all":
        generate_sample_data(Path(args.data_dir), args.train_count, args.test_count, args.size, args.seed)
        model = train_threshold_model(
            Path(args.data_dir) / "train" / "before",
            Path(args.data_dir) / "train" / "after",
            Path(args.data_dir) / "train" / "labels",
            Path(args.model_path),
        )
        run_inference(
            Path(args.model_path),
            Path(args.data_dir) / "test" / "before",
            Path(args.data_dir) / "test" / "after",
            Path(args.output_dir),
        )
        eval_payload = evaluate_predictions(
            Path(args.output_dir),
            Path(args.data_dir) / "test" / "labels",
            Path(args.eval_path),
        )
        print("Pipeline complete")
        print(f"Model: {args.model_path}")
        print(f"Predictions: {args.output_dir}")
        print(f"Evaluation: {args.eval_path}")
        print(f"Threshold: {model['threshold']:.2f}")
        print(f"F1: {eval_payload['average_metrics']['f1']:.4f}")
        return 0

    if args.command == "run-all-real":
        if args.download:
            cmd = [
                sys.executable,
                "download_real_dataset.py",
                "--data-dir",
                str(args.data_dir),
            ]
            if args.replace:
                cmd.append("--replace")
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                return completed.returncode

        model = train_threshold_model(
            Path(args.data_dir) / "train" / "before",
            Path(args.data_dir) / "train" / "after",
            Path(args.data_dir) / "train" / "labels",
            Path(args.model_path),
            strict=args.strict,
        )
        infer = run_inference(
            Path(args.model_path),
            Path(args.data_dir) / "test" / "before",
            Path(args.data_dir) / "test" / "after",
            Path(args.output_dir),
            strict=args.strict,
        )
        eval_payload = evaluate_predictions(
            Path(args.output_dir),
            Path(args.data_dir) / "test" / "labels",
            Path(args.eval_path),
            strict=args.strict,
        )
        print("Real-data pipeline complete")
        print(f"Model: {args.model_path}")
        print(f"Predictions: {args.output_dir}")
        print(f"Evaluation: {args.eval_path}")
        print(f"Train samples used: {model['train_samples']}")
        print(f"Test images processed: {infer['total_processed']}")
        print(f"Threshold: {model['threshold']:.2f}")
        print(f"F1: {eval_payload['average_metrics']['f1']:.4f}")
        print(f"IoU: {eval_payload['average_metrics']['iou']:.4f}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
