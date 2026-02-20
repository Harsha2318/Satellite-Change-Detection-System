from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")

@dataclass
class Paths:
    train_before: Path = Path("data/train/before")
    train_after: Path = Path("data/train/after")
    train_labels: Path = Path("data/train/labels")
    test_before: Path = Path("data/test/before")
    test_after: Path = Path("data/test/after")
    test_labels: Path = Path("data/test/labels")
    model_path: Path = Path("outputs/models/baseline_model.json")
    predictions_dir: Path = Path("outputs/predictions")
    evaluation_path: Path = Path("outputs/evaluation/evaluation_results.json")

def _list_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Image directory does not exist or is not a directory: {folder}")
    files: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(folder.glob(pattern))
    return sorted(files)

def _map_by_stem(files: list[Path]) -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    for path in files:
        if path.stem in by_stem:
            raise ValueError(f"Duplicate stem '{path.stem}' found in {path.parent}")
        by_stem[path.stem] = path
    return by_stem

def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

def _load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127).astype(np.uint8)

def _pair_lists(before_dir: Path, after_dir: Path, strict: bool = True) -> list[tuple[str, Path, Path]]:
    before_by_stem = _map_by_stem(_list_images(before_dir))
    after_by_stem = _map_by_stem(_list_images(after_dir))
    before_stems = set(before_by_stem)
    after_stems = set(after_by_stem)
    missing_after = sorted(before_stems - after_stems)
    missing_before = sorted(after_stems - before_stems)
    if strict and (missing_after or missing_before):
        details = []
        if missing_after:
            details.append(f"missing in after: {', '.join(missing_after[:10])}")
        if missing_before:
            details.append(f"missing in before: {', '.join(missing_before[:10])}")
        raise ValueError("Before/after filenames must match by stem; " + " | ".join(details))
    stems = sorted(before_stems & after_stems)
    return [(stem, before_by_stem[stem], after_by_stem[stem]) for stem in stems]

def _pair_with_labels(before_dir: Path, after_dir: Path, labels_dir: Path, strict: bool = True) -> list[tuple[str, Path, Path, Path]]:
    image_pairs = _pair_lists(before_dir, after_dir, strict=strict)
    labels_by_stem = _map_by_stem(_list_images(labels_dir))
    stems = {stem for stem, _, _ in image_pairs}
    label_stems = set(labels_by_stem)
    missing_labels = sorted(stems - label_stems)
    extra_labels = sorted(label_stems - stems)
    if strict and (missing_labels or extra_labels):
        details = []
        if missing_labels:
            details.append(f"missing labels: {', '.join(missing_labels[:10])}")
        if extra_labels:
            details.append(f"extra labels: {', '.join(extra_labels[:10])}")
        raise ValueError("Train image/label filenames must match by stem; " + " | ".join(details))
    valid_stems = sorted(stems & label_stems)
    return [(stem, b, a, labels_by_stem[stem]) for stem, b, a in image_pairs if stem in valid_stems]

def _compute_diff_map(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(after - before), axis=2)

def _confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int(np.sum(pred_b & gt_b))
    tn = int(np.sum(~pred_b & ~gt_b))
    fp = int(np.sum(pred_b & ~gt_b))
    fn = int(np.sum(~pred_b & gt_b))
    return tp, tn, fp, fn

def _metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou), "dice": float(dice)}

def _iter_thresholds() -> Iterable[float]:
    return np.linspace(0.05, 0.95, 19)

def train_threshold_model(before_dir: Path, after_dir: Path, labels_dir: Path, model_path: Path, strict: bool = True) -> dict:
    pairs = _pair_with_labels(before_dir, after_dir, labels_dir, strict=strict)
    if not pairs:
        raise ValueError("No valid training samples found")
    pos_hist = np.zeros(256, dtype=np.int64)
    neg_hist = np.zeros(256, dtype=np.int64)
    total_pixels = 0
    total_pos = 0
    for stem, before_path, after_path, label_path in pairs:
        before = _load_rgb(before_path)
        after = _load_rgb(after_path)
        gt = _load_mask(label_path)
        if before.shape[:2] != after.shape[:2] or before.shape[:2] != gt.shape[:2]:
            raise ValueError(f"Spatial dimension mismatch for sample '{stem}': before={before.shape}, after={after.shape}, label={gt.shape}")
        diff_u8 = np.clip(_compute_diff_map(before, after) * 255.0, 0.0, 255.0).astype(np.uint8)
        gt_b = gt.astype(bool)
        pos_hist += np.bincount(diff_u8[gt_b].ravel(), minlength=256)
        neg_hist += np.bincount(diff_u8[~gt_b].ravel(), minlength=256)
        total_pixels += int(gt.size)
        total_pos += int(np.sum(gt_b))
    total_neg = total_pixels - total_pos
    if total_pixels == 0:
        raise ValueError("Training data has zero pixels")
    pos_cumsum = np.cumsum(pos_hist[::-1])[::-1]
    neg_cumsum = np.cumsum(neg_hist[::-1])[::-1]
    best = {"threshold": 0.5, "f1": -1.0, "metrics": None}
    for thr in _iter_thresholds():
        thr_idx = int(np.clip(np.ceil(thr * 255.0), 0, 255))
        tp = int(pos_cumsum[thr_idx])
        fp = int(neg_cumsum[thr_idx])
        fn = int(total_pos - tp)
        tn = int(total_neg - fp)
        current = _metrics(tp, tn, fp, fn)
        if current["f1"] > best["f1"]:
            best = {"threshold": float(thr), "f1": current["f1"], "metrics": current}
    model_payload = {"model_type": "absdiff_threshold_baseline", "threshold": best["threshold"], "train_samples": len(pairs), "trained_at_utc": datetime.now(UTC).isoformat(), "train_metrics": best["metrics"]}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")
    return model_payload

def run_inference(model_path: Path, before_dir: Path, after_dir: Path, output_dir: Path, strict: bool = True) -> dict:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    threshold = float(model.get("threshold", 0.5))
    pairs = _pair_lists(before_dir, after_dir, strict=strict)
    if not pairs:
        raise ValueError("No valid inference samples found")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for stem, before_path, after_path in pairs:
        before = _load_rgb(before_path)
        after = _load_rgb(after_path)
        if before.shape != after.shape:
            raise ValueError(f"Before/after shape mismatch for '{stem}': {before.shape} vs {after.shape}")
        diff = _compute_diff_map(before, after)
        prob = np.clip(diff, 0.0, 1.0)
        pred = (prob >= threshold).astype(np.uint8)
        prob_path = output_dir / f"{stem}_probability.png"
        bin_path = output_dir / f"{stem}_binary.png"
        Image.fromarray((prob * 255).astype(np.uint8)).save(prob_path)
        Image.fromarray((pred * 255).astype(np.uint8)).save(bin_path)
        changed_pixels = int(np.sum(pred))
        total_pixels = int(pred.size)
        results.append({"image": stem, "probability_map": str(prob_path), "binary_map": str(bin_path), "changed_pixels": changed_pixels, "total_pixels": total_pixels, "change_percentage": float((changed_pixels / total_pixels) * 100.0)})
    summary = {"model": str(model_path), "model_type": model.get("model_type", "absdiff_threshold_baseline"), "threshold": threshold, "total_processed": len(results), "results": results}
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def evaluate_predictions(predictions_dir: Path, labels_dir: Path, output_path: Path, strict: bool = True) -> dict:
    if not predictions_dir.exists() or not predictions_dir.is_dir():
        raise ValueError(f"Predictions directory does not exist or is not a directory: {predictions_dir}")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise ValueError(f"Labels directory does not exist or is not a directory: {labels_dir}")
    pred_files = sorted(predictions_dir.glob("*_binary.png"))
    if not pred_files:
        raise ValueError(f"No predicted binary masks found in {predictions_dir}")
    labels_by_stem = _map_by_stem(_list_images(labels_dir))
    pred_by_stem = {p.stem.replace("_binary", ""): p for p in pred_files}
    pred_stems = set(pred_by_stem)
    label_stems = set(labels_by_stem)
    missing_labels = sorted(pred_stems - label_stems)
    if strict and missing_labels:
        raise ValueError("Missing ground-truth labels for predicted masks: " + ", ".join(missing_labels[:10]))
    eval_stems = sorted(pred_stems & label_stems)
    if not eval_stems:
        raise ValueError("No overlapping prediction/label stems to evaluate")
    all_results = []
    total_tp = total_tn = total_fp = total_fn = 0
    for stem in eval_stems:
        pred_file = pred_by_stem[stem]
        gt_file = labels_by_stem[stem]
        pred = _load_mask(pred_file)
        gt = _load_mask(gt_file)
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction/label shape mismatch for '{stem}': {pred.shape} vs {gt.shape}")
        tp, tn, fp, fn = _confusion(pred, gt)
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        per_img = _metrics(tp, tn, fp, fn)
        per_img["image"] = stem
        all_results.append(per_img)
    aggregate = _metrics(total_tp, total_tn, total_fp, total_fn)
    payload = {"total_evaluated": len(all_results), "individual_results": all_results, "average_metrics": aggregate}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
