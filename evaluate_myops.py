"""Evaluation script for IMS2Trans on MyoPS380 held-out test volumes (76 cases).

Protocol: myops380_voxel_v3 (voxel distance, empty mask policy, equal patient weight).
Exports canonical predictions to <output_dir>/test_predictions/<CaseID>_pred.npz with key 'prediction'.
Computes and saves per_case.csv and metrics.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_myops import (
    MyopsDataset,
    canonicalize_label,
    CLASS_NAMES,
)
from models.ims2trans_2d import IMS2Trans2D
from utils.metrics import (
    BENCHMARK_PROTOCOL,
    dataset_rows,
    summarize_dataset_rows,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to ims2trans_seed<seed>.pt checkpoint")
    parser.add_argument("--data_root", "--data-root",
                        default=os.environ.get("MYOPS380_DATA_ROOT", "/content/MyoPS380_dataset/Processed_data"),
                        help="Root folder containing bSSFP, LGE, T2w subdirectories")
    parser.add_argument("--list_dir", "--list-dir", default=str(PROJECT_ROOT / "splits" / "myops380_no_val"),
                        help="Directory containing test_vol.txt")
    parser.add_argument("--output_dir", "--output-dir", default=None,
                        help="Evaluation output directory; defaults to checkpoint run directory / evaluation")
    parser.add_argument("--split", default="test_vol", help="Split name to evaluate (default: test_vol)")
    parser.add_argument("--dataset_id", "--dataset-id", default="myops380", help="Dataset identifier")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8, help="Inference slices per batch")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto", help="AMP mode")
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True,
                        help="Save <CaseID>_pred.npz files")
    parser.add_argument("--img_size", type=int, default=128, help="Model image size")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of canonical classes")
    return parser


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this environment.")
    return device


def resolve_amp(mode: str, device: torch.device):
    if mode == "none" or (mode == "auto" and device.type == "cpu"):
        return None
    if device.type != "cuda":
        raise ValueError("Mixed precision requires CUDA; use --amp none on CPU.")
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 is not supported on this CUDA device.")
        return torch.bfloat16
    if mode == "auto" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.inference_mode()
def predict_volume(model: nn.Module, images: list[np.ndarray], img_size: int = 128, batch_size: int = 8, device=None, amp_dtype=None) -> np.ndarray:
    """Predict on 3 aligned (128, 128, D) modalities and return canonical (128, 128, D) uint8 labels."""
    bSSFP, LGE, T2w = [np.asarray(m, dtype=np.float32) for m in images]
    height, width, depth = bSSFP.shape
    prediction = np.empty((height, width, depth), dtype=np.uint8)

    model.eval()
    model.is_training = False

    for start in range(0, depth, batch_size):
        stop = min(start + batch_size, depth)

        # Transpose to (B, H, W)
        s_bssfp = torch.from_numpy(np.ascontiguousarray(bSSFP[:, :, start:stop].transpose(2, 0, 1))).to(device)
        s_lge = torch.from_numpy(np.ascontiguousarray(LGE[:, :, start:stop].transpose(2, 0, 1))).to(device)
        s_t2w = torch.from_numpy(np.ascontiguousarray(T2w[:, :, start:stop].transpose(2, 0, 1))).to(device)

        # Stack along channel dimension -> (B, 3, 128, 128)
        batch_input = torch.stack([s_bssfp, s_lge, s_t2w], dim=1)

        if batch_input.shape[-2:] != (img_size, img_size):
            batch_input = F.interpolate(batch_input, (img_size, img_size), mode="bilinear", align_corners=False)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            out = model(batch_input)
            logits = out[0] if isinstance(out, tuple) else out

        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits encountered during volume inference.")

        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits.float(), (height, width), mode="bilinear", align_corners=False)

        # Canonical argmax directly: 0: bg, 1: myo, 2: edema, 3: scar
        pred_slices = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        prediction[:, :, start:stop] = pred_slices.transpose(1, 2, 0)

    return prediction


def main(argv=None):
    args = build_parser().parse_args(argv)

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "test_predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = output_dir / "test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=[
            logging.FileHandler(str(log_file), mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp, device)
    amp_str = "none" if amp_dtype is None else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")

    logging.info(f"=== Starting IMS2Trans Evaluation on MyoPS380 ({args.split}) ===")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Device: {device} | AMP: {amp_str}")

    # Compute checkpoint SHA-256
    with open(checkpoint_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()
    logging.info(f"Checkpoint SHA-256: {ckpt_sha256}")

    # Load checkpoint
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_protocol = ckpt_data.get("training_protocol", {})
    completed_epochs = ckpt_data.get("completed_epochs", ckpt_data.get("epoch", -1) + 1)
    planned_epochs = ckpt_data.get("planned_epochs", 300)
    training_complete = ckpt_data.get("training_complete", (completed_epochs == planned_epochs))

    logging.info(f"Checkpoint Epoch: {completed_epochs}/{planned_epochs} | Training Complete: {training_complete}")

    # Build model
    model = IMS2Trans2D(
        num_cls=args.num_classes,
        img_size=args.img_size,
        depths=(2, 2, 2),
        model_num_heads=(3, 6, 12),
        feature_size=24,
    )
    weights = ckpt_data.get("model_state_dict", ckpt_data.get("model"))
    if weights is not None:
        model.load_state_dict(weights)
    else:
        raise KeyError("Checkpoint missing 'model_state_dict' or 'model' weights.")

    model.to(device)
    model.eval()

    # Load test dataset
    actual_list_dir = Path(args.list_dir)
    if not (actual_list_dir / f"{args.split}.txt").exists():
        fallback = PROJECT_ROOT / "lists" / "myops380_no_val"
        if (fallback / f"{args.split}.txt").exists():
            actual_list_dir = fallback

    test_dataset = MyopsDataset(
        base_dir=args.data_root,
        list_dir=str(actual_list_dir),
        split=args.split,
        label_order="legacy",
    )
    logging.info(f"Test cases to evaluate: {len(test_dataset)} from {actual_list_dir}/{args.split}.txt")

    all_rows = []
    start_eval_time = time.time()

    for idx in range(len(test_dataset)):
        case_data = test_dataset[idx]
        case_name = case_data["case_name"]
        images = [case_data["image"], case_data["image1"], case_data["image2"]]
        gt = case_data["label"]  # Canonical in memory

        t0 = time.time()
        pred = predict_volume(
            model=model,
            images=images,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=device,
            amp_dtype=amp_dtype,
        )
        elapsed = time.time() - t0

        # Save predictions in canonical format
        if args.save_predictions:
            pred_file = pred_dir / f"{case_name}_pred.npz"
            np.savez_compressed(
                pred_file,
                prediction=pred,
                exported_label_order="canonical",
                adapter_version="canonical_v1",
                model_id="ims2trans",
                case=case_name,
            )

        # Compute benchmark metric rows
        if gt is not None:
            case_rows = dataset_rows(prediction=pred, target=gt, case=case_name, compute_distance=True)
            all_rows.extend(case_rows)

            dice_scar = next((r["dice"] for r in case_rows if r["region"] == "scar"), None)
            dice_edema = next((r["dice"] for r in case_rows if r["region"] == "edema"), None)
            dice_myo = next((r["dice"] for r in case_rows if r["region"] == "normal_myocardium"), None)
            ds_str = f"{dice_scar:.4f}" if dice_scar is not None else "N/A"
            de_str = f"{dice_edema:.4f}" if dice_edema is not None else "N/A"
            dm_str = f"{dice_myo:.4f}" if dice_myo is not None else "N/A"

            logging.info(
                f"[{idx + 1:02d}/{len(test_dataset):02d}] {case_name} | "
                f"Scar: {ds_str} | Edema: {de_str} | Myo: {dm_str} | ({elapsed:.2f}s)"
            )
        else:
            logging.info(f"[{idx + 1:02d}/{len(test_dataset):02d}] {case_name} | Predicted in {elapsed:.2f}s (No GT)")

    eval_duration = time.time() - start_eval_time
    logging.info(f"=== Inference complete for {len(test_dataset)} cases in {eval_duration:.2f}s ===")

    # Aggregate metrics
    if all_rows:
        summary = summarize_dataset_rows(all_rows)

        # Export per_case.csv
        per_case_file = output_dir / "per_case.csv"
        fieldnames = list(all_rows[0].keys())
        with open(per_case_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logging.info(f"Saved per-case metrics to: {per_case_file}")

        # Assemble metrics.json
        metrics_payload = {
            "dataset": args.dataset_id,
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": ckpt_sha256,
            "completed_epochs": completed_epochs,
            "planned_epochs": planned_epochs,
            "training_complete": training_complete,
            "model_id": "ims2trans",
            "benchmark_protocol": BENCHMARK_PROTOCOL,
            "training_protocol": training_protocol,
            "summary": summary,
            "evaluation_duration_seconds": round(eval_duration, 2),
        }

        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        logging.info(f"Saved summary metrics to: {metrics_file}")

        # Summary Log
        logging.info("=" * 60)
        logging.info("MYOPS380 BENCHMARK SUMMARY RESULTS (IMS2Trans)")
        logging.info(f"  Scar Dice:                     {summary['scar']['mean_dice']:.4f}")
        logging.info(f"  Edema Dice (Exclusive):        {summary['edema']['mean_dice']:.4f}")
        logging.info(f"  Edema-Inclusive Dice:          {summary['edema_inclusive']['mean_dice']:.4f}")
        logging.info(f"  Normal Myocardium Dice:        {summary['normal_myocardium']['mean_dice']:.4f}")
        logging.info(f"  Avg Pathology Dice:            {summary['avg_pathology_dice']:.4f}")
        logging.info(f"  Avg Pathology Inclusive Dice:  {summary['avg_pathology_inclusive_dice']:.4f}")
        if summary['scar']['mean_hd95_voxel'] is not None:
            logging.info(f"  Scar HD95 (voxel):             {summary['scar']['mean_hd95_voxel']:.2f}")
        if summary['edema']['mean_hd95_voxel'] is not None:
            logging.info(f"  Edema HD95 (voxel):            {summary['edema']['mean_hd95_voxel']:.2f}")
        logging.info("=" * 60)


if __name__ == "__main__":
    main()
