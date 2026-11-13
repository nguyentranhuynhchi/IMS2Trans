"""Entry point for training IMS2Trans on MyoPS380 without validation.

Protocol: myops380_train304_test76_last_v1 (304 train patients / 1,972 slices, 300 epochs, no validation).
Total updates: 300 epochs * 124 batches = 37,200 updates.
Optimizer: AdamW (base_lr=0.0003333, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-8).
Scheduler: Polynomial decay power 0.9 with zero-based offset (im_mseg_adamw_poly_v1).
Loss: IMS2Trans multiclass objective: CE + Dice + lambda_con * Contrastive + lambda_cos * Cosine + Deep Supervision.
Checkpointing: Atomic write to <run_dir>/checkpoints/ims2trans_seed<seed>.pt after each epoch.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure local imports work cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_myops import (
    MyopsDataset,
    RandomGenerator,
    CLASS_NAMES,
    CANONICAL_LABEL_ORDER,
    LEGACY_LABEL_ORDER,
)
from models.ims2trans_2d import IMS2Trans2D
from utils.criterions import (
    ContrastiveLoss,
    CosineSimilarityLoss,
    dice_loss,
    softmax_weighted_loss,
)
from utils.efficiency import (
    IMS2TransInferenceWrapper,
    build_efficiency_payload,
    check_rule_compliance,
    count_parameters,
    measure_flops,
    measure_inference_efficiency,
    save_efficiency_report,
    summarize_training_efficiency,
    sync_if_cuda,
)
from utils.metrics import BENCHMARK_PROTOCOL
from utils.optimization import (
    OPTIMIZATION_PROTOCOL,
    build_optimizer,
    build_scheduler,
    validate_optimization_checkpoint,
)

NO_VAL_PROTOCOL = {
    "id": "myops380_train304_test76_last_v1",
    "validation": False,
    "checkpoint_selection": "last_completed_epoch",
    "budget_unit": "epochs",
    "train_patients": 304,
    "test_patients": 76,
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", "--data-root",
                        default=os.environ.get("MYOPS380_DATA_ROOT", "/content/MyoPS380_dataset/Processed_data"),
                        help="Root folder containing bSSFP, LGE, T2w subdirectories")
    parser.add_argument("--list_dir", "--list-dir", default=str(PROJECT_ROOT / "splits" / "myops380_no_val"),
                        help="Directory containing train.txt and test_vol.txt")
    parser.add_argument("--output_dir", "--output-dir", "--run_dir", "--run-dir", default=None,
                        help="Run output directory; defaults to runs/ims2trans_seed<seed>")
    parser.add_argument("--run_root", "--run-root", default=None,
                        help="Optional run root directory when --output_dir is not provided")
    parser.add_argument("--run_name", "--run-name", default=None,
                        help="Optional run folder name under --run_root (default: ims2trans_seed<seed>)")
    parser.add_argument("--model_id", default="ims2trans", help="Canonical lowercase model ID")
    parser.add_argument("--dataset_id", "--dataset-id", default="myops380", help="Dataset identifier")
    parser.add_argument("--img_size", "--img-size", type=int, default=128, help="Model input spatial size (128x128)")
    parser.add_argument("--num_classes", "--num-classes", type=int, default=4, help="Number of canonical classes (4)")
    parser.add_argument("--max_epochs", "--epochs", type=int, default=300, help="Planned training epochs (300)")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16, help="Physical batch size on device (16)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps (1)")
    parser.add_argument("--base_lr", "--lr", type=float, default=0.0003333, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for AdamW")
    parser.add_argument("--clip_grad", type=float, default=0.0, help="Gradient clipping norm (0.0: disabled)")
    parser.add_argument("--lambda_con", type=float, default=0.1, help="Weight for Contrastive Loss")
    parser.add_argument("--lambda_cos", type=float, default=0.1, help="Weight for Cosine Similarity Loss")
    parser.add_argument("--lambda_prm", type=float, default=0.5, help="Weight for Deep Supervision Loss")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=4,
                        help="DataLoader worker processes (use 0 for efficiency benchmark)")
    parser.add_argument("--pin_memory", "--pin-memory", dest="pin_memory",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Pin host memory for CUDA transfers (default: True on CUDA)")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto", help="Mixed precision mode")
    parser.add_argument("--no_val", "--no-val", action="store_true", default=True, help="Fixed no-val training protocol")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Deterministic algorithms")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Path to checkpoint to resume training from (or pass without path for auto-resume)")
    parser.add_argument("--log_every", "--log-every", dest="log_every", type=int, default=20, help="Logging step frequency")
    parser.add_argument("--measure_efficiency", "--measure-efficiency", dest="measure_efficiency",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Measure Complexity, Inference, and Training efficiency metrics and export JSON")
    parser.add_argument("--no_measure_efficiency", dest="measure_efficiency", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--efficiency_output", "--efficiency-output", dest="efficiency_output",
                        default="outputs/runs/ims2trans/efficiency_metrics.json",
                        help="Primary JSON output path for efficiency metrics")
    parser.add_argument("--efficiency_only", "--efficiency-only", dest="efficiency_only",
                        action="store_true",
                        help="Measure Complexity + Inference efficiency only (skip training loop)")
    return parser


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


RESUMABLE_ARG_KEYS = (
    "data_root",
    "list_dir",
    "model_id",
    "dataset_id",
    "img_size",
    "num_classes",
    "max_epochs",
    "batch_size",
    "accum_steps",
    "base_lr",
    "weight_decay",
    "clip_grad",
    "lambda_con",
    "lambda_cos",
    "lambda_prm",
    "seed",
    "num_workers",
    "pin_memory",
    "amp",
)


def _parse_args_and_explicit(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, set[str]]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)

    flag_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for opt in action.option_strings:
            flag_to_dest[opt] = action.dest

    explicit_dests: set[str] = set()
    for token in raw_argv:
        flag = token.split("=")[0]
        if flag in flag_to_dest:
            explicit_dests.add(flag_to_dest[flag])

    return args, explicit_dests


def _inherit_resume_args(
    args: argparse.Namespace,
    explicit_dests: set[str],
    saved_args: dict[str, Any] | None,
    skip_keys: set[str] | None = None,
) -> None:
    if not saved_args:
        return
    skip = set() if skip_keys is None else set(skip_keys)
    for key in RESUMABLE_ARG_KEYS:
        if key in skip or key in explicit_dests:
            continue
        if key in saved_args and hasattr(args, key):
            setattr(args, key, saved_args[key])


def _validate_numeric_args(args: argparse.Namespace) -> None:
    if args.max_epochs < 1:
        raise ValueError(f"--max_epochs must be >= 1, got {args.max_epochs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch_size must be >= 1, got {args.batch_size}")
    if args.accum_steps < 1:
        raise ValueError(f"--accum_steps must be >= 1, got {args.accum_steps}")
    if args.num_workers < 0:
        raise ValueError(f"--num_workers must be >= 0, got {args.num_workers}")


def atomic_save(path: Path | str, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def main(argv=None):
    args, explicit_dests = _parse_args_and_explicit(argv)

    resume_target: Path | None = None
    resume_ckpt: dict[str, Any] | None = None
    if args.resume is not None and args.resume != "auto":
        resume_target = Path(args.resume)
        if not resume_target.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_target}")
        resume_ckpt = torch.load(resume_target, map_location="cpu", weights_only=False)
        validate_optimization_checkpoint(resume_ckpt)
        _inherit_resume_args(args, explicit_dests, resume_ckpt.get("args"))
        if not any(k in explicit_dests for k in ("output_dir", "run_root", "run_name")):
            inferred_run_dir = (
                resume_target.parent.parent if resume_target.parent.name == "checkpoints" else resume_target.parent
            )
            args.output_dir = str(inferred_run_dir)

    # Resolve paths
    if args.output_dir:
        run_dir = Path(args.output_dir)
    elif args.run_root or args.run_name:
        base_root = Path(args.run_root) if args.run_root else (PROJECT_ROOT / "runs")
        run_dir = base_root / (args.run_name or f"{args.model_id}_seed{args.seed}")
    else:
        run_dir = PROJECT_ROOT / "runs" / f"{args.model_id}_seed{args.seed}"

    final_ckpt_path = run_dir / "checkpoints" / f"{args.model_id}_seed{args.seed}.pt"
    auto_resume_missing = False
    if args.resume == "auto":
        if final_ckpt_path.is_file():
            resume_target = final_ckpt_path
            resume_ckpt = torch.load(resume_target, map_location="cpu", weights_only=False)
            validate_optimization_checkpoint(resume_ckpt)
            _inherit_resume_args(args, explicit_dests, resume_ckpt.get("args"), skip_keys={"model_id", "seed"})
        else:
            auto_resume_missing = True

    _validate_numeric_args(args)

    has_full_resume_weights = (
        resume_ckpt is not None
        and (resume_ckpt.get("model_state_dict") is not None or resume_ckpt.get("model") is not None)
    )

    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    for d in (checkpoints_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = logs_dir / "train.log"
    file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=[
            file_handler,
            logging.StreamHandler(sys.stdout),
        ],
    )

    try:
        _run_training(
            args=args,
            explicit_dests=explicit_dests,
            run_dir=run_dir,
            logs_dir=logs_dir,
            final_ckpt_path=final_ckpt_path,
            resume_target=resume_target,
            resume_ckpt=resume_ckpt,
            has_full_resume_weights=has_full_resume_weights,
            auto_resume_missing=auto_resume_missing,
        )
    finally:
        root_logger = logging.getLogger()
        if file_handler in root_logger.handlers:
            root_logger.removeHandler(file_handler)
        file_handler.close()


def _run_training(
    *,
    args: argparse.Namespace,
    explicit_dests: set[str],
    run_dir: Path,
    logs_dir: Path,
    final_ckpt_path: Path,
    resume_target: Path | None,
    resume_ckpt: dict[str, Any] | None,
    has_full_resume_weights: bool,
    auto_resume_missing: bool,
) -> None:
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp, device)
    amp_str = "none" if amp_dtype is None else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")
    pin_memory = (device.type == "cuda") if args.pin_memory is None else bool(args.pin_memory)
    if device.type != "cuda" and "pin_memory" not in explicit_dests:
        pin_memory = False

    logging.info(f"=== Starting IMS2Trans Training on MyoPS380 (Seed {args.seed}) ===")
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Data root: {args.data_root} | List dir: {args.list_dir}")
    if auto_resume_missing:
        logging.info("--resume auto requested, but no checkpoint exists at %s; starting fresh.", final_ckpt_path)

    rule_compliant, compliance_warnings, hardware_info = check_rule_compliance(
        device=device,
        amp_mode=args.amp,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        batch_size=args.batch_size,
        planned_epochs=args.max_epochs,
        accum_steps=args.accum_steps,
    )
    if args.measure_efficiency or args.efficiency_only:
        logging.info(
            "Efficiency Config | GPU: %s (count=%d) | Precision: %s | num_workers: %d | pin_memory: %s | batch: %d | epochs: %d",
            hardware_info["gpu_name"],
            hardware_info["visible_gpu_count"],
            hardware_info["precision"],
            args.num_workers,
            pin_memory,
            args.batch_size,
            args.max_epochs,
        )
        if not rule_compliant:
            for warn_msg in compliance_warnings:
                logging.warning("[EFFICIENCY RULE MISMATCH] %s", warn_msg)

    if resume_ckpt is not None:
        logging.info(f"Loaded checkpoint metadata from: {resume_target}")
        completed = int(resume_ckpt.get("completed_epochs", resume_ckpt.get("epoch", -1) + 1))
        if (resume_ckpt.get("training_complete", False) or completed >= args.max_epochs) and not args.efficiency_only:
            logging.info(f"Training was already marked complete at epoch {completed}/{args.max_epochs}. Nothing to run.")
            return

    # Build IMS2Trans 2D Model
    net = IMS2Trans2D(
        num_cls=args.num_classes,
        img_size=args.img_size,
        depths=(2, 2, 2),
        model_num_heads=(3, 6, 12),
        feature_size=24,
    )

    if resume_ckpt is not None:
        model_weights = resume_ckpt.get("model_state_dict", resume_ckpt.get("model"))
        if model_weights is not None:
            net.load_state_dict(model_weights)
            logging.info("Resumed model weights from checkpoint.")

    net.float().to(device)

    params_info = count_parameters(net)
    total_params = params_info["total_params"]
    trainable_params = params_info["trainable_params"]

    # Mandatory master benchmark log line
    logging.info(
        f"INFO | Device {device.type} | AMP {amp_str} | "
        f"parameters {total_params:,} (trainable: {trainable_params:,}) | "
        f"train/val slices 1972/0 | effective batch <= {args.batch_size}"
    )

    # Save initial configuration
    config_dict = {
        "model_id": args.model_id,
        "model_name": "IMS2Trans",
        "img_size": args.img_size,
        "num_classes": args.num_classes,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "base_lr": args.base_lr,
        "weight_decay": args.weight_decay,
        "lambda_con": args.lambda_con,
        "lambda_cos": args.lambda_cos,
        "lambda_prm": args.lambda_prm,
        "seed": args.seed,
        "amp": args.amp,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "measure_efficiency": bool(args.measure_efficiency),
        "no_val": True,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_protocol": NO_VAL_PROTOCOL,
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Setup loss functions
    contrastive_criterion = ContrastiveLoss(temperature=0.5)
    cosine_criterion = CosineSimilarityLoss()

    # Pre-training efficiency measurement (Complexity & Inference)
    flops_info: dict[str, Any] | None = None
    inference_info: dict[str, Any] | None = None

    if args.measure_efficiency or args.efficiency_only:
        logging.info("--- Starting Complexity & Inference Efficiency Benchmarking ---")
        flops_info = measure_flops(net, input_shape=(1, 3, args.img_size, args.img_size), device=device)
        logging.info(f"FLOPs: {flops_info['flops_g']:.2f} GFLOPs")

        inference_info = measure_inference_efficiency(
            net,
            input_shape=(1, 3, args.img_size, args.img_size),
            device=device,
            warmup=30,
            repeat=100,
        )
        logging.info(
            f"Inference Latency: {inference_info['latency_ms_per_slice']:.2f} ms/slice | "
            f"Peak VRAM: {inference_info['peak_vram_mb']:.2f} MB ({inference_info['peak_vram_gb']:.4f} GB)"
        )

        if args.efficiency_only:
            payload = build_efficiency_payload(
                model_name="IMS2Trans",
                model_id=args.model_id,
                seed=args.seed,
                params_info=params_info,
                flops_info=flops_info,
                inference_info=inference_info,
                training_info=None,
                rule_compliant=rule_compliant,
                compliance_warnings=compliance_warnings,
                hardware_info=hardware_info,
            )
            report_paths = _dedupe_paths([Path(args.efficiency_output), run_dir / "efficiency_metrics.json"])
            save_efficiency_report(payload, report_paths)
            logging.info("=== Efficiency-only benchmark finished successfully ===")
            return

    # Dataset & DataLoader
    # Check both splits and lists
    actual_list_dir = Path(args.list_dir)
    if not (actual_list_dir / "train.txt").exists():
        fallback_list_dir = PROJECT_ROOT / "lists" / "myops380_no_val"
        if (fallback_list_dir / "train.txt").exists():
            actual_list_dir = fallback_list_dir

    train_dataset = MyopsDataset(
        base_dir=args.data_root,
        list_dir=str(actual_list_dir),
        split="train",
        transform=RandomGenerator(output_size=[args.img_size, args.img_size]),
        label_order="legacy",
    )
    logging.info(f"Loaded train dataset: {len(train_dataset)} slices from {actual_list_dir}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    batches_per_epoch = len(train_loader)
    total_updates = args.max_epochs * batches_per_epoch
    logging.info(f"Batches per epoch: {batches_per_epoch} | Planned updates: {total_updates}")

    # Optimizer & Scheduler
    optimizer = build_optimizer(net, base_lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, total_updates=total_updates)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    start_epoch = 0
    total_steps = 0
    epoch_times: list[float] = []

    if resume_ckpt is not None:
        start_epoch = int(resume_ckpt.get("completed_epochs", resume_ckpt.get("epoch", -1) + 1))
        total_steps = int(resume_ckpt.get("total_updates", resume_ckpt.get("global_step", start_epoch * batches_per_epoch)))
        if "optimizer" in resume_ckpt or "optimizer_state_dict" in resume_ckpt:
            opt_st = resume_ckpt.get("optimizer_state_dict", resume_ckpt.get("optimizer"))
            optimizer.load_state_dict(opt_st)
            # Ensure optimizer tensors are on correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        if "scheduler" in resume_ckpt or "scheduler_state_dict" in resume_ckpt:
            sched_st = resume_ckpt.get("scheduler_state_dict", resume_ckpt.get("scheduler"))
            scheduler.load_state_dict(sched_st)
        if "scaler" in resume_ckpt or "scaler_state_dict" in resume_ckpt:
            sc_st = resume_ckpt.get("scaler_state_dict", resume_ckpt.get("scaler"))
            scaler.load_state_dict(sc_st)
        if "epoch_times" in resume_ckpt:
            epoch_times = list(resume_ckpt["epoch_times"])
        logging.info(f"Resumed training state: Epoch {start_epoch}, Update {total_steps}")

    # Training Loop
    training_wall_start = time.time()
    if device.type == "cuda" and torch.cuda.is_available() and start_epoch == 0:
        torch.cuda.reset_peak_memory_stats(device)

    net.train()
    net.is_training = True

    for epoch in range(start_epoch, args.max_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        running_fuse_loss = 0.0
        running_dis_loss = 0.0
        running_cos_loss = 0.0
        running_prm_loss = 0.0
        epoch_steps = 0

        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=pin_memory)  # (B, 3, 128, 128)
            targets = batch["label"].to(device, non_blocking=pin_memory) # (B, 128, 128) canonical 0..3

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                out = net(images)
                logits, modal_feats, deep_preds = out
                fuse_prob = F.softmax(logits, dim=1)

                # 1. Main Segmentation Loss: CE + Dice
                fuse_ce = softmax_weighted_loss(fuse_prob, targets, num_cls=args.num_classes)
                fuse_dc = dice_loss(fuse_prob, targets, num_cls=args.num_classes)
                fuse_loss = fuse_ce + fuse_dc

                # 2. Contrastive Loss against average representation
                bssfp_f, lge_f, t2_f, avg_f = modal_feats
                dis_fdc = (
                    contrastive_criterion(bssfp_f, avg_f)
                    + contrastive_criterion(lge_f, avg_f)
                    + contrastive_criterion(t2_f, avg_f)
                ) / 3.0
                dis_loss = args.lambda_con * dis_fdc

                # 3. Cosine Similarity Loss across pairs (bssfp-lge, bssfp-t2, lge-t2)
                cos_loss = args.lambda_cos * cosine_criterion([bssfp_f, lge_f, t2_f])

                # 4. Deep Supervision Loss across 4 decoder stages
                prm_loss = torch.tensor(0.0, device=device)
                if deep_preds:
                    for dp in deep_preds:
                        dp_prob = F.softmax(dp, dim=1)
                        prm_loss = prm_loss + (
                            softmax_weighted_loss(dp_prob, targets, num_cls=args.num_classes)
                            + dice_loss(dp_prob, targets, num_cls=args.num_classes)
                        )
                    prm_loss = args.lambda_prm * (prm_loss / len(deep_preds))

                total_loss = fuse_loss + dis_loss + cos_loss + prm_loss

            scaler.scale(total_loss).backward()

            if args.clip_grad > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_steps += 1
            epoch_steps += 1
            running_loss += total_loss.item()
            running_fuse_loss += fuse_loss.item()
            running_dis_loss += dis_loss.item()
            running_cos_loss += cos_loss.item()
            running_prm_loss += prm_loss.item()

            if (i + 1) % args.log_every == 0 or (i + 1) == batches_per_epoch:
                current_lr = scheduler.get_last_lr()[0]
                logging.info(
                    f"Epoch [{epoch + 1:03d}/{args.max_epochs}] Iter [{i + 1:03d}/{batches_per_epoch}] "
                    f"Step {total_steps:05d} | LR {current_lr:.6f} | "
                    f"Loss {total_loss.item():.4f} (Fuse: {fuse_loss.item():.4f}, "
                    f"Con: {dis_loss.item():.4f}, Cos: {cos_loss.item():.4f}, Prm: {prm_loss.item():.4f})"
                )

        sync_if_cuda(device)
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)

        avg_loss = running_loss / max(epoch_steps, 1)
        avg_fuse = running_fuse_loss / max(epoch_steps, 1)
        logging.info(
            f"=== Epoch [{epoch + 1:03d}/{args.max_epochs}] Complete | "
            f"Avg Loss: {avg_loss:.4f} (Fuse: {avg_fuse:.4f}) | Duration: {epoch_duration:.2f}s ==="
        )

        # Atomic Checkpointing per epoch
        is_final = (epoch + 1) == args.max_epochs
        ckpt_payload = {
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "planned_epochs": args.max_epochs,
            "total_updates": total_steps,
            "training_complete": is_final,
            "training_protocol": NO_VAL_PROTOCOL,
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "model_state_dict": net.state_dict(),
            "model": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch_times": epoch_times,
            "args": vars(args),
            "seed": args.seed,
            "model_id": args.model_id,
        }
        atomic_save(final_ckpt_path, ckpt_payload)

    total_wall_clock = time.time() - training_wall_start
    logging.info(
        f"=== Training Finished in {total_wall_clock / 3600.0:.2f} hours ({total_wall_clock:.1f}s) | "
        f"Checkpoint saved at: {final_ckpt_path} ==="
    )

    # Post-training Efficiency Report
    if (args.measure_efficiency or args.efficiency_only) and flops_info is not None and inference_info is not None:
        peak_vram_bytes = 0
        peak_res_bytes = 0
        if device.type == "cuda" and torch.cuda.is_available():
            peak_vram_bytes = torch.cuda.max_memory_allocated(device)
            peak_res_bytes = torch.cuda.max_memory_reserved(device)

        training_info = summarize_training_efficiency(
            epoch_times=epoch_times,
            total_wall_clock_s=total_wall_clock,
            peak_vram_bytes=peak_vram_bytes,
            peak_reserved_bytes=peak_res_bytes,
            warmup_epochs=5,
            planned_epochs=args.max_epochs,
        )

        payload = build_efficiency_payload(
            model_name="IMS2Trans",
            model_id=args.model_id,
            seed=args.seed,
            params_info=params_info,
            flops_info=flops_info,
            inference_info=inference_info,
            training_info=training_info,
            rule_compliant=rule_compliant,
            compliance_warnings=compliance_warnings,
            hardware_info=hardware_info,
        )
        report_paths = _dedupe_paths([Path(args.efficiency_output), run_dir / "efficiency_metrics.json"])
        save_efficiency_report(payload, report_paths)


if __name__ == "__main__":
    main()
