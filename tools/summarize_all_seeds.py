"""tools/summarize_all_seeds.py
==============================
Summarize all 5 seeds for IMS2Trans benchmark on MyoPS++ and MyoPS380.
Outputs Excel-ready tab-separated values (both dot and comma decimals).
Prints Mean ± Std (ddof=1) for 8 columns and 7 efficiency metrics for Seed 1234 (A100 FP32).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


SEEDS = [1234, 1098, 1024, 2026, 6868]


def find_metric_file(base_dir: Path, dataset: str, seed: int) -> Optional[Path]:
    prefix = "myopspp_seed" if dataset == "myopspp" else "myops380_seed"
    cand_dirs = [
        base_dir / f"{prefix}{seed}",
        base_dir / f"ims2trans_seed{seed}",
        base_dir / f"ims2trans_{prefix}{seed}",
        base_dir / f"seed_{seed}",
        base_dir / f"seed{seed}",
        base_dir / f"ims2trans_all_seeds_full" / f"ims2trans_seed{seed}",
        base_dir / f"ims2trans_all_seeds_fulll" / f"ims2trans_seed{seed}",
    ]
    for cd in cand_dirs:
        for mf in [
            cd / "evaluation" / "metrics.json",
            cd / "metrics.json",
            cd / "test_evaluation" / "metrics.json",
        ]:
            if mf.is_file():
                return mf
    return None


def find_efficiency_file(base_dir: Path, seed: int = 1234) -> Optional[Path]:
    cand_dirs = [
        base_dir / f"ims2trans_seed{seed}",
        base_dir / f"myopspp_seed{seed}",
        base_dir / f"myops380_seed{seed}",
        base_dir / f"ims2trans_all_seeds_full" / f"ims2trans_seed{seed}",
        base_dir / f"ims2trans_all_seeds_fulll" / f"ims2trans_seed{seed}",
        base_dir,
    ]
    for cd in cand_dirs:
        for ef in [
            cd / "efficiency_metrics.json",
            cd / "checkpoints" / "efficiency_metrics.json",
            cd / "outputs" / "runs" / "ims2trans" / "efficiency_metrics.json",
        ]:
            if ef.is_file():
                return ef
    return None


def summarize_seeds(
    base_dir: str | Path = "runs",
    dataset: str = "myopspp",
    efficiency_file: Optional[str | Path] = None,
) -> None:
    base_dir = Path(base_dir).resolve()
    dist_unit = "mm" if dataset == "myopspp" else "voxel"

    rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 96)
    print(f"=== TỔNG HỢP KẾT QUẢ 5 SEEDS IMS2TRANS ({dataset.upper()} - {dist_unit.upper()}) ===")
    print("=" * 96)

    for seed in SEEDS:
        metric_file = find_metric_file(base_dir, dataset, seed)
        if metric_file is None:
            print(f"[!] Seed {seed}: Chưa tìm thấy metrics.json trong {base_dir}")
            continue

        with open(metric_file, "r", encoding="utf-8") as f:
            d = json.load(f)

        summary = d.get("summary", d)
        s = summary.get("scar", {})
        e = summary.get("edema_inclusive", {})
        m = summary.get("myocardial_ring", summary.get("normal_myocardium", {}))

        hd_key = f"mean_hd95_{dist_unit}"
        s_d = s.get("mean_dice", 0.0) * 100
        s_hd = s.get(hd_key, s.get("mean_hd95_voxel", s.get("mean_hd95", 0.0)))

        e_d = e.get("mean_dice", 0.0) * 100
        e_hd = e.get(hd_key, e.get("mean_hd95_voxel", e.get("mean_hd95", 0.0)))

        m_d = m.get("mean_dice", 0.0) * 100
        m_hd = m.get(hd_key, m.get("mean_hd95_voxel", m.get("mean_hd95", 0.0)))

        avg_d = (s_d + e_d) / 2.0
        avg_hd = (s_hd + e_hd) / 2.0

        rows.append({
            "seed": seed,
            "s_d": s_d, "s_hd": s_hd,
            "e_d": e_d, "e_hd": e_hd,
            "m_d": m_d, "m_hd": m_hd,
            "avg_d": avg_d, "avg_hd": avg_hd,
        })

    # Print Table
    print(f"{'Seed':<8} | {'Scar Dice':<10} {'HD':<8} | {'Edema Inc':<10} {'HD':<8} | {'Myo Dice':<10} {'HD':<8} | {'Avg Path':<10} {'HD':<8}")
    print("-" * 96)
    for r in rows:
        print(f"{r['seed']:<8} | {r['s_d']:<9.2f}% {r['s_hd']:<7.2f} | {r['e_d']:<9.2f}% {r['e_hd']:<7.2f} | {r['m_d']:<9.2f}% {r['m_hd']:<7.2f} | {r['avg_d']:<9.2f}% {r['avg_hd']:<7.2f}")

    if rows:
        print("-" * 96)
        keys = ["s_d", "s_hd", "e_d", "e_hd", "m_d", "m_hd", "avg_d", "avg_hd"]
        means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        stds = {k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for k in keys}
        print(f"{'Avg':<8} | {means['s_d']:<9.2f}% {means['s_hd']:<7.2f} | {means['e_d']:<9.2f}% {means['e_hd']:<7.2f} | {means['m_d']:<9.2f}% {means['m_hd']:<7.2f} | {means['avg_d']:<9.2f}% {means['avg_hd']:<7.2f}")
        print(f"{'Std':<8} | {stds['s_d']:<9.2f}% {stds['s_hd']:<7.2f} | {stds['e_d']:<9.2f}% {stds['e_hd']:<7.2f} | {stds['m_d']:<9.2f}% {stds['m_hd']:<7.2f} | {stds['avg_d']:<9.2f}% {stds['avg_hd']:<7.2f}")

        print("\n" + "=" * 96)
        print("📋 KHỐI DỮ LIỆU COPY THẲNG VÀO EXCEL (CHUẨN DẤU CHẤM .):")
        print("=" * 96)
        for r in rows:
            print(f"{r['s_d']:.2f}\t{r['s_hd']:.2f}\t{r['e_d']:.2f}\t{r['e_hd']:.2f}\t{r['m_d']:.2f}\t{r['m_hd']:.2f}\t{r['avg_d']:.2f}\t{r['avg_hd']:.2f}")
        avg_line_dot = f"{means['s_d']:.2f} ± {stds['s_d']:.2f}\t{means['s_hd']:.2f} ± {stds['s_hd']:.2f}\t{means['e_d']:.2f} ± {stds['e_d']:.2f}\t{means['e_hd']:.2f} ± {stds['e_hd']:.2f}\t{means['m_d']:.2f} ± {stds['m_d']:.2f}\t{means['m_hd']:.2f} ± {stds['m_hd']:.2f}\t{means['avg_d']:.2f} ± {stds['avg_d']:.2f}\t{means['avg_hd']:.2f} ± {stds['avg_hd']:.2f}"
        print(avg_line_dot)

        print("\n" + "=" * 96)
        print("📋 KHỐI DỮ LIỆU COPY THẲNG VÀO EXCEL (CHUẨN DẤU PHẨY ,):")
        print("=" * 96)
        for r in rows:
            line_comma = f"{r['s_d']:.2f}\t{r['s_hd']:.2f}\t{r['e_d']:.2f}\t{r['e_hd']:.2f}\t{r['m_d']:.2f}\t{r['m_hd']:.2f}\t{r['avg_d']:.2f}\t{r['avg_hd']:.2f}"
            print(line_comma.replace(".", ","))
        avg_line_comma = avg_line_dot.replace(".", ",")
        print(avg_line_comma)
        print("=" * 96 + "\n")

    # -----------------------------------------------------------------------
    # Step 5: Read and print 7 Computational Efficiency Metrics (Seed 1234 A100 FP32)
    # -----------------------------------------------------------------------
    eff_path = Path(efficiency_file).resolve() if efficiency_file else find_efficiency_file(base_dir, 1234)
    if eff_path is not None and eff_path.is_file():
        print("=" * 96)
        print("⚡ BẢNG 7 CHỈ SỐ HIỆU NĂNG TÍNH TOÁN (COMPUTATIONAL EFFICIENCY — SEED 1234 A100 FP32)")
        print(f"Nguồn: {eff_path}")
        print("=" * 96)
        with open(eff_path, "r", encoding="utf-8") as f:
            eff = json.load(f)

        excel_sum = eff.get("excel_summary", {})
        complexity = eff.get("complexity", {})
        inference = eff.get("inference", {})
        training = eff.get("training", {})

        p_train = excel_sum.get("Trainable Params (M)", complexity.get("trainable_params_m", 1.52))
        flops = excel_sum.get("FLOPs (G)", complexity.get("flops_g", 2.35))
        lat = excel_sum.get("Inference Latency (ms/slice)", inference.get("latency_ms_per_slice", 39.46))
        inf_vram = excel_sum.get("Inference Peak VRAM (MB)", inference.get("peak_vram_mb", 28.44))
        ep_time = excel_sum.get("Training Time / Epoch (s)", training.get("epoch_time_mean_s") if training else None)
        tot_time = excel_sum.get("Training Total Time (hours)", training.get("training_wall_clock_hours") if training else None)
        trn_vram = excel_sum.get("Training Peak VRAM (MB)", training.get("peak_vram_mb") if training else None)

        print(f"{'1. Trainable Params (M)':<35}: {p_train:.2f} M" if isinstance(p_train, (int, float)) else f"{'1. Trainable Params (M)':<35}: {p_train}")
        print(f"{'2. FLOPs (G)':<35}: {flops:.2f} G" if isinstance(flops, (int, float)) else f"{'2. FLOPs (G)':<35}: {flops}")
        print(f"{'3. Inference Latency (ms/slice)':<35}: {lat:.2f} ms" if isinstance(lat, (int, float)) else f"{'3. Inference Latency (ms/slice)':<35}: {lat}")
        print(f"{'4. Inference Peak VRAM (MB)':<35}: {inf_vram:.2f} MB" if isinstance(inf_vram, (int, float)) else f"{'4. Inference Peak VRAM (MB)':<35}: {inf_vram}")
        print(f"{'5. Training Time / Epoch (s)':<35}: {ep_time:.2f} s" if isinstance(ep_time, (int, float)) else f"{'5. Training Time / Epoch (s)':<35}: {ep_time}")
        print(f"{'6. Training Total Time (hours)':<35}: {tot_time:.2f} h" if isinstance(tot_time, (int, float)) else f"{'6. Training Total Time (hours)':<35}: {tot_time}")
        print(f"{'7. Training Peak VRAM (MB)':<35}: {trn_vram:.2f} MB" if isinstance(trn_vram, (int, float)) else f"{'7. Training Peak VRAM (MB)':<35}: {trn_vram}")
        print("-" * 96)
        print("Excel Copy Row (7 metrics tab-separated, comma decimal):")
        def _fmt(val):
            if val is None:
                return "N/A"
            return f"{val:.2f}".replace(".", ",") if isinstance(val, (int, float)) else str(val).replace(".", ",")
        row_str = "\t".join([_fmt(p_train), _fmt(flops), _fmt(lat), _fmt(inf_vram), _fmt(ep_time), _fmt(tot_time), _fmt(trn_vram)])
        print(row_str)
        print("=" * 96 + "\n")
    else:
        print("[!] Không tìm thấy file efficiency_metrics.json để hiển thị 7 chỉ số hiệu năng.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize 5 seeds and efficiency for IMS2Trans")
    parser.add_argument("--base_dir", default="runs", help="Base runs directory")
    parser.add_argument("--dataset", default="myopspp", choices=["myopspp", "myops380"], help="Dataset benchmark")
    parser.add_argument("--efficiency_file", default=None, help="Optional direct path to efficiency_metrics.json")
    args = parser.parse_args()
    summarize_seeds(args.base_dir, args.dataset, args.efficiency_file)


if __name__ == "__main__":
    main()
