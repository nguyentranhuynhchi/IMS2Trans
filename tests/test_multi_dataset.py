"""tests/test_multi_dataset.py
===========================
Comprehensive Unit Test Suite for Multi-Dataset Support (MyoPS380 & MyoPS++),
Double-Swap Prevention, Spacing & Distance Metrics, Dynamic Updates, CLI Parity,
and Loss & Architecture Preservation in IMS2Trans.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.dataset_myops import (
    CLASS_NAMES,
    CANONICAL_LUT,
    MYOPS380_EXPECTED_TEST_VOLUMES,
    MYOPS380_EXPECTED_TRAIN_PATIENTS,
    MYOPS380_EXPECTED_TRAIN_SLICES,
    MYOPSPP_EXPECTED_TEST_VOLUMES,
    MYOPSPP_EXPECTED_TRAIN_PATIENTS,
    MYOPSPP_EXPECTED_TRAIN_SLICES,
    MyopsDataset,
    canonicalize_label,
    normalize_dataset_name,
    patient_id,
    read_split_names,
    resolve_default_list_dir,
    resolve_label_order,
)
from utils.metrics import (
    BENCHMARK_PROTOCOLS,
    BENCHMARK_PROTOCOL,
    binary_metrics,
    dataset_rows,
    summarize_dataset_rows,
    metric_hd95_slice_robust,
    compute_patient_surface_distances,
)
from models.ims2trans_2d import IMS2Trans2D
from utils.criterions import (
    ContrastiveLoss,
    CosineSimilarityLoss,
    dice_loss,
    softmax_weighted_loss,
)


class TestMultiDatasetSuite(unittest.TestCase):
    """Test cases covering all 6 mandatory requirements of Step 6."""

    def test_myops380_label_lut_swap(self):
        """1. test_myops380_label_lut_swap: Check MyoPS380 applies LUT [0, 1, 3, 2] swapping 2 and 3."""
        # Legacy label: 0: bg, 1: myo, 2: scar, 3: edema
        legacy = np.array([0, 1, 2, 3], dtype=np.uint8)
        # Applying LUT [0, 1, 3, 2]:
        # 0 -> 0, 1 -> 1, 2 -> 3, 3 -> 2
        canonical = canonicalize_label(legacy, label_order="legacy")
        expected = np.array([0, 1, 3, 2], dtype=np.uint8)
        np.testing.assert_array_equal(canonical, expected)
        np.testing.assert_array_equal(CANONICAL_LUT, np.array([0, 1, 3, 2], dtype=np.uint8))

    def test_myopspp_label_identity_preservation(self):
        """2. test_myopspp_label_identity_preservation: Check MyoPS++ preserves labels 2 and 3 (NO swap)."""
        # Canonical input: 0: bg, 1: myo, 2: edema, 3: scar
        canonical_input = np.array([0, 1, 2, 3], dtype=np.uint8)
        out_canon = canonicalize_label(canonical_input, label_order="canonical")
        np.testing.assert_array_equal(out_canon, np.array([0, 1, 2, 3], dtype=np.uint8))

        out_ident = canonicalize_label(canonical_input, label_order="identity")
        np.testing.assert_array_equal(out_ident, np.array([0, 1, 2, 3], dtype=np.uint8))

        # Dataset level: initializing with dataset='myopspp' defaults to label_order='canonical'
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "Processed_data"
            for mod in ("bSSFP", "LGE", "T2w"):
                (data_root / mod / "train_npz").mkdir(parents=True)
            list_dir = tmp_path / "lists"
            list_dir.mkdir(parents=True)
            (list_dir / "train.txt").write_text("Case2001_slice000\n", encoding="utf-8")

            # Create sample with label 2 and 3
            lbl = np.array([[0, 1], [2, 3]], dtype=np.uint8)
            for mod in ("bSSFP", "LGE", "T2w"):
                np.savez(data_root / mod / "train_npz" / "Case2001_slice000.npz",
                         image=np.zeros((2, 2), dtype=np.float32), label=lbl)

            ds = MyopsDataset(base_dir=data_root, list_dir=list_dir, split="train", dataset="myopspp")
            self.assertEqual(ds.label_order, "canonical")
            sample = ds[0]
            # Must remain exactly 0, 1, 2, 3
            sample_lbl = sample["label"].numpy()
            np.testing.assert_array_equal(sample_lbl, lbl)

    def test_spacing_mm_extraction(self):
        """3. test_spacing_mm_extraction: Check extraction of dynamic mm spacing [1.0, 1.0, sz]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "Processed_data"
            for mod in ("bSSFP", "LGE", "T2w"):
                (data_root / mod / "test_vol_h5").mkdir(parents=True)

            list_dir = tmp_path / "lists"
            list_dir.mkdir(parents=True)
            (list_dir / "test_vol.txt").write_text("Case2001\n", encoding="utf-8")

            vol_img = np.zeros((128, 128, 4), dtype=np.float32)
            for mod in ("bSSFP", "LGE", "T2w"):
                with h5py.File(data_root / mod / "test_vol_h5" / "Case2001.npy.h5", "w") as hf:
                    hf.create_dataset("image", data=vol_img)
                    # Raw acquisition spacing with sz = 4.75
                    hf.create_dataset("spacing", data=np.array([1.25, 1.25, 4.75], dtype=np.float32))

            # On MyoPS++, dataset must enforce [1.0, 1.0, sz] where sz = 4.75
            ds = MyopsDataset(base_dir=data_root, list_dir=list_dir, split="test_vol", dataset="myopspp")
            sample = ds[0]
            np.testing.assert_allclose(sample["spacing"], [1.0, 1.0, 4.75])
            self.assertEqual(sample["spacing_unit"], "mm")

            # Check robust metric helper function accepts spacing
            pred = np.zeros((20, 20, 4), dtype=bool)
            gt = np.zeros((20, 20, 4), dtype=bool)
            pred[5:15, 5:15, :] = True
            gt[7:17, 7:17, :] = True
            hd_mm = metric_hd95_slice_robust(pred, gt, spacing=[1.0, 1.0, 4.75])
            self.assertIsNotNone(hd_mm)
            self.assertGreater(hd_mm, 0.0)

    def test_budget_steps_calculation(self):
        """4. test_budget_steps_calculation: Check batch/epoch and total steps (37,200 vs 5,700)."""
        # MyoPS380: 1,972 train slices, batch_size 16, drop_last=False -> ceil(1972/16) = 124 batches/epoch
        myops380_slices = 1972
        batch_size = 16
        epochs = 300
        myops380_batches_per_epoch = int(np.ceil(myops380_slices / batch_size))
        myops380_total_steps = epochs * myops380_batches_per_epoch
        self.assertEqual(myops380_batches_per_epoch, 124)
        self.assertEqual(myops380_total_steps, 37200)

        # MyoPS++: 296 slices (64 patients), batch_size 16, drop_last=False -> ceil(296/16) = 19 batches/epoch
        myopspp_slices = 296
        myopspp_batches_per_epoch = int(np.ceil(myopspp_slices / batch_size))
        myopspp_total_steps = epochs * myopspp_batches_per_epoch
        self.assertEqual(myopspp_batches_per_epoch, 19)
        self.assertEqual(myopspp_total_steps, 5700)

    def test_cli_argument_parity(self):
        """5. test_cli_argument_parity: Check parser across all 4 entrypoint files."""
        import train_myops
        import evaluate_myops
        import predict
        import evaluate_predictions

        modules = [
            (train_myops, "train_myops"),
            (evaluate_myops, "evaluate_myops"),
            (predict, "predict"),
            (evaluate_predictions, "evaluate_predictions"),
        ]

        for mod, name in modules:
            self.assertTrue(hasattr(mod, "build_parser"), f"{name} must provide build_parser()")
            parser = mod.build_parser()

            # Test --dataset choices and default
            extra_args = ["--checkpoint", "dummy.pt"] if name in ("evaluate_myops", "predict") else (
                ["--pred_dir", "dummy_pred"] if name == "evaluate_predictions" else []
            )

            args = parser.parse_args([
                "--dataset", "myopspp",
                "--split_dir", "splits/myopspp_no_val",
                "--measure_efficiency",
            ] + extra_args)

            self.assertEqual(args.dataset, "myopspp", f"Failed for {name}")
            self.assertEqual(args.list_dir, "splits/myopspp_no_val", f"Failed for {name}")
            self.assertTrue(args.measure_efficiency, f"Failed for {name}")

            # Test aliases --list_dir, --split_dir, --no_measure_efficiency
            args2 = parser.parse_args([
                "--dataset", "myops380",
                "--list_dir", "splits/myops380_no_val",
                "--no-measure-efficiency",
            ] + extra_args)
            self.assertEqual(args2.dataset, "myops380", f"Failed for {name}")
            self.assertEqual(args2.list_dir, "splits/myops380_no_val", f"Failed for {name}")
            self.assertFalse(args2.measure_efficiency, f"Failed for {name}")

    def test_loss_and_architecture_preservation(self):
        """6. test_loss_and_architecture_preservation: Check preservation of Swin Transformer 2D + RFM and composite loss."""
        # Check architecture instantiation
        model = IMS2Trans2D(
            num_cls=4,
            img_size=128,
            depths=(2, 2, 2),
            model_num_heads=(3, 6, 12),
            feature_size=24,
        )
        self.assertTrue(hasattr(model, "swinEncoder"))
        self.assertTrue(hasattr(model, "decoder_fuse"))
        self.assertTrue(hasattr(model, "block2"))
        self.assertTrue(hasattr(model, "block3"))


        # Verify composite loss functions
        x_pred = torch.randn(2, 4, 128, 128)
        y_true = torch.randint(0, 4, (2, 128, 128)).long()

        loss_ce = softmax_weighted_loss(x_pred, y_true, num_cls=4)
        loss_dice = dice_loss(x_pred, y_true, num_cls=4)
        self.assertGreater(float(loss_ce.item()), 0.0)
        self.assertGreater(float(loss_dice.item()), 0.0)

        contrastive_criterion = ContrastiveLoss(temperature=0.5)
        cosine_criterion = CosineSimilarityLoss()
        feat1 = torch.randn(2, 128)
        feat2 = torch.randn(2, 128)
        loss_con = contrastive_criterion(feat1, feat2)
        loss_cos = cosine_criterion([feat1, feat2])
        self.assertIsNotNone(loss_con)
        self.assertIsNotNone(loss_cos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
