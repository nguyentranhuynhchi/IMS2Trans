"""End-to-end integration and smoke tests for IMS2Trans training and evaluation pipelines."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_myops
import evaluate_myops


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="ims2trans_test_")
        self.data_root = Path(self.temp_dir) / "Processed_data"
        self.list_dir = Path(self.temp_dir) / "splits"
        self.list_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy training data (2 slices)
        for mod in ["bSSFP", "LGE", "T2w"]:
            train_dir = self.data_root / mod / "train_npz"
            train_dir.mkdir(parents=True, exist_ok=True)
            for name in ["case0001_slice001", "case0002_slice001"]:
                np.savez_compressed(
                    train_dir / f"{name}.npz",
                    image=np.random.rand(128, 128).astype(np.float32),
                    label=np.random.randint(0, 4, (128, 128)).astype(np.uint8),
                )

            # Create dummy test volume data (1 volume with 4 slices)
            test_dir = self.data_root / mod / "test_vol_h5"
            test_dir.mkdir(parents=True, exist_ok=True)
            import h5py
            with h5py.File(test_dir / "case0003.npy.h5", "w") as f:
                f.create_dataset("image", data=np.random.rand(128, 128, 4).astype(np.float32))
                f.create_dataset("label", data=np.random.randint(0, 4, (128, 128, 4)).astype(np.uint8))

        # Create split files
        (self.list_dir / "train.txt").write_text("case0001_slice001\ncase0002_slice001\n", encoding="utf-8")
        (self.list_dir / "test_vol.txt").write_text("case0003\n", encoding="utf-8")

        self.run_dir = Path(self.temp_dir) / "runs" / "test_run"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_and_eval_smoke(self):
        # 1. Run 1 epoch of training
        train_args = [
            "--data_root", str(self.data_root),
            "--list_dir", str(self.list_dir),
            "--output_dir", str(self.run_dir),
            "--seed", "1234",
            "--batch_size", "2",
            "--max_epochs", "1",
            "--amp", "none",
            "--num_workers", "0",
            "--no_measure_efficiency",
            "--device", "cpu",
        ]
        train_myops.main(train_args)

        ckpt_file = self.run_dir / "checkpoints" / "ims2trans_seed1234.pt"
        self.assertTrue(ckpt_file.is_file(), "Checkpoint file was not created!")

        # Verify checkpoint payload
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt["completed_epochs"], 1)
        self.assertTrue(ckpt["training_complete"])
        self.assertIn("model_state_dict", ckpt)

        # 2. Run evaluation
        eval_output_dir = self.run_dir / "evaluation"
        eval_args = [
            "--checkpoint", str(ckpt_file),
            "--data_root", str(self.data_root),
            "--list_dir", str(self.list_dir),
            "--output_dir", str(eval_output_dir),
            "--split", "test_vol",
            "--batch_size", "2",
            "--device", "cpu",
            "--amp", "none",
        ]
        evaluate_myops.main(eval_args)

        # Verify evaluation artifacts
        pred_file = eval_output_dir / "test_predictions" / "case0003_pred.npz"
        self.assertTrue(pred_file.is_file(), "Prediction npz file was not created!")

        with np.load(pred_file) as data:
            self.assertIn("prediction", data)
            pred = data["prediction"]
            self.assertEqual(pred.shape, (128, 128, 4))
            self.assertEqual(pred.dtype, np.uint8)
            self.assertTrue(np.isin(pred, [0, 1, 2, 3]).all())

        per_case_file = eval_output_dir / "per_case.csv"
        self.assertTrue(per_case_file.is_file(), "per_case.csv was not created!")

        metrics_file = eval_output_dir / "metrics.json"
        self.assertTrue(metrics_file.is_file(), "metrics.json was not created!")

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        self.assertIn("checkpoint_sha256", metrics)
        self.assertIn("summary", metrics)
        self.assertIn("avg_pathology_dice", metrics["summary"])
        self.assertIn("avg_pathology_inclusive_dice", metrics["summary"])


if __name__ == "__main__":
    unittest.main()
