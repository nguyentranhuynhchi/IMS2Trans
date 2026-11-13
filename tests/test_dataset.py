"""Unit tests for dataset adapter and canonical label handling."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.dataset_myops import (
    canonicalize_label,
    RandomGenerator,
    ResizeGenerator,
    CANONICAL_LABEL_ORDER,
    LEGACY_LABEL_ORDER,
)


class TestDatasetAdapter(unittest.TestCase):
    def test_canonicalize_label(self):
        # Legacy: 0: bg, 1: myo, 2: scar, 3: edema
        # Canonical: 0: bg, 1: myo, 2: edema, 3: scar
        legacy_mask = np.array([0, 1, 2, 3], dtype=np.uint8)
        canonical_mask = canonicalize_label(legacy_mask, label_order="legacy")

        self.assertEqual(canonical_mask[0], 0)  # BG -> BG
        self.assertEqual(canonical_mask[1], 1)  # Myo -> Myo
        self.assertEqual(canonical_mask[2], 3)  # Scar (2) -> Scar (3)
        self.assertEqual(canonical_mask[3], 2)  # Edema (3) -> Edema (2)

        # Ensure original array was not mutated
        self.assertEqual(legacy_mask.tolist(), [0, 1, 2, 3])

    def test_random_generator_shapes_and_types(self):
        rg = RandomGenerator(output_size=[128, 128])
        sample = {
            "image": np.random.rand(128, 128).astype(np.float32),
            "image1": np.random.rand(128, 128).astype(np.float32),
            "image2": np.random.rand(128, 128).astype(np.float32),
            "label": np.random.randint(0, 4, (128, 128)).astype(np.int64),
            "case_name": "test_case",
        }

        augmented = rg(sample)
        self.assertEqual(augmented["image"].shape, (1, 128, 128))
        self.assertEqual(augmented["image1"].shape, (1, 128, 128))
        self.assertEqual(augmented["image2"].shape, (1, 128, 128))
        self.assertEqual(augmented["label"].shape, (128, 128))
        self.assertEqual(augmented["image"].dtype, torch.float32)
        self.assertEqual(augmented["label"].dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
