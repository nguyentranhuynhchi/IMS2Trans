"""Unit tests for IMS2Trans 2D architecture adaptation."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ims2trans_2d import IMS2Trans2D, SwinTransformer2D, DecoderFuse2D


class TestIMS2Trans2D(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = IMS2Trans2D(
            num_cls=4,
            img_size=128,
            depths=(2, 2, 2),
            model_num_heads=(3, 6, 12),
            feature_size=24,
        ).to(self.device)

    def test_parameter_counts(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(total_params, trainable_params)
        self.assertGreater(total_params, 1_000_000)
        self.assertLess(total_params, 5_000_000)

    def test_eval_forward_pass(self):
        self.model.eval()
        x = torch.randn(2, 3, 128, 128, device=self.device)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 4, 128, 128))
        self.assertTrue(torch.isfinite(out).all())

    def test_batch_size_one_eval(self):
        self.model.eval()
        x = torch.randn(1, 3, 128, 128, device=self.device)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (1, 4, 128, 128))

    def test_train_forward_and_backward(self):
        self.model.train()
        x = torch.randn(2, 3, 128, 128, device=self.device)
        logits, modal_feats, deep_preds = self.model(x)

        # Logits
        self.assertEqual(logits.shape, (2, 4, 128, 128))

        # Modal features: bssfp, lge, t2, average
        self.assertEqual(len(modal_feats), 4)
        for f in modal_feats:
            self.assertEqual(f.shape, (2, 192 * 8 * 8))

        # Deep supervision predictions: 4 scales, all upsampled to (2, 4, 128, 128)
        self.assertEqual(len(deep_preds), 4)
        for dp in deep_preds:
            self.assertEqual(dp.shape, (2, 4, 128, 128))

        # Backward pass
        loss = logits.sum() + sum(f.sum() for f in modal_feats) + sum(d.sum() for d in deep_preds)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_tuple_input_modality(self):
        self.model.eval()
        x_bssfp = torch.randn(1, 1, 128, 128)
        x_lge = torch.randn(1, 1, 128, 128)
        x_t2 = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = self.model((x_bssfp, x_lge, x_t2))
        self.assertEqual(out.shape, (1, 4, 128, 128))


if __name__ == "__main__":
    unittest.main()
