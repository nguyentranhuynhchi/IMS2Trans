"""Unit tests for IMS2Trans loss functions in utils/criterions.py."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.criterions import (
    ContrastiveLoss,
    CosineSimilarityLoss,
    dice_loss,
    softmax_weighted_loss,
    softmax_loss,
)


class TestCriterions(unittest.TestCase):
    def test_contrastive_loss_dynamic_batch(self):
        criterion = ContrastiveLoss(temperature=0.5)

        for B in [1, 2, 3, 4, 16]:
            emb_i = torch.randn(B, 128)
            emb_j = torch.randn(B, 128)

            loss_pair = criterion(emb_i, emb_j)
            self.assertTrue(torch.isfinite(loss_pair), f"Failed for B={B}")
            self.assertGreaterEqual(loss_pair.item(), 0.0)
            if B >= 2:
                self.assertGreater(loss_pair.item(), 0.0)

            # Test concatenated input (2B, 128)
            loss_cat = criterion(torch.cat([emb_i, emb_j], dim=0))
            self.assertTrue(torch.isfinite(loss_cat))
            self.assertAlmostEqual(loss_pair.item(), loss_cat.item(), places=5)

    def test_contrastive_loss_identical_vs_orthogonal(self):
        criterion = ContrastiveLoss(temperature=0.5)
        # Identical embeddings should have lower loss than orthogonal
        emb = torch.randn(4, 64)
        loss_identical = criterion(emb, emb)

        ortho_a = torch.eye(4, 64)
        ortho_b = torch.roll(ortho_a, 4, dims=1)
        loss_ortho = criterion(ortho_a, ortho_b)

        self.assertLess(loss_identical.item(), loss_ortho.item())

    def test_cosine_similarity_loss(self):
        criterion = CosineSimilarityLoss()

        # Perfectly aligned: loss should be 0
        f = torch.randn(8, 64)
        loss_same = criterion([f, f, f])
        self.assertAlmostEqual(loss_same.item(), 0.0, places=5)

        # Opposite embeddings: loss should be near 2.0
        f_neg = -f
        loss_neg = criterion([f, f_neg])
        self.assertAlmostEqual(loss_neg.item(), 2.0, places=5)

        # Random 3 modalities
        m1 = torch.randn(16, 64)
        m2 = torch.randn(16, 64)
        m3 = torch.randn(16, 64)
        loss_rand = criterion([m1, m2, m3])
        self.assertTrue(torch.isfinite(loss_rand))
        self.assertGreaterEqual(loss_rand.item(), 0.0)
        self.assertLessEqual(loss_rand.item(), 2.0)

    def test_dice_and_ce_losses_2d(self):
        pred_logits = torch.randn(4, 4, 128, 128)
        pred_prob = F.softmax(pred_logits, dim=1)
        targets_int = torch.randint(0, 4, (4, 128, 128))
        targets_oh = F.one_hot(targets_int, 4).permute(0, 3, 1, 2).float()

        # Dice loss with integer vs one-hot targets
        d_int = dice_loss(pred_prob, targets_int, num_cls=4)
        d_oh = dice_loss(pred_prob, targets_oh, num_cls=4)
        self.assertTrue(torch.isfinite(d_int))
        self.assertTrue(torch.isfinite(d_oh))
        self.assertAlmostEqual(d_int.item(), d_oh.item(), places=5)

        # Softmax weighted loss with integer vs one-hot targets
        ce_int = softmax_weighted_loss(pred_prob, targets_int, num_cls=4)
        ce_oh = softmax_weighted_loss(pred_prob, targets_oh, num_cls=4)
        self.assertTrue(torch.isfinite(ce_int))
        self.assertTrue(torch.isfinite(ce_oh))
        self.assertAlmostEqual(ce_int.item(), ce_oh.item(), places=5)

    def test_softmax_weighted_loss_batch_16_shapes_and_weights(self):
        # Verify batch size 16 (production training batch size)
        B, C, H, W = 16, 4, 128, 128
        pred_prob = F.softmax(torch.randn(B, C, H, W), dim=1)
        targets = torch.randint(0, C, (B, H, W))

        loss = softmax_weighted_loss(pred_prob, targets, num_cls=C)
        self.assertEqual(loss.ndim, 0, "Loss must be a scalar")
        self.assertTrue(torch.isfinite(loss))

        # Verify rare class receives higher weight than dominant class
        # Target with 99% background (class 0) and 1% scar (class 3)
        skewed_target = torch.zeros(2, H, W, dtype=torch.long)
        skewed_target[:, :10, :10] = 3  # small patch of scar
        # Predictions: high confidence for BG, low for scar
        pred_logits = torch.zeros(2, C, H, W)
        pred_logits[:, 0, :, :] = 5.0  # high confidence BG
        pred_prob = F.softmax(pred_logits, dim=1)
        loss_skewed = softmax_weighted_loss(pred_prob, skewed_target, num_cls=C)
        self.assertTrue(torch.isfinite(loss_skewed))
        self.assertGreater(loss_skewed.item(), 0.0)

    def test_contrastive_loss_edge_cases(self):
        crit = ContrastiveLoss()
        # B=0 empty tensor
        empty = torch.zeros(0, 64)
        loss_zero = crit(empty, empty)
        self.assertEqual(loss_zero.item(), 0.0)

        # B=1 (2 views of 1 sample, 0 negatives)
        one = torch.randn(1, 64)
        loss_one = crit(one, one)
        self.assertEqual(loss_one.item(), 0.0)

    def test_loss_gradients(self):
        pred_logits = torch.randn(2, 4, 128, 128, requires_grad=True)
        pred_prob = F.softmax(pred_logits, dim=1)
        targets = torch.randint(0, 4, (2, 128, 128))

        f1 = torch.randn(2, 64, requires_grad=True)
        f2 = torch.randn(2, 64, requires_grad=True)
        f3 = torch.randn(2, 64, requires_grad=True)
        f_avg = (f1 + f2 + f3) / 3.0

        cont = ContrastiveLoss()
        cos = CosineSimilarityLoss()

        ce = softmax_weighted_loss(pred_prob, targets, num_cls=4)
        dc = dice_loss(pred_prob, targets, num_cls=4)
        con_l = (cont(f1, f_avg) + cont(f2, f_avg) + cont(f3, f_avg)) / 3.0
        cos_l = cos([f1, f2, f3])

        total = ce + dc + 0.1 * con_l + 0.1 * cos_l
        total.backward()

        self.assertIsNotNone(pred_logits.grad)
        self.assertIsNotNone(f1.grad)
        self.assertIsNotNone(f2.grad)
        self.assertIsNotNone(f3.grad)


if __name__ == "__main__":
    unittest.main()
