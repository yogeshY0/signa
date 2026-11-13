"""SupCon loss regression tests (incl. forged batches)."""
import unittest
import torch
from training.stage3_metric_learning import supcon_loss

class SupConTests(unittest.TestCase):
    def test_supcon_accepts_forged_samples(self):
        embeddings = torch.randn(6, 8)
        signer_ids = torch.tensor([0, 0, 1, 1, 0, 1])
        is_forgery = torch.tensor([False, False, False, False, True, True])
        loss, metrics = supcon_loss(embeddings, signer_ids, is_forgery)
        self.assertEqual(loss.shape, ())
        for value in metrics.values():
            self.assertEqual(value.shape, ())

    def test_supcon_lower_loss_when_same_signer_agrees(self):
        signer_ids = torch.tensor([0, 0, 1, 1])
        is_forgery = torch.tensor([False] * 4)
        tight = torch.tensor([[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0], [-0.99, 0.01]])
        spread = torch.tensor([[1.0, 0.0], [-0.5, 0.8], [-1.0, 0.0], [0.5, -0.8]])
        loss_tight = supcon_loss(tight, signer_ids, is_forgery)[0]
        loss_spread = supcon_loss(spread, signer_ids, is_forgery)[0]
        self.assertLess(loss_tight.item(), loss_spread.item())

if __name__ == "__main__":
    unittest.main()
