import unittest

try:
    import torch
    import torch.nn.functional as F
except (ImportError, OSError) as exc:
    raise unittest.SkipTest(f"PyTorch runtime is unavailable: {exc}") from exc

from specforge.core.dlite import compute_stage1_distillation_loss


class Stage1LossTest(unittest.TestCase):
    def test_stage1_is_weighted_kl_only(self):
        student = torch.tensor([[1.0, 0.0], [0.5, -0.5]])
        teacher = torch.tensor([[0.0, 1.0], [-0.5, 0.5]])
        weights = torch.tensor([1.0, 3.0])
        total, kl, numerator, denominator = compute_stage1_distillation_loss(
            student_serial_logits=student,
            teacher_serial_logits=teacher,
            raw_weight_mask=weights,
            kl_weight=2.5,
            loss_decay_gamma=None,
        )
        per_row = F.kl_div(
            F.log_softmax(student.float(), dim=-1),
            F.softmax(teacher.float(), dim=-1),
            reduction="none",
        ).sum(dim=-1)
        expected_kl = (per_row * weights).sum() / weights.sum()
        torch.testing.assert_close(kl, expected_kl)
        torch.testing.assert_close(total, expected_kl * 2.5)
        torch.testing.assert_close(denominator, weights.sum())
        torch.testing.assert_close(numerator, expected_kl * weights.sum() * 2.5)


if __name__ == "__main__":
    unittest.main()
