import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SFTTrainingSurfaceTest(unittest.TestCase):
    def test_sft_has_independent_entrypoints_and_unprefixed_losses(self):
        trainer = (ROOT / "scripts/train_dlite_sft.py").read_text(encoding="utf-8")
        launcher = (ROOT / "scripts/run_training_dlite_sft.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('model_role="pivot_q_student"', trainer)
        self.assertIn("-m scripts.train_dlite_sft", launcher)
        self.assertNotIn("train_dlite_two_stage", trainer + launcher)
        self.assertNotIn("--stage1-", trainer + launcher)
        self.assertNotIn("--stage2-", trainer + launcher)
        self.assertNotIn("teacher-draft", trainer + launcher)

        for option in (
            "--num-epochs",
            "--loss-decay-gamma",
            "--final-ce-weight",
            "--tv-loss-weight",
            "--base-lm-ce-weight",
            "--base-lm-ce-decay-gamma",
            "--local-position",
        ):
            self.assertIn(option, trainer)
            self.assertIn(option, launcher)


if __name__ == "__main__":
    unittest.main()
