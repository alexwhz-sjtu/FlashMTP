import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TrainingSurfaceTest(unittest.TestCase):
    def test_removed_legacy_options_do_not_return(self):
        sources = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in (
                "scripts/dlite_training.py",
                "scripts/train_dlite_teacher.py",
                "scripts/run_training_dlite_teacher.sh",
                "scripts/train_dlite_two_stage.py",
                "scripts/run_training_dlite_two_stage.sh",
                "specforge/core/dlite.py",
                "specforge/modeling/draft/sequential_head.py",
            )
        )
        for legacy in (
            "rnn_easy",
            "markov_head",
            "stage1-smooth-l1-beta",
            "stage1-hidden-weight",
            "stage1-ce-weight",
            "stage1-train-data-path",
            "stage2-train-data-path",
            "student-init-mode",
            "sequential-teacher-forcing-ratio",
            "sequential_teacher_forcing_ratio",
            "SEQUENTIAL_TEACHER_FORCING_RATIO",
            "forward_scheduled_sampling",
            "anchor-group-size",
            "anchor_group_size",
            "ANCHOR_GROUP_SIZE",
            "initial_prev_token_ids",
        ):
            self.assertNotIn(legacy, sources)
        self.assertIn("--train-data-path", sources)
        self.assertIn("--stage1-kl-weight", sources)


if __name__ == "__main__":
    unittest.main()
