import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_DIR / "scripts" / "run_training_flashmtp_two_stage.sh"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        PYTHON_BIN=sys.executable,
        NPROC_PER_NODE="2",
        NNODES="1",
        NODE_RANK="0",
        TP_SIZE="2",
        TARGET_MODEL="/models/target",
        TEACHER_DRAFT_PATH="/models/teacher/final",
        TRAIN_DATA_PATH="/data/train.jsonl",
        OUTPUT_DIR="/output/student",
        STAGE1_EPOCHS="2",
        STAGE1_LEARNING_RATE="5e-4",
        STAGE2_EPOCHS="6",
        STAGE2_LEARNING_RATE="2e-4",
        MASK_TOKEN_ID="151669",
        CACHE_DIR="/cache/student",
        DRY_RUN="1",
    )
    return env


class TwoStageLauncherTest(unittest.TestCase):
    def test_dry_run_forwards_portable_environment(self):
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "--report-to", "none"],
            cwd=PROJECT_DIR,
            env=_base_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        command = completed.stdout
        self.assertIn("--teacher-draft-path /models/teacher/final", command)
        self.assertIn("--mask-token-id 151669", command)
        self.assertIn("--cache-dir /cache/student", command)
        self.assertIn("--nproc_per_node 2", command)
        self.assertIn("--tp-size 2", command)
        self.assertIn("--report-to none", command)

    def test_multinode_rejects_loopback_master(self):
        env = _base_env()
        env.update(NNODES="2", MASTER_ADDR="127.0.0.1")
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("reachable MASTER_ADDR", completed.stderr)

    def test_fresh_run_requires_teacher_checkpoint(self):
        env = _base_env()
        env.pop("TEACHER_DRAFT_PATH")
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("requires TEACHER_DRAFT_PATH", completed.stderr)

    def test_tp_draft_sharding_expands_target_batch(self):
        env = _base_env()
        env.update(
            TARGET_MODEL_BACKEND="sglang",
            SHARD_DRAFT_BY_TP="1",
            BATCH_SIZE="1",
        )
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        self.assertIn("--target-model-backend sglang", completed.stdout)
        self.assertIn("--batch-size 2", completed.stdout)
        self.assertIn("--shard-draft-by-tp", completed.stdout)

    def test_tp_draft_sharding_requires_sglang(self):
        env = _base_env()
        env["SHARD_DRAFT_BY_TP"] = "1"
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("requires TARGET_MODEL_BACKEND=sglang", completed.stderr)


if __name__ == "__main__":
    unittest.main()
