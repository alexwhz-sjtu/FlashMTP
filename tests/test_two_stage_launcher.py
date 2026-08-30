import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_DIR / "scripts" / "run_training_flashmtp_two_stage.sh"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "PET_NNODES",
        "PET_NODE_RANK",
        "PET_NPROC_PER_NODE",
        "PET_MASTER_ADDR",
        "PET_MASTER_PORT",
        "RUN_SUFFIX",
        "WANDB_RUN_NAME",
        "STAGE1_TRAIN_DATA_PATH",
        "STAGE2_TRAIN_DATA_PATH",
        "STAGE1_BUILD_DATASET_NUM_PROC",
        "STAGE2_BUILD_DATASET_NUM_PROC",
        "STUDENT_NUM_DRAFT_LAYERS",
        "NUM_DRAFT_LAYERS",
    ):
        env.pop(name, None)
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
        REPORT_TO="none",
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
        self.assertIn("--learning-rate 5e-4", command)
        self.assertIn("--warmup-ratio 0.04", command)
        self.assertIn("--stage1-kl-weight 1.0", command)
        self.assertNotIn("--stage2-learning-rate", command)
        self.assertIn("Ignoring STAGE2_LEARNING_RATE=2e-4", completed.stderr)

    def test_separate_stage_datasets_and_build_workers_are_forwarded(self):
        env = _base_env()
        env.pop("TRAIN_DATA_PATH")
        env.update(
            STAGE1_TRAIN_DATA_PATH="/data/distill_aug1.jsonl",
            STAGE2_TRAIN_DATA_PATH="/data/supervised_pb.jsonl",
            STAGE1_BUILD_DATASET_NUM_PROC="12",
            STAGE2_BUILD_DATASET_NUM_PROC="24",
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

        command = completed.stdout
        self.assertIn("Stage 1 dataset: /data/distill_aug1.jsonl", command)
        self.assertIn("Stage 2 dataset: /data/supervised_pb.jsonl", command)
        self.assertIn(
            "--stage1-train-data-path /data/distill_aug1.jsonl", command
        )
        self.assertIn(
            "--stage2-train-data-path /data/supervised_pb.jsonl", command
        )
        self.assertIn("--stage1-build-dataset-num-proc 12", command)
        self.assertIn("--stage2-build-dataset-num-proc 24", command)

    def test_shared_partial_forwards_student_depth(self):
        env = _base_env()
        env.update(
            STUDENT_INIT_MODE="shared_partial",
            STUDENT_NUM_DRAFT_LAYERS="3",
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

        self.assertIn("--student-init-mode shared_partial", completed.stdout)
        self.assertIn("--student-num-draft-layers 3", completed.stdout)

    def test_fresh_shared_partial_requires_student_depth(self):
        env = _base_env()
        env["STUDENT_INIT_MODE"] = "shared_partial"

        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("requires STUDENT_NUM_DRAFT_LAYERS", completed.stderr)

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

    def test_pet_node_defaults_match_v2_launcher(self):
        env = _base_env()
        for name in (
            "NPROC_PER_NODE",
            "NNODES",
            "NODE_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            env.pop(name, None)
        env.update(
            PET_NPROC_PER_NODE="8",
            PET_NNODES="3",
            PET_NODE_RANK="2",
            PET_MASTER_ADDR="10.2.3.4",
            PET_MASTER_PORT="29666",
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

        command = completed.stdout
        self.assertIn("nodes=3 rank=2 gpus/node=8 world=24 tp=2", command)
        self.assertIn("--nnodes 3 --node_rank 2", command)
        self.assertIn("--master_addr 10.2.3.4 --master_port 29666", command)

    def test_default_output_mask_and_wandb_identifiers_are_generated(self):
        env = _base_env()
        for name in (
            "OUTPUT_DIR",
            "CACHE_DIR",
            "MASK_TOKEN_ID",
            "REPORT_TO",
            "WANDB_PROJECT",
            "WANDB_NAME",
            "WANDB_RUN_ID",
        ):
            env.pop(name, None)
        env["TRAIN_DATA_PATH"] = "/data/aug1.jsonl"
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "--dt", "qz"],
            cwd=PROJECT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        command = completed.stdout
        self.assertIn("dt=qz", command)
        self.assertIn("MASK token id: 151669", command)
        self.assertIn("/cache/models/v23s_qz_", command)
        self.assertIn("W&B project: flashmtp-training-v2.3-student", command)
        self.assertIn("--mask-token-id 151669", command)
        self.assertIn("--report-to wandb", command)
        self.assertIn("--wandb-name v23s_qz_", command)
        self.assertIn("--wandb-run-id v23s-qz-", command)
        self.assertNotIn("--dt qz", command)

    def test_generated_paths_and_wandb_ids_are_node_rank_independent(self):
        env = _base_env()
        for name in (
            "OUTPUT_DIR",
            "CACHE_DIR",
            "REPORT_TO",
            "WANDB_PROJECT",
            "WANDB_NAME",
            "WANDB_RUN_ID",
        ):
            env.pop(name, None)
        env.update(NNODES="3", MASTER_ADDR="10.2.3.4")

        generated = []
        for node_rank in ("0", "2"):
            env["NODE_RANK"] = node_rank
            completed = subprocess.run(
                ["bash", str(LAUNCHER), "--dt", "qz"],
                cwd=PROJECT_DIR,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            stable_lines = tuple(
                line
                for line in completed.stdout.splitlines()
                if line.startswith(
                    (
                        "Output directory:",
                        "Dataset cache:",
                        "W&B name:",
                        "W&B run id:",
                    )
                )
            )
            generated.append(stable_lines)

        self.assertEqual(generated[0], generated[1])


if __name__ == "__main__":
    unittest.main()
