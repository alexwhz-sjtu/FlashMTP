import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts" / "config" / "launch_dflash_family.py"
SPEC = importlib.util.spec_from_file_location("launch_dflash_family", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.mark.parametrize(
    ("algorithm", "filename", "block_size"),
    [
        ("dflash", "dflash.yaml", "16"),
        ("dflash2", "dflash2.yaml", "16"),
        ("dspark", "dspark.yaml", "7"),
    ],
)
def test_repository_family_yaml_loads_expected_values(
    algorithm, filename, block_size
):
    path = ROOT / "scripts/config" / filename

    env, cli = launcher.load_config(path, algorithm)

    assert env["TARGET_MODEL"] == "/data/wanghanzhen/models/Qwen3-4B"
    assert env["NPROC_PER_NODE"] == "8"
    assert "wildchat_chinese_turns_le2" in env["TRAIN_DATA"]
    assert cli[cli.index("--max-length") + 1] == "4096"
    assert cli[cli.index("--block-size") + 1] == block_size
    assert "--enable-thinking" not in cli


def test_dflash2_yaml_keeps_selector_top_k_distinct_from_sampling_top_k():
    path = ROOT / "scripts/config/dflash2.yaml"

    _, cli = launcher.load_config(path, "dflash2")

    assert cli[cli.index("--selector-top-k") + 1] == "16"


def test_yaml_requires_exactly_one_data_source(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "algorithm: dflash2\n"
        "launcher:\n"
        "  target_model: /model\n"
        "  output_dir: /output\n",
        encoding="utf-8",
    )

    with pytest.raises(launcher.ConfigError, match="exactly one"):
        launcher.load_config(path, "dflash2")
