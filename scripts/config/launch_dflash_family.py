#!/usr/bin/env python3
"""Launch a DFlash-family training job from a YAML configuration."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


ALGORITHMS = {"dflash", "dflash2", "dspark"}
LAUNCHER_KEYS = {
    "target_model",
    "train_data",
    "train_hidden_states",
    "output_dir",
    "nproc_per_node",
    "master_port",
}


class ConfigError(ValueError):
    pass


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be a mapping")
    return value


def load_config(path: Path, algorithm: str) -> tuple[dict[str, str], list[str]]:
    if algorithm not in ALGORITHMS:
        raise ConfigError(f"unsupported algorithm: {algorithm}")
    if not path.is_file():
        raise ConfigError(f"config does not exist: {path}")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read {path}: {exc}") from exc
    config = _mapping(config, "config")

    configured_algorithm = config.get("algorithm", algorithm)
    if configured_algorithm != algorithm:
        raise ConfigError(
            f"config algorithm {configured_algorithm!r} does not match launcher {algorithm!r}"
        )

    launcher = _mapping(config.get("launcher"), "launcher")
    unknown = sorted(set(launcher) - LAUNCHER_KEYS)
    if unknown:
        raise ConfigError(f"unknown launcher keys: {unknown}")
    missing = [key for key in ("target_model", "output_dir") if not launcher.get(key)]
    if missing:
        raise ConfigError(f"missing launcher keys: {missing}")
    if bool(launcher.get("train_data")) == bool(launcher.get("train_hidden_states")):
        raise ConfigError(
            "set exactly one of launcher.train_data and launcher.train_hidden_states"
        )

    env = {
        "TARGET_MODEL": str(launcher["target_model"]),
        "OUTPUT_DIR": str(launcher["output_dir"]),
        "NPROC_PER_NODE": str(launcher.get("nproc_per_node", 8)),
        "MASTER_PORT": str(launcher.get("master_port", 29500)),
    }
    if launcher.get("train_data"):
        env["TRAIN_DATA"] = str(launcher["train_data"])
    else:
        env["TRAIN_HIDDEN_STATES"] = str(launcher["train_hidden_states"])

    arguments = _mapping(config.get("arguments", {}), "arguments")
    cli: list[str] = []
    for raw_name, value in arguments.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ConfigError("argument names must be non-empty strings")
        option = "--" + raw_name.strip().replace("_", "-")
        if value is None or value is False:
            continue
        cli.append(option)
        if value is True:
            continue
        if isinstance(value, list):
            if not value:
                raise ConfigError(f"{raw_name} list must not be empty")
            cli.extend(str(item) for item in value)
        elif isinstance(value, (str, int, float)):
            cli.append(str(value))
        else:
            raise ConfigError(
                f"argument {raw_name!r} has unsupported value type {type(value).__name__}"
            )
    return env, cli


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(
            "usage: launch_dflash_family.py ALGORITHM CONFIG.yaml", file=sys.stderr
        )
        return 2
    algorithm, config_value = argv
    project_dir = Path(__file__).resolve().parents[2]
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    try:
        configured_env, cli = load_config(config_path, algorithm)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.update(configured_env)
    venv_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    launcher = project_dir / "scripts/dflash_family/run_training_dflash_family.sh"
    os.execvpe(
        "bash",
        ["bash", str(launcher), algorithm, *cli],
        env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
