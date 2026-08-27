from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_foreign_site_packages(entry: str, interpreter_prefix: Path) -> bool:
    """Return whether a sys.path entry injects another Python environment."""
    if not entry:
        return False
    try:
        path = Path(entry).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    if "site-packages" not in path.parts:
        return False
    try:
        path.relative_to(interpreter_prefix)
    except ValueError:
        return True
    return False


def _prepare_environment(argv: list[str]) -> list[str]:
    rewritten = list(argv)
    found = False
    for index, value in enumerate(rewritten):
        if value.startswith("--speculative-algorithm="):
            found = True
            algorithm = value.split("=", 1)[1]
            if algorithm.upper() != "FLASHMTP":
                raise ValueError(
                    "The FlashMTP launcher requires --speculative-algorithm FLASHMTP."
                )
            rewritten[index] = "--speculative-algorithm=DFLASH"
            continue
        if value == "--speculative-algorithm":
            if index + 1 >= len(rewritten):
                raise ValueError("--speculative-algorithm requires FLASHMTP.")
            found = True
            if rewritten[index + 1].upper() != "FLASHMTP":
                raise ValueError(
                    "The FlashMTP launcher requires --speculative-algorithm FLASHMTP."
                )
            rewritten[index + 1] = "DFLASH"
    if not found and not any(value in ("-h", "--help") for value in rewritten):
        raise ValueError("Missing --speculative-algorithm FLASHMTP.")

    os.environ["SGLANG_FLASHMTP_ACTIVE"] = "1"
    overlap = "--disable-overlap-schedule" not in rewritten
    os.environ["SGLANG_ENABLE_SPEC_V2"] = "True" if overlap else "False"
    os.environ["SGLANG_ENABLE_DFLASH_SPEC_V2"] = "True" if overlap else "False"

    # Python multiprocessing uses fresh interpreters. sitecustomize makes every
    # child install the same adapter without changing the conda environment.
    bootstrap_dir = Path(__file__).resolve().parent / "_bootstrap"
    project_root = Path(__file__).resolve().parents[2]
    adapter_entries = [str(bootstrap_dir), str(project_root)]

    # Do not pass an activated uv/conda environment's PYTHONPATH to SGLang
    # children.  In particular, a foreign ``.../site-packages`` entry can make
    # a process launched by mtp-sglang import SGLang from the older mtp env.
    # sys.path is already initialized for this process, so sanitize both it and
    # the environment inherited by multiprocessing children.
    interpreter_prefix = Path(sys.prefix).resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if not _is_foreign_site_packages(entry, interpreter_prefix)
    ]
    for entry in reversed(adapter_entries):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    os.environ["PYTHONPATH"] = os.pathsep.join(adapter_entries)
    return rewritten


def main() -> None:
    argv = _prepare_environment(sys.argv[1:])
    from .bootstrap import install

    install()
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(argv)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
