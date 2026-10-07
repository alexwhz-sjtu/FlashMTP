"""Reader and batching utilities for versioned ``regen_full`` tar caches."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

REGEN_FULL_SCHEMA_VERSION = 2


class HiddenCacheError(ValueError):
    """Raised when a regen_full cache cannot safely be used for training."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_model(expected: str, actual: str) -> bool:
    expected_path = Path(expected)
    actual_path = Path(actual)
    if expected_path.exists() and actual_path.exists():
        return expected_path.resolve() == actual_path.resolve()
    return expected.rstrip("/\\") == actual.rstrip("/\\")


def load_hidden_cache_manifest(
    root: str | Path,
    *,
    target_model: str,
    num_hidden_layers: int,
    hidden_size: int,
    required_layer_ids: set[int],
    max_length: int,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise HiddenCacheError(f"regen_full manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HiddenCacheError(f"cannot read regen_full manifest: {exc}") from exc

    if manifest.get("schema_version") != REGEN_FULL_SCHEMA_VERSION:
        raise HiddenCacheError(
            f"regen_full schema must be {REGEN_FULL_SCHEMA_VERSION}; regenerate the cache"
        )
    if manifest.get("save_mode") != "regen_full" or not manifest.get("complete"):
        raise HiddenCacheError("regen_full cache is not marked complete")
    if not _same_model(target_model, str(manifest.get("target_model", ""))):
        raise HiddenCacheError(
            "regen_full target model does not match training target: "
            f"cache={manifest.get('target_model')!r}, training={target_model!r}"
        )
    if int(manifest.get("num_hidden_layers", -1)) != int(num_hidden_layers):
        raise HiddenCacheError("regen_full target layer count does not match training")
    if int(manifest.get("hidden_size", -1)) != int(hidden_size):
        raise HiddenCacheError("regen_full hidden size does not match training")
    if int(max_length) > int(manifest.get("max_length", -1)):
        raise HiddenCacheError(
            f"training max_length={max_length} exceeds cache max_length="
            f"{manifest.get('max_length')}"
        )

    final_layer_id = int(manifest.get("final_norm_layer_id", -1))
    if final_layer_id != int(num_hidden_layers) - 1:
        raise HiddenCacheError("regen_full final-norm layer ID is invalid")
    stored = {int(value) for value in manifest.get("stored_hidden_layer_ids", [])}
    missing = sorted(set(required_layer_ids) - stored)
    if final_layer_id not in stored:
        missing = sorted(set(missing) | {final_layer_id})
    if missing:
        needed = sorted(set(required_layer_ids) | {final_layer_id})
        raise HiddenCacheError(
            f"regen_full is missing required hidden layers {missing}; "
            f"regenerate with --target-layer-ids {','.join(map(str, needed))}"
        )

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise HiddenCacheError("regen_full manifest contains no shards")
    expected_start = 0
    counted = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise HiddenCacheError("regen_full shard entry must be an object")
        path = root / str(shard.get("path", ""))
        if not path.is_file():
            raise HiddenCacheError(f"regen_full shard does not exist: {path}")
        start = int(shard.get("success_start", -1))
        end = int(shard.get("success_end", -1))
        count = int(shard.get("sample_count", -1))
        if start != expected_start or end != start + count - 1 or count <= 0:
            raise HiddenCacheError(f"regen_full shard range is invalid: {path}")
        source_indices = shard.get("source_indices", [])
        if len(source_indices) != count:
            raise HiddenCacheError(
                f"regen_full shard source index count is invalid: {path}"
            )
        if verify_checksums and _sha256(path) != shard.get("sha256"):
            raise HiddenCacheError(f"regen_full shard checksum mismatch: {path}")
        expected_start = end + 1
        counted += count
    if counted != int(manifest.get("successful_samples", -1)):
        raise HiddenCacheError("regen_full successful sample count is inconsistent")
    return manifest


class RegenFullDataset(Dataset):
    """Map-style dataset backed by regen_full tar members."""

    def __init__(
        self,
        root: str | Path,
        manifest: dict[str, Any],
        *,
        required_layer_ids: set[int],
        max_length: int,
    ) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.required_layer_ids = set(required_layer_ids)
        self.final_layer_id = int(manifest["final_norm_layer_id"])
        self.hidden_size = int(manifest["hidden_size"])
        self.max_length = int(max_length)
        self.entries: list[tuple[Path, str, int]] = []
        for shard in manifest["shards"]:
            path = self.root / shard["path"]
            start = int(shard["success_start"])
            for offset, source_index in enumerate(shard["source_indices"]):
                ordinal = start + offset
                self.entries.append(
                    (path, f"samples/{ordinal:012d}.pt", int(source_index))
                )
        self._archives: dict[Path, tarfile.TarFile] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_archives"] = {}
        return state

    def close(self) -> None:
        archives = getattr(self, "_archives", {})
        for archive in archives.values():
            archive.close()
        archives.clear()

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_payload(self, index: int) -> dict[str, Any]:
        path, member_name, source_index = self.entries[index]
        archive = self._archives.get(path)
        if archive is None:
            # Kept open per DataLoader worker and closed by close()/__del__.
            archive = tarfile.open(path, "r:*")  # noqa: SIM115
            self._archives[path] = archive
        try:
            member = archive.getmember(member_name)
            handle = archive.extractfile(member)
            if handle is None:
                raise HiddenCacheError(f"cannot read {member_name} from {path}")
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        except (KeyError, OSError, RuntimeError, tarfile.TarError) as exc:
            raise HiddenCacheError(
                f"cannot load {member_name} from {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise HiddenCacheError(f"sample {member_name} is not an object")
        if payload.get("schema_version") != REGEN_FULL_SCHEMA_VERSION:
            raise HiddenCacheError(f"sample {member_name} has an unsupported schema")
        if int(payload.get("sample_index", -1)) != source_index:
            raise HiddenCacheError(f"sample index mismatch in {member_name}")
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        payload = self._load_payload(index)
        input_ids = torch.as_tensor(payload.get("input_ids"))[: self.max_length]
        attention_mask = torch.as_tensor(payload.get("attention_mask"))[
            : self.max_length
        ]
        loss_mask = torch.as_tensor(payload.get("loss_mask"))[: self.max_length]
        length = input_ids.numel()
        if input_ids.ndim != 1 or length == 0:
            raise HiddenCacheError("cached input_ids must be a non-empty vector")
        if (
            attention_mask.shape != input_ids.shape
            or loss_mask.shape != input_ids.shape
        ):
            raise HiddenCacheError("cached token and mask shapes do not match")

        raw_hidden = payload.get("hidden_states")
        if not isinstance(raw_hidden, dict):
            raise HiddenCacheError("cached hidden_states must be a dictionary")
        hidden_states = {int(key): value for key, value in raw_hidden.items()}
        final_hidden = payload.get("final_hidden_state")
        if not isinstance(final_hidden, torch.Tensor):
            raise HiddenCacheError("cached final_hidden_state is missing")
        hidden_states[self.final_layer_id] = final_hidden
        selected = {}
        for layer_id in sorted(self.required_layer_ids | {self.final_layer_id}):
            value = hidden_states.get(layer_id)
            if not isinstance(value, torch.Tensor):
                raise HiddenCacheError(f"cached layer {layer_id} is missing")
            value = value[: self.max_length]
            if (
                value.ndim != 2
                or value.shape[0] != length
                or value.shape[1] != self.hidden_size
            ):
                raise HiddenCacheError(f"cached layer {layer_id} has invalid shape")
            if not torch.isfinite(value).all():
                raise HiddenCacheError(f"cached layer {layer_id} contains NaN or Inf")
            selected[layer_id] = value
        return {
            "sample_index": int(payload["sample_index"]),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "hidden_states": selected,
            "final_hidden_state": selected[self.final_layer_id],
            "final_norm_layer_id": self.final_layer_id,
        }


class RegenFullCollator:
    def __init__(self, pad_token_id: int, pad_to_length: int | None = None) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_length = pad_to_length

    @staticmethod
    def _pad_to(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.shape[1] == length:
            return tensor
        shape = (tensor.shape[0], length - tensor.shape[1], *tensor.shape[2:])
        return torch.cat((tensor, tensor.new_zeros(shape)), dim=1)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise HiddenCacheError("cannot collate an empty regen_full batch")
        sequence_length = max(item["input_ids"].numel() for item in features)
        if self.pad_to_length is not None:
            if sequence_length > self.pad_to_length:
                raise HiddenCacheError("cached sequence exceeds fixed padding length")
            sequence_length = self.pad_to_length

        def pad_vectors(key: str, padding_value: int) -> torch.Tensor:
            value = pad_sequence(
                [item[key] for item in features],
                batch_first=True,
                padding_value=padding_value,
            )
            return self._pad_to(value, sequence_length)

        layer_ids = sorted(features[0]["hidden_states"])
        if any(sorted(item["hidden_states"]) != layer_ids for item in features):
            raise HiddenCacheError("cached samples contain different hidden layers")
        final_layer_id = int(features[0]["final_norm_layer_id"])
        if any(int(item["final_norm_layer_id"]) != final_layer_id for item in features):
            raise HiddenCacheError("cached samples disagree on final-norm layer ID")
        hidden_states = {}
        for layer_id in layer_ids:
            value = pad_sequence(
                [item["hidden_states"][layer_id] for item in features],
                batch_first=True,
                padding_value=0.0,
            )
            hidden_states[layer_id] = self._pad_to(value, sequence_length)
        return {
            "sample_index": torch.tensor(
                [item["sample_index"] for item in features], dtype=torch.long
            ),
            "input_ids": pad_vectors("input_ids", self.pad_token_id),
            "attention_mask": pad_vectors("attention_mask", 0),
            "loss_mask": pad_vectors("loss_mask", 0),
            "hidden_states": hidden_states,
            "final_hidden_state": hidden_states[final_layer_id],
            "final_norm_layer_id": final_layer_id,
        }
